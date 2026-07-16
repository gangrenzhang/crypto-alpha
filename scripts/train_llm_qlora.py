"""专家4 训练脚本: 在单张 80GB 卡上用 QLoRA(4-bit) 微调 Qwen2.5-72B。

目标: 监督微调(SFT) 让模型对 "指标摘要(+可选新闻) + 方向" 输出 verbalizer 答案
"1(会盈利)/0(不会盈利)"; 推理时读取该位置 token 概率得到 P(盈利)(见 experts/llm.py)。

数据: 复用 pipeline.prepare_dataset 产出的三重障碍元标签(严格无泄漏)。
      训练/验证按时间切分(最后 val_frac 作验证), 杜绝未来信息泄漏。
损失: 仅对答案 token 计损, 并可用样本唯一性权重加权。

依赖: pip install -e ".[llm]"   (transformers peft bitsandbytes accelerate datasets)
运行: python scripts/train_llm_qlora.py            # 用 config.yaml 的 experts.llm
      python scripts/train_llm_qlora.py --dry-run  # 只构造/统计数据集, 不加载大模型

显存参考(Qwen2.5-72B, 4-bit, lora_r=16, seq=1024, bs=4*grad8):约 55-70GB, ≤100GB。
"""
import _bootstrap  # noqa: F401

import argparse

import numpy as np

from crypto_alpha.config import Config, set_global_seed


# --------------------------------------------------------------------------
# 数据集构造 (轻依赖, 可 --dry-run 独立验证)
# --------------------------------------------------------------------------
def build_sft_records(cfg: Config, symbols=None) -> list[dict]:
    """从元标签数据集构造 SFT 样本(messages + 答案 + 权重), 跨币种按时间排序。"""
    from crypto_alpha.pipeline import prepare_dataset
    from crypto_alpha.experts.llm import build_messages
    from crypto_alpha.data import load_news_for_events

    records: list[dict] = []
    for symbol in symbols or cfg["data"]["symbols"]:
        ds = prepare_dataset(cfg, symbol)
        news = load_news_for_events(cfg, symbol, ds.X.index)  # 无泄漏 as-of 对齐
        for i, ts in enumerate(ds.X.index):
            row = ds.panel.loc[ts]
            side = int(ds.X["side"].iloc[i])
            text = news.get(ts, "") if news else ""
            records.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "messages": build_messages(row, side, text),
                    "answer": str(int(ds.y[i])),
                    "weight": float(ds.sample_weight[i]),
                }
            )
    records.sort(key=lambda r: r["timestamp"])
    return records


def chronological_split(records: list[dict], val_frac: float):
    n_val = int(len(records) * val_frac)
    return records[: len(records) - n_val], records[len(records) - n_val :]


# --------------------------------------------------------------------------
# Tokenization + 训练 (重依赖, 需 GPU)
# --------------------------------------------------------------------------
def _encode(tok, rec: dict, max_len: int) -> dict:
    """把一条记录编码为 input_ids/labels; 仅答案 token 参与损失。"""
    prompt_ids = tok.apply_chat_template(
        rec["messages"], add_generation_prompt=True, tokenize=True
    )
    answer_ids = tok.encode(rec["answer"], add_special_tokens=False) + [tok.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels, "weight": rec["weight"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只构造数据集并统计, 不加载模型")
    args = parser.parse_args()

    cfg = Config.load()
    set_global_seed(cfg.seed)
    lc = cfg["experts"]["llm"]

    print("[1/5] 构造 SFT 数据集 ...")
    records = build_sft_records(cfg)
    train_recs, val_recs = chronological_split(records, float(lc.get("val_frac", 0.15)))
    pos = np.mean([int(r["answer"]) for r in records])
    print(f"  样本总数={len(records)} (train={len(train_recs)}, val={len(val_recs)}), 正类占比={pos:.3f}")
    print(f"  示例 prompt:\n---\n{records[0]['messages'][1]['content']}\n答案={records[0]['answer']}\n---")

    if args.dry_run:
        print("[dry-run] 数据集构造完成, 未加载模型。")
        return

    # ---- 重依赖 ----
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = lc.get("model_name", "Qwen/Qwen2.5-72B-Instruct")
    max_len = int(lc.get("max_seq_len", 1024))

    print(f"[2/5] 加载分词器与 4-bit 模型: {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(lc.get("load_in_4bit", True)),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(lc.get("gradient_checkpointing", True))
    )
    if bool(lc.get("gradient_checkpointing", True)):
        model.config.use_cache = False

    print("[3/5] 注入 LoRA 适配器 ...")
    lora = LoraConfig(
        r=int(lc.get("lora_r", 16)),
        lora_alpha=int(lc.get("lora_alpha", 32)),
        lora_dropout=float(lc.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lc.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("[4/5] 编码数据集 ...")
    use_w = bool(lc.get("use_sample_weight", True))
    train_enc = [_encode(tok, r, max_len) for r in train_recs]
    val_enc = [_encode(tok, r, max_len) for r in val_recs]

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    def collate(batch):
        maxb = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        input_ids, labels, attn, weights = [], [], [], []
        for b in batch:
            pad = maxb - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
            weights.append(b["weight"] if use_w else 1.0)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weight": torch.tensor(weights, dtype=torch.float),
        }

    class WeightedSFTTrainer(Trainer):
        """仅对答案 token 计损, 并按样本权重加权(唯一性权重)。"""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("weight", None)
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            vocab = shift_logits.size(-1)
            ce = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, vocab), shift_labels.view(-1),
                reduction="none", ignore_index=-100,
            ).view(shift_labels.size())
            mask = (shift_labels != -100).float()
            per_sample = (ce * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            if weights is not None:
                per_sample = per_sample * weights.to(per_sample.device)
            loss = per_sample.mean()
            return (loss, outputs) if return_outputs else loss

    out_dir = str((cfg.root / lc.get("adapter_path", "artifacts/qwen_qlora_adapter")))
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(lc.get("epochs", 2)),
        per_device_train_batch_size=int(lc.get("per_device_batch_size", 4)),
        gradient_accumulation_steps=int(lc.get("grad_accum", 8)),
        learning_rate=float(lc.get("lr", 1e-4)),
        warmup_ratio=float(lc.get("warmup_ratio", 0.03)),
        weight_decay=float(lc.get("weight_decay", 0.0)),
        lr_scheduler_type="cosine",
        bf16=bool(lc.get("bf16", True)),
        gradient_checkpointing=bool(lc.get("gradient_checkpointing", True)),
        logging_steps=int(lc.get("logging_steps", 20)),
        save_steps=int(lc.get("save_steps", 200)),
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=int(lc.get("save_steps", 200)),
        report_to=[],
        optim="paged_adamw_8bit",
    )

    trainer = WeightedSFTTrainer(
        model=model,
        args=targs,
        train_dataset=SFTDataset(train_enc),
        eval_dataset=SFTDataset(val_enc),
        data_collator=collate,
    )

    print("[5/5] 开始 QLoRA 微调 ...")
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[ok] 适配器已保存 -> {out_dir}")
    print("     在 config.yaml 的 experts.enabled 加入 'llm' 即可让集成使用该专家。")


if __name__ == "__main__":
    main()
