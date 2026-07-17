# crypto-alpha — BTC/ETH 多专家集成：方向概率 + 风控

面向 **BTC / ETH** 的量化研究工程：输出 **做多 / 做空 / 观望 + 校准后盈利概率 + ATR 止损止盈 + 建议仓位**。

核心不是「预测涨跌」，而是 **三重障碍 + 元标签 + 多专家 Stacking + 防泄漏验证 + 组合级回测**。

> ⚠️ 仅供研究学习。须经 CPCV/PBO 与纸面交易验证后，才可考虑极小资金实盘。  
> 📖 **完整架构 / 技术细节 / 使用说明**：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 架构一览

```
行情(+衍生品) + 辅周期(4h/1d MTF) + 新闻
        ↓
特征(技术指标 / FFD / MTF as-of / 新闻数值)
        ↓
CUSUM → 三重障碍(ATR) → 元标签 + 样本权重
        ↓
┌─ GBDT (LightGBM)          本机 CPU ──┐
├─ DeepTS (小 PatchTST)     本机 CPU/GPU┤→ Stacking(nested OOF)
├─ TSFM (Chronos+浅头)[可选] 本机 ──────┤→ 校准 + 保形
└─ LLM (Qwen QLoRA)[可选]   需大卡单独训┘
        ↓
组合级含成本回测 / CPCV·DSR·PBO
        ↓
结构化决策 → HTML / Canvas / Telegram
```

默认启用：`gbdt` + `deep_ts`（本机可训完）。详见 [ARCHITECTURE §8 / §16.8](docs/ARCHITECTURE.md)。

---

## 目录

```
config/config.yaml       全局配置（唯一参数入口）
docs/ARCHITECTURE.md     架构 + 实现 + 使用手册
src/crypto_alpha/        主包
scripts/                 01–11 分阶段脚本 + train_llm_qlora.py
tests/                   smoke / leakage / design_fixes / mtf
data/                    运行时缓存（gitignore）
artifacts/               模型与面板产出
```

---

## 安装

```powershell
pip install -e .                 # 最小：GBDT + 集成 + 回测
pip install torch                # DeepTS（按本机 CUDA 选官方源）
pip install -e ".[data]"         # ccxt 真实行情
pip install -e ".[tsfm]"         # Chronos
pip install -e ".[llm]"          # QLoRA 依赖
```

---

## 快速开始

### 冒烟（强制合成，无需网络）

```powershell
python tests/test_smoke.py
pytest -q
```

### 研究主路径（默认真实数据）

配置默认：`data.use_synthetic: false`、`news.use_synthetic: false`。无网络时主周期会 warn 后降级合成。

```powershell
pip install -e ".[data]"
python scripts/01_fetch_data.py              # 主+辅周期缓存
# 可选：python scripts/09_backfill_news.py 后设 news.use_history: true
python scripts/04_train_and_backtest.py      # 训练+组合回测+决策
python scripts/05_cpcv_report.py             # 发布前严谨评估
python scripts/10_run_all.py --cpcv --open   # 联跑 + HTML
python scripts/11_make_canvas.py             # Cursor 交互面板
```

### 实时服务

```powershell
python scripts/07_serve.py --once
python scripts/07_serve.py --loop
```

### LLM（可选，需大显存）

```powershell
python scripts/train_llm_qlora.py --dry-run
python scripts/train_llm_qlora.py
# 再在 config.experts.enabled 中加入 "llm"
```

---

## 决策输出示例

```json
{
  "signal": "LONG",
  "win_probability": 0.63,
  "entry_price": 65000.0,
  "suggested_position_pct": 0.12,
  "stop_loss": 64100.0,
  "take_profit": 65900.0,
  "confident": true
}
```

`HOLD` 时不输出止损止盈（避免误挂单）。完整字段与口径见 [ARCHITECTURE §12 / §16.6](docs/ARCHITECTURE.md)。

---

## 关键配置（摘录）

| 项 | 当前默认 |
|----|----------|
| 专家 | `["gbdt", "deep_ts"]` |
| 数据 | 真实行情；MTF `4h`/`1d` |
| 标注障碍 | `barrier_vol: atr`，与 decide 共用 `pt_sl` |
| 回测 | `portfolio_mode: true`（组合资金占用） |
| LLM | `Qwen2.5-32B-Instruct`（未默认启用） |

全部键说明见 [ARCHITECTURE §14](docs/ARCHITECTURE.md) 与 `config/config.yaml`。

---

## 脚本速查

| 脚本 | 作用 |
|------|------|
| `01`–`03` | 数据 / 特征 / 标注 |
| `04` | 训练+回测+决策 |
| `05` | CPCV / DSR / PBO |
| `06`–`07` | 单次决策 / 常驻服务 |
| `08`–`09` | 新闻采集 / 历史回填 |
| `10`–`11` | HTML 联跑 / Canvas |
| `train_llm_qlora.py` | LLM 唯一独立训练入口 |

---

## 许可与免责

研究用途，非投资建议。市场有风险，回测不等于未来表现。
