# crypto-alpha — BTC/ETH 多专家集成 方向概率 + 风控 预测系统

一个面向 **BTC / ETH** 的量化研究工程骨架, 目标是输出 **做多/做空/观望信号 + 校准后的盈利概率 + 波动率自适应止损/止盈 + 建议仓位**。核心思想不是"预测涨跌", 而是 **三重障碍标注 + 元标签 + 多专家集成 + 极致严谨的防泄漏验证**。

> ⚠️ 本项目用于研究与学习。金融市场可预测 edge 小且不稳定, 任何结果都必须经 CPCV/PBO 检验并纸面交易验证后, 才可考虑实盘小资金。

> 📖 **完整架构文档**(每个技术选型的作用/优势 + 详细使用说明): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 一、整体架构

```
多模态数据 → 多视角特征(含分数阶差分) → 三重障碍标注 + 元标签
      │
      ├── 专家1 GBDT (LightGBM)            —— 表格非线性交互, 压舱石
      ├── 专家2 时序基础模型 (Chronos/TimesFM) —— 纯数值时序表征 [可插拔]
      ├── 专家3 深度时序 (PatchTST 风格)    —— 注意力时序
      └── 专家4 LLM (Qwen2.5-72B QLoRA)    —— 新闻/事件推理 [可插拔]
            │
      Stacking 元学习器 → 概率校准 → 保形预测(可弃权)
            │
      Purged K-Fold / CPCV 无泄漏验证 + 去偏夏普(DSR) + 过拟合概率(PBO)
            │
      分数 Kelly 仓位 + ATR 止损 → 结构化交易决策
```

## 二、目录结构

```
config/config.yaml         全局配置(唯一参数入口)
src/crypto_alpha/
  data/         数据采集(ccxt) + 合成数据兜底 + parquet
  features/     技术指标 + 分数阶差分(FFD) + 特征装配
  labeling/     三重障碍 + 元标签 + 样本唯一性权重
  validation/   Purged K-Fold + CPCV
  experts/      四专家 + 统一接口(BaseExpert)
  ensemble/     Stacking 元学习器
  calibration/  Isotonic/Platt 校准 + 保形预测
  backtest/     含成本回测 + DSR + PBO
  risk/         分数 Kelly 仓位 + ATR 止损 + 决策
  pipeline/     端到端编排 + CPCV 评估
scripts/        01~11 分阶段脚本 + train_llm_qlora(…历史回填/一键联跑/HTML面板/交互Canvas)
tests/          主干冒烟测试
```

## 三、安装

```powershell
# 1) 最小依赖(主干: GBDT + 集成 + 回测, 可跑通全流程)
pip install -e .

# 2) 深度时序专家(PatchTST) 需要 torch(按你的 CUDA 版本安装)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3) 真实数据 / TSFM / LLM 专家(按需)
pip install -e ".[data]"      # ccxt 拉真实行情
pip install -e ".[tsfm]"      # Chronos / TimesFM
pip install -e ".[llm]"       # Qwen2.5-72B QLoRA
```

## 四、快速开始(合成数据, 无需网络/GPU)

```powershell
python tests/test_smoke.py                 # 30 秒内跑通主干
python scripts/01_fetch_data.py            # 数据(默认合成)
python scripts/02_build_features.py        # 特征
python scripts/03_label.py                 # 标注概览
python scripts/04_train_and_backtest.py    # 训练+校准+回测+决策+净值图
python scripts/05_cpcv_report.py           # CPCV: 夏普分布 + DSR + PBO
python scripts/06_decide.py                # 输出最新结构化交易决策(单次)
python scripts/07_serve.py --once          # 实时服务: 跑一轮并播报
python scripts/07_serve.py --loop          # 实时服务: 常驻定时轮询 + 播报
python scripts/08_fetch_news.py            # 采集/合成多源新闻(供 LLM 专家)
python scripts/09_backfill_news.py         # 历史新闻回填(多年期回测语料)
python scripts/10_run_all.py               # ★一键全专家联跑 + 生成 HTML 结果面板
python scripts/11_make_canvas.py           # ★把结果生成可交互 Cursor Canvas 面板
```

默认已是真实数据模式(`data.use_synthetic: false` / `news.use_synthetic: false`)。离线冒烟请在测试里显式打开合成, 或临时改配置。依赖: `pip install -e ".[data]"`。

### 真实 OHLCV 多年回测(缓存 + 增量更新)

真实模式下, `load_symbol_data` 会:
1. 用 ccxt **分页拉取** `data.since`(默认 2020-01-01)至今的多年 K 线;
2. **落盘缓存** `data/raw/<SYMBOL>.parquet`(`data.cache: true`), 之后复用避免重复拉取;
3. **增量更新**(`data.incremental_update: true`): 每次只拉缓存最后一根 bar 之后的新数据并合并。

配合 `news.use_history: true`(见"历史新闻回填"), 即得到**价格 + 新闻都覆盖多年**的真实回测。网络/依赖缺失时自动降级合成数据, 不阻断流程。合成模式为确定性重建、不缓存(尊重 `synthetic_bars` 改动)。

### ★ 一键全专家联跑面板 (`scripts/10_run_all.py`)

一条命令跑完整主干(数据→特征含新闻→三重障碍标注→**全专家 Stacking**→校准→含成本回测→最新决策), 并产出**自包含 HTML 结果面板**(离线可看)。

- **自动优雅降级**: 逐个探测专家运行时可用性 —— `gbdt`(lightgbm)、`deep_ts`(torch)、`tsfm`(缺 chronos/timesfm 自动回退 naive)、`llm`(需 transformers+CUDA+微调 adapter)。不可用者跳过并在面板标注原因(如"无 CUDA GPU")。
- **结果面板**(`artifacts/dashboard.html`): 每币种含①决策卡(方向/概率/入场/止损止盈/仓位)②集成+回测 KPI(AUC/Brier/Sharpe/回撤/Calmar/胜率)③**专家 vs 集成对比表**(高亮最优 AUC)④OOF 净值曲线⑤可选 CPCV(DSR/PBO + 各配置夏普)。同时落 `artifacts/run_all_summary.json`。

```powershell
python scripts/10_run_all.py                       # 全专家(自动降级) + 面板
python scripts/10_run_all.py --cpcv --open         # 叠加 CPCV 严谨评估, 生成后打开
python scripts/10_run_all.py --experts gbdt tsfm   # 只跑指定专家(快)
python scripts/10_run_all.py --symbols BTC/USDT
```

实测(本机无 GPU, 合成数据): 自动跑 `gbdt+deep_ts+tsfm(naive)`、跳过 `llm`, 面板正确渲染两币种决策卡/对比表/内嵌净值图, 单文件 ~86KB。上 GPU 机器并备好 QLoRA adapter 后, `llm` 会自动纳入四专家联跑。

### ★ 可交互 Cursor Canvas 面板 (`scripts/11_make_canvas.py`)

在 Cursor 里, 除静态 HTML 外还可生成**可并排打开的实时交互面板**。脚本读取 `artifacts/run_all_summary.json`(`10_run_all` 已把降采样净值曲线一并写入), 把数据**内联**进单文件 `crypto-alpha-dashboard.canvas.tsx` 并写到 Cursor 托管的 `canvases/` 目录:

- **交互**: 币种切换(BTC/ETH)、指标切换(AUC / 准确率 / Brier)、双币净值对比开关(状态持久化)。
- **视图**: 最新决策卡、含成本回测 KPI、"专家 vs 集成"判别力对比图(带随机基线参考线)、OOF 净值曲线(本金=1.0 参考线)、各专家 OOF 指标表(高亮最优 AUC)。
- **刷新**: 重跑 `10_run_all.py` 后再跑 `11_make_canvas.py` 即更新面板数据(数据驱动、可复现)。

```powershell
python scripts/10_run_all.py          # 先产出 run_all_summary.json(含净值曲线)
python scripts/11_make_canvas.py      # 生成/刷新 canvas, 在 Cursor 点击路径并排打开
```

## 五、如何使用(问什么答什么)

系统不是聊天机器人。它在每根 K 线收盘后, 用最新市场状态自动产出一条决策:

```json
{
  "symbol": "BTC/USDT",
  "signal": "LONG",              // 做多 / 做空 / 观望(HOLD)
  "win_probability": 0.63,       // 校准后的盈利概率
  "entry_price": 64200.0,
  "stop_loss": 62850.0,          // ATR 自适应止损
  "take_profit": 66600.0,
  "suggested_position_pct": 0.12 // 半 Kelly 建议仓位
}
```

你(或自动化程序)据此挂单/复核。

### 实时播报服务 (`scripts/07_serve.py`)

服务会"训练一次 → 周期性拉最新数据出决策 → 去重 → 播报", 并按 `serve.retrain_every_cycles` 周期自动重训以应对概念漂移。对最新一根 bar 出决策**不需要标签**, 因此复用历史训练好的集成+校准器, 每轮只重算最新特征并推理。

- **默认只推 LONG/SHORT**, 观望(HOLD) 静默(可 `serve.notify_hold: true` 打开)。
- **去重**: 同一币种相同信号不重复推送(`serve.dedupe`)。
- **调度**: `--once` 适合 cron / Windows 任务计划; `--loop` 常驻, 间隔由 `serve.poll_seconds` 控制(建议对齐主时间框架)。

**接入 Telegram**(未配置则自动回退控制台打印):

```powershell
# 1) config.yaml: serve.telegram.enabled: true
# 2) 设置环境变量(勿把 token 写进代码/配置)
$env:TELEGRAM_BOT_TOKEN="123456:ABC..."
$env:TELEGRAM_CHAT_ID="987654321"
python scripts/07_serve.py --loop
```

## 六、逐步接入高阶专家

- **深度时序(专家3)**: 安装 torch 后, 在 `config.yaml` 的 `experts.enabled` 加入 `"deep_ts"`。
- **时序基础模型(专家2)**: `pip install -e ".[tsfm]"`, `enabled` 加 `"tsfm"`。零样本即可用, 也可微调。
  - **新闻协变量**: TSFM 预测分与新闻协变量(`news_sentiment` 等)经**协变量融合头**(logistic/gbdt)得到概率(`experts.tsfm.covariate_cols: auto`, `head: logistic`)。注意 **Chronos 是单变量模型不吃原生协变量**, 故用融合头注入; **TimesFM 2.0+** 可走原生协变量路径。
  - 无 chronos/timesfm 依赖时自动回退到内置 `naive` 动量基线, 便于离线打通链路。
- **LLM(专家4)**: `pip install -e ".[llm]"`, 用 `scripts/train_llm_qlora.py` 在单张 80GB 卡上 QLoRA 微调 Qwen2.5-72B(4-bit, ≤100GB 显存):

```powershell
python scripts/train_llm_qlora.py --dry-run   # 先验证数据集构造(无需GPU)
python scripts/train_llm_qlora.py             # 正式微调(需 80GB GPU)
```

  微调产物(LoRA adapter)保存到 `config.yaml` 的 `experts.llm.adapter_path`; 之后在
  `experts.enabled` 加 `"llm"` 即可让集成加载该专家。概率来源为 verbalizer(读取答案
  token "1" vs "0" 的 softmax 概率), 训练目标与推理口径一致, 且可被下游 isotonic 校准。
  可选: 在 `experts.llm.news_path` 指定 "时间->新闻文本" 的 parquet, 训练/推理会 as-of
  对齐(只用过去的新闻), 从而让 LLM 融合事件信息。

## 六点五、新闻数据源(喂给 LLM 专家)

LLM 专家的独特价值在于**读懂新闻/事件**。新闻质量直接决定这块 alpha 的成色, 因此对数据源做了多维度设计。

### 为什么要接多个来源

单一来源有致命缺陷: 覆盖不全、单点偏见、可被操纵(加密媒体存在付费/喊单稿)、单点故障。多源的价值在于**交叉互证**——同一事件被多个独立、权威来源报道时可信度更高; 同时多源提高覆盖率与冗余。代价是需要**去重**(同一新闻常被转载)与**权威加权**(避免小道消息与权威公告等权重)。

### 权威性的多维评估(本项目落地为 `tier` 分层)

| 维度 | 说明 | 落地方式 |
|---|---|---|
| 权威/可信度 | 官方监管/交易所公告 > 一线财经(Reuters) > 一线加密媒体(CoinDesk) > 聚合/社媒 | `tier` 1→4, 映射为 `tier_weights` |
| 时效/延迟 | 越快越有交易价值 | GDELT/交易所公告延迟低 |
| 时间戳准确性 | **防前视偏差的命门** | 统一取 `published_at`(UTC) |
| 覆盖面 | 主题/资产广度 | 多源合并 + 市场级关键词 |
| 偏见/操纵 | 付费稿、喊单 | 低 tier 降权, 依赖互证 |
| 同质化 | 转载重复 | 标题 Jaccard 去重 |
| 成本/许可 | API 额度/合规 | 免费源优先, key 走环境变量 |

### 已内置的来源(分层)

- **T1 权威**: SEC 官方 RSS(监管)。
- **T2 一线/聚合**: CoinDesk RSS、CryptoCompare News、CryptoPanic(带来源与投票)、GDELT(全球、时间戳可靠)。
- **T3 加密媒体**: Cointelegraph RSS。
- **T4 社媒/弱源**: 仅作情绪参考, 低权重。

聚合逻辑: 去重归并 → 记录**互证来源数(corroboration)** → `authority = tier_weight × (1+0.3×(证数−1))` → 每个时间桶按权威加权计算情绪、取 Top-K 头条生成摘要。

### 防泄漏(强制)

只使用 `published_at + buffer_minutes ≤ bar_time` 的新闻(新闻需传播时间才可交易)。对齐用 `merge_asof(backward)`, 并由 `tests/test_leakage.py::test_news_asof_no_future_leak` 验证(事件发生在新闻发布+缓冲之前时取不到该新闻)。

> ⚠️ **合成新闻守卫**: 合成新闻的情绪由**未来收益**构造(仅用于纯合成链路跑通)。若 `data.use_synthetic=false` 而 `news.use_synthetic=true`, 系统会**直接报错**, 防止把未来信息注入真实行情(前视泄漏)。真实模式请用 `news.use_history=true`(先跑 `09_backfill_news`)或配置真实新闻源。

### 用法

```powershell
# 合成新闻(默认, 离线可跑):
python scripts/08_fetch_news.py
# 接真实来源: config.yaml 设 news.use_synthetic: false, 并按需配置 key:
$env:CRYPTOPANIC_KEY="..."; $env:CRYPTOCOMPARE_KEY="..."
python scripts/08_fetch_news.py
```

产物 `data/news/<SYMBOL>.parquet` 会被 QLoRA 训练脚本与 LLM 专家推理**自动 as-of 对齐**使用。想扩展来源, 在 `config.news.sources` 增加一条(支持 `rss`/`cryptopanic`/`cryptocompare`/`gdelt`)即可。

### 历史新闻回填(多年期回测)`scripts/09_backfill_news.py`

`08` 只抓"当前快照", 无法支撑几年期回测。`09` 负责**把多年历史新闻分页/分窗抓下来**, 追加去重进**原始语料库**, 再重建各币种面板:

- **原始库与面板分离**: `data/news_raw/corpus.parquet`(一行=一篇原始报道)+ `backfill_state.json`(续跑检查点)。抓取**可增量、可续跑**(追加天然幂等: 按 `source+title+published_at` 去重); 面板可随时用当前库重建。
- **分页/分窗抓取**:
  - `cryptocompare`: 用 `lTs` 参数向历史方向**逐页翻**(每页约 50 条)直到覆盖 `start`, 需 `$env:CRYPTOCOMPARE_KEY`。
  - `gdelt`: 把 `[start,end]` 切成 `window_days` 窗口, 每窗最多 250 条**逐窗抓**, 免费无需 key。
  - `synthetic`: 离线兜底, 在区间内生成多年合成语料(无网络即可打通全链路)。
- **限速友好**: `rate_limit_sec` 控制请求间隔, 任何来源失败不阻断其他来源。
- **时间窗互证**: `dedup_window_hours`(默认 48h)只归并**窗内**多源报道 —— 互证是"多源近乎同时报道同一事件", 跨月同名标题**不合并**(否则多年语料会塌缩到最早的少数簇, 破坏历史回测)。

```powershell
# 离线合成多年语料(打通链路):
python scripts/09_backfill_news.py --start 2022-01-01 --end 2024-12-31 --providers synthetic
# 接真实历史源(逐年回填):
$env:CRYPTOCOMPARE_KEY="..."
python scripts/09_backfill_news.py --start 2021-01-01 --providers gdelt cryptocompare
```

回填并重建面板后, 把 `config.news.use_history: true` 打开, 后续所有专家(GBDT/深度时序/TSFM/LLM)与回测都会消费这份多年语料。实测合成回填 8904 条 → 面板覆盖 2022-01~2024-12 共 ~2355 桶, as-of 对齐在多年月度网格上 35/35 全命中。

### 情绪打分: CryptoBERT / FinBERT

情绪打分是可插拔后端(`config.news.sentiment.backend`):

- `lexicon`(默认, 零依赖兜底)、`cryptobert`(英文加密, `ElKulako/cryptobert`)、`finbert`(英文金融)、`chinese`(中文金融, `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`)、**`multilingual`(中英双模型融合)**。
- Transformer 后端输出**概率加权有符分数** `score = P(看多) − P(看空)`(而非 argmax); **通用标签映射**自动识别 bull/positive/正面、bear/negative/负面; 对 `LABEL_0/1` 这类无语义标签支持显式 `label_map`, 并对二/三分类提供有序兜底。
- **中英双模型融合(`multilingual`)**: 按文本语言**路由**到母语模型(英→CryptoBERT, 中→中文模型), 混合语言按 `fusion_weights` 加权融合; 各语言独立回退词典。
- **批量推理 + 磁盘缓存**(`artifacts/sentiment_cache.json`) + 设备自适应(cuda/cpu)。缺依赖/加载失败**优雅回退词典**。

```powershell
pip install -e ".[sentiment]"     # transformers + torch
# 纯英文: news.sentiment.backend: cryptobert
# 中英混合(推荐): news.sentiment.backend: multilingual  (有 GPU 时 device: cuda)
python scripts/08_fetch_news.py
```

实测(CPU): 中文 "监管...交易所被起诉...暴跌" → **−0.94**; 英文 "ETF approved, inflows" → **+0.55**; 混合文本自动融合两模型。中文母语模型对中文的判别力显著强于用英文模型硬跑中文, 这正是接入双模型的价值。

### 新闻作为全体专家的共享特征

新闻不止喂 LLM: 其数值信号也以**无泄漏 as-of + 时间衰减**并入特征面板, 供 GBDT/深度时序共享(`news.as_feature: true`)。产出列: `news_sentiment`(衰减后)、`news_sentiment_raw`、`news_corroboration`、`news_n_items`、`news_max_authority`、`news_age_hours`、`has_recent_news`、`news_sent_ema`。

防泄漏三重保证:
1. 新闻桶用**桶末(可用时刻)**标记(而非桶左沿), 根除同期前视;
2. 再加 `buffer_minutes` 传播缓冲;
3. `merge_asof(backward)` 只对齐"可用时刻 ≤ bar 时间"的最新新闻。

时间衰减: 影响按 `feature_halflife_hours` 指数衰减, 超过 `feature_ttl_hours` 归零(避免陈旧新闻污染当前判断)。

## 七、关键方法出处

三重障碍 / 元标签 / 样本唯一性 / 分数阶差分 / Purged CV / CPCV / DSR / PBO 均来自
Marcos López de Prado, *Advances in Financial Machine Learning* (2018)。

## 八、防坑清单

- 严禁普通随机交叉验证(必用 Purged/CPCV), 否则数据泄漏导致虚高分数。
- 只看 accuracy 不够, 必看 **Brier/校准 + DSR + PBO + 含成本回测**。
- 先纸面交易验证 1~3 个月再谈实盘。
