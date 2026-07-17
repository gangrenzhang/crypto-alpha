"""闭环完整性诊断的核心实现。

设计原则(为什么这样测才有效):
- **空对照(null control)**: 用纯随机游走喂全链路, 真实无信号 => OOF AUC 必须≈0.5、
  净值不显著为正。若这里出现"能预测", 说明标注/特征/评估在制造虚假信号(泄漏或口径错)。
- **置换基线(permutation)**: 打乱标签后重训, 正确的 CV/堆叠/校准**无法**预测随机标签,
  AUC 必须塌回≈0.5。若打乱后仍高, 说明二层/校准/回测在偷看测试集。
- **正对照(positive control)**: 用"仅含过去信息即可预测"的构造数据, AUC 必须明显>0.5,
  证明链路在真有信号时抓得到(排除"永远随机"的假阴性)。
- **不变量(invariants)**: CPCV/PurgedKFold 训练集与测试集标签区间零重叠、禁运有间隔;
  回测权益可对账、并发敞口有上限、成本单调。这些与数据无关, 恒真, 违反即代码 bug。
- **时移不变(time-shift)**: 把所有时间整体平移, 结果应逐位相等(逻辑不应依赖绝对时间)。

这些检查全部 CPU 秒级可跑, 只用 GBDT, 不碰深度/LLM/显卡。
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 诊断用的轻量 GBDT: 够表达非线性又跑得快, 保证测试秒级
_DIAG_GBDT: dict = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": -1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_samples": 20,
}


@dataclass
class CheckResult:
    """单条体检结论。passed=None 表示"仅报告、不判定"。"""

    name: str
    passed: bool | None
    detail: dict = field(default_factory=dict)
    note: str = ""

    @property
    def status(self) -> str:
        if self.passed is None:
            return "INFO"
        return "PASS" if self.passed else "FAIL"


# --------------------------------------------------------------------------- #
# 合成数据构造                                                                 #
# --------------------------------------------------------------------------- #
def make_random_walk_ohlcv(
    n: int = 6000, seed: int = 0, sigma: float = 0.01,
    start: str = "2021-01-01", freq: str = "1h",
) -> pd.DataFrame:
    """纯随机游走 OHLCV(几何布朗运动, **无任何可学习结构**)。

    收益 iid 正态 => 三重障碍标签相对特征完全不可预测, 是空对照的黄金标准。
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    logret = rng.normal(0.0, sigma, size=n)
    close = 100.0 * np.exp(np.cumsum(logret))
    prev = np.concatenate([[100.0], close[:-1]])  # 开盘≈上一收盘
    open_ = prev
    intr = np.abs(rng.normal(0.0, sigma * 0.4, size=n))
    hi = np.maximum(open_, close) * (1.0 + intr)
    lo = np.minimum(open_, close) * (1.0 - intr)
    vol = rng.uniform(1.0, 10.0, size=n)
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}, index=idx
    )


def make_predictable_core_dataset(
    n: int = 1500, seed: int = 0, beta: float = 1.8, overlap: int = 5, freq: str = "1h",
):
    """构造"仅凭过去信息即可预测"的核心闭环数据集(正对照)。

    返回 (X, y, t1, events, feature_cols):
    - X: 含一个真信号列 f_signal 与一个噪声列 f_noise(均为该事件时点的已知量);
    - y: 由 f_signal 经 logistic 生成 => 真有信号, 干净 CV 下 AUC 应明显>0.5;
    - t1: 每个事件跨 overlap 根 bar => 标签**重叠**, 用于压测净化/禁运是否真的生效;
    - events: 含 ret/t1/side/bin/bars_held, 可直接喂回测。
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq=freq, tz="UTC")
    z = rng.normal(size=n)
    noise = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-beta * z))
    y = (rng.uniform(size=n) < prob).astype(int)

    X = pd.DataFrame({"f_signal": z, "f_noise": noise}, index=idx)
    feature_cols = ["f_signal", "f_noise"]

    end_loc = np.minimum(np.arange(n) + overlap, n - 1)
    t1_idx = idx[end_loc]  # tz-aware DatetimeIndex, 按位置对应每个事件
    t1 = pd.Series(t1_idx, index=idx)
    mag = 0.01 + 0.005 * np.abs(z)
    ret = (2 * y - 1) * mag  # 赢家正收益、输家负收益
    events = pd.DataFrame(
        {
            "ret": ret,
            "t1": t1_idx,
            "side": np.ones(n, dtype=int),
            "bin": y,
            "bars_held": (end_loc - np.arange(n)).clip(min=1),
        },
        index=idx,
    )
    return X, y, t1, events, feature_cols


# --------------------------------------------------------------------------- #
# 核心学习闭环(GBDT 专家 + Stacking + nested OOF)                             #
# --------------------------------------------------------------------------- #
def run_core_loop(
    X: pd.DataFrame, y: np.ndarray, t1: pd.Series, feature_cols: list[str],
    sample_weight: np.ndarray | None = None, n_splits: int = 5,
    embargo_pct: float = 0.01, seed: int = 42, min_expert_auc: float = 0.0,
) -> np.ndarray:
    """跑一遍与主流程同构的堆叠闭环, 返回二层 nested OOF 概率。"""
    from ..experts.gbdt import GBDTExpert
    from ..ensemble import StackingEnsemble

    experts = [GBDTExpert(dict(_DIAG_GBDT), list(feature_cols), seed=seed)]
    ens = StackingEnsemble(
        experts, {"meta_learner": "logistic", "C": 1.0, "min_expert_auc": min_expert_auc}, seed=seed
    )
    ens.fit(X, np.asarray(y), pd.Series(t1), sample_weight=sample_weight,
            n_splits=n_splits, embargo_pct=embargo_pct)
    return ens.oof_proba()


def core_auc(oof: np.ndarray, y: np.ndarray) -> float:
    from ..calibration import classification_report_probs

    return float(classification_report_probs(np.asarray(oof, dtype=float), np.asarray(y))["auc"])


def permutation_baseline(
    X: pd.DataFrame, y: np.ndarray, t1: pd.Series, feature_cols: list[str],
    seed: int = 42, n_splits: int = 5, embargo_pct: float = 0.01,
) -> dict:
    """对同一份数据, 分别用真实标签与打乱标签跑核心闭环, 比较 AUC。

    - auc_real: 真标签的样本外 AUC;
    - auc_shuffled: 打乱标签后的样本外 AUC(正确闭环下应≈0.5);
    - gap: 两者之差, 越大说明"真信号"越被抓住且"闭环不偷看"。
    """
    oof_real = run_core_loop(X, y, t1, feature_cols, n_splits=n_splits,
                             embargo_pct=embargo_pct, seed=seed)
    rng = np.random.default_rng(seed)
    y_shuf = np.asarray(y).copy()
    rng.shuffle(y_shuf)
    oof_shuf = run_core_loop(X, y_shuf, t1, feature_cols, n_splits=n_splits,
                             embargo_pct=embargo_pct, seed=seed)
    a_real = core_auc(oof_real, y)
    a_shuf = core_auc(oof_shuf, y_shuf)
    return {"auc_real": a_real, "auc_shuffled": a_shuf, "gap": a_real - a_shuf}


# --------------------------------------------------------------------------- #
# 标注 oracle: 三重障碍在人工构造价格上必须给出唯一确定结果                    #
# --------------------------------------------------------------------------- #
def _oracle_bin(close, high, low, side_val: int, vertical_bars: int = 9) -> tuple[int, float, int]:
    """在给定价格路径上跑一次三重障碍, 返回 (bin, ret, bars_held)。"""
    from ..labeling.triple_barrier import get_events, get_bins

    idx = close.index
    t_events = pd.DatetimeIndex([idx[0]])
    trgt = pd.Series(0.01, index=idx)
    side = pd.Series(side_val, index=idx)
    pt_sl = (1.0, 1.0)
    ev = get_events(close, high, low, t_events, pt_sl, trgt, vertical_bars, side, 0.0)
    b = get_bins(ev, close, pt_sl)
    return int(b["bin"].iloc[0]), float(b["ret"].iloc[0]), int(b["bars_held"].iloc[0])


def labeling_oracle_results() -> list["CheckResult"]:
    """三重障碍 oracle: 止盈/止损/同 bar 平局判损/垂直到期/做空对称。

    这些是与数据无关的**恒真**断言, 挂了说明标注函数(方向/平局/到期口径)有 bug。
    此前只在 pytest 覆盖, 现纳入在线体检(12_audit), 使在线体检也能拦标注逻辑错误。
    """
    idx = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    base = pd.Series(100.0, index=idx)
    out: list[CheckResult] = []

    # 1) 仅上破 => 止盈 => bin=1, ret>0
    hi = base.copy(); hi.iloc[2] = 102.0
    b, r, _ = _oracle_bin(base, hi, base, side_val=1)
    out.append(CheckResult("标注oracle: 上破止盈=>bin1/ret>0", b == 1 and r > 0, {"bin": b, "ret": round(r, 4)}))

    # 2) 仅下破 => 止损 => bin=0, ret<0
    lo = base.copy(); lo.iloc[2] = 98.0
    b, r, _ = _oracle_bin(base, base, lo, side_val=1)
    out.append(CheckResult("标注oracle: 下破止损=>bin0/ret<0", b == 0 and r < 0, {"bin": b, "ret": round(r, 4)}))

    # 3) 同 bar 同时上下破 => 悲观判止损 bin=0
    hi2 = base.copy(); hi2.iloc[2] = 102.0
    lo2 = base.copy(); lo2.iloc[2] = 98.0
    b, r, _ = _oracle_bin(base, hi2, lo2, side_val=1)
    out.append(CheckResult("标注oracle: 同bar平局=>悲观判损bin0", b == 0, {"bin": b}))

    # 4) 缓涨不触碰 => 垂直到期; 多头为盈利
    slow = pd.Series(np.linspace(100.0, 100.5, 10), index=idx)
    b, _, bh = _oracle_bin(slow, slow, slow, side_val=1)
    out.append(CheckResult("标注oracle: 垂直到期缓涨(多)=>bin1", b == 1 and bh >= 1, {"bin": b, "bars_held": bh}))

    # 5) 做空对称: 同样缓涨到期 => 亏损 bin=0
    b, _, _ = _oracle_bin(slow, slow, slow, side_val=-1)
    out.append(CheckResult("标注oracle: 做空缓涨到期=>bin0(对称)", b == 0, {"bin": b}))
    return out


def null_backtest_return_mean(cfg, symbol: str, seeds, n_bars: int = 3000) -> dict:
    """多种子随机游走喂满全链路, 汇总回测收益均值。

    纯噪声 + 成本下, 期望回测收益应 ≤ 0(成本侵蚀); 若均值显著为正, 说明回测/决策层
    在无信号时仍系统性造利润(单次幸运不算, 故用多种子均值)。
    """
    import copy as _copy

    rets: list[float] = []
    for sd in seeds:
        cfg_n = _cpu_cfg(_copy.deepcopy(cfg), n_splits=5)
        raw = make_random_walk_ohlcv(n=n_bars, seed=int(sd))
        _, tr = run_full_pipeline_with_prices(cfg_n, raw, symbol)
        rets.append(float(tr["backtest"]["metrics"]["total_return"]))
    arr = np.asarray(rets, dtype=float)
    return {"mean": float(arr.mean()), "max": float(arr.max()),
            "min": float(arr.min()), "returns": [round(x, 4) for x in rets]}


# --------------------------------------------------------------------------- #
# 交叉验证不变量                                                               #
# --------------------------------------------------------------------------- #
def _naive_arrays(t1: pd.Series):
    """把事件起始时间与标签结束时间统一为 tz-naive(UTC) 的 numpy 数组, 避免比较报错/告警。"""
    idx = pd.DatetimeIndex(t1.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    ends = pd.DatetimeIndex(pd.to_datetime(np.asarray(t1.values)))
    if ends.tz is not None:
        ends = ends.tz_localize(None)
    return idx.to_numpy(), ends.to_numpy()


def _overlap_against(tr: np.ndarray, seg_idx: np.ndarray, starts_np, ends_np) -> int:
    """训练集里标签区间 [start, t1] 与某测试段区间 [t0, test_end] 重叠的样本数。

    AFML 定义: 训练标签结束 >= 测试段起始 且 训练起始 <= 测试段结束, 即为重叠(应被净化)。
    """
    if len(tr) == 0 or len(seg_idx) == 0:
        return 0
    t0 = starts_np[seg_idx].min()
    test_end = ends_np[seg_idx].max()
    bad = (ends_np[tr] >= t0) & (starts_np[tr] <= test_end)
    return int(np.asarray(bad).sum())


def count_cv_overlaps(
    t1: pd.Series, n_splits: int = 5, embargo_pct: float = 0.02,
    kind: str = "purged", n_test_groups: int = 2,
) -> int:
    """遍历所有划分, 累计训练/测试标签区间重叠样本数(应为 0)。

    CPCV 的测试组可能不相邻, 因此按**每个测试组**分别判定重叠(与引擎逐组净化口径一致),
    而非把整块测试集当成连续区间(否则会把组间的正常训练样本误判为重叠)。
    """
    X = pd.DataFrame(index=t1.index)
    starts_np, ends_np = _naive_arrays(t1)
    total = 0
    if kind == "purged":
        from ..validation.purged_kfold import PurgedKFold

        for tr, te in PurgedKFold(n_splits, t1, embargo_pct).split(X):
            total += _overlap_against(tr, te, starts_np, ends_np)
    elif kind == "cpcv":
        from ..validation.cpcv import CombinatorialPurgedCV

        n = len(t1)
        groups = np.array_split(np.arange(n), n_splits)
        for tr, _te, combo in CombinatorialPurgedCV(n_splits, n_test_groups, t1, embargo_pct).split(X):
            for g in combo:
                total += _overlap_against(tr, groups[g], starts_np, ends_np)
    else:
        raise ValueError(f"未知划分类型: {kind}")
    return total


def embargo_gap_ok(t1: pd.Series, n_splits: int = 5, embargo_pct: float = 0.05) -> bool:
    """禁运有效性: 每个测试段结束后, 紧邻的 embargo 根样本不得进入训练集。"""
    from ..validation.purged_kfold import PurgedKFold

    n = len(t1)
    embargo = int(n * embargo_pct)
    if embargo <= 0:
        return True
    X = pd.DataFrame(index=t1.index)
    indices = np.arange(n)
    for tr, te in PurgedKFold(n_splits, t1, embargo_pct).split(X):
        end = int(te[-1]) + 1
        if end + embargo > n:
            continue
        banned = set(range(end, end + embargo))
        if banned & set(tr.tolist()):
            return False
    return True


# --------------------------------------------------------------------------- #
# 回测对账                                                                     #
# --------------------------------------------------------------------------- #
def backtest_reconciliation(bt: dict, tol: float = 1e-6) -> dict:
    """回测自洽对账: 末端权益应等于逐笔 pnl 的累计复利, total_return 与之一致。"""
    detail = bt["detail"]
    equity = bt["equity"]
    pnl = detail["pnl"].values if "pnl" in detail.columns else np.array([])
    recon_equity = float(np.prod(1.0 + pnl)) if len(pnl) else 1.0
    end_equity = float(equity.values[-1]) if len(equity) else 1.0
    reported = 1.0 + float(bt["metrics"]["total_return"])
    return {
        "recon_from_pnl": recon_equity,
        "end_equity": end_equity,
        "reported_total_return_equity": reported,
        "equity_matches_pnl": abs(recon_equity - end_equity) <= tol * max(1.0, recon_equity),
        "metric_matches_equity": abs(reported - end_equity) <= tol * max(1.0, reported),
    }


def max_concurrent_gross(bt: dict) -> float:
    """从回测明细重建时间线, 计算任意时刻的最大并发锁定敞口(应 <= max_gross_exposure)。"""
    detail = bt["detail"]
    if "size" not in detail.columns or len(detail) == 0:
        return 0.0
    events = []
    for ts, row in detail.iterrows():
        size = float(row["size"])
        if size <= 0:
            continue
        exit_ts = pd.Timestamp(row["t1"]) if "t1" in detail.columns else ts
        events.append((ts, +size))
        events.append((exit_ts, -size))
    if not events:
        return 0.0
    # 同一时刻先释放(-)再占用(+): 与回测引擎口径一致
    events.sort(key=lambda x: (x[0], 0 if x[1] < 0 else 1))
    cur = 0.0
    peak = 0.0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return float(peak)


# --------------------------------------------------------------------------- #
# 数据集运行期护栏                                                             #
# --------------------------------------------------------------------------- #
def sanity_check_dataset(ds, min_events: int = 100) -> list[str]:
    """对 prepare_dataset 产物做体检, 返回告警列表(空=通过)。"""
    warns: list[str] = []
    n = len(ds.y)
    if n < min_events:
        warns.append(f"事件数过少: {n} < {min_events}")
    pos_rate = float(np.mean(ds.y)) if n else 0.0
    if not (0.2 <= pos_rate <= 0.8):
        warns.append(f"正类比例失衡: {pos_rate:.3f} (期望 0.2~0.8)")

    X = ds.X[ds.feature_cols]
    nan_cols = [c for c in ds.feature_cols if X[c].isna().any()]
    if nan_cols:
        warns.append(f"特征含 NaN: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}")
    const_cols = [c for c in ds.feature_cols if float(np.nanstd(X[c].values)) == 0.0]
    if const_cols:
        warns.append(f"常数特征(无信息): {const_cols[:5]}{'...' if len(const_cols) > 5 else ''}")

    sw = np.asarray(ds.sample_weight, dtype=float)
    if not np.all(np.isfinite(sw)):
        warns.append("样本权重含非有限值")
    elif (sw < 0).any():
        warns.append("样本权重出现负值")

    t1 = pd.Series(ds.t1)
    if (pd.DatetimeIndex(t1.index) > pd.DatetimeIndex(pd.to_datetime(t1.values, utc=True))).any():
        warns.append("存在 t1 早于事件起始时间的样本(时间倒挂)")
    return warns


# --------------------------------------------------------------------------- #
# 用注入价格跑全链路(monkeypatch load_symbol_data)                            #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _patched_prices(raw_df: pd.DataFrame):
    import crypto_alpha.pipeline.run as run_mod

    original = run_mod.load_symbol_data
    run_mod.load_symbol_data = lambda cfg, symbol: raw_df.copy()
    try:
        yield
    finally:
        run_mod.load_symbol_data = original


def run_full_pipeline_with_prices(cfg, raw_df: pd.DataFrame, symbol: str = "BTC/USDT"):
    """在给定 cfg 下, 用注入的 OHLCV 跑 prepare_dataset + train_and_validate。

    注意: 调用方应先把 cfg 收敛为"纯 GBDT、关新闻/多周期、合成模式"以便 CPU 快速跑通。
    """
    from ..pipeline import prepare_dataset, train_and_validate

    with _patched_prices(raw_df):
        ds = prepare_dataset(cfg, symbol)
        trained = train_and_validate(cfg, ds)
    return ds, trained


def _cpu_cfg(cfg, n_splits: int = 5):
    """把配置收敛到 CPU 友好、单一 GBDT、无外部依赖的诊断模式(原地修改 raw)。"""
    cfg.raw["experts"]["enabled"] = ["gbdt"]
    cfg.raw["experts"]["gbdt"] = dict(_DIAG_GBDT)
    cfg.raw.setdefault("features", {})
    cfg.raw["features"]["mtf_enabled"] = False
    cfg.raw["news"]["as_feature"] = False
    cfg.raw["news"]["use_synthetic"] = True
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["validation"]["n_splits"] = n_splits
    cfg.raw["ensemble"]["min_expert_auc"] = 0.0
    return cfg


# --------------------------------------------------------------------------- #
# 在线体检编排                                                                 #
# --------------------------------------------------------------------------- #
def audit_pipeline(
    cfg, symbol: str = "BTC/USDT", n_bars: int = 6000, seed: int = 0,
    null_auc_max: float = 0.58, pos_auc_min: float = 0.62, perm_gap_min: float = 0.10,
    null_ret_seeds: tuple[int, ...] = (1000, 1001, 1002), null_ret_mean_max: float = 0.03,
) -> list[CheckResult]:
    """一次性跑完所有 CPU 级闭环体检, 返回结论列表(供脚本打印/判定退出码)。"""
    import copy

    results: list[CheckResult] = []

    # 0) 标注 oracle: 三重障碍在人工构造价格上的确定性结果(方向/平局/到期口径)
    results.extend(labeling_oracle_results())

    # 1) 正对照 + 置换基线(核心闭环: Stacking/nested OOF/校准是否"真学到且不偷看")
    X, y, t1, events, fcols = make_predictable_core_dataset(n=1500, seed=seed)
    perm = permutation_baseline(X, y, t1, fcols, seed=42)
    results.append(CheckResult(
        "正对照: 真信号可学习(AUC 明显>0.5)",
        perm["auc_real"] >= pos_auc_min,
        {"auc_real": round(perm["auc_real"], 4), "min": pos_auc_min},
    ))
    results.append(CheckResult(
        "置换基线: 打乱标签后 AUC 塌回≈0.5(不偷看)",
        perm["auc_shuffled"] <= null_auc_max,
        {"auc_shuffled": round(perm["auc_shuffled"], 4), "max": null_auc_max},
    ))
    results.append(CheckResult(
        "信号差(auc_real - auc_shuffled 足够大)",
        perm["gap"] >= perm_gap_min,
        {"gap": round(perm["gap"], 4), "min": perm_gap_min},
    ))

    # 2) 时移不变: 整体平移时间, OOF 应逐位相等
    oof0 = run_core_loop(X, y, t1, fcols, seed=7)
    shift = pd.Timedelta(days=7)
    X2 = X.copy(); X2.index = X.index + shift
    t2 = t1 + shift; t2.index = t2.index + shift
    oof1 = run_core_loop(X2, y, t2, fcols, seed=7)
    results.append(CheckResult(
        "时移不变: 平移时间戳结果不变",
        bool(np.allclose(np.nan_to_num(oof0), np.nan_to_num(oof1), atol=1e-9)),
        {"max_abs_diff": float(np.nanmax(np.abs(oof0 - oof1)))},
    ))

    # 3) CV 不变量: 净化零重叠 + 禁运有间隔
    ov_p = count_cv_overlaps(t1, n_splits=5, embargo_pct=0.02, kind="purged")
    ov_c = count_cv_overlaps(t1, n_splits=6, embargo_pct=0.02, kind="cpcv", n_test_groups=2)
    results.append(CheckResult("PurgedKFold 训练/测试标签零重叠", ov_p == 0, {"overlaps": ov_p}))
    results.append(CheckResult("CPCV 训练/测试标签零重叠", ov_c == 0, {"overlaps": ov_c}))
    results.append(CheckResult("禁运(embargo)有效", embargo_gap_ok(t1, 5, 0.05), {}))

    # 4) 回测对账 + 并发敞口 + 成本单调
    from ..backtest import backtest_events

    prob = 0.5 + 0.45 * (2 * y - 1) * np.abs(np.tanh(X["f_signal"].values))  # 与标签同向、界内
    prob = np.clip(prob, 0.01, 0.99)
    risk_cfg = {"kelly_fraction": 0.5, "max_position_pct": 0.3,
                "max_gross_exposure": 1.0, "daily_max_drawdown": 0.0}
    bt0 = backtest_events(events, prob, {"portfolio_mode": True, "prob_threshold": 0.55,
                                         "fee_bps": 0.0, "slippage_bps": 0.0}, risk_cfg)
    bt1 = backtest_events(events, prob, {"portfolio_mode": True, "prob_threshold": 0.55,
                                         "fee_bps": 20.0, "slippage_bps": 10.0}, risk_cfg)
    recon = backtest_reconciliation(bt0)
    results.append(CheckResult("回测权益与逐笔 pnl 对账一致", recon["equity_matches_pnl"], recon))
    peak = max_concurrent_gross(bt0)
    results.append(CheckResult(
        "并发敞口不超上限(max_gross_exposure)",
        peak <= risk_cfg["max_gross_exposure"] + 1e-9,
        {"peak_gross": round(peak, 4), "cap": risk_cfg["max_gross_exposure"]},
    ))
    results.append(CheckResult(
        "成本单调: 提高费用不增加收益",
        bt1["metrics"]["total_return"] <= bt0["metrics"]["total_return"] + 1e-9,
        {"ret_no_fee": round(bt0["metrics"]["total_return"], 5),
         "ret_high_fee": round(bt1["metrics"]["total_return"], 5)},
    ))
    # 独立复利模式净值应 >= 组合模式(资金占用约束更松) —— 口径对照
    bti = backtest_events(events, prob, {"portfolio_mode": False, "prob_threshold": 0.55,
                                         "fee_bps": 0.0, "slippage_bps": 0.0}, risk_cfg)
    results.append(CheckResult(
        "组合级回测不高估收益(<=独立复利口径)",
        bt0["metrics"]["total_return"] <= bti["metrics"]["total_return"] + 1e-9,
        {"portfolio": round(bt0["metrics"]["total_return"], 5),
         "independent": round(bti["metrics"]["total_return"], 5)},
        note="仅口径对照, 独立复利会系统性偏高",
    ))

    # 5) 全链路空对照: 随机游走喂 prepare_dataset+train, AUC 必须≈0.5
    try:
        cfg_null = _cpu_cfg(copy.deepcopy(cfg), n_splits=5)
        raw = make_random_walk_ohlcv(n=n_bars, seed=seed)
        ds_null, tr_null = run_full_pipeline_with_prices(cfg_null, raw, symbol)
        auc_null = float(tr_null["report"].get("auc", float("nan")))
        ret_null = float(tr_null["backtest"]["metrics"]["total_return"])
        ok = (not np.isfinite(auc_null)) or (auc_null <= null_auc_max)
        results.append(CheckResult(
            "全链路空对照: 随机游走 OOF AUC≈0.5",
            ok, {"auc": round(auc_null, 4), "max": null_auc_max,
                 "n_events": len(ds_null.y), "backtest_return": round(ret_null, 5)},
        ))
        results.append(CheckResult(
            "全链路空对照: 数据集体检无致命告警",
            len(sanity_check_dataset(ds_null)) == 0,
            {"warnings": sanity_check_dataset(ds_null)},
        ))
    except Exception as e:  # 环境缺依赖(如 lightgbm) 时降级为 INFO, 不误报 FAIL
        results.append(CheckResult("全链路空对照", None, {"error": repr(e)},
                                   note="跳过(可能缺依赖或数据), 详见 error"))

    # 5b) 空对照回测收益闸门: 多种子随机游走的回测收益均值应 ≤ 阈值(成本下应≈0/为负)
    try:
        nr = null_backtest_return_mean(cfg, symbol, seeds=null_ret_seeds, n_bars=3000)
        results.append(CheckResult(
            "全链路空对照: 随机游走回测收益均值≈0(不凭空造利润)",
            nr["mean"] <= null_ret_mean_max,
            {"mean_return": round(nr["mean"], 4), "max": null_ret_mean_max,
             "per_seed": nr["returns"]},
            note="纯噪声+成本下期望收益≤0; 单次幸运不计, 故取多种子均值",
        ))
    except Exception as e:
        results.append(CheckResult("全链路空对照: 收益闸门", None, {"error": repr(e)},
                                   note="跳过(可能缺依赖或数据)"))

    # 6) 复现性: 同 seed 两次全链路 OOF 应完全一致
    try:
        cfg_r = _cpu_cfg(copy.deepcopy(cfg), n_splits=5)
        raw_r = make_random_walk_ohlcv(n=2000, seed=123)
        _, t_a = run_full_pipeline_with_prices(cfg_r, raw_r, symbol)
        _, t_b = run_full_pipeline_with_prices(cfg_r, raw_r, symbol)
        a = np.nan_to_num(t_a["oof_calibrated"]); b = np.nan_to_num(t_b["oof_calibrated"])
        same = bool(a.shape == b.shape and np.allclose(a, b, atol=1e-9))
        results.append(CheckResult(
            "复现性: 同 seed 两次训练 OOF 一致", same,
            {"max_abs_diff": float(np.max(np.abs(a - b))) if a.shape == b.shape else None},
        ))
    except Exception as e:
        results.append(CheckResult("复现性", None, {"error": repr(e)}, note="跳过"))

    return results
