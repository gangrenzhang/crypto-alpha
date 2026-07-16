"""含成本回测 + 去偏夏普(DSR) + 回测过拟合概率(PBO)。

回测: 对每个事件, 若校准概率 > 阈值则按分数 Kelly 下注, 计入手续费/滑点/资金费。
PnL 以事件为单位, 按入场时间排序做复利, 得到净值曲线与风险指标。
DSR / PBO 用于判断 "看起来不错的夏普" 是不是多次尝试或过拟合造成的假象。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..risk.sizing import position_size


def backtest_events(
    events: pd.DataFrame,
    prob: np.ndarray,
    bt_cfg: dict,
    risk_cfg: dict,
    payoff: float = 1.0,
) -> dict:
    """events 需含: ret(含方向的对数收益), t1。prob 为校准后概率(与 events 对齐)。"""
    df = events.copy()
    df["prob"] = prob
    df = df.dropna(subset=["prob", "ret"]).sort_index()

    thr = float(bt_cfg.get("prob_threshold", 0.55))
    fee = float(bt_cfg.get("fee_bps", 5.0)) / 1e4
    slip = float(bt_cfg.get("slippage_bps", 2.0)) / 1e4
    funding = float(bt_cfg.get("funding_bps_per_bar", 0.0)) / 1e4

    kf = float(risk_cfg.get("kelly_fraction", 0.5))
    maxp = float(risk_cfg.get("max_position_pct", 0.3))

    sizes, rets = [], []
    for _, r in df.iterrows():
        if r["prob"] < thr:
            sizes.append(0.0)
            rets.append(0.0)
            continue
        size = position_size(r["prob"], payoff, kf, maxp)
        cost = size * (2 * (fee + slip) + funding)  # 开平各一次
        pnl = size * (np.exp(r["ret"]) - 1.0) - cost
        sizes.append(size)
        rets.append(pnl)

    df["size"] = sizes
    df["pnl"] = rets
    equity = (1.0 + df["pnl"]).cumprod()

    traded = df[df["size"] > 0]
    metrics = {
        "n_events": int(len(df)),
        "n_trades": int(len(traded)),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": sharpe_ratio(df["pnl"].values),
        "max_drawdown": max_drawdown(equity.values),
        "win_rate": float((traded["pnl"] > 0).mean()) if len(traded) else 0.0,
        "avg_pnl": float(traded["pnl"].mean()) if len(traded) else 0.0,
    }
    mdd = abs(metrics["max_drawdown"]) + 1e-9
    metrics["calmar"] = metrics["total_return"] / mdd
    return {"metrics": metrics, "equity": equity, "detail": df}


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[r != 0] if (r != 0).any() else r
    if len(r) < 2 or r.std() == 0:
        return 0.0
    sr = r.mean() / (r.std() + 1e-12)
    if periods_per_year:
        sr *= np.sqrt(periods_per_year)
    return float(sr)


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def deflated_sharpe_ratio(
    observed_sr: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """去偏夏普比率(López de Prado)。在多次试验下, 期望最大夏普会虚高;
    DSR 给出 "真实夏普 > 0" 的概率(近似)。"""
    from scipy.stats import norm

    if n_trials < 1 or n_obs < 2:
        return float("nan")
    emc = 0.5772156649
    # 期望最大夏普(在 n_trials 次独立试验下, SR~N(0,1) 的期望最大值)
    max_z = (1 - emc) * norm.ppf(1 - 1.0 / n_trials) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_std = np.sqrt((1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr**2) / (n_obs - 1))
    dsr = norm.cdf((observed_sr - max_z * sr_std) / (sr_std + 1e-12))
    return float(dsr)


def probability_of_backtest_overfitting(perf_matrix: np.ndarray) -> float:
    """PBO: 输入形状 (n_configs, n_splits) 的绩效矩阵(每列一个 CPCV 划分)。
    通过组合式划分比较 "样本内最优配置" 在样本外的排名, 估计过拟合概率。"""
    from itertools import combinations

    M = np.asarray(perf_matrix, dtype=float)
    n_cfg, n_split = M.shape
    if n_split < 2:
        return float("nan")
    half = n_split // 2
    logits = []
    for is_cols in combinations(range(n_split), half):
        oos_cols = [c for c in range(n_split) if c not in is_cols]
        is_perf = M[:, list(is_cols)].mean(axis=1)
        oos_perf = M[:, oos_cols].mean(axis=1)
        best_is = int(np.argmax(is_perf))
        # 该配置在样本外的相对排名
        rank = (oos_perf <= oos_perf[best_is]).mean()
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())
