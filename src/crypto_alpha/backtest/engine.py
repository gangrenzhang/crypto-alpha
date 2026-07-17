"""含成本回测 + 去偏夏普(DSR) + 回测过拟合概率(PBO)。

回测: 对每个事件, 若校准概率 > 阈值则按分数 Kelly 下注, 计入手续费/滑点/资金费。
PnL 以事件为单位, 按入场时间排序做复利, 得到净值曲线与风险指标。
DSR / PBO 用于判断 "看起来不错的夏普" 是不是多次尝试或过拟合造成的假象。

重要口径与局限(务必知悉, 勿当作可交易净值):
- **事件级、假设顺序独立**: 三重障碍事件在时间上会重叠, 这里把每笔当独立资本回合做
  复利, 未建模并发持仓/组合资金约束, 因此**会高估收益、低估真实回撤**。
- **资金费按持有 bar 数累计**(funding × bars_held), 手续费/滑点开平各一次。
- **夏普为"每笔"口径**, 另给按成交频率年化的 sharpe_annualized 便于跨频比较。
- **日内熔断**(risk.daily_max_drawdown): 当日回撤触及阈值后, 当日剩余事件停止开仓。
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
    daily_max_dd = float(risk_cfg.get("daily_max_drawdown", 0.0) or 0.0)

    has_bars = "bars_held" in df.columns
    equity_run = 1.0            # 运行净值(用于日内熔断)
    day_key = None              # 当前 UTC 日期
    day_start_equity = 1.0      # 当日起始净值
    halted_today = False

    sizes, rets, halted_flags = [], [], []
    for ts, r in df.iterrows():
        # 日内熔断状态维护(按 UTC 自然日)
        d = ts.date() if hasattr(ts, "date") else None
        if d != day_key:
            day_key = d
            day_start_equity = equity_run
            halted_today = False
        if daily_max_dd > 0 and not halted_today:
            if equity_run <= day_start_equity * (1.0 - daily_max_dd):
                halted_today = True

        if halted_today or r["prob"] < thr:
            sizes.append(0.0)
            rets.append(0.0)
            halted_flags.append(bool(halted_today))
            continue
        size = position_size(r["prob"], payoff, kf, maxp)
        bars = float(r["bars_held"]) if has_bars else 1.0
        cost = size * (2 * (fee + slip) + funding * bars)  # 手续费/滑点开平各一次; 资金费按持有 bar 累计
        pnl = size * (np.exp(r["ret"]) - 1.0) - cost
        equity_run *= (1.0 + pnl)
        sizes.append(size)
        rets.append(pnl)
        halted_flags.append(False)

    df["size"] = sizes
    df["pnl"] = rets
    df["halted"] = halted_flags
    equity = (1.0 + df["pnl"]).cumprod()

    traded = df[df["size"] > 0]
    ppy = _periods_per_year(df.index, len(traded))
    metrics = {
        "n_events": int(len(df)),
        "n_trades": int(len(traded)),
        "n_halted": int(sum(halted_flags)),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": sharpe_ratio(df["pnl"].values),                       # 每笔口径
        "sharpe_annualized": sharpe_ratio(df["pnl"].values, ppy),        # 按成交频率年化
        "max_drawdown": max_drawdown(equity.values),
        "win_rate": float((traded["pnl"] > 0).mean()) if len(traded) else 0.0,
        "avg_pnl": float(traded["pnl"].mean()) if len(traded) else 0.0,
    }
    mdd = abs(metrics["max_drawdown"]) + 1e-9
    metrics["calmar"] = metrics["total_return"] / mdd
    return {"metrics": metrics, "equity": equity, "detail": df}


def _periods_per_year(index, n_trades: int) -> float:
    """按成交时间跨度估计"每年成交笔数", 用于把每笔夏普年化。"""
    try:
        if len(index) < 2 or n_trades < 2:
            return 0.0
        span_days = (index[-1] - index[0]).total_seconds() / 86400.0
        if span_days <= 0:
            return 0.0
        return float(n_trades / (span_days / 365.25))
    except Exception:
        return 0.0


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 0.0) -> float:
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
