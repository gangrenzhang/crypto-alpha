"""含成本回测 + 去偏夏普(DSR) + 回测过拟合概率(PBO)。

默认 **组合级资金占用** 回测:
- 重叠的三重障碍事件共享同一权益池; 开仓锁定仓位比例, 平仓后释放。
- 单笔 ≤ max_position_pct, 并发合计 ≤ max_gross_exposure, 避免旧版"独立复利"虚高。
- 可选 portfolio_mode=false 回退到旧的事件独立复利(仅作对照, 勿当可交易净值)。

其它口径:
- 资金费按持有 bar 数累计(funding × bars_held), 手续费/滑点开平各一次。
- 夏普为"每笔"口径, 另给按成交频率年化的 sharpe_annualized。
- 日内熔断(risk.daily_max_drawdown): 当日权益回撤触及阈值后, 当日停止新开仓。
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
    if bool(bt_cfg.get("portfolio_mode", True)):
        return _backtest_portfolio(events, prob, bt_cfg, risk_cfg, payoff)
    return _backtest_independent(events, prob, bt_cfg, risk_cfg, payoff)


def _cost(size: float, bars: float, fee: float, slip: float, funding: float) -> float:
    return size * (2 * (fee + slip) + funding * bars)


def _backtest_independent(
    events: pd.DataFrame,
    prob: np.ndarray,
    bt_cfg: dict,
    risk_cfg: dict,
    payoff: float,
) -> dict:
    """旧口径: 每笔独立对全权益复利(会高估收益)。仅作对照。"""
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
    equity_run = 1.0
    day_key = None
    day_start_equity = 1.0
    halted_today = False

    sizes, rets, halted_flags = [], [], []
    for ts, r in df.iterrows():
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
        cost = _cost(size, bars, fee, slip, funding)
        pnl = size * (np.exp(r["ret"]) - 1.0) - cost
        equity_run *= (1.0 + pnl)
        sizes.append(size)
        rets.append(pnl)
        halted_flags.append(False)

    df["size"] = sizes
    df["pnl"] = rets
    df["halted"] = halted_flags
    equity = (1.0 + df["pnl"]).cumprod()
    return _pack_result(df, equity, halted_flags, mode="independent")


def _backtest_portfolio(
    events: pd.DataFrame,
    prob: np.ndarray,
    bt_cfg: dict,
    risk_cfg: dict,
    payoff: float,
) -> dict:
    """组合级: 锁定并发仓位, 平仓释放后再开; 权益仅在出场时更新。"""
    df = events.copy()
    df["prob"] = np.asarray(prob, dtype=float)
    need = ["prob", "ret", "t1"]
    df = df.dropna(subset=need).sort_index()
    if len(df) == 0:
        empty = df.copy()
        empty["size"] = []
        empty["pnl"] = []
        empty["halted"] = []
        return _pack_result(empty, pd.Series(dtype=float), [], mode="portfolio")

    thr = float(bt_cfg.get("prob_threshold", 0.55))
    fee = float(bt_cfg.get("fee_bps", 5.0)) / 1e4
    slip = float(bt_cfg.get("slippage_bps", 2.0)) / 1e4
    funding = float(bt_cfg.get("funding_bps_per_bar", 0.0)) / 1e4
    min_size = float(bt_cfg.get("min_position_pct", 0.01))
    kf = float(risk_cfg.get("kelly_fraction", 0.5))
    maxp = float(risk_cfg.get("max_position_pct", 0.3))
    max_gross = float(risk_cfg.get("max_gross_exposure", 1.0))
    daily_max_dd = float(risk_cfg.get("daily_max_drawdown", 0.0) or 0.0)
    has_bars = "bars_held" in df.columns

    n = len(df)
    sizes = np.zeros(n, dtype=float)
    pnls = np.zeros(n, dtype=float)
    halted_flags = np.zeros(n, dtype=bool)
    skipped_cap = np.zeros(n, dtype=bool)

    # 时间线: 同一时刻先平仓释放资金, 再考虑开仓
    entries = [(df.index[i], 1, i) for i in range(n)]  # kind=1 entry
    exits = [(pd.Timestamp(df["t1"].iloc[i]), 0, i) for i in range(n)]  # kind=0 exit first
    timeline = sorted(entries + exits, key=lambda x: (x[0], x[1]))

    equity = 1.0
    locked = 0.0
    open_pos: dict[int, dict] = {}
    day_key = None
    day_start_equity = 1.0
    halted_today = False
    equity_pts: list[tuple[pd.Timestamp, float]] = []

    for ts, kind, i in timeline:
        d = ts.date() if hasattr(ts, "date") else None
        if d != day_key:
            day_key = d
            day_start_equity = equity
            halted_today = False
        if daily_max_dd > 0 and not halted_today:
            if equity <= day_start_equity * (1.0 - daily_max_dd):
                halted_today = True

        if kind == 0:  # exit
            if i not in open_pos:
                continue
            pos = open_pos.pop(i)
            size = pos["size"]
            bars = pos["bars"]
            ret = pos["ret"]
            cost = _cost(size, bars, fee, slip, funding)
            pnl = size * (np.exp(ret) - 1.0) - cost
            equity *= (1.0 + pnl)
            locked = max(0.0, locked - size)
            pnls[i] = pnl
            equity_pts.append((ts, equity))
            continue

        # entry
        row = df.iloc[i]
        if halted_today or float(row["prob"]) < thr:
            halted_flags[i] = bool(halted_today)
            continue
        want = position_size(float(row["prob"]), payoff, kf, maxp)
        avail = max(0.0, max_gross - locked)
        size = min(want, avail)
        if size < min_size:
            skipped_cap[i] = True
            continue
        bars = float(row["bars_held"]) if has_bars else 1.0
        sizes[i] = size
        locked += size
        open_pos[i] = {"size": size, "bars": bars, "ret": float(row["ret"])}

    # 若回测窗口结束仍有未平仓(t1 超出样本), 按标签收益强制了结
    for i, pos in list(open_pos.items()):
        size = pos["size"]
        cost = _cost(size, pos["bars"], fee, slip, funding)
        pnl = size * (np.exp(pos["ret"]) - 1.0) - cost
        equity *= (1.0 + pnl)
        pnls[i] = pnl
        locked = max(0.0, locked - size)
        equity_pts.append((df.index[i], equity))
    open_pos.clear()

    df = df.copy()
    df["size"] = sizes
    df["pnl"] = pnls
    df["halted"] = halted_flags
    df["skipped_capacity"] = skipped_cap

    if equity_pts:
        eq_ser = pd.Series({t: e for t, e in equity_pts}).sort_index()
        # 同时间多笔出场时保留最后净值
        eq_ser = eq_ser[~eq_ser.index.duplicated(keep="last")]
    else:
        eq_ser = pd.Series([1.0], index=[df.index[0]] if len(df) else pd.DatetimeIndex([]))

    return _pack_result(df, eq_ser, halted_flags.tolist(), mode="portfolio")


def _pack_result(df: pd.DataFrame, equity: pd.Series, halted_flags, mode: str) -> dict:
    traded = df[df["size"] > 0] if "size" in df.columns and len(df) else df.iloc[0:0]
    ppy = _periods_per_year(df.index if len(df) else equity.index, len(traded))
    eq_vals = equity.values if len(equity) else np.array([1.0])
    metrics = {
        "n_events": int(len(df)),
        "n_trades": int(len(traded)),
        "n_halted": int(sum(halted_flags)) if halted_flags is not None else 0,
        "total_return": float(eq_vals[-1] - 1.0) if len(eq_vals) else 0.0,
        "sharpe": sharpe_ratio(df["pnl"].values) if len(df) and "pnl" in df.columns else 0.0,
        "sharpe_annualized": sharpe_ratio(
            df["pnl"].values if len(df) and "pnl" in df.columns else np.array([]), ppy
        ),
        "max_drawdown": max_drawdown(eq_vals),
        "win_rate": float((traded["pnl"] > 0).mean()) if len(traded) else 0.0,
        "avg_pnl": float(traded["pnl"].mean()) if len(traded) else 0.0,
        "portfolio_mode": mode,
    }
    if "skipped_capacity" in df.columns:
        metrics["n_skipped_capacity"] = int(df["skipped_capacity"].sum())
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
        rank = (oos_perf <= oos_perf[best_is]).mean()
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())
