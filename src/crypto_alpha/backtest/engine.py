"""含成本回测 + 去偏夏普(DSR) + 回测过拟合概率(PBO)。

默认 **组合级资金占用** 回测:
- 重叠的三重障碍事件共享同一权益池; 开仓锁定仓位比例, 平仓后释放。
- 单笔 ≤ max_position_pct, 并发合计 ≤ max_gross_exposure, 避免旧版"独立复利"虚高。
- 出场按**入场名义加性**记账(ΔE = E_entry × pnl_frac), 避免重叠仓乘积复利虚高。
- 可选 portfolio_mode=false 回退到旧的事件独立复利(仅作对照, 勿当可交易净值)。
- 可选传入 confident 掩码与实盘保形弃权对齐。

其它口径:
- 资金费按持有 bar 数累计(funding × bars_held), 手续费/滑点开平各一次。
- 夏普为"每笔"口径, 年化用**平均唯一性折算的有效独立成交数**(而非直接 √成交数),
  因为三重障碍事件持有期高度重叠、并不独立, 直接年化会系统性高估。
  字段 ``sharpe`` / ``sharpe_annualized`` 语义不变; 另**并行**输出权益曲线夏普
  (``sharpe_equity*`` / ``sharpe_equity_mtm*``), 不替换旧字段、不改 CPCV/DSR 输入。
- **盯市(mark-to-market)**: 若传入 `prices`(收盘价序列)且事件含 side, 组合模式会按
  收盘价重建含持仓浮盈亏的权益曲线, 用于 MDD 与日内熔断——避免"只在出场记账"漏掉
  并发持仓的浮动回撤。浮动与标签/已实现同形:
  ``size × side × (P_t/P_entry − 1)``(入场名义简单收益), **禁止**
  ``(P_t/P_entry)^side − 1``(空头几何口径会系统性偏离)。未传 prices 时回退到
  仅出场记账的旧口径(偏乐观)。
- 日内熔断(risk.daily_max_drawdown): 当日(盯市)权益回撤触及阈值后, 当日停止新开仓。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..risk.sizing import kelly_fraction, position_size, resolve_roundtrip_cost


def backtest_events(
    events: pd.DataFrame,
    prob: np.ndarray,
    bt_cfg: dict,
    risk_cfg: dict,
    payoff: float = 1.0,
    prices: pd.Series | None = None,
    confident: np.ndarray | None = None,
    ref_trgt: float | None = None,
) -> dict:
    """events 需含: ret(含方向的对数收益), t1。prob 为校准后概率(与 events 对齐)。

    prices: 可选收盘价序列(索引=bar 时间)。传入且 events 含 side 时, 组合模式启用盯市
    权益曲线(MDD/日内熔断更真实); 独立复利模式忽略之。
    confident: 可选布尔数组(与 events 对齐); False 视为保形弃权, 不开仓(与实盘 HOLD 对齐)。
    ref_trgt: 可选的波动滑点参考(相对 ATR 中位数)。调用方应传入**训练窗**冻结值,
    使成本模型与 decide/serve 同口径; 省略则退回本次样本中位数(见 _ref_trgt_from_events)。
    """
    if bool(bt_cfg.get("portfolio_mode", True)):
        return _backtest_portfolio(
            events, prob, bt_cfg, risk_cfg, payoff, prices, confident=confident,
            ref_trgt=ref_trgt,
        )
    return _backtest_independent(
        events, prob, bt_cfg, risk_cfg, payoff, confident=confident,
        ref_trgt=ref_trgt,
    )


def _cost(size: float, bars: float, roundtrip_cost: float, funding: float) -> float:
    """成交总成本(相对入场名义的分数) = 往返成本 + 持有期资金费。

    ``roundtrip_cost`` 必须来自 ``resolve_roundtrip_cost``, 与 Kelly 扣成本、
    ``decide`` 用的是**同一个数**。旧实现在这里硬算 ``2*(fee+slip)``, 只在
    ``risk.roundtrip_cost_frac`` 为 null(默认)时与 Kelly 口径巧合一致; 一旦显式配置
    该值, 就会出现「按高成本压缩仓位、按低成本扣 PnL」的偏乐观组合。
    """
    return size * (float(roundtrip_cost) + funding * bars)


def resolve_event_slippage(
    base_slip: float,
    trgt: float | None,
    ref_trgt: float,
    bt_cfg: dict,
) -> float:
    """事件级滑点: 可选随相对波动(trgt=atr/close)放大, 永不低于 base。

    CUSUM 偏好波动放大窗; 常数 slip 会在最常开仓时低估成本。
    ``slippage_vol_scale: true`` 时 ``slip = base * clip(trgt/ref, 1, cap)``。
    """
    if not bool(bt_cfg.get("slippage_vol_scale", False)):
        return float(base_slip)
    cap = float(bt_cfg.get("slippage_vol_mult_cap", 3.0) or 3.0)
    cap = max(cap, 1.0)
    if trgt is None or not np.isfinite(trgt) or ref_trgt <= 0 or not np.isfinite(ref_trgt):
        return float(base_slip)
    scale = float(trgt) / float(ref_trgt)
    scale = float(np.clip(scale, 1.0, cap))
    return float(base_slip * scale)


def _ref_trgt_from_events(df: pd.DataFrame, ref_trgt: float | None = None) -> float:
    """相对波动的中位数参考(用于 vol-scale slip)。

    ``ref_trgt`` 有值(有限正数)时直接采用: 研究/WF 回测应传入**训练窗**冻结的参考值,
    与 ``decide``/``serve`` 用的 ``trained["slip_ref_trgt"]`` 同一个数。
    否则回退到本次回测样本的中位数——这在评估窗上是「用被评估区间自身的统计量定成本」,
    虽不能被用来预测方向, 但会让研究回测与部署的成本模型不同口径。
    """
    if ref_trgt is not None:
        v = float(ref_trgt)
        if np.isfinite(v) and v > 0:
            return v
    if "trgt" not in df.columns:
        return float("nan")
    v = pd.to_numeric(df["trgt"], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return float("nan")
    return float(np.median(v))


def _backtest_independent(
    events: pd.DataFrame,
    prob: np.ndarray,
    bt_cfg: dict,
    risk_cfg: dict,
    payoff: float,
    confident: np.ndarray | None = None,
    ref_trgt: float | None = None,
) -> dict:
    """旧口径: 每笔独立对全权益复利(会高估收益)。仅作对照。"""
    df = events.copy()
    df["prob"] = prob
    if confident is not None:
        df["confident"] = np.asarray(confident, dtype=bool)
    df = df.dropna(subset=["prob", "ret"]).sort_index()

    thr = float(bt_cfg.get("prob_threshold", 0.55))
    fee = float(bt_cfg.get("fee_bps", 5.0)) / 1e4
    slip = float(bt_cfg.get("slippage_bps", 2.0)) / 1e4
    funding = float(bt_cfg.get("funding_bps_per_bar", 0.0)) / 1e4
    kf = float(risk_cfg.get("kelly_fraction", 0.5))
    maxp = float(risk_cfg.get("max_position_pct", 0.3))
    daily_max_dd = float(risk_cfg.get("daily_max_drawdown", 0.0) or 0.0)
    has_conf = "confident" in df.columns
    slip_ref = _ref_trgt_from_events(df, ref_trgt)

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

        conf_ok = bool(r["confident"]) if has_conf else True
        if halted_today or (not conf_ok) or r["prob"] < thr:
            sizes.append(0.0)
            rets.append(0.0)
            halted_flags.append(bool(halted_today))
            continue
        trgt_i = float(r["trgt"]) if "trgt" in r.index and pd.notna(r["trgt"]) else None
        slip_i = resolve_event_slippage(slip, trgt_i, slip_ref, bt_cfg)
        rt_cost = resolve_roundtrip_cost(risk_cfg, fee=fee, slip=slip_i)
        min_edge = float(risk_cfg.get("min_kelly_edge", 0.0) or 0.0)
        if min_edge > 0.0 and kelly_fraction(float(r["prob"]), payoff, cost=rt_cost) < min_edge:
            sizes.append(0.0)
            rets.append(0.0)
            halted_flags.append(False)
            continue
        size = position_size(r["prob"], payoff, kf, maxp, cost=rt_cost)
        bars = float(r["bars_held"]) if has_bars else 1.0
        cost = _cost(size, bars, rt_cost, funding)
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
    prices: pd.Series | None = None,
    confident: np.ndarray | None = None,
    ref_trgt: float | None = None,
) -> dict:
    """组合级: 锁定并发仓位, 平仓释放后再开。

    - 已实现权益(equity)在出场时按**入场名义加性**记账:
      Δequity = entry_equity × (size×(e^ret-1) - cost), 避免重叠仓乘积复利虚高。
    - pnl 列仍为相对入场权益的分数贡献(供夏普/胜率); entry_equity 列供对账。
    - 盯市权益(mark)在每个时间线节点按收盘价重估持仓浮盈亏, 用于 MDD 与日内熔断。
    - 并发上限按**名义额**记账: 每仓名义 = size × 入场时权益, 上限 =
      max_gross_exposure × 当前权益。旧实现直接累加 ``size``(各自相对不同的入场权益),
      权益漂移后 Σsize 既不是当下杠杆也不是入场杠杆, 会随净值涨跌系统性放松/收紧闸门。
    """
    df = events.copy()
    df["prob"] = np.asarray(prob, dtype=float)
    if confident is not None:
        df["confident"] = np.asarray(confident, dtype=bool)
    need = ["prob", "ret", "t1"]
    df = df.dropna(subset=need).sort_index()
    if len(df) == 0:
        empty = df.copy()
        empty["size"] = []
        empty["pnl"] = []
        empty["halted"] = []
        empty["entry_equity"] = []
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
    has_conf = "confident" in df.columns
    # 名字必须与循环内的**盯市权益**参考区分开: 两者曾同叫 ``ref``, 循环里的
    # ``ref = _mark(ts)`` 会覆盖波动参考, 使 vol-scale 滑点按 trgt/权益(≈1.0) 算
    # 倍数 → clip 后恒为 1, 组合模式下的波动放大滑点静默失效(成本被系统性低估)。
    slip_ref = _ref_trgt_from_events(df, ref_trgt)

    # 盯市: 需要收盘价 + 事件方向 side(用于按收盘价重估持仓浮盈亏)
    mtm_enabled = prices is not None and "side" in df.columns
    px: pd.Series | None = None
    if mtm_enabled:
        p = pd.Series(prices).astype(float)
        p = p[~p.index.duplicated(keep="last")]
        px = p.where(p > 0).dropna()

    def _px_at(ts):
        if px is None or ts not in px.index:
            return None
        return float(px.loc[ts])

    n = len(df)
    sizes = np.zeros(n, dtype=float)
    pnls = np.zeros(n, dtype=float)
    entry_equities = np.zeros(n, dtype=float)
    halted_flags = np.zeros(n, dtype=bool)
    skipped_cap = np.zeros(n, dtype=bool)

    # 时间线: 同一时刻先平仓释放资金, 再考虑开仓
    entries = [(df.index[i], 1, i) for i in range(n)]  # kind=1 entry
    exits = [(pd.Timestamp(df["t1"].iloc[i]), 0, i) for i in range(n)]  # kind=0 exit first
    timeline = sorted(entries + exits, key=lambda x: (x[0], x[1]))

    equity = 1.0          # 已实现权益(加性记账)
    locked_notional = 0.0  # 未平仓名义额合计(单位=初始权益)
    open_pos: dict[int, dict] = {}
    day_key = None
    day_start_ref = 1.0   # 当日起始参考权益(盯市优先, 否则已实现)
    halted_today = False
    equity_pts: list[tuple[pd.Timestamp, float]] = []
    mtm_pts: list[tuple[pd.Timestamp, float]] = []
    # 盯市浮动的增量累加器(见 _mark): floating(P) = P·mtm_a − mtm_b
    mtm_a = 0.0           # Σ wᵢ / P_entryᵢ
    mtm_b = 0.0           # Σ wᵢ,   wᵢ = entry_equityᵢ × sizeᵢ × sideᵢ

    def _mark(ts) -> float:
        """当前盯市权益 = 已实现权益 + 各仓按入场权益计的浮动盈亏。

        浮动口径与 ``_position_log_return`` / 已实现 ``expm1(ret)`` **同形**:
        ``frac = size × side × (P_t / P_entry − 1)``。
        旧实现 ``size × ((P_t/P_entry)^side − 1)`` 多头巧合一致、空头系统性偏离
        (下跌浮盈偏高、上涨浮亏偏低), 会扭曲 MDD / 日内熔断。

        实现上把求和拆开(代数恒等, 数值同形)::

            Σ wᵢ(P/P₀ᵢ − 1) = P·Σ(wᵢ/P₀ᵢ) − Σwᵢ

        两个和在开/平仓时增量维护, 于是每个时间线节点是 O(1) 而不是 O(并发仓数);
        原实现在 2N 个节点上各遍历一次持仓, 重叠越重越慢(vertical_barrier_bars=48 的
        CUSUM 事件流并发常有数十仓)。
        """
        if not mtm_enabled:
            return equity
        pt = _px_at(ts)
        if pt is None:
            return equity
        return equity + (pt * mtm_a - mtm_b)

    for ts, kind, i in timeline:
        d = ts.date() if hasattr(ts, "date") else None
        # 先按当前持仓的盯市权益判定熔断(捕捉并发持仓的浮动回撤)
        mark_ref = _mark(ts)
        if d != day_key:
            day_key = d
            day_start_ref = mark_ref
            halted_today = False
        if daily_max_dd > 0 and not halted_today:
            if mark_ref <= day_start_ref * (1.0 - daily_max_dd):
                halted_today = True
        mtm_pts.append((ts, mark_ref))

        if kind == 0:  # exit
            if i not in open_pos:
                continue
            pos = open_pos.pop(i)
            size = pos["size"]
            bars = pos["bars"]
            ret = pos["ret"]
            entry_eq = pos["entry_equity"]
            cost = _cost(size, bars, pos["rt_cost"], funding)
            pnl_frac = size * (np.exp(ret) - 1.0) - cost
            equity += entry_eq * pnl_frac
            locked_notional = max(0.0, locked_notional - size * entry_eq)
            mtm_a -= pos.get("mtm_a", 0.0)
            mtm_b -= pos.get("mtm_b", 0.0)
            pnls[i] = pnl_frac
            entry_equities[i] = entry_eq
            equity_pts.append((ts, equity))
            mtm_pts.append((ts, _mark(ts)))  # 出场后重估
            continue

        # entry
        row = df.iloc[i]
        conf_ok = bool(row["confident"]) if has_conf else True
        if halted_today or (not conf_ok) or float(row["prob"]) < thr:
            halted_flags[i] = bool(halted_today)
            continue
        trgt_i = float(row["trgt"]) if "trgt" in row.index and pd.notna(row["trgt"]) else None
        slip_i = resolve_event_slippage(slip, trgt_i, slip_ref, bt_cfg)
        rt_cost = resolve_roundtrip_cost(risk_cfg, fee=fee, slip=slip_i)
        min_edge = float(risk_cfg.get("min_kelly_edge", 0.0) or 0.0)
        if min_edge > 0.0 and kelly_fraction(
            float(row["prob"]), payoff, cost=rt_cost,
        ) < min_edge:
            continue
        want = position_size(float(row["prob"]), payoff, kf, maxp, cost=rt_cost)
        # 名义额口径: 可用名义 = max_gross×当前权益 − 已锁定名义, 再折回「相对入场权益」。
        # 分母用**盯市**权益 mark_ref(无价格序列时即已实现权益), 与 MDD/日内熔断同一基准:
        # 浮亏未了结时真实保证金已经缩水, 用已实现权益当分母会在回撤中偏松放仓。
        entry_eq = float(equity)
        avail_notional = max(0.0, max_gross * float(mark_ref) - locked_notional)
        avail = avail_notional / max(entry_eq, 1e-12)
        size = min(want, avail)
        if size < min_size:
            skipped_cap[i] = True
            continue
        bars = float(row["bars_held"]) if has_bars else 1.0
        sizes[i] = size
        locked_notional += size * entry_eq
        pos = {
            "size": size, "bars": bars, "ret": float(row["ret"]),
            "entry_equity": entry_eq,
            "slip": slip_i,
            "rt_cost": rt_cost,
            "mtm_a": 0.0, "mtm_b": 0.0,
        }
        if mtm_enabled:
            pos["side"] = float(row["side"])
            ep = _px_at(ts)
            pos["entry_px"] = ep  # None ⇒ 该仓不参与盯市
            pos["mtm"] = ep is not None
            if ep is None:
                pos["side"] = 0.0  # 无入场价 => 浮动恒 0, 不污染 mark
            elif ep > 0:
                w = entry_eq * size * pos["side"]
                pos["mtm_a"] = w / ep
                pos["mtm_b"] = w
                mtm_a += pos["mtm_a"]
                mtm_b += pos["mtm_b"]
        open_pos[i] = pos

    # 若回测窗口结束仍有未平仓(t1 超出样本), 按标签收益强制了结(记账在实际了结时刻)
    for i, pos in list(open_pos.items()):
        size = pos["size"]
        entry_eq = pos["entry_equity"]
        cost = _cost(size, pos["bars"], pos["rt_cost"], funding)
        pnl_frac = size * (np.exp(pos["ret"]) - 1.0) - cost
        equity += entry_eq * pnl_frac
        pnls[i] = pnl_frac
        entry_equities[i] = entry_eq
        locked_notional = max(0.0, locked_notional - size * entry_eq)
        mtm_a -= pos.get("mtm_a", 0.0)
        mtm_b -= pos.get("mtm_b", 0.0)
        close_ts = pd.Timestamp(df["t1"].iloc[i])
        equity_pts.append((close_ts, equity))
        mtm_pts.append((close_ts, equity))
    open_pos.clear()

    df = df.copy()
    df["size"] = sizes
    df["pnl"] = pnls
    df["entry_equity"] = entry_equities
    df["halted"] = halted_flags
    df["skipped_capacity"] = skipped_cap

    if equity_pts:
        eq_ser = pd.Series({t: e for t, e in equity_pts}).sort_index()
        # 同时间多笔出场时保留最后净值
        eq_ser = eq_ser[~eq_ser.index.duplicated(keep="last")]
    else:
        eq_ser = pd.Series([1.0], index=[df.index[0]] if len(df) else pd.DatetimeIndex([]))

    mtm_ser = None
    if mtm_enabled and mtm_pts:
        mtm_ser = pd.Series({t: e for t, e in mtm_pts}).sort_index()
        mtm_ser = mtm_ser[~mtm_ser.index.duplicated(keep="last")]

    return _pack_result(df, eq_ser, halted_flags.tolist(), mode="portfolio", mtm_equity=mtm_ser)


def _pack_result(
    df: pd.DataFrame, equity: pd.Series, halted_flags, mode: str,
    mtm_equity: pd.Series | None = None,
) -> dict:
    traded = df[df["size"] > 0] if "size" in df.columns and len(df) else df.iloc[0:0]
    # 年化: 用平均唯一性折算的**有效独立成交数**, 避免重叠事件把年化夏普放大
    avg_uniq = _avg_uniqueness_from_intervals(traded) if len(traded) else 1.0
    ppy = _periods_per_year(df.index if len(df) else equity.index, len(traded), avg_uniq)
    eq_vals = equity.values if len(equity) else np.array([1.0])
    # MDD: 优先用盯市权益(含持仓浮动回撤), 否则回退到出场记账的已实现权益
    dd_vals = mtm_equity.values if (mtm_equity is not None and len(mtm_equity)) else eq_vals
    metrics = {
        "n_events": int(len(df)),
        "n_trades": int(len(traded)),
        "n_halted": int(sum(halted_flags)) if halted_flags is not None else 0,
        "total_return": float(eq_vals[-1] - 1.0) if len(eq_vals) else 0.0,
        # 每笔口径(历史字段, 语义冻结; CPCV/DSR 仍吃此口径)
        "sharpe": sharpe_ratio(df["pnl"].values) if len(df) and "pnl" in df.columns else 0.0,
        "sharpe_annualized": sharpe_ratio(
            df["pnl"].values if len(df) and "pnl" in df.columns else np.array([]), ppy
        ),
        "max_drawdown": max_drawdown(dd_vals),
        "max_drawdown_realized": max_drawdown(eq_vals),
        "mark_to_market": bool(mtm_equity is not None and len(mtm_equity) > 0),
        "avg_uniqueness": float(avg_uniq),
        "n_trades_effective": float(len(traded) * avg_uniq),
        "win_rate": float((traded["pnl"] > 0).mean()) if len(traded) else 0.0,
        "avg_pnl": float(traded["pnl"].mean()) if len(traded) else 0.0,
        "portfolio_mode": mode,
    }
    # 权益曲线夏普: 纯增量字段, 不改 sharpe / sharpe_annualized / 成交明细
    eq_sr = equity_curve_sharpe(equity)
    metrics["sharpe_equity"] = eq_sr["sharpe"]
    metrics["sharpe_equity_annualized"] = eq_sr["sharpe_annualized"]
    if mtm_equity is not None and len(mtm_equity) > 0:
        mtm_sr = equity_curve_sharpe(mtm_equity)
        metrics["sharpe_equity_mtm"] = mtm_sr["sharpe"]
        metrics["sharpe_equity_mtm_annualized"] = mtm_sr["sharpe_annualized"]
    else:
        metrics["sharpe_equity_mtm"] = metrics["sharpe_equity"]
        metrics["sharpe_equity_mtm_annualized"] = metrics["sharpe_equity_annualized"]
    if "skipped_capacity" in df.columns:
        metrics["n_skipped_capacity"] = int(df["skipped_capacity"].sum())
    mdd = abs(metrics["max_drawdown"]) + 1e-9
    metrics["calmar"] = metrics["total_return"] / mdd
    out_equity = mtm_equity if (mtm_equity is not None and len(mtm_equity)) else equity
    return {"metrics": metrics, "equity": equity, "equity_mtm": out_equity, "detail": df}


def _avg_uniqueness_from_intervals(traded: pd.DataFrame) -> float:
    """成交事件的平均唯一性(时间加权 1/并发数)。

    并发度越高(持有期重叠越多), 唯一性越低; 用于把"每笔夏普"年化时折算有效独立样本数,
    避免把强重叠的成交当成独立观测而系统性高估年化夏普。返回 (0, 1] 的标量。
    """
    if "t1" not in traded.columns or len(traded) == 0:
        return 1.0
    try:
        starts = pd.DatetimeIndex(traded.index).asi8.astype(np.int64)
        ends = pd.DatetimeIndex(pd.to_datetime(traded["t1"].values, utc=True)).asi8.astype(np.int64)
    except Exception:
        return 1.0
    m = len(starts)
    valid = ends > starts
    if not valid.any():
        return 1.0
    starts, ends = starts[valid], ends[valid]
    m = len(starts)
    pts = np.unique(np.concatenate([starts, ends]))
    if len(pts) < 2:
        return 1.0
    seg_len = np.diff(pts).astype(float)                       # 每段时长
    left = np.searchsorted(pts, starts, side="left")
    right = np.searchsorted(pts, ends, side="left")            # 覆盖段 [left, right)
    # 并发数: 差分数组累计
    diff = np.zeros(len(pts), dtype=float)
    np.add.at(diff, left, 1.0)
    np.add.at(diff, right, -1.0)
    conc = np.cumsum(diff)[:-1]                                # 每段并发数
    inv = np.divide(1.0, conc, out=np.zeros_like(conc, dtype=float), where=conc > 0)
    # 时间加权段内均值用前缀和 O(1) 取(等价于逐仓 slice 求和, 去掉 O(成交数×段数) 循环)
    c_dur = np.concatenate([[0.0], np.cumsum(seg_len)])
    c_wsum = np.concatenate([[0.0], np.cumsum(inv * seg_len)])
    lo = np.clip(left, 0, len(seg_len))
    hi = np.clip(right, 0, len(seg_len))
    dur = c_dur[hi] - c_dur[lo]
    wsum = c_wsum[hi] - c_wsum[lo]
    uniq = np.divide(wsum, dur, out=np.ones(m, dtype=float), where=dur > 0)
    u = float(np.mean(uniq))
    return u if np.isfinite(u) and u > 0 else 1.0


def _periods_per_year(index, n_trades: int, avg_uniqueness: float = 1.0) -> float:
    """按成交时间跨度估计"每年**有效独立**成交数", 用于把每笔夏普年化。

    有效独立数 = 成交数 × 平均唯一性(重叠越重、折算越多), 而非直接用成交数。
    """
    try:
        n_eff = max(n_trades * float(avg_uniqueness), 0.0)
        if len(index) < 2 or n_eff < 2:
            return 0.0
        span_days = (index[-1] - index[0]).total_seconds() / 86400.0
        if span_days <= 0:
            return 0.0
        return float(n_eff / (span_days / 365.25))
    except Exception:
        return 0.0


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 0.0) -> float:
    """每笔 ``pnl_frac`` 夏普(历史口径)。

    丢弃恰好为 0 的收益后再算 mean/std——与早期实现一致, **请勿改此行为**
    (CPCV/DSR/既有实验依赖)。账户级表现请看 ``equity_curve_sharpe``。
    """
    r = np.asarray(returns, dtype=float)
    r = r[r != 0] if (r != 0).any() else r
    if len(r) < 2 or r.std() == 0:
        return 0.0
    sr = r.mean() / (r.std() + 1e-12)
    if periods_per_year:
        sr *= np.sqrt(periods_per_year)
    return float(sr)


def equity_curve_sharpe(equity: pd.Series | np.ndarray) -> dict:
    """由权益曲线相邻点简单收益计算夏普(与每笔 ``sharpe_ratio`` 独立)。

    - 保留 0 收益段(平坦权益), **不**套用 ``sharpe_ratio`` 的去零逻辑。
    - 年化: ``(mean/std) * sqrt(n_periods / years)``, years 由首末时间戳跨度估计;
      无时间索引或跨度无效时 ``sharpe_annualized=0``。
    - 不修改任何回测成交/仓位逻辑; 仅供 ``metrics`` 增量字段。

    返回 ``{sharpe, sharpe_annualized, n_periods, periods_per_year}``。
    """
    out = {
        "sharpe": 0.0,
        "sharpe_annualized": 0.0,
        "n_periods": 0,
        "periods_per_year": 0.0,
    }
    if isinstance(equity, pd.Series):
        eq = equity.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        idx = eq.index
        vals = eq.values
    else:
        vals = np.asarray(equity, dtype=float)
        vals = vals[np.isfinite(vals)]
        idx = None
    if len(vals) < 2:
        return out

    # 简单收益; 权益非正则跳过该段(防御)
    prev = vals[:-1]
    curr = vals[1:]
    ok = (np.abs(prev) > 1e-15) & np.isfinite(prev) & np.isfinite(curr)
    if not ok.any():
        return out
    rets = (curr[ok] - prev[ok]) / prev[ok]
    out["n_periods"] = int(len(rets))
    if len(rets) < 2 or float(np.std(rets)) == 0.0:
        return out
    raw = float(np.mean(rets) / (float(np.std(rets)) + 1e-12))
    out["sharpe"] = raw

    ppy = 0.0
    if idx is not None and len(idx) >= 2:
        try:
            t0 = pd.Timestamp(idx[0])
            t1 = pd.Timestamp(idx[-1])
            if t0.tzinfo is None:
                t0 = t0.tz_localize("UTC")
            if t1.tzinfo is None:
                t1 = t1.tz_localize("UTC")
            span_days = (t1 - t0).total_seconds() / 86400.0
            if span_days > 0:
                ppy = float(len(rets) / (span_days / 365.25))
        except Exception:
            ppy = 0.0
    out["periods_per_year"] = ppy
    if ppy > 0:
        out["sharpe_annualized"] = float(raw * np.sqrt(ppy))
    return out


def max_drawdown(equity: np.ndarray) -> float:
    """最大回撤(负数)。

    ``peak<=0``(权益被打穿到 0 以下的极端路径)时该段回撤记为 -1(全损), 而不是让
    ``(eq-peak)/peak`` 产生 inf/nan 静默污染 metrics 与 calmar。
    """
    eq = np.asarray(equity, dtype=float)
    eq = eq[np.isfinite(eq)]
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = np.divide(
        eq - peak, peak,
        out=np.where(eq < peak, -1.0, 0.0),
        where=peak > 0,
    )
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
    # 高偏度/高夏普时括号内可为负; clamp 后再开方, 避免 NaN 污染报告
    var_term = (1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr**2) / (n_obs - 1)
    sr_std = float(np.sqrt(max(var_term, 0.0)))
    if sr_std <= 0.0:
        return float("nan")
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
