"""实时决策服务: 训练一次 -> 周期性刷新数据出决策 -> 去重 -> 播报, 并按周期重训。

关键: 对"最新一根 bar"出决策**不需要标签**(三重障碍无法标注最近的 bar)。
因此服务复用历史训练好的集成+校准器, 每个周期只重算最新特征并推理。
与训练一致: CUSUM 事件采样时非事件 bar 强制 HOLD; 每轮刷新 LLM 新闻。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..data import load_symbol_data, refresh_market_data
from ..features.build import build_feature_matrix
from ..features.news_features import add_news_features
from ..labeling.meta_labeling import primary_signal
from ..pipeline import prepare_dataset, train_and_validate
from ..backtest.engine import resolve_event_slippage
from ..diagnostics.decision_audit import attach_decision_audit, build_decision_audit
from ..diagnostics.env_guard import hold_for_environment, should_hold_for_environment
from ..pipeline.run import (
    _attach_news_to_llm,
    _is_tradable_event,
    align_feature_schema,
    hold_for_schema_mismatch,
)
from ..risk.sizing import decide
from .notifier import attach_decision_description, format_decision


@dataclass
class ModelBundle:
    ensemble: object
    calibrator: object
    feature_cols: list
    conformal: object = None
    cusum_full_sampling: bool = False
    data_source: str = "real"
    prob_threshold_effective: float | None = None
    slip_ref_trgt: float | None = None
    train_degradations: list | None = None


class DecisionService:
    def __init__(self, cfg: Config, notifier):
        self.cfg = cfg
        self.notifier = notifier
        self.models: dict[str, ModelBundle] = {}
        self.last_signal: dict[str, str] = {}
        self._symbols = cfg["serve"].get("symbols") or cfg["data"]["symbols"]

    # ---------------- 训练 ----------------
    def train(self, symbol: str) -> None:
        ds = prepare_dataset(self.cfg, symbol)
        trained = train_and_validate(self.cfg, ds)
        _srt = trained.get("slip_ref_trgt")
        self.models[symbol] = ModelBundle(
            ensemble=trained["ensemble"],
            calibrator=trained["calibrator"],
            feature_cols=ds.feature_cols,
            conformal=trained.get("conformal"),
            cusum_full_sampling=bool(trained.get("cusum_full_sampling", ds.cusum_full_sampling)),
            data_source=ds.data_source,
            prob_threshold_effective=(
                float(trained["prob_threshold_effective"])
                if trained.get("prob_threshold_effective") is not None
                else None
            ),
            slip_ref_trgt=float(_srt) if _srt is not None and np.isfinite(float(_srt)) else None,
            train_degradations=list(trained.get("degradations") or ds.degradations or []),
        )
        m = trained["backtest"]["metrics"]
        md = (trained.get("backtest_deploy") or {}).get("metrics") or {}
        print(
            f"[train] {symbol}: 研究夏普={m['sharpe']:.3f} 胜率={m['win_rate']:.3f} "
            f"部署成交={md.get('n_trades', '?')} "
            f"thr_r={trained.get('prob_threshold_research')} "
            f"thr_d={trained.get('prob_threshold_effective', self.cfg['backtest'].get('prob_threshold'))} "
            f"数据={trained.get('data_mode_zh', ds.data_source)}"
        )

    def train_all(self) -> None:
        for s in self._symbols:
            self.train(s)

    # ---------------- 实时推理 ----------------
    def decide_live(self, symbol: str) -> dict | None:
        bundle = self.models[symbol]
        fcols = bundle.feature_cols

        dcfg = self.cfg["data"]
        if (
            bool(dcfg.get("refresh_before_decide", True))
            and not dcfg.get("use_synthetic", False)
        ):
            raw = refresh_market_data(self.cfg, symbol)  # 刷到当下已收盘 tip
        else:
            raw = load_symbol_data(self.cfg, symbol)
        data_source = str(getattr(raw, "attrs", {}).get("data_source", bundle.data_source))
        feat = build_feature_matrix(raw, self.cfg, symbol=symbol)
        feat["close"] = raw["close"]
        feat["high"] = raw["high"] if "high" in raw.columns else raw["close"]
        feat["low"] = raw["low"] if "low" in raw.columns else raw["close"]
        feat = add_news_features(feat, self.cfg, symbol)  # 与训练特征保持一致
        # 环境 HOLD 只计「本次 tip/特征装配」标签; 训练期质量标签仅并入展示
        live_deg: list[str] = []
        for t in list(getattr(raw, "attrs", {}).get("degradations") or []):
            if t not in live_deg:
                live_deg.append(t)
        if getattr(raw, "attrs", {}).get("tip_exchange_mismatch"):
            tag = "ohlcv_tip_exchange_fallback"
            if tag not in live_deg:
                live_deg.append(tag)
        for t in list(getattr(feat, "attrs", {}).get("degradations") or []):
            if t not in live_deg:
                live_deg.append(t)
        if data_source == "synthetic_fallback" and "ohlcv_synthetic_fallback" not in live_deg:
            live_deg.append("ohlcv_synthetic_fallback")
        deg_all: list[str] = list(live_deg)
        for t in list(bundle.train_degradations or []):
            if t not in deg_all:
                deg_all.append(t)

        lc = self.cfg["labeling"]
        side_ser = primary_signal(feat["close"], kind=lc["primary_signal"],
                                  lookback=int(lc["primary_lookback"]))
        # 与训练 prepare_dataset 一致: side 在建模特征中时写入面板
        if "side" in fcols:
            feat = feat.copy()
            feat["side"] = side_ser.astype(float)
            if "liq_align" in fcols and "liq_imbalance" in feat.columns:
                feat["liq_align"] = feat["side"].astype(float) * feat["liq_imbalance"].astype(float)

        def _audit(d: dict) -> dict:
            trained_like = {
                "prob_threshold_effective": bundle.prob_threshold_effective,
                "cusum_full_sampling": bundle.cusum_full_sampling,
                "data_source": data_source,
                "degradations": list(d.get("degradations") or deg_all),
                "ensemble": bundle.ensemble,
            }
            audit = build_decision_audit(
                self.cfg, panel=feat, feature_cols=fcols, trained=trained_like,
                degradations=list(d.get("degradations") or deg_all),
            )
            return attach_decision_audit(attach_decision_description(d), audit)

        # 辅周期/新闻装配失败会导致训练期 feature_cols 整列缺失 → 补 0 防 KeyError,
        # 但分布已偏移: 强制 HOLD, 绝不在坏 schema 上推理开仓。
        feat, missing = align_feature_schema(feat, fcols)
        if missing:
            print(
                f"[warn] {symbol}: 特征列与训练 schema 不一致, 强制 HOLD; "
                f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
            _ts = feat.index[-1] if len(feat) else None
            _close = float(feat["close"].iloc[-1]) if len(feat) and "close" in feat.columns else None
            return hold_for_schema_mismatch(
                symbol=symbol, missing_cols=missing, risk_cfg=self.cfg["risk"],
                timestamp=_ts, close=_close,
                data_source=data_source,
            )

        valid = feat[fcols].notna().all(axis=1)
        if not valid.any():
            print(f"[warn] {symbol}: 无有效特征行, 跳过。")
            return None
        ts = feat.index[valid][-1]  # 最新一根特征完整的 bar
        bar_close = float(feat["close"].loc[ts])

        env_thr = self.cfg["risk"].get("env_degradation_hold_score", 50)
        # 只对 live_deg 计分(不含 train_degradations)
        hold_env, env_score, env_tag = should_hold_for_environment(live_deg, threshold=env_thr)
        if hold_env and env_tag:
            return _audit(hold_for_environment(
                symbol=symbol, score=env_score, reason_tag=env_tag,
                risk_cfg=self.cfg["risk"], timestamp=ts, close=bar_close,
                data_source=data_source, degradations=deg_all,
            ))

        if not _is_tradable_event(self.cfg, feat, ts, bundle.cusum_full_sampling):
            from ..risk.sizing import resolve_execution_assumption

            return _audit({
                "signal": "HOLD",
                "symbol": symbol,
                "timestamp": str(ts),
                "close": bar_close,
                "reason": "not_cusum_event",
                "win_probability": None,
                "suggested_position_pct": 0.0,
                "stop_loss": None,
                "take_profit": None,
                "confident": False,
                "data_source": data_source,
                "execution_assumption": resolve_execution_assumption(self.cfg["risk"]),
                "degradations": deg_all,
            })

        # 刷新需要时序面板的专家 + LLM 新闻快照
        for e in bundle.ensemble.experts:
            if getattr(e, "needs_panel", False):
                e.set_panel(feat)
        _attach_news_to_llm(self.cfg, symbol, bundle.ensemble.experts)

        X_last = feat.loc[[ts], fcols].copy()
        if "side" not in X_last.columns:
            X_last["side"] = side_ser.loc[ts]
        prob = float(bundle.calibrator.transform(bundle.ensemble.predict_proba(X_last))[0])

        confident = True
        if bundle.conformal is not None:
            confident = bool(bundle.conformal.predict_set(np.array([prob]))["confident"][0])

        side = int(side_ser.loc[ts])
        entry = bar_close
        atr = float(feat["atr_14"].loc[ts]) if "atr_14" in feat.columns else entry * 0.01
        pt_sl = (float(lc["pt_sl"][0]), float(lc["pt_sl"][1]))
        payoff = pt_sl[0] / pt_sl[1]
        fee = float(self.cfg["backtest"].get("fee_bps", 5.0)) / 1e4
        slip = float(self.cfg["backtest"].get("slippage_bps", 2.0)) / 1e4
        trgt_now = atr / entry if entry > 0 else None
        ref = bundle.slip_ref_trgt if bundle.slip_ref_trgt is not None else float("nan")
        slip = resolve_event_slippage(slip, trgt_now, float(ref), self.cfg["backtest"])
        risk_cfg = dict(self.cfg["risk"])
        thr_eff = bundle.prob_threshold_effective
        if thr_eff is None:
            thr_eff = float(self.cfg["backtest"]["prob_threshold"])
        d = decide(
            prob, side, entry, atr, risk_cfg,
            prob_threshold=float(thr_eff), payoff=payoff,
            confident=confident, pt_sl=pt_sl, fee=fee, slip=slip,
        )
        d["prob_threshold_effective"] = float(thr_eff)
        d["symbol"] = symbol
        d["timestamp"] = str(ts)
        d["close"] = bar_close
        d["data_source"] = data_source
        d["degradations"] = deg_all
        d["env_degradation_score"] = env_score
        return _audit(d)

    # ---------------- 播报(含去重) ----------------
    def _should_notify(self, symbol: str, signal: str) -> bool:
        scfg = self.cfg["serve"]
        if signal == "HOLD" and not scfg.get("notify_hold", False):
            return False
        if scfg.get("dedupe", True) and self.last_signal.get(symbol) == signal:
            return False
        return True

    def run_once(self) -> list[dict]:
        out = []
        for s in self._symbols:
            if s not in self.models:
                self.train(s)
            d = self.decide_live(s)
            if d is None:
                continue
            out.append(d)
            if self._should_notify(s, d["signal"]):
                self.notifier.send(d.get("description") or format_decision(d))
            self.last_signal[s] = d["signal"]
        return out

    # ---------------- 循环调度 ----------------
    def run_forever(self) -> None:
        scfg = self.cfg["serve"]
        poll = int(scfg.get("poll_seconds", 3600))
        retrain_every = int(scfg.get("retrain_every_cycles", 24))
        cycle = 0
        self.train_all()
        while True:
            self.run_once()
            cycle += 1
            if retrain_every > 0 and cycle % retrain_every == 0:
                print("[serve] 周期重训…")
                self.train_all()
            time.sleep(poll)
