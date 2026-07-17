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
from ..data import load_symbol_data
from ..features.build import build_feature_matrix
from ..features.news_features import add_news_features
from ..labeling.meta_labeling import primary_signal
from ..pipeline import prepare_dataset, train_and_validate
from ..pipeline.run import _attach_news_to_llm, _is_tradable_event
from ..risk.sizing import decide


@dataclass
class ModelBundle:
    ensemble: object
    calibrator: object
    feature_cols: list
    conformal: object = None
    cusum_full_sampling: bool = False
    data_source: str = "real"


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
        self.models[symbol] = ModelBundle(
            ensemble=trained["ensemble"],
            calibrator=trained["calibrator"],
            feature_cols=ds.feature_cols,
            conformal=trained.get("conformal"),
            cusum_full_sampling=bool(trained.get("cusum_full_sampling", ds.cusum_full_sampling)),
            data_source=ds.data_source,
        )
        m = trained["backtest"]["metrics"]
        print(
            f"[train] {symbol}: 事件={m['n_events']} 夏普={m['sharpe']:.3f} "
            f"胜率={m['win_rate']:.3f} 数据={trained.get('data_mode_zh', ds.data_source)}"
        )

    def train_all(self) -> None:
        for s in self._symbols:
            self.train(s)

    # ---------------- 实时推理 ----------------
    def decide_live(self, symbol: str) -> dict | None:
        bundle = self.models[symbol]
        fcols = bundle.feature_cols

        raw = load_symbol_data(self.cfg, symbol)  # 拉取最新数据(合成或真实)
        data_source = str(getattr(raw, "attrs", {}).get("data_source", bundle.data_source))
        feat = build_feature_matrix(raw, self.cfg, symbol=symbol)
        feat["close"] = raw["close"]
        feat["high"] = raw["high"] if "high" in raw.columns else raw["close"]
        feat["low"] = raw["low"] if "low" in raw.columns else raw["close"]
        feat = add_news_features(feat, self.cfg, symbol)  # 与训练特征保持一致

        lc = self.cfg["labeling"]
        side_ser = primary_signal(feat["close"], kind=lc["primary_signal"],
                                  lookback=int(lc["primary_lookback"]))
        # 与训练 prepare_dataset 一致: side 在建模特征中时写入面板
        if "side" in fcols:
            feat = feat.copy()
            feat["side"] = side_ser.astype(float)

        valid = feat[fcols].notna().all(axis=1)
        if not valid.any():
            print(f"[warn] {symbol}: 无有效特征行, 跳过。")
            return None
        ts = feat.index[valid][-1]  # 最新一根特征完整的 bar

        if not _is_tradable_event(self.cfg, feat, ts, bundle.cusum_full_sampling):
            return {
                "signal": "HOLD",
                "symbol": symbol,
                "timestamp": str(ts),
                "reason": "not_cusum_event",
                "win_probability": None,
                "suggested_position_pct": 0.0,
                "stop_loss": None,
                "take_profit": None,
                "confident": False,
                "data_source": data_source,
                "execution_assumption": str(
                    self.cfg["risk"].get("execution_assumption", "close_fill")
                ),
            }

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
        entry = float(feat["close"].loc[ts])
        atr = float(feat["atr_14"].loc[ts]) if "atr_14" in feat.columns else entry * 0.01
        pt_sl = (float(lc["pt_sl"][0]), float(lc["pt_sl"][1]))
        payoff = pt_sl[0] / pt_sl[1]
        fee = float(self.cfg["backtest"].get("fee_bps", 5.0)) / 1e4
        slip = float(self.cfg["backtest"].get("slippage_bps", 2.0)) / 1e4
        risk_cfg = dict(self.cfg["risk"])
        d = decide(
            prob, side, entry, atr, risk_cfg,
            prob_threshold=float(self.cfg["backtest"]["prob_threshold"]), payoff=payoff,
            confident=confident, pt_sl=pt_sl, fee=fee, slip=slip,
        )
        d["symbol"] = symbol
        d["timestamp"] = str(ts)
        d["data_source"] = data_source
        return d

    # ---------------- 播报(含去重) ----------------
    def _should_notify(self, symbol: str, signal: str) -> bool:
        scfg = self.cfg["serve"]
        if signal == "HOLD" and not scfg.get("notify_hold", False):
            return False
        if scfg.get("dedupe", True) and self.last_signal.get(symbol) == signal:
            return False
        return True

    def run_once(self) -> list[dict]:
        from .notifier import format_decision

        out = []
        for s in self._symbols:
            if s not in self.models:
                self.train(s)
            d = self.decide_live(s)
            if d is None:
                continue
            out.append(d)
            if self._should_notify(s, d["signal"]):
                self.notifier.send(format_decision(d))
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
