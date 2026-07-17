"""实时决策服务: 训练一次 -> 周期性刷新数据出决策 -> 去重 -> 播报, 并按周期重训。

关键: 对"最新一根 bar"出决策**不需要标签**(三重障碍无法标注最近的 bar)。
因此服务复用历史训练好的集成+校准器, 每个周期只重算最新特征并推理。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data import load_symbol_data
from ..features.build import build_feature_matrix, feature_columns
from ..features.news_features import add_news_features
from ..labeling.meta_labeling import primary_signal
from ..pipeline import prepare_dataset, train_and_validate
from ..risk.sizing import decide


@dataclass
class ModelBundle:
    ensemble: object
    calibrator: object
    feature_cols: list
    conformal: object = None


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
        )
        m = trained["backtest"]["metrics"]
        print(f"[train] {symbol}: 事件={m['n_events']} 夏普={m['sharpe']:.3f} 胜率={m['win_rate']:.3f}")

    def train_all(self) -> None:
        for s in self._symbols:
            self.train(s)

    # ---------------- 实时推理 ----------------
    def decide_live(self, symbol: str) -> dict | None:
        bundle = self.models[symbol]
        fcols = bundle.feature_cols

        raw = load_symbol_data(self.cfg, symbol)  # 拉取最新数据(合成或真实)
        feat = build_feature_matrix(raw, self.cfg, symbol=symbol)
        feat["close"] = raw["close"]
        feat = add_news_features(feat, self.cfg, symbol)  # 与训练特征保持一致

        lc = self.cfg["labeling"]
        side_ser = primary_signal(feat["close"], kind=lc["primary_signal"],
                                  lookback=int(lc["primary_lookback"]))

        valid = feat[fcols].notna().all(axis=1)
        if not valid.any():
            print(f"[warn] {symbol}: 无有效特征行, 跳过。")
            return None
        ts = feat.index[valid][-1]  # 最新一根特征完整的 bar

        # 刷新需要时序面板的专家, 使其看到截至最新 bar 的历史
        for e in bundle.ensemble.experts:
            if getattr(e, "needs_panel", False):
                e.set_panel(feat)

        X_last = feat.loc[[ts], fcols].copy()
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
        d = decide(
            prob, side, entry, atr, self.cfg["risk"],
            prob_threshold=float(self.cfg["backtest"]["prob_threshold"]), payoff=payoff,
            confident=confident, pt_sl=pt_sl,
        )
        d["symbol"] = symbol
        d["timestamp"] = str(ts)
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
        print(f"[serve] 启动: 币种={self._symbols} 轮询={poll}s 重训周期={retrain_every} 次")
        self.train_all()
        cycle = 0
        while True:
            cycle += 1
            if cycle > 1 and retrain_every > 0 and cycle % retrain_every == 0:
                print(f"[serve] 第 {cycle} 轮: 触发重训。")
                self.train_all()
            try:
                self.run_once()
            except Exception as e:  # 单轮失败不应终止服务
                print(f"[error] 第 {cycle} 轮出错: {e}")
            time.sleep(poll)
