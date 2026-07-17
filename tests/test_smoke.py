"""主干冒烟测试: 在小规模合成数据上跑通 数据->特征->标注->集成->校准->回测。

只启用 gbdt 专家以保证无 torch 也能跑; deep_ts 在完整流程中另测。
运行: pytest -q  或  python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset, train_and_validate, latest_decision


def _small_cfg() -> Config:
    cfg = Config.load()
    cfg.raw["data"]["use_synthetic"] = True
    cfg.raw["news"]["use_synthetic"] = True  # 与合成行情配对, 守卫允许
    cfg.raw["data"]["synthetic_bars"] = 4000
    cfg.raw["experts"]["enabled"] = ["gbdt"]  # 冒烟只用 GBDT
    cfg.raw["validation"]["n_splits"] = 4
    cfg.raw["backtest"]["portfolio_mode"] = True
    return cfg


def test_trunk_runs():
    cfg = _small_cfg()
    ds = prepare_dataset(cfg, "BTC/USDT")
    assert len(ds.y) > 100, "事件太少"
    assert ds.X.shape[0] == len(ds.y)

    trained = train_and_validate(cfg, ds)
    rep = trained["report"]
    assert 0.0 <= rep["auc"] <= 1.0 or rep["auc"] != rep["auc"]  # 允许 nan
    assert rep["n"] > 0

    bt = trained["backtest"]["metrics"]
    assert "sharpe" in bt and "max_drawdown" in bt

    assert ds.data_source == "synthetic"
    d = latest_decision(cfg, ds, trained)
    assert d["signal"] in {"LONG", "SHORT", "HOLD"}
    if d.get("reason") == "not_cusum_event":
        assert d["signal"] == "HOLD"  # 与训练 CUSUM 事件对齐
    else:
        assert d["win_probability"] is not None
        assert 0.0 <= d["win_probability"] <= 1.0
    print("SMOKE OK:", {"auc": rep["auc"], "sharpe": bt["sharpe"], "decision": d["signal"]})


if __name__ == "__main__":
    test_trunk_runs()
    print("所有冒烟测试通过。")
