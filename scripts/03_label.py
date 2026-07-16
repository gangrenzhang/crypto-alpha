"""阶段3: 三重障碍 + 元标签, 打印标签分布与样本权重概览。"""
import _bootstrap  # noqa: F401

import numpy as np

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset


def main():
    cfg = Config.load()
    for symbol in cfg["data"]["symbols"]:
        ds = prepare_dataset(cfg, symbol)
        pos = float(np.mean(ds.y))
        print(f"[ok] {symbol}: 事件数={len(ds.y)}, 正类(盈利)占比={pos:.3f}, "
              f"平均收益={ds.events['ret'].mean():.5f}, "
              f"样本权重均值={ds.sample_weight.mean():.3f}")


if __name__ == "__main__":
    main()
