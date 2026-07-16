"""阶段5: CPCV 严谨评估 —— 夏普分布 + 去偏夏普(DSR) + 过拟合概率(PBO)。"""
import _bootstrap  # noqa: F401

import json

import numpy as np

from crypto_alpha.config import Config
from crypto_alpha.pipeline import prepare_dataset, build_experts, cpcv_report


def main():
    cfg = Config.load()
    for symbol in cfg["data"]["symbols"]:
        print(f"\n===== {symbol} CPCV =====")
        ds = prepare_dataset(cfg, symbol)
        rep = cpcv_report(cfg, ds, build_experts)
        print(f"回测路径数 φ = {rep['n_paths']}")
        print(f"路径夏普: 均值={rep['mean_sharpe']:.3f} 标准差={rep['std_sharpe']:.3f}")
        print(f"去偏夏普 DSR(真实夏普>0 的概率) = {rep['deflated_sharpe']:.3f}")
        print(f"过拟合概率 PBO = {rep['pbo']:.3f}  (越低越好, <0.5 较可信)")
        print(f"各配置平均夏普: "
              f"{dict(zip(rep['config_names'], np.round(rep['perf_matrix'].mean(1), 3)))}")


if __name__ == "__main__":
    main()
