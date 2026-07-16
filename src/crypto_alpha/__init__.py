"""crypto_alpha: 加密货币(BTC/ETH)多专家集成 方向概率 + 风控 预测系统。

模块划分:
    data        数据采集与存储
    features    特征工程 (含分数阶差分)
    labeling    三重障碍标注 + 元标签 + 样本权重
    validation  Purged K-Fold / CPCV 防泄漏验证
    experts     四类专家模型 (GBDT / 深度时序 / TSFM / LLM)
    ensemble    Stacking 元学习器
    calibration 概率校准 + 保形预测
    backtest    含成本回测 + 去偏夏普 / PBO
    risk        分数 Kelly 仓位 + 波动率止损
    pipeline    端到端编排
"""

__version__ = "0.1.0"
