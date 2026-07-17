"""闭环完整性诊断: 控制实验(空/正对照)、不变量校验、对账工具、运行期护栏。

用于在**不依赖显卡**的前提下, 快速判断"标注/回测/验证闭环"的代码逻辑是否严谨:
- 离线单测(tests/test_pipeline_integrity.py) 把这些函数当断言闸门;
- 在线体检(scripts/12_audit.py) 在每次训练前后跑一遍并出报告。
"""
from .integrity import (
    CheckResult,
    make_random_walk_ohlcv,
    make_predictable_core_dataset,
    run_core_loop,
    core_auc,
    permutation_baseline,
    count_cv_overlaps,
    embargo_gap_ok,
    backtest_reconciliation,
    max_concurrent_gross,
    sanity_check_dataset,
    run_full_pipeline_with_prices,
    audit_pipeline,
)

__all__ = [
    "CheckResult",
    "make_random_walk_ohlcv",
    "make_predictable_core_dataset",
    "run_core_loop",
    "core_auc",
    "permutation_baseline",
    "count_cv_overlaps",
    "embargo_gap_ok",
    "backtest_reconciliation",
    "max_concurrent_gross",
    "sanity_check_dataset",
    "run_full_pipeline_with_prices",
    "audit_pipeline",
]
