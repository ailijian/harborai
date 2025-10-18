"""
数据一致性模块

该模块提供数据一致性检查和修正功能，包括：
- 数据一致性检查器
- 数据库约束管理器
- 自动修正服务
- 实时监控
"""

from .data_consistency_checker import (
    DataConsistencyChecker,
    ConsistencyIssue,
    ConsistencyReport,
    IssueType,
    IssueSeverity
)
from .database_constraint_manager import (
    DatabaseConstraintManager,
    ConstraintViolation,
    ConstraintType,
    ViolationSeverity
)
from .auto_correction_service import (
    AutoCorrectionService,
    CorrectionAction,
    CorrectionResult,
    ActionType,
    CorrectionStatus
)

__all__ = [
    'DataConsistencyChecker',
    'ConsistencyIssue',
    'ConsistencyReport',
    'IssueType',
    'IssueSeverity',
    'DatabaseConstraintManager',
    'ConstraintViolation',
    'ConstraintType',
    'ViolationSeverity',
    'AutoCorrectionService',
    'CorrectionAction',
    'CorrectionResult',
    'ActionType',
    'CorrectionStatus'
]