"""
蒸馏模块
提供教师模型集合管理、软标签蒸馏、层输出分布对齐、权重特征对齐功能。
"""

from .teacher_ensemble import TeacherEnsemble
from .soft_label_distiller import SoftLabelDistiller
from .feature_distiller import FeatureDistiller
from .weight_distiller import WeightDistiller

__all__ = [
    'TeacherEnsemble',
    'SoftLabelDistiller',
    'FeatureDistiller',
    'WeightDistiller',
]
