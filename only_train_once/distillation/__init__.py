"""
蒸馏模块
提供4个核心类：
1. TeacherEnsemble - 教师集合管理
2. SoftLabelDistiller - 软标签蒸馏
3. FeatureDistiller - 每层输出分布对齐
4. WeightDistiller - 每层权重特征对齐
5. Distiller - 统一蒸馏器（整合接口）
"""

from .core import (
    TeacherEnsemble,
    SoftLabelDistiller,
    FeatureDistiller,
    WeightDistiller,
    Distiller,
)

__all__ = [
    'TeacherEnsemble',
    'SoftLabelDistiller',
    'FeatureDistiller',
    'WeightDistiller',
    'Distiller',
]
