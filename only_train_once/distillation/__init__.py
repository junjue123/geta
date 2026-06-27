"""
蒸馏模块
提供教师模型集合管理、软标签蒸馏、层输出分布对齐、权重特征对齐功能。
"""

from .teacher_ensemble import TeacherEnsemble
from .soft_label_distiller import SoftLabelDistiller
from .feature_distiller import FeatureDistiller
from .weight_distiller import WeightDistiller
from .layer_wise_distiller import LayerWiseDistiller, ProgressiveLayerDistiller
from .advanced_distiller import (
    DynamicLayerWeighting,
    AttentionDistiller,
    TemperatureAnnealer,
    SampleCurriculum,
    AdapterDistiller,
    AdvancedDistiller,
)
from .pruning_aware_distiller import (
    PruningAwareDistiller,
    PruningAwareTeacherEnsemble,
    PruningStageScheduler,
    IntegratedPruningDistiller,
)

__all__ = [
    'TeacherEnsemble',
    'SoftLabelDistiller',
    'FeatureDistiller',
    'WeightDistiller',
    'LayerWiseDistiller',
    'ProgressiveLayerDistiller',
    'DynamicLayerWeighting',
    'AttentionDistiller',
    'TemperatureAnnealer',
    'SampleCurriculum',
    'AdapterDistiller',
    'AdvancedDistiller',
    'PruningAwareDistiller',
    'PruningAwareTeacherEnsemble',
    'PruningStageScheduler',
    'IntegratedPruningDistiller',
]
