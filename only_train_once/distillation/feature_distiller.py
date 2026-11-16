"""
层输出分布对齐模块
使用前向/反向KL散度对齐学生和教师的中间层输出分布。

关键优化:
- 使用 PyTorch hooks 捕获中间层输出，避免修改模型代码
- 使用通道维度的统计量而非全量特征图，减少计算量
- 支持自动匹配学生和教师的对应层
"""

import logging
from typing import List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureDistiller:
    """层输出分布对齐器

    使用KL散度对齐学生和教师的中间层输出分布。

    Args:
        mode: KL散度模式
            - 'forward': KL(p_teacher || p_student) - 模式覆盖
            - 'reverse': KL(p_student || p_teacher) - 模式寻找
            - 'js': Jensen-Shannon散度 - 平衡两者
        feature_weight: 特征对齐loss权重
        channel_reduce: 通道降维方式
            - 'mean': 对通道取均值
            - 'max': 对通道取最大值
            - 'none': 保留所有通道
        normalize: 是否对特征进行归一化
    """

    def __init__(
        self,
        mode: str = 'forward',
        feature_weight: float = 0.1,
        channel_reduce: str = 'mean',
        normalize: bool = True
    ):
        self.mode = mode
        self.feature_weight = feature_weight
        self.channel_reduce = channel_reduce
        self.normalize = normalize

        if mode not in ('forward', 'reverse', 'js'):
            raise ValueError(f"mode 必须是 'forward', 'reverse' 或 'js'，收到 '{mode}'")

        # 存储中间层输出
        self.student_features: Dict[str, torch.Tensor] = {}
        self.teacher_features: Dict[str, torch.Tensor] = {}

        # 钩子句柄
        self._student_hooks: List = []
        self._teacher_hooks: List = []

    def register_hooks(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        layer_names: Optional[List[str]] = None
    ):
        """注册前向钩子捕获中间层输出

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            layer_names: 要捕获的层名称列表，None则自动选择
        """
        # 清除旧钩子
        self.remove_hooks()

        # 自动选择层（如果未指定）
        if layer_names is None:
            layer_names = self._auto_select_layers(student_model, teacher_model)

        # 注册学生钩子
        for name in layer_names:
            module = self._get_module_by_name(student_model, name)
            if module is not None:
                hook = module.register_forward_hook(
                    self._make_hook_fn(self.student_features, name)
                )
                self._student_hooks.append(hook)

        # 注册教师钩子
        for name in layer_names:
            module = self._get_module_by_name(teacher_model, name)
            if module is not None:
                hook = module.register_forward_hook(
                    self._make_hook_fn(self.teacher_features, name)
                )
                self._teacher_hooks.append(hook)

        logger.info(f"注册了 {len(self._student_hooks)} 个学生钩子, {len(self._teacher_hooks)} 个教师钩子")

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self._student_hooks:
            hook.remove()
        for hook in self._teacher_hooks:
            hook.remove()
        self._student_hooks.clear()
        self._teacher_hooks.clear()
        self.student_features.clear()
        self.teacher_features.clear()

    def _make_hook_fn(self, feature_dict: Dict, name: str) -> Callable:
        """创建钩子函数"""
        def hook_fn(module, input, output):
            feature_dict[name] = output
        return hook_fn

    def _auto_select_layers(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> List[str]:
        """自动选择要对齐的层

        选择策略：选择相同类型的模块（Conv, Linear, BN等）
        """
        student_layers = self._get_named_modules(student_model)
        teacher_layers = self._get_named_modules(teacher_model)

        # 找到相同名称的层
        common_names = set(student_layers.keys()) & set(teacher_layers.keys())

        # 过滤出有意义的层（Conv, Linear, BN）
        meaningful_types = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
        selected = []
        for name in sorted(common_names):
            student_module = student_layers[name]
            teacher_module = teacher_layers[name]
            if isinstance(student_module, meaningful_types) and \
               isinstance(teacher_module, meaningful_types):
                selected.append(name)

        logger.info(f"自动选择了 {len(selected)} 个层进行对齐")
        return selected

    def _get_named_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """获取所有命名模块"""
        return dict(model.named_modules())

    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """根据名称获取模块"""
        try:
            return dict(model.named_modules()).get(name)
        except Exception:
            return None

    def compute_loss(self) -> torch.Tensor:
        """计算特征对齐loss

        Returns:
            特征对齐loss (标量)
        """
        if not self.student_features or not self.teacher_features:
            logger.warning("特征字典为空，返回0")
            return torch.tensor(0.0)

        losses = []

        # 对每个匹配的层计算loss
        common_layers = set(self.student_features.keys()) & set(self.teacher_features.keys())

        for layer_name in common_layers:
            student_feat = self.student_features[layer_name]
            teacher_feat = self.teacher_features[layer_name]

            # 检查形状是否兼容
            if student_feat.shape != teacher_feat.shape:
                logger.debug(f"层 {layer_name} 形状不匹配: student={student_feat.shape}, teacher={teacher_feat.shape}")
                continue

            loss = self._compute_layer_loss(student_feat, teacher_feat)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.feature_weight

    def _compute_layer_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """计算单层的特征对齐loss

        Args:
            student_feat: 学生特征 [B, C, ...]
            teacher_feat: 教师特征 [B, C, ...]

        Returns:
            单层loss
        """
        # 通道降维
        if self.channel_reduce == 'mean':
            # 对通道取均值，保留空间维度
            student_stat = student_feat.mean(dim=1, keepdim=True)
            teacher_stat = teacher_feat.mean(dim=1, keepdim=True)
        elif self.channel_reduce == 'max':
            # 对通道取最大值
            student_stat = student_feat.max(dim=1, keepdim=True)[0]
            teacher_stat = teacher_feat.max(dim=1, keepdim=True)[0]
        else:  # 'none'
            student_stat = student_feat
            teacher_stat = teacher_feat

        # 展平空间维度
        student_flat = student_stat.flatten(start_dim=1)  # [B, D]
        teacher_flat = teacher_stat.flatten(start_dim=1)  # [B, D]

        # 归一化
        if self.normalize:
            student_flat = F.normalize(student_flat, dim=-1)
            teacher_flat = F.normalize(teacher_flat, dim=-1)

        # 计算分布
        student_dist = F.softmax(student_flat, dim=-1)
        teacher_dist = F.softmax(teacher_flat, dim=-1)

        # 计算KL散度
        student_log_dist = F.log_softmax(student_flat, dim=-1)
        teacher_log_dist = F.log_softmax(teacher_flat, dim=-1)

        if self.mode == 'forward':
            # KL(p_teacher || p_student)
            loss = F.kl_div(student_log_dist, teacher_dist, reduction='batchmean')
        elif self.mode == 'reverse':
            # KL(p_student || p_teacher)
            loss = F.kl_div(teacher_log_dist, student_dist, reduction='batchmean')
        else:  # js
            # Jensen-Shannon散度
            m = 0.5 * (student_dist + teacher_dist)
            loss = 0.5 * F.kl_div(student_log_dist, m, reduction='batchmean') + \
                   0.5 * F.kl_div(teacher_log_dist, m, reduction='batchmean')

        return loss

    def compute_loss_with_hooks(
        self,
        student_model: nn.Module,
        teacher_models: List[nn.Module],
        x: torch.Tensor
    ) -> torch.Tensor:
        """计算特征对齐loss（自动注册钩子）

        Args:
            student_model: 学生模型
            teacher_models: 教师模型列表
            x: 输入数据

        Returns:
            特征对齐loss
        """
        total_loss = torch.tensor(0.0, device=x.device)

        for teacher_model in teacher_models:
            # 注册钩子
            self.register_hooks(student_model, teacher_model)

            # 前向传播
            with torch.no_grad():
                teacher_model(x)
            student_model(x)

            # 计算loss
            loss = self.compute_loss()
            total_loss = total_loss + loss

            # 清除钩子
            self.remove_hooks()

        # 多教师取平均
        if len(teacher_models) > 0:
            total_loss = total_loss / len(teacher_models)

        return total_loss
