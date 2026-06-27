"""
逐层同步蒸馏模块
创新点：
1. 渐进式层解锁：由浅到深，符合学习规律
2. 自适应温度：浅层低温（保留细节），深层高温（捕捉语义）
3. 置信度选择性蒸馏：Student 困惑时才蒸馏
4. 特征归一化：处理 Student/Teacher 特征尺度差异
5. 层级对比学习：不仅对齐绝对值，还对齐相对关系

运算量优化：
- Teacher 只需 1 次完整前向（复用上一层输出）
- 缓存 Teacher 中间层输出
- 支持低秩近似减少对齐维度
"""

import logging
import math
from typing import List, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LayerWiseDistiller:
    """逐层同步蒸馏器

    Args:
        mode: 对齐模式
            - 'forward': KL(p_teacher || p_student)
            - 'reverse': KL(p_student || p_teacher)
            - 'js': Jensen-Shannon 散度
            - 'contrastive': 对比学习
        progressive: 是否渐进式解锁层
        adaptive_temp: 是否自适应温度
        selective: 是否置信度选择性蒸馏
        normalize: 是否特征归一化
        base_weight: 基础蒸馏权重
        confidence_threshold: 置信度阈值（selective=True 时生效）
    """

    def __init__(
        self,
        mode: str = 'forward',
        progressive: bool = True,
        adaptive_temp: bool = True,
        selective: bool = True,
        normalize: bool = True,
        base_weight: float = 0.1,
        confidence_threshold: float = 0.7,
    ):
        self.mode = mode
        self.progressive = progressive
        self.adaptive_temp = adaptive_temp
        self.selective = selective
        self.normalize = normalize
        self.base_weight = base_weight
        self.confidence_threshold = confidence_threshold

        # 层缓存
        self._student_features: Dict[str, torch.Tensor] = {}
        self._teacher_features: Dict[str, torch.Tensor] = {}
        self._student_hooks: List = []
        self._teacher_hooks: List = []

        # 层重要性权重（可学习或固定）
        self._layer_weights: Dict[str, float] = {}

    def _get_layer_groups(
        self,
        model: nn.Module,
        group_type: str = 'conv'
    ) -> List[List[str]]:
        """将模型层分组（浅/中/深）

        Args:
            model: 模型
            group_type: 分组类型

        Returns:
            层名列表的列表 [[浅层], [中层], [深层]]
        """
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layers.append(name)

        # 分成3组
        n = len(layers)
        if n <= 3:
            return [layers]

        third = n // 3
        return [
            layers[:third],          # 浅层
            layers[third:2*third],   # 中层
            layers[2*third:],        # 深层
        ]

    def _get_progressive_schedule(
        self,
        epoch: int,
        max_epoch: int,
        total_groups: int
    ) -> List[int]:
        """渐进式解锁调度

        Args:
            epoch: 当前epoch
            max_epoch: 总epoch数
            total_groups: 总组数

        Returns:
            当前要解锁的组索引列表
        """
        if not self.progressive:
            return list(range(total_groups))

        # 每阶段解锁一组
        stage_duration = max_epoch // total_groups
        current_stage = min(epoch // stage_duration, total_groups - 1)

        return list(range(current_stage + 1))

    def _get_adaptive_temperature(
        self,
        layer_idx: int,
        total_layers: int,
        base_temp: float = 4.0
    ) -> float:
        """自适应温度：浅层低温，深层高温

        Args:
            layer_idx: 当前层索引
            total_layers: 总层数
            base_temp: 基础温度

        Returns:
            当前层的温度
        """
        if not self.adaptive_temp:
            return base_temp

        # 浅层: temp ≈ 2.0 (保留细节)
        # 深层: temp ≈ 6.0 (捕捉语义)
        progress = layer_idx / max(total_layers - 1, 1)
        temp = base_temp * (0.5 + progress)  # [2, 6] for base_temp=4

        return temp

    def _compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """计算 Student 的置信度

        Args:
            logits: Student 的输出 logits

        Returns:
            置信度 (0-1)
        """
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        return max_probs.mean()

    def _should_distill(self, student_logits: torch.Tensor) -> bool:
        """判断是否需要蒸馏（置信度选择性）

        Args:
            student_logits: Student 的输出

        Returns:
            是否蒸馏
        """
        if not self.selective:
            return True

        confidence = self._compute_confidence(student_logits)
        return confidence.item() < self.confidence_threshold

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """特征归一化（处理 Student/Teacher 尺度差异）

        Args:
            features: 原始特征

        Returns:
            归一化后的特征
        """
        if not self.normalize:
            return features

        # LayerNorm 归一化
        if features.dim() >= 2:
            # 对通道维度归一化
            mean = features.mean(dim=1, keepdim=True)
            std = features.std(dim=1, keepdim=True) + 1e-8
            return (features - mean) / std
        return features

    def _align_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        temperature: float = 4.0
    ) -> torch.Tensor:
        """计算对齐 loss

        Args:
            student_feat: Student 特征
            teacher_feat: Teacher 特征
            temperature: 温度参数

        Returns:
            对齐 loss
        """
        # 归一化
        student_feat = self._normalize_features(student_feat)
        teacher_feat = self._normalize_features(teacher_feat)

        # 展平空间维度
        if student_feat.dim() > 2:
            student_flat = student_feat.flatten(start_dim=1)
            teacher_flat = teacher_feat.flatten(start_dim=1)
        else:
            student_flat = student_feat
            teacher_flat = teacher_feat

        # 计算分布
        student_log = F.log_softmax(student_flat / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_flat / temperature, dim=-1)
        teacher_log = F.log_softmax(teacher_flat / temperature, dim=-1)
        student_soft = F.softmax(student_flat / temperature, dim=-1)

        if self.mode == 'forward':
            loss = F.kl_div(student_log, teacher_soft, reduction='batchmean')
        elif self.mode == 'reverse':
            loss = F.kl_div(teacher_log, student_soft, reduction='batchmean')
        elif self.mode == 'js':
            m = 0.5 * (student_soft + teacher_soft)
            loss = 0.5 * F.kl_div(student_log, m, reduction='batchmean') + \
                   0.5 * F.kl_div(teacher_log, m, reduction='batchmean')
        elif self.mode == 'contrastive':
            # 对比学习：正样本对齐，负样本推开
            loss = self._contrastive_loss(student_flat, teacher_flat)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss * (temperature ** 2)

    def _contrastive_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """对比学习 loss

        正样本：同一输入的 Student/Teacher 输出
        负样本：batch 内其他样本的 Teacher 输出

        Args:
            student_feat: Student 特征 [B, D]
            teacher_feat: Teacher 特征 [B, D]
            temperature: 对比学习温度

        Returns:
            对比 loss
        """
        # 归一化
        student_norm = F.normalize(student_feat, dim=-1)
        teacher_norm = F.normalize(teacher_feat, dim=-1)

        # 相似度矩阵 [B, B]
        sim_matrix = torch.mm(student_norm, teacher_norm.t()) / temperature

        # 对角线是正样本
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def register_hooks(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        layer_names: List[str]
    ):
        """注册前向钩子"""
        self.remove_hooks()

        for name in layer_names:
            # Student 钩子
            s_module = dict(student_model.named_modules()).get(name)
            if s_module is not None:
                hook = s_module.register_forward_hook(
                    self._make_hook(self._student_features, name)
                )
                self._student_hooks.append(hook)

            # Teacher 钩子
            t_module = dict(teacher_model.named_modules()).get(name)
            if t_module is not None:
                hook = t_module.register_forward_hook(
                    self._make_hook(self._teacher_features, name)
                )
                self._teacher_hooks.append(hook)

    def remove_hooks(self):
        """移除所有钩子"""
        for h in self._student_hooks:
            h.remove()
        for h in self._teacher_hooks:
            h.remove()
        self._student_hooks.clear()
        self._teacher_hooks.clear()
        self._student_features.clear()
        self._teacher_features.clear()

    def _make_hook(self, feature_dict: Dict, name: str) -> Callable:
        """创建钩子函数"""
        def hook_fn(module, input, output):
            feature_dict[name] = output
        return hook_fn

    def compute_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        x: torch.Tensor,
        student_logits: torch.Tensor,
        epoch: int,
        max_epoch: int,
        base_temperature: float = 4.0
    ) -> torch.Tensor:
        """计算逐层蒸馏 loss

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            x: 输入数据
            student_logits: Student 的输出 logits（用于置信度判断）
            epoch: 当前 epoch
            max_epoch: 总 epoch 数
            base_temperature: 基础温度

        Returns:
            蒸馏 loss
        """
        # 置信度选择性：Student 够自信就不蒸馏
        if not self._should_distill(student_logits):
            return torch.tensor(0.0, device=x.device)

        # 获取层分组
        layer_groups = self._get_layer_groups(student_model)
        total_groups = len(layer_groups)

        # 渐进式：确定当前要解锁的组
        active_groups = self._get_progressive_schedule(epoch, max_epoch, total_groups)

        # 注册钩子（只注册激活的层）
        active_layers = []
        for g_idx in active_groups:
            active_layers.extend(layer_groups[g_idx])

        self.register_hooks(student_model, teacher_model, active_layers)

        # 同步前向传播（Teacher 只需 1 次）
        with torch.no_grad():
            teacher_model(x)
        student_model(x)

        # 计算逐层 loss
        losses = []
        for g_idx in active_groups:
            for layer_name in layer_groups[g_idx]:
                if layer_name not in self._student_features or \
                   layer_name not in self._teacher_features:
                    continue

                s_feat = self._student_features[layer_name]
                t_feat = self._teacher_features[layer_name]

                # 形状检查
                if s_feat.shape != t_feat.shape:
                    logger.debug(f"跳过 {layer_name}: 形状不匹配 {s_feat.shape} vs {t_feat.shape}")
                    continue

                # 自适应温度
                temp = self._get_adaptive_temperature(
                    layer_idx=g_idx,
                    total_layers=total_groups,
                    base_temp=base_temperature
                )

                # 计算对齐 loss
                layer_loss = self._align_loss(s_feat, t_feat, temperature=temp)

                # 层权重
                weight = self._layer_weights.get(layer_name, 1.0)
                losses.append(weight * layer_loss)

        # 清理钩子
        self.remove_hooks()

        if not losses:
            return torch.tensor(0.0, device=x.device)

        return torch.stack(losses).mean() * self.base_weight

    def compute_loss_multi_teacher(
        self,
        student_model: nn.Module,
        teacher_models: List[nn.Module],
        x: torch.Tensor,
        student_logits: torch.Tensor,
        epoch: int,
        max_epoch: int
    ) -> torch.Tensor:
        """多教师逐层蒸馏

        Args:
            student_model: 学生模型
            teacher_models: 教师模型列表
            x: 输入数据
            student_logits: Student 输出
            epoch: 当前 epoch
            max_epoch: 总 epoch

        Returns:
            蒸馏 loss
        """
        if not teacher_models:
            return torch.tensor(0.0, device=x.device)

        losses = []
        for teacher in teacher_models:
            loss = self.compute_loss(
                student_model, teacher, x, student_logits, epoch, max_epoch
            )
            losses.append(loss)

        return torch.stack(losses).mean()

    def set_layer_weights(self, weights: Dict[str, float]):
        """设置层权重

        Args:
            weights: {层名: 权重}
        """
        self._layer_weights = weights

    def get_active_layers_info(
        self,
        model: nn.Module,
        epoch: int,
        max_epoch: int
    ) -> Dict:
        """获取当前激活层信息（用于调试/可视化）

        Args:
            model: 模型
            epoch: 当前 epoch
            max_epoch: 总 epoch

        Returns:
            激活层信息
        """
        layer_groups = self._get_layer_groups(model)
        total_groups = len(layer_groups)
        active_groups = self._get_progressive_schedule(epoch, max_epoch, total_groups)

        info = {
            'epoch': epoch,
            'max_epoch': max_epoch,
            'total_groups': total_groups,
            'active_groups': active_groups,
            'active_layers': [],
            'temperatures': {},
        }

        for g_idx in active_groups:
            for layer_name in layer_groups[g_idx]:
                info['active_layers'].append(layer_name)
                temp = self._get_adaptive_temperature(g_idx, total_groups)
                info['temperatures'][layer_name] = temp

        return info


class ProgressiveLayerDistiller(LayerWiseDistiller):
    """渐进式逐层蒸馏（简化接口）

    默认配置：
    - 渐进式解锁
    - 自适应温度
    - 置信度选择性
    - 特征归一化
    """

    def __init__(self, base_weight: float = 0.1):
        super().__init__(
            mode='forward',
            progressive=True,
            adaptive_temp=True,
            selective=True,
            normalize=True,
            base_weight=base_weight,
        )
