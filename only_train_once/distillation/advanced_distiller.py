"""
高级蒸馏技术模块
实现多种创新蒸馏方法：

1. 动态层权重（Dynamic Layer Weighting）
2. 注意力对齐（Attention Alignment）
3. 温度退火（Temperature Annealing）
4. 样本级课程蒸馏（Sample-level Curriculum）
5. 适配器蒸馏（Adapter Distillation）

所有方法可独立使用或组合使用。
"""

import logging
import math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 1. 动态层权重
# ============================================================================

class DynamicLayerWeighting:
    """动态层权重：根据 Student 学习困难度自动调整

    困难度越高 -> 权重越大 -> 更多关注

    Args:
        temperature: 权重锐化温度
        update_freq: 更新频率（每N个batch更新一次）
        momentum: 权重更新动量
    """

    def __init__(
        self,
        temperature: float = 5.0,
        update_freq: int = 10,
        momentum: float = 0.9,
    ):
        self.temperature = temperature
        self.update_freq = update_freq
        self.momentum = momentum
        self._step = 0
        self._layer_weights: Dict[str, float] = {}

    def compute_weights(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算动态层权重

        Args:
            student_features: Student 中间层特征
            teacher_features: Teacher 中间层特征

        Returns:
            层权重字典
        """
        self._step += 1

        # 每隔 update_freq 步更新一次
        if self._step % self.update_freq != 0 and self._layer_weights:
            return self._layer_weights

        difficulties = {}
        common_layers = set(student_features.keys()) & set(teacher_features.keys())

        for layer_name in common_layers:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            if s_feat.shape != t_feat.shape:
                continue

            # 计算余弦距离作为困难度
            s_flat = s_feat.flatten(start_dim=1).mean(dim=0)
            t_flat = t_feat.flatten(start_dim=1).mean(dim=0)

            cos_sim = F.cosine_similarity(s_flat.unsqueeze(0), t_flat.unsqueeze(0))
            difficulty = 1.0 - cos_sim.item()  # [0, 2]
            difficulties[layer_name] = difficulty

        if not difficulties:
            return {}

        # Softmax 归一化
        layer_names = list(difficulties.keys())
        diff_values = torch.tensor([difficulties[n] for n in layer_names])
        weights = F.softmax(diff_values * self.temperature, dim=0)

        # 动量更新
        new_weights = {}
        for i, name in enumerate(layer_names):
            new_w = weights[i].item()
            old_w = self._layer_weights.get(name, new_w)
            self._layer_weights[name] = self.momentum * old_w + (1 - self.momentum) * new_w
            new_weights[name] = self._layer_weights[name]

        return new_weights

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self._layer_weights.copy()


# ============================================================================
# 2. 注意力对齐
# ============================================================================

class AttentionDistiller:
    """注意力对齐：对齐 Student/Teacher 的注意力图

    注意力图捕捉"哪些区域更重要"，比原始特征更具语义

    Args:
        weight: 注意力对齐权重
        normalize: 是否归一化注意力图
    """

    def __init__(self, weight: float = 0.1, normalize: bool = True):
        self.weight = weight
        self.normalize = normalize

    def _compute_attention(self, feature_map: torch.Tensor) -> torch.Tensor:
        """计算注意力图

        Args:
            feature_map: 特征图 [B, C, H, W] 或 [B, C]

        Returns:
            注意力图 [B, 1, H, W] 或 [B, 1]
        """
        if feature_map.dim() == 4:
            # [B, C, H, W] -> [B, 1, H, W]
            attention = feature_map.abs().mean(dim=1, keepdim=True)
        elif feature_map.dim() == 3:
            # [B, C, L] -> [B, 1, L]
            attention = feature_map.abs().mean(dim=1, keepdim=True)
        else:
            # [B, C] -> [B, 1]
            attention = feature_map.abs().mean(dim=1, keepdim=True)

        if self.normalize:
            # 归一化到 [0, 1]
            min_val = attention.min()
            max_val = attention.max()
            if max_val - min_val > 1e-8:
                attention = (attention - min_val) / (max_val - min_val)

        return attention

    def compute_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算注意力对齐 loss

        Args:
            student_features: Student 中间层特征
            teacher_features: Teacher 中间层特征

        Returns:
            注意力对齐 loss
        """
        losses = []
        common_layers = set(student_features.keys()) & set(teacher_features.keys())

        for layer_name in common_layers:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            if s_feat.shape != t_feat.shape:
                continue

            s_attention = self._compute_attention(s_feat)
            t_attention = self._compute_attention(t_feat)

            # MSE 对齐注意力
            loss = F.mse_loss(s_attention, t_attention)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight


# ============================================================================
# 3. 温度退火
# ============================================================================

class TemperatureAnnealer:
    """温度退火：随训练进度逐渐降低温度

    早期高温 -> 软标签，探索更多
    后期低温 -> 硬标签，精炼细节

    Args:
        init_temp: 初始温度
        min_temp: 最小温度
        anneal_type: 退火类型
            - 'linear': 线性退火
            - 'cosine': 余弦退火
            - 'exponential': 指数退火
    """

    def __init__(
        self,
        init_temp: float = 10.0,
        min_temp: float = 2.0,
        anneal_type: str = 'cosine',
    ):
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.anneal_type = anneal_type

    def get_temperature(self, epoch: int, max_epoch: int) -> float:
        """获取当前温度

        Args:
            epoch: 当前 epoch
            max_epoch: 总 epoch 数

        Returns:
            当前温度
        """
        progress = epoch / max(max_epoch - 1, 1)

        if self.anneal_type == 'linear':
            temp = self.init_temp - (self.init_temp - self.min_temp) * progress
        elif self.anneal_type == 'cosine':
            temp = self.min_temp + 0.5 * (self.init_temp - self.min_temp) * \
                   (1 + math.cos(math.pi * progress))
        elif self.anneal_type == 'exponential':
            decay = 0.01  # 衰减率
            temp = self.min_temp + (self.init_temp - self.min_temp) * math.exp(-decay * epoch)
        else:
            temp = self.init_temp

        return max(temp, self.min_temp)


# ============================================================================
# 4. 样本级课程蒸馏
# ============================================================================

class SampleCurriculum:
    """样本级课程蒸馏：先简单后困难

    Args:
        warmup_epochs: 预热阶段（只蒸馏简单样本）
        difficulty_threshold: 困难度阈值
        adaptive: 是否自适应调整阈值
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        difficulty_threshold: float = 0.5,
        adaptive: bool = True,
    ):
        self.warmup_epochs = warmup_epochs
        self.difficulty_threshold = difficulty_threshold
        self.adaptive = adaptive
        self._epoch_difficulties: List[float] = []

    def get_sample_mask(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        epoch: int
    ) -> torch.Tensor:
        """获取样本蒸馏掩码

        Args:
            student_logits: Student 输出 [B, C]
            labels: 真实标签 [B]
            epoch: 当前 epoch

        Returns:
            掩码 [B]，1 表示蒸馏，0 表示跳过
        """
        # 计算每个样本的困难度
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(
                student_logits, labels, reduction='none'
            )

        # 归一化到 [0, 1]
        max_loss = per_sample_loss.max()
        if max_loss > 0:
            difficulties = per_sample_loss / max_loss
        else:
            difficulties = per_sample_loss

        # 预热阶段：只蒸馏简单样本
        if epoch < self.warmup_epochs:
            threshold = self.difficulty_threshold * (1 - epoch / self.warmup_epochs)
        else:
            threshold = self.difficulty_threshold

        # 自适应调整阈值
        if self.adaptive:
            mean_diff = difficulties.mean().item()
            self._epoch_difficulties.append(mean_diff)
            # 使用历史平均值作为阈值参考
            if len(self._epoch_difficulties) > 10:
                threshold = sum(self._epoch_difficulties[-10:]) / 10

        # 生成掩码
        mask = (difficulties < threshold).float()

        # 确保至少有一些样本被选中
        if mask.sum() == 0:
            mask = torch.ones_like(mask)

        return mask

    def compute_weighted_loss(
        self,
        losses: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        epoch: int
    ) -> torch.Tensor:
        """计算加权蒸馏 loss

        Args:
            losses: 每个样本的 loss [B]
            student_logits: Student 输出
            labels: 真实标签
            epoch: 当前 epoch

        Returns:
            加权 loss
        """
        mask = self.get_sample_mask(student_logits, labels, epoch)

        # 简单样本权重小，困难样本权重大（但在阈值内）
        weights = mask * (1.0 + mask * 0.5)  # 被选中的样本权重 1.5

        if weights.sum() > 0:
            weighted_loss = (losses * weights).sum() / weights.sum()
        else:
            weighted_loss = losses.mean()

        return weighted_loss


# ============================================================================
# 5. 适配器蒸馏
# ============================================================================

class AdapterDistiller(nn.Module):
    """适配器蒸馏：Student 特征通过小网络映射到 Teacher 空间

    优势：
    - 处理 Student/Teacher 特征维度不同
    - 适配器可以学习最优映射
    - 适配器参数可训练

    Args:
        student_dims: Student 各层特征维度
        teacher_dims: Teacher 各层特征维度
        adapter_hidden: 适配器隐藏层维度
        weight: 蒸馏权重
    """

    def __init__(
        self,
        student_dims: Dict[str, int],
        teacher_dims: Dict[str, int],
        adapter_hidden: int = 64,
        weight: float = 0.1,
    ):
        super().__init__()
        self.weight = weight

        # 为每对匹配层创建适配器
        self.adapters = nn.ModuleDict()
        common_layers = set(student_dims.keys()) & set(teacher_dims.keys())

        for layer_name in common_layers:
            s_dim = student_dims[layer_name]
            t_dim = teacher_dims[layer_name]

            if s_dim != t_dim:
                self.adapters[layer_name] = nn.Sequential(
                    nn.Linear(s_dim, adapter_hidden),
                    nn.ReLU(),
                    nn.Linear(adapter_hidden, t_dim),
                )

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算适配器蒸馏 loss

        Args:
            student_features: Student 中间层特征
            teacher_features: Teacher 中间层特征

        Returns:
            蒸馏 loss
        """
        losses = []

        for layer_name, adapter in self.adapters.items():
            if layer_name not in student_features or \
               layer_name not in teacher_features:
                continue

            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            # 展平
            if s_feat.dim() > 2:
                s_flat = s_feat.flatten(start_dim=1).mean(dim=0, keepdim=True)
                t_flat = t_feat.flatten(start_dim=1).mean(dim=0, keepdim=True)
            else:
                s_flat = s_feat
                t_flat = t_feat

            # 适配器映射
            s_adapted = adapter(s_flat)

            # MSE 对齐
            loss = F.mse_loss(s_adapted, t_flat.detach())
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def get_adapted_features(
        self,
        student_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """获取适配后的 Student 特征

        Args:
            student_features: Student 中间层特征

        Returns:
            适配后的特征
        """
        adapted = {}
        for layer_name, adapter in self.adapters.items():
            if layer_name not in student_features:
                continue

            s_feat = student_features[layer_name]
            if s_feat.dim() > 2:
                s_flat = s_feat.flatten(start_dim=1).mean(dim=0, keepdim=True)
            else:
                s_flat = s_feat

            adapted[layer_name] = adapter(s_flat)

        return adapted


# ============================================================================
# 组合蒸馏器
# ============================================================================

class AdvancedDistiller:
    """高级蒸馏器：组合多种技术

    Args:
        use_dynamic_weight: 使用动态层权重
        use_attention: 使用注意力对齐
        use_temperature_anneal: 使用温度退火
        use_curriculum: 使用样本级课程
        use_adapter: 使用适配器
    """

    def __init__(
        self,
        use_dynamic_weight: bool = True,
        use_attention: bool = True,
        use_temperature_anneal: bool = True,
        use_curriculum: bool = True,
        use_adapter: bool = False,
        base_weight: float = 0.1,
    ):
        self.base_weight = base_weight

        # 动态层权重
        self.dynamic_weight = DynamicLayerWeighting() if use_dynamic_weight else None

        # 注意力对齐
        self.attention_distiller = AttentionDistiller(weight=0.1) if use_attention else None

        # 温度退火
        self.temp_annealer = TemperatureAnnealer() if use_temperature_anneal else None

        # 样本级课程
        self.curriculum = SampleCurriculum() if use_curriculum else None

        # 适配器（需要额外初始化）
        self.adapter = None

    def compute_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        student_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        epoch: int,
        max_epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算组合蒸馏 loss

        Args:
            student_features: Student 中间层特征
            teacher_features: Teacher 中间层特征
            student_logits: Student 输出
            labels: 真实标签
            epoch: 当前 epoch
            max_epoch: 总 epoch

        Returns:
            (总loss, 各部分loss字典)
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)

        # 1. 动态层权重
        layer_weights = None
        if self.dynamic_weight is not None:
            layer_weights = self.dynamic_weight.compute_weights(
                student_features, teacher_features
            )

        # 2. 注意力对齐
        if self.attention_distiller is not None:
            att_loss = self.attention_distiller.compute_loss(
                student_features, teacher_features
            )
            total_loss = total_loss + att_loss
            loss_dict['attention'] = att_loss.item()

        # 3. 特征对齐（带动态权重）
        feat_loss = self._compute_feature_loss(
            student_features, teacher_features, layer_weights
        )
        total_loss = total_loss + feat_loss
        loss_dict['feature'] = feat_loss.item()

        # 4. 适配器蒸馏
        if self.adapter is not None:
            adapter_loss = self.adapter(student_features, teacher_features)
            total_loss = total_loss + adapter_loss
            loss_dict['adapter'] = adapter_loss.item()

        # 5. 样本级课程（如果提供标签）
        if self.curriculum is not None and labels is not None:
            per_sample_loss = F.cross_entropy(
                student_logits, labels, reduction='none'
            )
            curriculum_loss = self.curriculum.compute_weighted_loss(
                per_sample_loss, student_logits, labels, epoch
            )
            total_loss = total_loss + curriculum_loss
            loss_dict['curriculum'] = curriculum_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss * self.base_weight, loss_dict

    def _compute_feature_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        layer_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """计算特征对齐 loss"""
        losses = []
        common_layers = set(student_features.keys()) & set(teacher_features.keys())

        for layer_name in common_layers:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            if s_feat.shape != t_feat.shape:
                continue

            # 展平
            s_flat = s_feat.flatten(start_dim=1)
            t_flat = t_feat.flatten(start_dim=1)

            # 归一化
            s_norm = F.normalize(s_flat, dim=-1)
            t_norm = F.normalize(t_flat, dim=-1)

            # MSE
            loss = F.mse_loss(s_norm, t_norm)

            # 动态权重
            weight = layer_weights.get(layer_name, 1.0) if layer_weights else 1.0
            losses.append(weight * loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean()

    def get_temperature(self, epoch: int, max_epoch: int) -> float:
        """获取当前温度"""
        if self.temp_annealer:
            return self.temp_annealer.get_temperature(epoch, max_epoch)
        return 4.0

    def init_adapter(
        self,
        student_dims: Dict[str, int],
        teacher_dims: Dict[str, int]
    ):
        """初始化适配器"""
        self.adapter = AdapterDistiller(
            student_dims=student_dims,
            teacher_dims=teacher_dims,
            adapter_hidden=64,
            weight=0.1
        )
