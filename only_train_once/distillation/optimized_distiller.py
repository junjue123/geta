"""
优化蒸馏模块
实现多种方法优化：

1. 在线蒸馏（Online Distillation）
2. 自蒸馏（Self-Distillation）
3. 特征图重加权（Feature Map Reweighting）
4. 分布匹配（Distribution Matching）
5. 架构感知蒸馏（Architecture-Aware Distillation）
6. 缓存优化（Caching Optimization）
"""

import logging
import math
from typing import List, Dict, Optional, Tuple, Set
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 1. 在线蒸馏
# ============================================================================

class OnlineDistiller:
    """在线蒸馏：教师和Student同时训练，互相学习

    优势：
    - 教师不断适应Student，提供更相关的知识
    - 避免教师和Student分布不匹配
    - 训练过程中教师也在进化

    Args:
        alpha: Student向Teacher学习的权重
        beta: Teacher向Student学习的权重
        temperature: 温度参数
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        temperature: float = 4.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def compute_mutual_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算互相蒸馏loss

        Args:
            student_logits: Student输出
            teacher_logits: Teacher输出
            labels: 真实标签

        Returns:
            (student_loss, teacher_loss)
        """
        T = self.temperature

        # Student向Teacher学习
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        s_kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)

        # Teacher向Student学习
        teacher_log_soft = F.log_softmax(teacher_logits / T, dim=-1)
        student_soft_detach = F.softmax(student_logits.detach() / T, dim=-1)
        t_kd_loss = F.kl_div(teacher_log_soft, student_soft_detach, reduction='batchmean') * (T ** 2)

        # 分类损失
        s_ce_loss = F.cross_entropy(student_logits, labels)
        t_ce_loss = F.cross_entropy(teacher_logits, labels)

        # 总损失
        student_loss = s_ce_loss + self.alpha * s_kd_loss
        teacher_loss = t_ce_loss + self.beta * t_kd_loss

        return student_loss, teacher_loss


# ============================================================================
# 2. 自蒸馏
# ============================================================================

class SelfDistiller:
    """自蒸馏：Student从自己的深层学习浅层

    原理：
    - 深层特征包含更高级的语义信息
    - 浅层通过学习深层，获得更好的表示
    - 不需要外部教师模型

    Args:
        weight: 自蒸馏权重
        projection_dim: 投影维度
    """

    def __init__(
        self,
        weight: float = 0.1,
        projection_dim: Optional[int] = None,
    ):
        self.weight = weight
        self.projection_dim = projection_dim
        self._projection_heads: Dict[str, nn.Linear] = {}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        layer_order: List[str]
    ) -> torch.Tensor:
        """计算自蒸馏loss

        Args:
            features: 各层特征 {层名: 特征}
            layer_order: 层顺序（从浅到深）

        Returns:
            自蒸馏loss
        """
        if len(layer_order) < 2:
            return torch.tensor(0.0)

        losses = []

        # 浅层学习深层
        for i in range(len(layer_order) - 1):
            shallow_name = layer_order[i]
            deep_name = layer_order[i + 1]

            if shallow_name not in features or deep_name not in features:
                continue

            shallow_feat = features[shallow_name]
            deep_feat = features[deep_name]

            # 展平
            if shallow_feat.dim() > 2:
                shallow_flat = shallow_feat.flatten(start_dim=1).mean(dim=1, keepdim=True)
                deep_flat = deep_feat.flatten(start_dim=1).mean(dim=1, keepdim=True)
            else:
                shallow_flat = shallow_feat
                deep_flat = deep_feat

            # 投影（如果维度不同）
            if shallow_flat.shape[-1] != deep_flat.shape[-1]:
                proj = self._get_projection(
                    shallow_name,
                    shallow_flat.shape[-1],
                    deep_flat.shape[-1],
                    shallow_flat.device
                )
                shallow_flat = proj(shallow_flat)

            # 归一化
            shallow_norm = F.normalize(shallow_flat, dim=-1)
            deep_norm = F.normalize(deep_flat.detach(), dim=-1)

            # MSE对齐
            loss = F.mse_loss(shallow_norm, deep_norm)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def _get_projection(
        self,
        layer_name: str,
        in_dim: int,
        out_dim: int,
        device: torch.device
    ) -> nn.Linear:
        """获取或创建投影头"""
        key = f"{layer_name}_{in_dim}_{out_dim}"
        if key not in self._projection_heads:
            proj = nn.Linear(in_dim, out_dim, bias=False).to(device)
            nn.init.xavier_uniform_(proj.weight)
            self._projection_heads[key] = proj
        return self._projection_heads[key]

    def get_projection_parameters(self):
        """获取投影头参数"""
        params = []
        for proj in self._projection_heads.values():
            params.extend(proj.parameters())
        return params


# ============================================================================
# 3. 特征图重加权
# ============================================================================

class FeatureReweighter:
    """特征图重加权：重要区域权重更大

    原理：
    - 不同空间位置的重要性不同
    - 通过注意力机制自动学习重要区域
    - 重要区域的对齐权重更大

    Args:
        weight: 重加权权重
        attention_type: 注意力类型
            - 'channel': 通道注意力
            - 'spatial': 空间注意力
            - 'both': 通道+空间注意力
    """

    def __init__(
        self,
        weight: float = 0.1,
        attention_type: str = 'spatial',
    ):
        self.weight = weight
        self.attention_type = attention_type

    def _compute_attention(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """计算注意力图

        Args:
            features: 特征图 [B, C, H, W]

        Returns:
            注意力权重 [B, 1, H, W] 或 [B, C, 1, 1]
        """
        if features.dim() != 4:
            return torch.ones(features.shape[0], 1, 1, 1, device=features.device)

        if self.attention_type == 'channel':
            # 通道注意力：GAP -> [B, C, 1, 1]
            attention = F.adaptive_avg_pool2d(features, 1)
        elif self.attention_type == 'spatial':
            # 空间注意力：通道平均 -> [B, 1, H, W]
            attention = features.mean(dim=1, keepdim=True)
        else:  # 'both'
            # 通道+空间
            channel_att = F.adaptive_avg_pool2d(features, 1)
            spatial_att = features.mean(dim=1, keepdim=True)
            attention = channel_att * spatial_att

        # 归一化到 [0, 1]
        attention = torch.sigmoid(attention)

        return attention

    def compute_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算重加权对齐loss

        Args:
            student_features: Student特征
            teacher_features: Teacher特征

        Returns:
            重加权loss
        """
        losses = []
        common_layers = set(student_features.keys()) & set(teacher_features.keys())

        for layer_name in common_layers:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            if s_feat.shape != t_feat.shape:
                continue

            # 计算注意力
            attention = self._compute_attention(t_feat)

            # 重加权
            s_weighted = s_feat * attention
            t_weighted = t_feat * attention

            # 归一化
            s_norm = F.normalize(s_weighted.flatten(start_dim=1), dim=-1)
            t_norm = F.normalize(t_weighted.flatten(start_dim=1), dim=-1)

            # MSE对齐
            loss = F.mse_loss(s_norm, t_norm)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight


# ============================================================================
# 4. 分布匹配
# ============================================================================

class DistributionMatcher:
    """分布匹配：使用MMD匹配特征分布

    原理：
    - 点对点对齐忽略分布差异
    - MMD（最大均值差异）匹配整个分布
    - 更鲁棒，对异常值不敏感

    Args:
        weight: 分布匹配权重
        kernel: 核函数类型
            - 'rbf': RBF核
            - 'linear': 线性核
            - 'polynomial': 多项式核
        bandwidth: RBF核带宽
    """

    def __init__(
        self,
        weight: float = 0.1,
        kernel: str = 'rbf',
        bandwidth: float = 1.0,
    ):
        self.weight = weight
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF核函数"""
        xx = (x * x).sum(dim=-1, keepdim=True)
        yy = (y * y).sum(dim=-1, keepdim=True)
        dist = xx + yy.T - 2 * x @ y.T
        return torch.exp(-dist / (2 * self.bandwidth ** 2))

    def _linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """线性核函数"""
        return x @ y.T

    def _compute_mmd(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """计算MMD

        Args:
            x: 样本1 [N, D]
            y: 样本2 [M, D]

        Returns:
            MMD值
        """
        if self.kernel == 'rbf':
            kernel_fn = self._rbf_kernel
        elif self.kernel == 'linear':
            kernel_fn = self._linear_kernel
        else:
            kernel_fn = self._rbf_kernel

        k_xx = kernel_fn(x, x)
        k_yy = kernel_fn(y, y)
        k_xy = kernel_fn(x, y)

        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        return mmd

    def compute_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算分布匹配loss

        Args:
            student_features: Student特征
            teacher_features: Teacher特征

        Returns:
            分布匹配loss
        """
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

            # 计算MMD
            mmd = self._compute_mmd(s_flat, t_flat)
            losses.append(mmd)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight


# ============================================================================
# 5. 架构感知蒸馏
# ============================================================================

class ArchitectureAwareDistiller:
    """架构感知蒸馏：根据架构类型调整策略

    原理：
    - CNN: 对齐空间特征
    - ViT: 对齐注意力图
    - 轻量级: 只对齐关键层

    Args:
        default_weight: 默认权重
    """

    def __init__(self, default_weight: float = 0.1):
        self.default_weight = default_weight
        self._arch_cache: Dict[str, str] = {}

    def detect_architecture(self, model: nn.Module) -> str:
        """检测模型架构类型

        Args:
            model: 模型

        Returns:
            架构类型: 'cnn', 'vit', 'lightweight', 'unknown'
        """
        model_id = id(model)
        if model_id in self._arch_cache:
            return self._arch_cache[model_id]

        has_conv = False
        has_attention = False
        has_depthwise = False
        total_params = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                has_conv = True
                if module.groups > 1:
                    has_depthwise = True
            elif isinstance(module, nn.MultiheadAttention):
                has_attention = True
            total_params += sum(p.numel() for p in module.parameters())

        if has_attention and not has_conv:
            arch = 'vit'
        elif has_depthwise and total_params < 5_000_000:
            arch = 'lightweight'
        elif has_conv:
            arch = 'cnn'
        else:
            arch = 'unknown'

        self._arch_cache[model_id] = arch
        return arch

    def compute_loss(
        self,
        student: nn.Module,
        teacher: nn.Module,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        student_attention: Optional[Dict[str, torch.Tensor]] = None,
        teacher_attention: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """计算架构感知loss

        Args:
            student: Student模型
            teacher: Teacher模型
            student_features: Student特征
            teacher_features: Teacher特征
            student_attention: Student注意力图
            teacher_attention: Teacher注意力图

        Returns:
            架构感知loss
        """
        student_arch = self.detect_architecture(student)
        teacher_arch = self.detect_architecture(teacher)

        # 根据架构选择策略
        if student_arch == 'vit' and teacher_arch == 'vit':
            # ViT: 优先对齐注意力图
            if student_attention and teacher_attention:
                return self._attention_alignment(student_attention, teacher_attention)
            else:
                return self._feature_alignment(student_features, teacher_features)

        elif student_arch == 'lightweight':
            # 轻量级: 只对齐关键层
            return self._selective_alignment(student_features, teacher_features)

        else:
            # CNN等: 标准特征对齐
            return self._feature_alignment(student_features, teacher_features)

    def _feature_alignment(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """标准特征对齐"""
        losses = []
        common_layers = set(student_features.keys()) & set(teacher_features.keys())

        for layer_name in common_layers:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            if s_feat.shape != t_feat.shape:
                continue

            s_norm = F.normalize(s_feat.flatten(start_dim=1), dim=-1)
            t_norm = F.normalize(t_feat.flatten(start_dim=1), dim=-1)

            loss = F.mse_loss(s_norm, t_norm)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.default_weight

    def _attention_alignment(
        self,
        student_attention: Dict[str, torch.Tensor],
        teacher_attention: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """注意力图对齐"""
        losses = []
        common_layers = set(student_attention.keys()) & set(teacher_attention.keys())

        for layer_name in common_layers:
            s_attn = student_attention[layer_name]
            t_attn = teacher_attention[layer_name]

            if s_attn.shape != t_attn.shape:
                continue

            loss = F.mse_loss(s_attn, t_attn)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.default_weight

    def _selective_alignment(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """选择性对齐（只对齐关键层）"""
        # 选择特征图最大的层（通常包含最多信息）
        common_layers = set(student_features.keys()) & set(teacher_features.keys())

        if not common_layers:
            return torch.tensor(0.0)

        # 按特征图大小排序
        sorted_layers = sorted(
            common_layers,
            key=lambda name: student_features[name].numel(),
            reverse=True
        )

        # 只对齐前3层
        top_layers = sorted_layers[:3]

        losses = []
        for layer_name in top_layers:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            if s_feat.shape != t_feat.shape:
                continue

            s_norm = F.normalize(s_feat.flatten(start_dim=1), dim=-1)
            t_norm = F.normalize(t_feat.flatten(start_dim=1), dim=-1)

            loss = F.mse_loss(s_norm, t_norm)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.default_weight


# ============================================================================
# 6. 缓存优化
# ============================================================================

class CachedDistiller:
    """缓存优化蒸馏：避免重复计算教师特征

    Args:
        max_cache_size: 最大缓存大小
        cache_mode: 缓存模式
            - 'epoch': 按epoch缓存
            - 'batch': 按batch缓存
            - 'none': 不缓存
    """

    def __init__(
        self,
        max_cache_size: int = 10,
        cache_mode: str = 'epoch',
    ):
        self.max_cache_size = max_cache_size
        self.cache_mode = cache_mode

        # 缓存字典
        self._feature_cache: OrderedDict = OrderedDict()
        self._logits_cache: OrderedDict = OrderedDict()

    def get_cached_features(
        self,
        key: str,
        compute_fn: callable
    ) -> Dict[str, torch.Tensor]:
        """获取缓存的特征，如果不存在则计算

        Args:
            key: 缓存键
            compute_fn: 计算函数

        Returns:
            特征字典
        """
        if self.cache_mode == 'none':
            return compute_fn()

        if key in self._feature_cache:
            return self._feature_cache[key]

        # 计算并缓存
        features = compute_fn()

        # 管理缓存大小
        if len(self._feature_cache) >= self.max_cache_size:
            self._feature_cache.popitem(last=False)

        self._feature_cache[key] = features
        return features

    def get_cached_logits(
        self,
        key: str,
        compute_fn: callable
    ) -> torch.Tensor:
        """获取缓存的logits

        Args:
            key: 缓存键
            compute_fn: 计算函数

        Returns:
            logits
        """
        if self.cache_mode == 'none':
            return compute_fn()

        if key in self._logits_cache:
            return self._logits_cache[key]

        logits = compute_fn()

        if len(self._logits_cache) >= self.max_cache_size:
            self._logits_cache.popitem(last=False)

        self._logits_cache[key] = logits
        return logits

    def clear_cache(self):
        """清除缓存"""
        self._feature_cache.clear()
        self._logits_cache.clear()

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            'feature_cache_size': len(self._feature_cache),
            'logits_cache_size': len(self._logits_cache),
            'max_cache_size': self.max_cache_size,
        }


# ============================================================================
# 组合优化蒸馏器
# ============================================================================

class OptimizedDistiller:
    """组合优化蒸馏器

    Args:
        use_online: 使用在线蒸馏
        use_self: 使用自蒸馏
        use_reweight: 使用特征重加权
        use_distribution: 使用分布匹配
        use_architecture: 使用架构感知
        use_cache: 使用缓存优化
    """

    def __init__(
        self,
        use_online: bool = False,
        use_self: bool = True,
        use_reweight: bool = True,
        use_distribution: bool = False,
        use_architecture: bool = True,
        use_cache: bool = True,
        base_weight: float = 0.1,
    ):
        self.base_weight = base_weight

        # 初始化各组件
        self.online = OnlineDistiller() if use_online else None
        self.self_distiller = SelfDistiller(weight=0.1) if use_self else None
        self.reweighter = FeatureReweighter(weight=0.1) if use_reweight else None
        self.distribution = DistributionMatcher(weight=0.1) if use_distribution else None
        self.architecture = ArchitectureAwareDistiller() if use_architecture else None
        self.cache = CachedDistiller() if use_cache else None

    def compute_loss(
        self,
        student: nn.Module,
        teacher: nn.Module,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        layer_order: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算组合优化loss

        Args:
            student: Student模型
            teacher: Teacher模型
            student_features: Student特征
            teacher_features: Teacher特征
            student_logits: Student输出
            teacher_logits: Teacher输出
            labels: 真实标签
            layer_order: 层顺序（自蒸馏用）

        Returns:
            (总loss, 各部分loss字典)
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)

        # 1. 在线蒸馏
        if self.online and labels is not None:
            s_loss, t_loss = self.online.compute_mutual_loss(
                student_logits, teacher_logits, labels
            )
            total_loss = total_loss + s_loss
            loss_dict['online_student'] = s_loss.item()
            loss_dict['online_teacher'] = t_loss.item()

        # 2. 自蒸馏
        if self.self_distiller and layer_order:
            self_loss = self.self_distiller.compute_loss(student_features, layer_order)
            total_loss = total_loss + self_loss
            loss_dict['self'] = self_loss.item()

        # 3. 特征重加权
        if self.reweighter:
            reweight_loss = self.reweighter.compute_loss(student_features, teacher_features)
            total_loss = total_loss + reweight_loss
            loss_dict['reweight'] = reweight_loss.item()

        # 4. 分布匹配
        if self.distribution:
            dist_loss = self.distribution.compute_loss(student_features, teacher_features)
            total_loss = total_loss + dist_loss
            loss_dict['distribution'] = dist_loss.item()

        # 5. 架构感知
        if self.architecture:
            arch_loss = self.architecture.compute_loss(
                student, teacher, student_features, teacher_features
            )
            total_loss = total_loss + arch_loss
            loss_dict['architecture'] = arch_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss * self.base_weight, loss_dict

    def get_projection_parameters(self):
        """获取所有投影头参数"""
        params = []
        if self.self_distiller:
            params.extend(self.self_distiller.get_projection_parameters())
        return params

    def clear_cache(self):
        """清除缓存"""
        if self.cache:
            self.cache.clear_cache()
