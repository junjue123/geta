"""
剪枝感知蒸馏模块
针对训练中剪枝（GETA框架）设计的蒸馏方法。

核心挑战：
1. Student 特征维度随剪枝变化
2. Teacher 始终保持原始维度
3. 需要动态适应维度变化

解决方案：
1. 投影头（Projection Head）：动态映射不同维度
2. 剪枝掩码（Pruning Mask）：只对齐存活通道
3. 重要性加权（Importance Weighting）：重要通道权重更大
4. 渐进式对齐：随剪枝深入逐步调整
"""

import logging
import math
from typing import List, Dict, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PruningAwareDistiller:
    """剪枝感知蒸馏器

    核心思想：在剪枝过程中动态适应维度变化

    Args:
        weight: 蒸馏权重
        use_projection: 是否使用投影头映射维度
        use_mask: 是否使用剪枝掩码
        use_importance: 是否使用重要性加权
        projection_dim: 投影维度（None则使用Teacher维度）
    """

    def __init__(
        self,
        weight: float = 0.1,
        use_projection: bool = True,
        use_mask: bool = True,
        use_importance: bool = True,
        projection_dim: Optional[int] = None,
    ):
        self.weight = weight
        self.use_projection = use_projection
        self.use_mask = use_mask
        self.use_importance = use_importance
        self.projection_dim = projection_dim

        # 投影头：Student维度 -> Teacher维度
        self._projection_heads: Dict[str, nn.Linear] = {}

        # 剪枝掩码缓存
        self._pruning_masks: Dict[str, torch.Tensor] = {}

        # 重要性分数缓存
        self._importance_scores: Dict[str, torch.Tensor] = {}

        # 维度缓存
        self._student_dims: Dict[str, int] = {}
        self._teacher_dims: Dict[str, int] = {}

    def update_pruning_info(
        self,
        pruned_groups: List[int],
        param_groups: List[Dict],
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ):
        """更新剪枝信息

        Args:
            pruned_groups: 已剪枝的组索引
            param_groups: 参数组信息
            importance_scores: 重要性分数
        """
        # 更新剪枝掩码
        for group in param_groups:
            if not group.get('is_prunable', False):
                continue

            layer_name = self._get_layer_name(group)
            num_groups = group['num_groups']
            pruned_idxes = set(group.get('pruned_idxes', []))

            # 创建掩码：1表示存活，0表示已剪枝
            mask = torch.ones(num_groups)
            for idx in pruned_idxes:
                if idx < num_groups:
                    mask[idx] = 0.0

            self._pruning_masks[layer_name] = mask

        # 更新重要性分数
        if importance_scores:
            self._importance_scores.update(importance_scores)

        logger.debug(f"更新剪枝信息: {len(self._pruning_masks)} 层")

    def _get_layer_name(self, group: Dict) -> str:
        """从参数组获取层名"""
        p_names = group.get('p_names', [])
        if p_names:
            # 取第一个参数名，去掉最后的 .weight/.bias
            return '.'.join(p_names[0].split('.')[:-1])
        return 'unknown'

    def _get_or_create_projection(
        self,
        layer_name: str,
        student_dim: int,
        teacher_dim: int,
        device: torch.device
    ) -> nn.Linear:
        """获取或创建投影头

        Args:
            layer_name: 层名
            student_dim: Student特征维度
            teacher_dim: Teacher特征维度
            device: 设备

        Returns:
            投影头
        """
        key = f"{layer_name}_{student_dim}_{teacher_dim}"

        if key not in self._projection_heads:
            out_dim = self.projection_dim or teacher_dim
            proj = nn.Linear(student_dim, out_dim, bias=False).to(device)
            # Xavier初始化
            nn.init.xavier_uniform_(proj.weight)
            self._projection_heads[key] = proj
            logger.debug(f"创建投影头: {layer_name} ({student_dim} -> {out_dim})")

        return self._projection_heads[key]

    def _apply_mask(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        dim: int = 1
    ) -> torch.Tensor:
        """应用剪枝掩码

        Args:
            features: 特征 [B, C, ...]
            mask: 掩码 [C]
            dim: 通道维度

        Returns:
            掩码后的特征
        """
        if not self.use_mask:
            return features

        # 调整mask形状
        shape = [1] * features.dim()
        shape[dim] = -1
        mask = mask.view(shape).to(features.device)

        # 应用掩码
        return features * mask

    def _compute_importance_weight(
        self,
        layer_name: str,
        num_channels: int
    ) -> torch.Tensor:
        """计算重要性权重

        Args:
            layer_name: 层名
            num_channels: 通道数

        Returns:
            权重 [C]
        """
        if not self.use_importance:
            return torch.ones(num_channels)

        if layer_name in self._importance_scores:
            scores = self._importance_scores[layer_name]
            # 归一化到 [0.5, 1.5]
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
                scores = 0.5 + scores
            return scores
        else:
            return torch.ones(num_channels)

    def _align_features(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        layer_name: str,
        temperature: float = 4.0
    ) -> torch.Tensor:
        """对齐特征（处理维度不匹配）

        Args:
            student_feat: Student特征 [B, C_s, ...]
            teacher_feat: Teacher特征 [B, C_t, ...]
            layer_name: 层名
            temperature: 温度

        Returns:
            对齐loss
        """
        B = student_feat.shape[0]
        device = student_feat.device

        # 获取维度
        s_dim = student_feat.shape[1]
        t_dim = teacher_feat.shape[1]

        # 应用剪枝掩码（如果维度匹配）
        if s_dim == t_dim and layer_name in self._pruning_masks:
            mask = self._pruning_masks[layer_name]
            student_feat = self._apply_mask(student_feat, mask)
            teacher_feat = self._apply_mask(teacher_feat, mask)

        # 展平空间维度
        if student_feat.dim() > 2:
            s_flat = student_feat.flatten(start_dim=2).mean(dim=2)  # [B, C_s]
            t_flat = teacher_feat.flatten(start_dim=2).mean(dim=2)  # [B, C_t]
        else:
            s_flat = student_feat  # [B, C_s]
            t_flat = teacher_feat  # [B, C_t]

        # 维度不匹配时使用投影
        if s_dim != t_dim and self.use_projection:
            proj = self._get_or_create_projection(layer_name, s_dim, t_dim, device)
            s_flat = proj(s_flat)  # [B, C_t]

        # 归一化
        s_norm = F.normalize(s_flat, dim=-1)
        t_norm = F.normalize(t_flat, dim=-1)

        # 计算KL散度
        s_log = F.log_softmax(s_norm / temperature, dim=-1)
        t_soft = F.softmax(t_norm / temperature, dim=-1)

        loss = F.kl_div(s_log, t_soft, reduction='batchmean')

        # 重要性加权
        if self.use_importance and layer_name in self._importance_scores:
            imp_weight = self._compute_importance_weight(layer_name, s_flat.shape[1])
            # 对batch维度取均值，对通道维度加权
            if imp_weight.shape[0] == s_flat.shape[1]:
                loss = loss * imp_weight.mean()

        return loss * (temperature ** 2)

    def compute_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        temperature: float = 4.0
    ) -> torch.Tensor:
        """计算剪枝感知蒸馏loss

        Args:
            student_features: Student中间层特征
            teacher_features: Teacher中间层特征
            temperature: 温度

        Returns:
            蒸馏loss
        """
        losses = []
        common_layers = set(student_features.keys()) & set(teacher_features.keys())

        for layer_name in common_layers:
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            # 检查是否已完全剪枝
            if layer_name in self._pruning_masks:
                mask = self._pruning_masks[layer_name]
                if mask.sum() == 0:
                    # 该层已完全剪枝，跳过
                    continue

            loss = self._align_features(s_feat, t_feat, layer_name, temperature)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def get_projection_parameters(self):
        """获取投影头参数（用于优化器）"""
        params = []
        for proj in self._projection_heads.values():
            params.extend(proj.parameters())
        return params

    def clear_cache(self):
        """清除缓存"""
        self._projection_heads.clear()
        self._pruning_masks.clear()
        self._importance_scores.clear()


class PruningAwareTeacherEnsemble:
    """剪枝感知教师集合

    根据剪枝阶段自动选择合适的教师：
    1. 剪枝前：使用历史最优模型
    2. 剪枝中：使用上一轮剪枝后的模型
    3. 剪枝后：使用剪枝前的完整模型

    Args:
        max_history: 最大历史记录数
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history

        # 历史模型：(epoch, loss, state_dict, is_pruned)
        self._history: List[Tuple[int, float, Dict, bool]] = []

        # 最优模型
        self._best_model: Optional[Tuple[int, float, Dict]] = None

        # 上一轮剪枝后的模型
        self._last_pruned_model: Optional[Tuple[int, float, Dict]] = None

    def update(
        self,
        model: nn.Module,
        loss: float,
        epoch: int,
        is_pruning: bool = False,
        just_pruned: bool = False
    ):
        """更新教师集合

        Args:
            model: 当前模型
            loss: 当前损失
            epoch: 当前epoch
            is_pruning: 是否在剪枝阶段
            just_pruned: 是否刚刚完成剪枝
        """
        state_dict = self._safe_copy_state_dict(model)

        # 更新最优模型
        if self._best_model is None or loss < self._best_model[1]:
            self._best_model = (epoch, loss, state_dict)

        # 更新历史
        self._history.append((epoch, loss, state_dict, is_pruning))
        if len(self._history) > self.max_history:
            self._history.pop(0)

        # 更新剪枝后模型
        if just_pruned:
            self._last_pruned_model = (epoch, loss, state_dict)
            logger.info(f"记录剪枝后模型: epoch={epoch}, loss={loss:.4f}")

    def get_teachers(
        self,
        is_pruning: bool = False,
        pruning_period: int = 0
    ) -> List[Dict]:
        """获取教师模型列表

        Args:
            is_pruning: 是否在剪枝阶段
            pruning_period: 当前剪枝周期

        Returns:
            教师模型列表
        """
        teachers = []

        # 1. 始终包含最优模型
        if self._best_model:
            epoch, loss, state_dict = self._best_model
            teachers.append({
                'state_dict': state_dict,
                'loss': loss,
                'epoch': epoch,
                'type': 'best'
            })

        # 2. 剪枝阶段：包含上一轮剪枝后模型
        if is_pruning and self._last_pruned_model:
            epoch, loss, state_dict = self._last_pruned_model
            teachers.append({
                'state_dict': state_dict,
                'loss': loss,
                'epoch': epoch,
                'type': 'last_pruned'
            })

        # 3. 包含最近的历史模型（避免重复）
        seen_epochs = {t['epoch'] for t in teachers}
        for epoch, loss, state_dict, was_pruned in reversed(self._history):
            if epoch not in seen_epochs and len(teachers) < 5:
                teachers.append({
                    'state_dict': state_dict,
                    'loss': loss,
                    'epoch': epoch,
                    'type': 'history'
                })
                seen_epochs.add(epoch)

        return teachers

    def _safe_copy_state_dict(self, model: nn.Module) -> Dict:
        """安全复制模型状态"""
        return {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    def state_dict(self) -> Dict:
        """保存状态"""
        return {
            'history': self._history,
            'best_model': self._best_model,
            'last_pruned_model': self._last_pruned_model,
        }

    def load_state_dict(self, state_dict: Dict):
        """恢复状态"""
        self._history = state_dict.get('history', [])
        self._best_model = state_dict.get('best_model')
        self._last_pruned_model = state_dict.get('last_pruned_model')


class PruningStageScheduler:
    """剪枝阶段调度器

    根据剪枝进度自动调整蒸馏策略：

    阶段1（预热期，0-20%）：
    - 轻量蒸馏，主要让Student学习基础特征
    - 使用低权重，高温

    阶段2（剪枝期，20-80%）：
    - 重点蒸馏，对齐存活通道
    - 使用投影头，中等权重

    阶段3（精炼期，80-100%）：
    - 精确蒸馏，对齐最终结构
    - 使用高权重，低温

    Args:
        warmup_ratio: 预热期比例
        refine_ratio: 精炼期比例
        warmup_weight: 预热期权重
        pruning_weight: 剪枝期权重
        refine_weight: 精炼期权重
    """

    def __init__(
        self,
        warmup_ratio: float = 0.2,
        refine_ratio: float = 0.2,
        warmup_weight: float = 0.05,
        pruning_weight: float = 0.1,
        refine_weight: float = 0.2,
    ):
        self.warmup_ratio = warmup_ratio
        self.refine_ratio = refine_ratio
        self.warmup_weight = warmup_weight
        self.pruning_weight = pruning_weight
        self.refine_weight = refine_weight

    def get_stage_config(
        self,
        epoch: int,
        max_epoch: int,
        is_pruning: bool
    ) -> Dict:
        """获取当前阶段配置

        Args:
            epoch: 当前epoch
            max_epoch: 总epoch数
            is_pruning: 是否在剪枝阶段

        Returns:
            阶段配置字典
        """
        progress = epoch / max(max_epoch - 1, 1)

        if progress < self.warmup_ratio:
            # 预热期
            stage = 'warmup'
            weight = self.warmup_weight
            temperature = 6.0  # 高温
            use_projection = False
            use_mask = False
        elif progress < (1 - self.refine_ratio):
            # 剪枝期
            stage = 'pruning'
            weight = self.pruning_weight
            temperature = 4.0  # 中温
            use_projection = True
            use_mask = True
        else:
            # 精炼期
            stage = 'refine'
            weight = self.refine_weight
            temperature = 2.0  # 低温
            use_projection = False
            use_mask = True

        return {
            'stage': stage,
            'weight': weight,
            'temperature': temperature,
            'use_projection': use_projection,
            'use_mask': use_mask,
            'progress': progress,
        }


class IntegratedPruningDistiller:
    """集成剪枝蒸馏器

    将所有组件整合，提供统一接口

    Args:
        enabled: 是否启用蒸馏
        top_n_teachers: 教师数量
    """

    def __init__(
        self,
        enabled: bool = True,
        top_n_teachers: int = 3,
    ):
        self.enabled = enabled

        # 组件
        self.teacher_ensemble = PruningAwareTeacherEnsemble(max_history=20)
        self.distiller = PruningAwareDistiller(
            weight=0.1,
            use_projection=True,
            use_mask=True,
            use_importance=True,
        )
        self.scheduler = PruningStageScheduler()

        # 钩子
        self._student_hooks = []
        self._teacher_hooks = []
        self._student_features = {}
        self._teacher_features = {}

    def update(
        self,
        model: nn.Module,
        loss: float,
        epoch: int,
        is_pruning: bool = False,
        just_pruned: bool = False,
        param_groups: Optional[List[Dict]] = None,
        importance_scores: Optional[Dict] = None
    ):
        """更新蒸馏器状态

        Args:
            model: 当前模型
            loss: 当前损失
            epoch: 当前epoch
            is_pruning: 是否在剪枝阶段
            just_pruned: 是否刚刚完成剪枝
            param_groups: 参数组信息
            importance_scores: 重要性分数
        """
        if not self.enabled:
            return

        # 更新教师集合
        self.teacher_ensemble.update(model, loss, epoch, is_pruning, just_pruned)

        # 更新剪枝信息
        if param_groups:
            self.distiller.update_pruning_info(
                pruned_groups=[],
                param_groups=param_groups,
                importance_scores=importance_scores
            )

    def compute_loss(
        self,
        student_model: nn.Module,
        x: torch.Tensor,
        epoch: int,
        max_epoch: int,
        is_pruning: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """计算蒸馏loss

        Args:
            student_model: Student模型
            x: 输入数据
            epoch: 当前epoch
            max_epoch: 总epoch数
            is_pruning: 是否在剪枝阶段

        Returns:
            (loss, 详细信息)
        """
        if not self.enabled:
            return torch.tensor(0.0), {}

        # 获取阶段配置
        config = self.scheduler.get_stage_config(epoch, max_epoch, is_pruning)

        # 获取教师
        teachers = self.teacher_ensemble.get_teachers(is_pruning)

        if not teachers:
            return torch.tensor(0.0), config

        # 动态调整蒸馏器参数
        self.distiller.weight = config['weight']
        self.distiller.use_projection = config['use_projection']
        self.distiller.use_mask = config['use_mask']

        # 计算蒸馏loss
        total_loss = torch.tensor(0.0, device=x.device)
        teacher_losses = []

        for teacher_info in teachers:
            # 加载教师模型
            teacher_model = self._load_teacher(student_model, teacher_info['state_dict'])
            teacher_model = teacher_model.to(x.device)

            # 注册钩子
            self._register_hooks(student_model, teacher_model)

            # 前向传播
            with torch.no_grad():
                teacher_model(x)
            student_model(x)

            # 计算loss
            loss = self.distiller.compute_loss(
                self._student_features,
                self._teacher_features,
                temperature=config['temperature']
            )

            teacher_losses.append(loss.item())
            total_loss = total_loss + loss

            # 清理
            self._remove_hooks()

        # 多教师取平均
        if teachers:
            total_loss = total_loss / len(teachers)

        # 详细信息
        info = {
            **config,
            'num_teachers': len(teachers),
            'teacher_losses': teacher_losses,
            'total_loss': total_loss.item(),
        }

        return total_loss, info

    def _load_teacher(
        self,
        student_model: nn.Module,
        state_dict: Dict
    ) -> nn.Module:
        """加载教师模型"""
        import copy
        teacher = copy.deepcopy(student_model)
        teacher.load_state_dict(state_dict)
        teacher.eval()
        return teacher

    def _register_hooks(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ):
        """注册钩子"""
        self._remove_hooks()
        self._student_features.clear()
        self._teacher_features.clear()

        # 选择要对齐的层
        layer_names = self._select_layers(student_model, teacher_model)

        for name in layer_names:
            s_module = dict(student_model.named_modules()).get(name)
            t_module = dict(teacher_model.named_modules()).get(name)

            if s_module is not None:
                hook = s_module.register_forward_hook(
                    self._make_hook(self._student_features, name)
                )
                self._student_hooks.append(hook)

            if t_module is not None:
                hook = t_module.register_forward_hook(
                    self._make_hook(self._teacher_features, name)
                )
                self._teacher_hooks.append(hook)

    def _remove_hooks(self):
        """移除钩子"""
        for h in self._student_hooks:
            h.remove()
        for h in self._teacher_hooks:
            h.remove()
        self._student_hooks.clear()
        self._teacher_hooks.clear()

    def _make_hook(self, feature_dict: Dict, name: str):
        """创建钩子函数"""
        def hook_fn(module, input, output):
            feature_dict[name] = output
        return hook_fn

    def _select_layers(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> List[str]:
        """选择要对齐的层"""
        s_layers = set()
        t_layers = set()

        for name, module in student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                s_layers.add(name)

        for name, module in teacher_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                t_layers.add(name)

        return sorted(s_layers & t_layers)

    def get_projection_parameters(self):
        """获取投影头参数"""
        return self.distiller.get_projection_parameters()

    def state_dict(self) -> Dict:
        """保存状态"""
        return {
            'teacher_ensemble': self.teacher_ensemble.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        """恢复状态"""
        self.teacher_ensemble.load_state_dict(state_dict.get('teacher_ensemble', {}))
