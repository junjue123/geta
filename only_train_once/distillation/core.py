"""
蒸馏核心模块
整合所有蒸馏方法到4个核心类中：

1. TeacherEnsemble - 教师集合管理
2. SoftLabelDistiller - 软标签蒸馏
3. FeatureDistiller - 每层输出分布对齐
4. WeightDistiller - 每层权重特征对齐

每个类内部包含相关优化方法，保持接口简洁。
"""

import copy
import heapq
import itertools
import random
import logging
import math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 1. 教师集合管理
# ============================================================================

class TeacherEnsemble:
    """教师集合管理

    管理三类教师模型：
    1. pre_pruning_best: 剪枝前最优模型
    2. pruning_top_n: 剪枝过程中前N个epoch的模型
    3. pruning_random: 剪枝过程中随机采样的模型

    Args:
        top_n: 剪枝过程中保留的前N个最优模型数量
    """

    def __init__(self, top_n: int = 3):
        self.top_n = top_n

        # 剪枝前最优模型
        self.pre_pruning_best: Optional[Dict] = None

        # 剪枝中前N个模型 (最大堆，保留loss最小的N个)
        self.pruning_top_n: List[tuple] = []
        self._counter = itertools.count()

        # 剪枝中随机采样 (蓄水池采样)
        self.pruning_random: Optional[Dict] = None
        self.pruning_sample_count = 0

        # 当前是否在剪枝阶段
        self.is_pruning_phase = False

    def update(
        self,
        model: nn.Module,
        loss: float,
        epoch: int,
        is_pruning: bool = False
    ):
        """更新教师集合

        Args:
            model: 当前模型
            loss: 当前损失值
            epoch: 当前epoch
            is_pruning: 是否在剪枝阶段
        """
        # 首次进入剪枝阶段
        if is_pruning and not self.is_pruning_phase:
            self.is_pruning_phase = True
            logger.info(f"进入剪枝阶段，锁定剪枝前最优模型 (epoch={epoch})")

        if not is_pruning:
            # 更新剪枝前最优
            self._update_pre_pruning_best(model, loss, epoch)
        else:
            # 更新剪枝中的模型
            self._update_pruning_top_n(model, loss, epoch)
            self._update_pruning_random(model, loss, epoch)

    def _update_pre_pruning_best(self, model: nn.Module, loss: float, epoch: int):
        """更新剪枝前最优模型"""
        if self.pre_pruning_best is None or loss < self.pre_pruning_best['loss']:
            self.pre_pruning_best = {
                'state_dict': self._copy_state_dict(model),
                'loss': loss,
                'epoch': epoch
            }

    def _update_pruning_top_n(self, model: nn.Module, loss: float, epoch: int):
        """更新剪枝中前N个模型"""
        counter = next(self._counter)
        neg_loss = -loss

        if len(self.pruning_top_n) < self.top_n:
            heapq.heappush(self.pruning_top_n, (neg_loss, counter, epoch, self._copy_state_dict(model)))
        elif neg_loss > self.pruning_top_n[0][0]:
            heapq.heapreplace(self.pruning_top_n, (neg_loss, counter, epoch, self._copy_state_dict(model)))

    def _update_pruning_random(self, model: nn.Module, loss: float, epoch: int):
        """更新剪枝中随机采样 (蓄水池采样)"""
        self.pruning_sample_count += 1
        if self.pruning_random is None:
            self.pruning_random = {
                'state_dict': self._copy_state_dict(model),
                'loss': loss,
                'epoch': epoch
            }
        elif random.random() < 1.0 / self.pruning_sample_count:
            self.pruning_random = {
                'state_dict': self._copy_state_dict(model),
                'loss': loss,
                'epoch': epoch
            }

    def get_teachers(self) -> List[Dict]:
        """获取所有教师模型

        Returns:
            教师模型列表，每个元素包含 state_dict, loss, epoch
        """
        teachers = []

        # 1. 剪枝前最优
        if self.pre_pruning_best is not None:
            teachers.append(self.pre_pruning_best)

        # 2. 剪枝中前N个 (按loss排序)
        for neg_loss, counter, epoch, state_dict in sorted(self.pruning_top_n, reverse=True):
            teachers.append({
                'state_dict': state_dict,
                'loss': -neg_loss,
                'epoch': epoch
            })

        # 3. 剪枝中随机采样
        if self.pruning_random is not None:
            teachers.append(self.pruning_random)

        return teachers

    def _copy_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """安全复制模型状态"""
        return {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    def state_dict(self) -> Dict:
        """保存状态"""
        return {
            'pre_pruning_best': self.pre_pruning_best,
            'pruning_top_n': self.pruning_top_n,
            'pruning_random': self.pruning_random,
            'pruning_sample_count': self.pruning_sample_count,
            'is_pruning_phase': self.is_pruning_phase,
        }

    def load_state_dict(self, state_dict: Dict):
        """恢复状态"""
        self.pre_pruning_best = state_dict.get('pre_pruning_best')
        self.pruning_top_n = state_dict.get('pruning_top_n', [])
        self.pruning_random = state_dict.get('pruning_random')
        self.pruning_sample_count = state_dict.get('pruning_sample_count', 0)
        self.is_pruning_phase = state_dict.get('is_pruning_phase', False)

    def __len__(self) -> int:
        """返回教师模型数量"""
        count = 0
        if self.pre_pruning_best is not None:
            count += 1
        count += len(self.pruning_top_n)
        if self.pruning_random is not None:
            count += 1
        return count

    def get_weighted_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """根据教师置信度加权平均logits

        置信度越高（max probability越大）的教师权重越大。

        Args:
            student_logits: 学生logits [B, C]
            teacher_logits_list: 教师logits列表

        Returns:
            加权平均后的logits [B, C]
        """
        if not teacher_logits_list:
            return student_logits

        # 计算每个教师的置信度
        confidences = []
        for t_logits in teacher_logits_list:
            probs = F.softmax(t_logits, dim=-1)
            conf = probs.max(dim=-1)[0].mean()  # 平均最大概率
            confidences.append(conf)

        # 归一化权重
        conf_tensor = torch.stack(confidences)
        weights = F.softmax(conf_tensor, dim=0)

        # 加权平均
        weighted_logits = torch.zeros_like(teacher_logits_list[0])
        for w, t_logits in zip(weights, teacher_logits_list):
            weighted_logits += w * t_logits

        return weighted_logits

    def get_ensemble_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """获取所有教师的集成状态（平均权重）

        Args:
            model: 学生模型（用于获取结构）

        Returns:
            平均后的state_dict
        """
        teachers = self.get_teachers()
        if not teachers:
            return model.state_dict()

        # 收集所有state_dict
        all_state_dicts = [t['state_dict'] for t in teachers]

        # 计算平均
        avg_state_dict = {}
        for key in all_state_dicts[0].keys():
            values = [sd[key].float() for sd in all_state_dicts if key in sd]
            if values:
                avg_state_dict[key] = torch.stack(values).mean(dim=0)

        return avg_state_dict


# ============================================================================
# 2. 软标签蒸馏
# ============================================================================

class SoftLabelDistiller:
    """软标签蒸馏 (经典KD)

    公式: L_KD = α * T² * KL(softmax(z_t/T) || softmax(z_s/T)) + (1-α) * CE(y, z_s)

    Args:
        temperature: 温度参数，控制softmax的平滑程度
        alpha: 蒸馏loss权重
        mode: KL散度模式
            - 'forward': KL(p_teacher || p_student) - 模式覆盖
            - 'reverse': KL(p_student || p_teacher) - 模式寻找
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        mode: str = 'forward'
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.mode = mode

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算软标签蒸馏loss

        Args:
            student_logits: 学生模型输出 [B, C]
            teacher_logits_list: 教师模型输出列表，每个 [B, C]
            labels: 真实标签 [B]

        Returns:
            蒸馏loss
        """
        if not teacher_logits_list:
            return torch.tensor(0.0, device=student_logits.device)

        T = self.temperature
        kd_losses = []

        for teacher_logits in teacher_logits_list:
            if self.mode == 'forward':
                # KL(p_teacher || p_student)
                student_log = F.log_softmax(student_logits / T, dim=-1)
                teacher_soft = F.softmax(teacher_logits / T, dim=-1)
                kd = F.kl_div(student_log, teacher_soft, reduction='batchmean')
            else:
                # KL(p_student || p_teacher)
                teacher_log = F.log_softmax(teacher_logits / T, dim=-1)
                student_soft = F.softmax(student_logits / T, dim=-1)
                kd = F.kl_div(teacher_log, student_soft, reduction='batchmean')

            kd_losses.append(kd)

        # 多教师取平均
        kd_loss = torch.stack(kd_losses).mean() * (T ** 2)

        # 混合CE loss
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return kd_loss

    def get_temperature(self, epoch: int, max_epoch: int) -> float:
        """获取当前温度（支持温度退火）

        Args:
            epoch: 当前epoch
            max_epoch: 总epoch数

        Returns:
            当前温度
        """
        # 余弦退火：高温 -> 低温
        progress = epoch / max(max_epoch - 1, 1)
        temp = 2.0 + 0.5 * (self.temperature - 2.0) * (1 + math.cos(math.pi * progress))
        return max(temp, 2.0)

    def compute_confidence_weights(
        self,
        teacher_logits_list: List[torch.Tensor]
    ) -> List[float]:
        """根据教师置信度计算权重

        置信度越高（熵越低）的教师权重越大。

        Args:
            teacher_logits_list: 教师logits列表

        Returns:
            权重列表
        """
        if not teacher_logits_list:
            return []

        entropies = []
        for t_logits in teacher_logits_list:
            probs = F.softmax(t_logits, dim=-1)
            # 计算熵（越低越自信）
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            entropies.append(entropy.item())

        # 熵越低权重越大
        inv_entropies = [1.0 / (e + 1e-8) for e in entropies]
        total = sum(inv_entropies)
        weights = [ie / total for ie in inv_entropies]

        return weights

    def compute_loss_weighted(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算加权软标签蒸馏loss

        使用教师置信度作为权重。

        Args:
            student_logits: 学生logits
            teacher_logits_list: 教师logits列表
            labels: 真实标签

        Returns:
            加权蒸馏loss
        """
        if not teacher_logits_list:
            return torch.tensor(0.0, device=student_logits.device)

        weights = self.compute_confidence_weights(teacher_logits_list)
        T = self.temperature

        kd_losses = []
        for w, teacher_logits in zip(weights, teacher_logits_list):
            if self.mode == 'forward':
                student_log = F.log_softmax(student_logits / T, dim=-1)
                teacher_soft = F.softmax(teacher_logits / T, dim=-1)
                kd = F.kl_div(student_log, teacher_soft, reduction='batchmean')
            else:
                teacher_log = F.log_softmax(teacher_logits / T, dim=-1)
                student_soft = F.softmax(student_logits / T, dim=-1)
                kd = F.kl_div(teacher_log, student_soft, reduction='batchmean')

            kd_losses.append(w * kd)

        kd_loss = torch.stack(kd_losses).sum() * (T ** 2)

        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return kd_loss


# ============================================================================
# 3. 每层输出分布对齐
# ============================================================================

class FeatureDistiller:
    """每层输出分布对齐

    使用前向/反向KL散度对齐学生和教师的中间层输出。

    Args:
        mode: KL散度模式
            - 'forward': KL(p_teacher || p_student)
            - 'reverse': KL(p_student || p_teacher)
        weight: 特征对齐loss权重
        use_projection: 是否使用投影头处理维度不匹配
        progressive: 是否渐进式解锁层
        adaptive_temp: 是否自适应温度
    """

    def __init__(
        self,
        mode: str = 'forward',
        weight: float = 0.1,
        use_projection: bool = True,
        progressive: bool = True,
        adaptive_temp: bool = True
    ):
        self.mode = mode
        self.weight = weight
        self.use_projection = use_projection
        self.progressive = progressive
        self.adaptive_temp = adaptive_temp

        self._projection_heads: Dict[str, nn.Linear] = {}
        self._student_features: Dict[str, torch.Tensor] = {}
        self._teacher_features: Dict[str, torch.Tensor] = {}
        self._student_hooks: List = []
        self._teacher_hooks: List = []

    def register_hooks(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        layer_names: Optional[List[str]] = None
    ):
        """注册前向钩子捕获中间层输出"""
        self.remove_hooks()

        if layer_names is None:
            layer_names = self._auto_select_layers(student_model, teacher_model)

        for name in layer_names:
            s_module = dict(student_model.named_modules()).get(name)
            if s_module is not None:
                hook = s_module.register_forward_hook(
                    self._make_hook(self._student_features, name)
                )
                self._student_hooks.append(hook)

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

    def _make_hook(self, feature_dict: Dict, name: str):
        """创建钩子函数"""
        def hook_fn(module, input, output):
            feature_dict[name] = output
        return hook_fn

    def _auto_select_layers(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> List[str]:
        """自动选择要对齐的层（Conv2d和Linear）"""
        s_layers = {name for name, m in student_model.named_modules()
                    if isinstance(m, (nn.Conv2d, nn.Linear))}
        t_layers = {name for name, m in teacher_model.named_modules()
                    if isinstance(m, (nn.Conv2d, nn.Linear))}
        return sorted(s_layers & t_layers)

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
        return base_temp * (0.5 + progress)

    def _get_progressive_layers(
        self,
        all_layers: List[str],
        epoch: int,
        max_epoch: int
    ) -> List[str]:
        """渐进式解锁层

        Args:
            all_layers: 所有层列表
            epoch: 当前epoch
            max_epoch: 总epoch

        Returns:
            当前要解锁的层
        """
        if not self.progressive:
            return all_layers

        # 分成3组：浅/中/深
        n = len(all_layers)
        if n <= 3:
            return all_layers

        third = n // 3
        groups = [
            all_layers[:third],          # 浅层
            all_layers[third:2*third],   # 中层
            all_layers[2*third:],        # 深层
        ]

        # 根据epoch解锁
        progress = epoch / max(max_epoch - 1, 1)
        if progress < 0.33:
            return groups[0]
        elif progress < 0.66:
            return groups[0] + groups[1]
        else:
            return all_layers

    def compute_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        x: torch.Tensor,
        epoch: int = 0,
        max_epoch: int = 100,
        layer_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """计算特征对齐loss

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            x: 输入数据
            epoch: 当前epoch
            max_epoch: 总epoch
            layer_names: 要对齐的层名称

        Returns:
            特征对齐loss
        """
        # 获取所有层
        if layer_names is None:
            layer_names = self._auto_select_layers(student_model, teacher_model)

        # 渐进式解锁
        active_layers = self._get_progressive_layers(layer_names, epoch, max_epoch)

        # 注册钩子
        self.register_hooks(student_model, teacher_model, active_layers)

        # 前向传播
        with torch.no_grad():
            teacher_model(x)
        student_model(x)

        # 计算loss
        losses = []
        total_layers = len(active_layers)

        for idx, layer_name in enumerate(active_layers):
            if layer_name not in self._student_features or \
               layer_name not in self._teacher_features:
                continue

            s_feat = self._student_features[layer_name]
            t_feat = self._teacher_features[layer_name]

            # 展平空间维度
            if s_feat.dim() > 2:
                s_flat = s_feat.flatten(start_dim=2).mean(dim=2)  # [B, C]
                t_flat = t_feat.flatten(start_dim=2).mean(dim=2)  # [B, C]
            else:
                s_flat = s_feat
                t_flat = t_feat

            # 维度不匹配时使用投影
            if s_flat.shape[-1] != t_flat.shape[-1] and self.use_projection:
                proj = self._get_projection(
                    layer_name, s_flat.shape[-1], t_flat.shape[-1], s_flat.device
                )
                s_flat = proj(s_flat)

            # 归一化
            s_norm = F.normalize(s_flat, dim=-1)
            t_norm = F.normalize(t_flat, dim=-1)

            # 自适应温度
            temp = self._get_adaptive_temperature(idx, total_layers)

            # 计算KL散度
            s_log = F.log_softmax(s_norm / temp, dim=-1)
            t_soft = F.softmax(t_norm / temp, dim=-1)

            if self.mode == 'forward':
                loss = F.kl_div(s_log, t_soft, reduction='batchmean')
            else:
                t_log = F.log_softmax(t_norm / temp, dim=-1)
                s_soft = F.softmax(s_norm / temp, dim=-1)
                loss = F.kl_div(t_log, s_soft, reduction='batchmean')

            losses.append(loss * (temp ** 2))

        # 清理钩子
        self.remove_hooks()

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def get_projection_parameters(self):
        """获取投影头参数"""
        params = []
        for proj in self._projection_heads.values():
            params.extend(proj.parameters())
        return params

    def compute_attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """计算特征图的注意力图

        Args:
            features: 特征图 [B, C, H, W] 或 [B, C]

        Returns:
            注意力权重 [B, 1, H, W] 或 [B, 1]
        """
        if features.dim() == 4:
            # 空间注意力：通道平均
            attention = features.abs().mean(dim=1, keepdim=True)
        elif features.dim() == 3:
            attention = features.abs().mean(dim=1, keepdim=True)
        else:
            attention = features.abs().mean(dim=1, keepdim=True)

        # 归一化到 [0, 1]
        min_val = attention.min()
        max_val = attention.max()
        if max_val - min_val > 1e-8:
            attention = (attention - min_val) / (max_val - min_val)

        return attention

    def compute_loss_attention_weighted(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        x: torch.Tensor,
        epoch: int = 0,
        max_epoch: int = 100,
        layer_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """计算注意力加权的特征对齐loss

        使用教师的注意力图加权，重要区域权重更大。

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            x: 输入数据
            epoch: 当前epoch
            max_epoch: 总epoch
            layer_names: 要对齐的层名称

        Returns:
            注意力加权的特征对齐loss
        """
        if layer_names is None:
            layer_names = self._auto_select_layers(student_model, teacher_model)

        active_layers = self._get_progressive_layers(layer_names, epoch, max_epoch)
        self.register_hooks(student_model, teacher_model, active_layers)

        with torch.no_grad():
            teacher_model(x)
        student_model(x)

        losses = []
        total_layers = len(active_layers)

        for idx, layer_name in enumerate(active_layers):
            if layer_name not in self._student_features or \
               layer_name not in self._teacher_features:
                continue

            s_feat = self._student_features[layer_name]
            t_feat = self._teacher_features[layer_name]

            if s_feat.shape != t_feat.shape:
                continue

            # 计算注意力图（使用教师）
            attention = self.compute_attention_map(t_feat)

            # 加权
            s_weighted = s_feat * attention
            t_weighted = t_feat * attention

            # 展平
            if s_weighted.dim() > 2:
                s_flat = s_weighted.flatten(start_dim=2).mean(dim=2)
                t_flat = t_weighted.flatten(start_dim=2).mean(dim=2)
            else:
                s_flat = s_weighted
                t_flat = t_weighted

            # 归一化
            s_norm = F.normalize(s_flat, dim=-1)
            t_norm = F.normalize(t_flat, dim=-1)

            # MSE对齐
            loss = F.mse_loss(s_norm, t_norm)
            losses.append(loss)

        self.remove_hooks()

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def get_layer_statistics(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """获取各层特征统计信息

        Args:
            model: 模型
            x: 输入数据

        Returns:
            统计信息字典
        """
        features = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        # 注册钩子
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # 前向传播
        with torch.no_grad():
            model(x)

        # 清理钩子
        for h in hooks:
            h.remove()

        # 计算统计信息
        stats = {}
        for name, feat in features.items():
            stats[name] = {
                'mean': feat.mean().item(),
                'std': feat.std().item(),
                'min': feat.min().item(),
                'max': feat.max().item(),
                'shape': list(feat.shape),
            }

        return stats


# ============================================================================
# 4. 每层权重特征对齐
# ============================================================================

class WeightDistiller:
    """每层权重特征对齐

    使用奇异值对齐学生和教师的权重特征。

    Args:
        weight: 权重对齐loss权重
        align_mode: 对齐模式
            - 'max': 只对齐最大奇异值
            - 'avg': 只对齐平均奇异值
            - 'both': 同时对齐最大和平均奇异值
        use_lowrank: 是否使用随机化SVD（适合大矩阵）
        lowrank_k: 随机化SVD保留的奇异值数量
    """

    def __init__(
        self,
        weight: float = 0.01,
        align_mode: str = 'both',
        use_lowrank: bool = True,
        lowrank_k: int = 10
    ):
        self.weight = weight
        self.align_mode = align_mode
        self.use_lowrank = use_lowrank
        self.lowrank_k = lowrank_k

    def _compute_svd(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        """计算权重的奇异值

        Args:
            weight: 权重张量 [out_channels, in_channels, ...]

        Returns:
            奇异值向量
        """
        try:
            # 展平为2D矩阵
            if weight.dim() > 2:
                weight_2d = weight.flatten(start_dim=1)
            else:
                weight_2d = weight

            # 选择SVD方法
            if self.use_lowrank and min(weight_2d.shape) > self.lowrank_k:
                k = min(self.lowrank_k, min(weight_2d.shape) - 1)
                _, sv, _ = torch.svd_lowrank(weight_2d, q=k)
            else:
                sv = torch.linalg.svdvals(weight_2d)

            return sv

        except Exception as e:
            logger.warning(f"SVD计算失败: {e}")
            return None

    def compute_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> torch.Tensor:
        """计算权重对齐loss

        Args:
            student_model: 学生模型
            teacher_model: 教师模型

        Returns:
            权重对齐loss
        """
        # 收集所有Conv2d和Linear层的权重
        student_weights = {}
        teacher_weights = {}

        for name, module in student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                student_weights[name] = module.weight

        for name, module in teacher_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                teacher_weights[name] = module.weight

        # 计算每层的loss
        losses = []
        common_layers = set(student_weights.keys()) & set(teacher_weights.keys())

        for layer_name in common_layers:
            s_weight = student_weights[layer_name]
            t_weight = teacher_weights[layer_name]

            # 计算奇异值
            s_sv = self._compute_svd(s_weight)
            t_sv = self._compute_svd(t_weight)

            if s_sv is None or t_sv is None:
                continue

            # 对齐长度
            min_len = min(len(s_sv), len(t_sv))
            s_sv = s_sv[:min_len]
            t_sv = t_sv[:min_len]

            # 计算loss
            layer_losses = []

            if self.align_mode in ('max', 'both'):
                # 最大奇异值对齐
                max_loss = F.mse_loss(s_sv[0:1], t_sv[0:1])
                layer_losses.append(max_loss)

            if self.align_mode in ('avg', 'both'):
                # 平均奇异值对齐
                avg_loss = F.mse_loss(s_sv.mean(), t_sv.mean())
                layer_losses.append(avg_loss)

            if layer_losses:
                losses.append(torch.stack(layer_losses).mean())

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def get_svd_statistics(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """获取模型的奇异值统计信息

        Args:
            model: 模型

        Returns:
            统计信息字典
        """
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                sv = self._compute_svd(module.weight)
                if sv is not None:
                    stats[name] = {
                        'max_sv': sv[0].item(),
                        'mean_sv': sv.mean().item(),
                        'min_sv': sv[-1].item() if len(sv) > 0 else 0.0,
                        'num_sv': len(sv)
                    }
        return stats

    def compute_frobenius_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> torch.Tensor:
        """计算Frobenius范数对齐loss

        直接对齐权重矩阵的Frobenius范数，计算量更小。

        Args:
            student_model: 学生模型
            teacher_model: 教师模型

        Returns:
            Frobenius范数对齐loss
        """
        student_weights = {}
        teacher_weights = {}

        for name, module in student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                student_weights[name] = module.weight

        for name, module in teacher_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                teacher_weights[name] = module.weight

        losses = []
        common_layers = set(student_weights.keys()) & set(teacher_weights.keys())

        for layer_name in common_layers:
            s_weight = student_weights[layer_name]
            t_weight = teacher_weights[layer_name]

            if s_weight.shape != t_weight.shape:
                continue

            # Frobenius范数
            s_norm = torch.norm(s_weight, p='fro')
            t_norm = torch.norm(t_weight, p='fro')

            loss = F.mse_loss(s_norm, t_norm)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def compute_distribution_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module
    ) -> torch.Tensor:
        """计算权重分布对齐loss

        对齐权重的均值和方差。

        Args:
            student_model: 学生模型
            teacher_model: 教师模型

        Returns:
            分布对齐loss
        """
        student_weights = {}
        teacher_weights = {}

        for name, module in student_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                student_weights[name] = module.weight

        for name, module in teacher_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                teacher_weights[name] = module.weight

        losses = []
        common_layers = set(student_weights.keys()) & set(teacher_weights.keys())

        for layer_name in common_layers:
            s_weight = student_weights[layer_name]
            t_weight = teacher_weights[layer_name]

            if s_weight.shape != t_weight.shape:
                continue

            # 均值对齐
            mean_loss = F.mse_loss(s_weight.mean(), t_weight.mean())

            # 方差对齐
            var_loss = F.mse_loss(s_weight.var(), t_weight.var())

            losses.append(mean_loss + var_loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight

    def compute_loss_combined(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        svd_weight: float = 0.5,
        frobenius_weight: float = 0.3,
        distribution_weight: float = 0.2
    ) -> torch.Tensor:
        """计算组合权重对齐loss

        结合SVD、Frobenius范数和分布对齐。

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            svd_weight: SVD loss权重
            frobenius_weight: Frobenius loss权重
            distribution_weight: 分布loss权重

        Returns:
            组合loss
        """
        svd_loss = self.compute_loss(student_model, teacher_model)
        frobenius_loss = self.compute_frobenius_loss(student_model, teacher_model)
        dist_loss = self.compute_distribution_loss(student_model, teacher_model)

        return (svd_weight * svd_loss +
                frobenius_weight * frobenius_loss +
                distribution_weight * dist_loss)


# ============================================================================
# 统一蒸馏器（整合接口）
# ============================================================================

class Distiller:
    """统一蒸馏器

    整合所有蒸馏方法，提供简洁接口。

    Args:
        temperature: 软标签蒸馏温度
        alpha: 软标签蒸馏权重
        feature_weight: 特征对齐权重
        weight_weight: 权重对齐权重
        mode: KL散度模式
        progressive: 是否渐进式解锁层
        adaptive_temp: 是否自适应温度
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_weight: float = 0.1,
        weight_weight: float = 0.01,
        mode: str = 'forward',
        progressive: bool = True,
        adaptive_temp: bool = True
    ):
        # 教师集合
        self.teacher_ensemble = TeacherEnsemble(top_n=3)

        # 三个核心蒸馏方法
        self.soft_label = SoftLabelDistiller(
            temperature=temperature,
            alpha=alpha,
            mode=mode
        )
        self.feature = FeatureDistiller(
            mode=mode,
            weight=feature_weight,
            progressive=progressive,
            adaptive_temp=adaptive_temp
        )
        self.weight = WeightDistiller(
            weight=weight_weight,
            align_mode='both'
        )

    def update_teacher(
        self,
        model: nn.Module,
        loss: float,
        epoch: int,
        is_pruning: bool = False
    ):
        """更新教师集合"""
        self.teacher_ensemble.update(model, loss, epoch, is_pruning)

    def compute_loss(
        self,
        student_model: nn.Module,
        x: torch.Tensor,
        student_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epoch: int = 100
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总蒸馏loss

        Args:
            student_model: 学生模型
            x: 输入数据
            student_logits: 学生输出
            labels: 真实标签
            epoch: 当前epoch
            max_epoch: 总epoch

        Returns:
            (总loss, 各部分loss字典)
        """
        teachers = self.teacher_ensemble.get_teachers()

        if not teachers:
            return torch.tensor(0.0, device=x.device), {}

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=x.device)

        # 加载教师模型
        teacher_models = []
        for teacher_info in teachers:
            teacher = copy.deepcopy(student_model)
            teacher.load_state_dict(teacher_info['state_dict'])
            teacher.eval()
            teacher = teacher.to(x.device)
            teacher_models.append(teacher)

        # 1. 软标签蒸馏
        teacher_logits_list = []
        for teacher in teacher_models:
            with torch.no_grad():
                teacher_logits_list.append(teacher(x))
        soft_loss = self.soft_label.compute_loss(
            student_logits, teacher_logits_list, labels
        )
        total_loss = total_loss + soft_loss
        loss_dict['soft_label'] = soft_loss.item()

        # 2. 特征对齐（只用第一个教师，避免重复计算）
        if teacher_models:
            feat_loss = self.feature.compute_loss(
                student_model, teacher_models[0], x, epoch, max_epoch
            )
            total_loss = total_loss + feat_loss
            loss_dict['feature'] = feat_loss.item()

        # 3. 权重对齐（只用第一个教师）
        if teacher_models:
            w_loss = self.weight.compute_loss(student_model, teacher_models[0])
            total_loss = total_loss + w_loss
            loss_dict['weight'] = w_loss.item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
