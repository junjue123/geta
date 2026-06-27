"""
权重特征对齐模块
使用奇异值对齐学生和教师的权重特征。

关键优化:
- 使用 torch.linalg.svdvals() 只计算奇异值，不计算U和V
- 对于大权重矩阵，使用随机化SVD (torch.svd_lowrank)
- 缓存教师模型的奇异值，避免重复计算
"""

import logging
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeightDistiller:
    """权重特征对齐器

    使用奇异值对齐学生和教师的权重特征。

    Args:
        weight_weight: 权重对齐loss权重
        align_mode: 对齐模式
            - 'max': 只对齐最大奇异值
            - 'avg': 只对齐平均奇异值
            - 'both': 同时对齐最大和平均奇异值
        svd_method: SVD计算方法
            - 'full': 完整SVD (精确但慢)
            - 'lowrank': 随机化SVD (近似但快，适合大矩阵)
        lowrank_k: 随机化SVD保留的奇异值数量
        cache_teachers: 是否缓存教师模型的奇异值
    """

    def __init__(
        self,
        weight_weight: float = 0.01,
        align_mode: str = 'both',
        svd_method: str = 'lowrank',
        lowrank_k: int = 10,
        cache_teachers: bool = True
    ):
        self.weight_weight = weight_weight
        self.align_mode = align_mode
        self.svd_method = svd_method
        self.lowrank_k = lowrank_k
        self.cache_teachers = cache_teachers

        if align_mode not in ('max', 'avg', 'both'):
            raise ValueError(f"align_mode 必须是 'max', 'avg' 或 'both'，收到 '{align_mode}'")

        # 教师奇异值缓存
        self._teacher_sv_cache: Dict[int, Dict[str, List[torch.Tensor]]] = {}

    def compute_loss(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        teacher_id: Optional[int] = None
    ) -> torch.Tensor:
        """计算权重对齐loss

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            teacher_id: 教师模型ID（用于缓存）

        Returns:
            权重对齐loss (标量)
        """
        # 获取教师奇异值（可能来自缓存）
        teacher_svs = self._get_teacher_svs(teacher_model, teacher_id)

        # 计算学生奇异值
        student_svs = self._compute_model_svs(student_model)

        # 检查数量是否匹配
        if len(student_svs) != len(teacher_svs):
            logger.warning(f"学生和教师的权重层数量不匹配: {len(student_svs)} vs {len(teacher_svs)}")
            min_len = min(len(student_svs), len(teacher_svs))
            student_svs = student_svs[:min_len]
            teacher_svs = teacher_svs[:min_len]

        # 计算每层的loss
        losses = []
        for sv_s, sv_t in zip(student_svs, teacher_svs):
            loss = self._compute_layer_loss(sv_s, sv_t)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean() * self.weight_weight

    def _get_teacher_svs(
        self,
        teacher_model: nn.Module,
        teacher_id: Optional[int]
    ) -> List[torch.Tensor]:
        """获取教师模型的奇异值（带缓存）"""
        if self.cache_teachers and teacher_id is not None:
            if teacher_id in self._teacher_sv_cache:
                logger.debug(f"使用缓存的教师奇异值 (id={teacher_id})")
                return self._teacher_sv_cache[teacher_id]['svs']

        # 计算奇异值
        svs = self._compute_model_svs(teacher_model)

        # 缓存
        if self.cache_teachers and teacher_id is not None:
            self._teacher_sv_cache[teacher_id] = {'svs': svs}
            logger.debug(f"缓存教师奇异值 (id={teacher_id})")

        return svs

    def _compute_model_svs(self, model: nn.Module) -> List[torch.Tensor]:
        """计算模型所有权重层的奇异值

        Args:
            model: 模型

        Returns:
            奇异值列表，每个元素是一层的奇异值
        """
        svs = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                sv = self._compute_svd(weight)
                if sv is not None:
                    svs.append(sv)
        return svs

    def _compute_svd(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        """计算权重的奇异值

        Args:
            weight: 权重张量 [out_channels, in_channels, ...]

        Returns:
            奇异值向量，或None（如果计算失败）
        """
        try:
            # 展平为2D矩阵
            if weight.dim() > 2:
                weight_2d = weight.flatten(start_dim=1)  # [out, in*h*w]
            else:
                weight_2d = weight

            # 选择SVD方法
            if self.svd_method == 'lowrank' and min(weight_2d.shape) > self.lowrank_k:
                # 随机化SVD，只计算前k个奇异值
                k = min(self.lowrank_k, min(weight_2d.shape) - 1)
                _, sv, _ = torch.svd_lowrank(weight_2d, q=k)
            else:
                # 完整SVD
                sv = torch.linalg.svdvals(weight_2d)

            return sv

        except Exception as e:
            logger.warning(f"SVD计算失败: {e}")
            return None

    def _compute_layer_loss(
        self,
        student_sv: torch.Tensor,
        teacher_sv: torch.Tensor
    ) -> torch.Tensor:
        """计算单层的奇异值对齐loss

        Args:
            student_sv: 学生奇异值 [k]
            teacher_sv: 教师奇异值 [k]

        Returns:
            单层loss
        """
        # 对齐长度
        min_len = min(len(student_sv), len(teacher_sv))
        student_sv = student_sv[:min_len]
        teacher_sv = teacher_sv[:min_len]

        losses = []

        if self.align_mode in ('max', 'both'):
            # 最大奇异值对齐
            max_loss = F.mse_loss(student_sv[0:1], teacher_sv[0:1])
            losses.append(max_loss)

        if self.align_mode in ('avg', 'both'):
            # 平均奇异值对齐
            avg_loss = F.mse_loss(student_sv.mean(), teacher_sv.mean())
            losses.append(avg_loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean()

    def compute_loss_multi_teacher(
        self,
        student_model: nn.Module,
        teacher_models: List[nn.Module],
        teacher_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """计算多教师的权重对齐loss

        Args:
            student_model: 学生模型
            teacher_models: 教师模型列表
            teacher_ids: 教师模型ID列表（用于缓存）

        Returns:
            权重对齐loss
        """
        if teacher_ids is None:
            teacher_ids = list(range(len(teacher_models)))

        losses = []
        for teacher_model, teacher_id in zip(teacher_models, teacher_ids):
            loss = self.compute_loss(student_model, teacher_model, teacher_id)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0)

        return torch.stack(losses).mean()

    def clear_cache(self):
        """清除教师奇异值缓存"""
        self._teacher_sv_cache.clear()
        logger.debug("清除教师奇异值缓存")

    def get_svd_statistics(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """获取模型的奇异值统计信息

        Args:
            model: 模型

        Returns:
            统计信息字典，key为层名，value为统计量
        """
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                sv = self._compute_svd(weight)
                if sv is not None:
                    stats[name] = {
                        'max_sv': sv[0].item(),
                        'mean_sv': sv.mean().item(),
                        'min_sv': sv[-1].item() if len(sv) > 0 else 0.0,
                        'num_sv': len(sv)
                    }
        return stats
