"""
软标签蒸馏模块
实现经典 KD 蒸馏 (Hinton et al., 2015)

公式:
L_KD = α * T² * mean_t[KL(softmax(z_t/T) || softmax(z_s/T))]
     + (1-α) * CE(y, z_s)

其中:
- z_t: 教师logits
- z_s: 学生logits
- T: 温度参数
- α: 蒸馏loss权重
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SoftLabelDistiller:
    """软标签蒸馏器

    实现经典 KD 蒸馏，支持多教师模型集合。

    Args:
        temperature: 温度参数，控制softmax的平滑程度
        alpha: 蒸馏loss权重，1-alpha 为CE loss权重
        kd_mode: KD散度模式
            - 'forward': KL(p_teacher || p_student) - 模式覆盖
            - 'reverse': KL(p_student || p_teacher) - 模式寻找
            - 'js': Jensen-Shannon散度 - 平衡两者
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        kd_mode: str = 'forward'
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.kd_mode = kd_mode

        if kd_mode not in ('forward', 'reverse', 'js'):
            raise ValueError(f"kd_mode 必须是 'forward', 'reverse' 或 'js'，收到 '{kd_mode}'")

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算软标签蒸馏loss

        Args:
            student_logits: 学生模型输出 logits [B, C]
            teacher_logits_list: 教师模型输出 logits 列表，每个 [B, C]
            labels: 真实标签 [B]，如果为None则只返回KD loss

        Returns:
            蒸馏loss (标量)
        """
        if not teacher_logits_list:
            logger.warning("教师模型列表为空，返回0")
            return torch.tensor(0.0, device=student_logits.device)

        T = self.temperature
        kd_losses = []

        # 计算学生softmax (带温度)
        student_soft = F.log_softmax(student_logits / T, dim=-1)

        for teacher_logits in teacher_logits_list:
            # 教师softmax (带温度)
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)

            # 计算KL散度
            if self.kd_mode == 'forward':
                # KL(p_teacher || p_student)
                kd = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            elif self.kd_mode == 'reverse':
                # KL(p_student || p_teacher)
                teacher_log_soft = F.log_softmax(teacher_logits / T, dim=-1)
                kd = F.kl_div(teacher_log_soft, F.softmax(student_logits / T, dim=-1), reduction='batchmean')
            else:  # js
                # Jensen-Shannon散度
                m = 0.5 * (teacher_soft + F.softmax(student_logits / T, dim=-1))
                kd = 0.5 * F.kl_div(student_soft, m, reduction='batchmean') + \
                     0.5 * F.kl_div(F.log_softmax(teacher_logits / T, dim=-1), m, reduction='batchmean')

            kd_losses.append(kd)

        # 多教师取平均
        kd_loss = torch.stack(kd_losses).mean()

        # 缩放
        kd_loss = kd_loss * (T ** 2)

        # 如果有标签，混合CE loss
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        else:
            total_loss = kd_loss

        return total_loss

    def compute_loss_with_features(
        self,
        student_model: nn.Module,
        teacher_models: List[nn.Module],
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算蒸馏loss（自动获取logits）

        Args:
            student_model: 学生模型
            teacher_models: 教师模型列表
            x: 输入数据
            labels: 真实标签

        Returns:
            蒸馏loss
        """
        student_logits = student_model(x)

        teacher_logits_list = []
        for teacher in teacher_models:
            with torch.no_grad():
                teacher_logits = teacher(x)
            teacher_logits_list.append(teacher_logits)

        return self.compute_loss(student_logits, teacher_logits_list, labels)
