"""
教师模型集合管理模块
管理剪枝前最优、剪枝中前N个、随机采样的教师模型。

关键优化:
- 使用堆维护前N个最优模型，O(logN) 更新
- 使用蓄水池采样算法处理随机采样
- 使用深拷贝避免模型引用问题
"""

import copy
import heapq
import random
import itertools
import logging
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TeacherEnsemble:
    """教师模型集合管理器

    管理三类教师模型:
    1. pre_pruning_best: 剪枝前最优模型
    2. pruning_top_n: 剪枝过程中前N个epoch的模型
    3. pruning_random: 剪枝过程中随机采样的模型

    Args:
        top_n: 剪枝过程中保留的前N个最优模型数量
        device: 模型存储设备
    """

    def __init__(self, top_n: int = 3, device: str = 'cpu'):
        self.top_n = top_n
        self.device = device

        # 剪枝前最优模型
        self.pre_pruning_best: Optional[Dict[str, Any]] = None

        # 剪枝中前N个模型 (使用最小堆，存储 (loss, counter, model_state, epoch))
        # counter 用于打破 loss 相同时的比较
        self.pruning_top_n: List[tuple] = []
        self._counter = itertools.count()

        # 剪枝中随机采样 (蓄水池采样)
        self.pruning_random: Optional[Dict[str, Any]] = None
        self.pruning_sample_count = 0

        # 当前是否在剪枝阶段
        self.is_pruning_phase = False

    def update(self, model: nn.Module, loss: float, epoch: int, is_pruning: bool = False):
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
                'state_dict': self._safe_copy_state_dict(model),
                'loss': loss,
                'epoch': epoch
            }
            logger.debug(f"更新剪枝前最优: epoch={epoch}, loss={loss:.4f}")

    def _update_pruning_top_n(self, model: nn.Module, loss: float, epoch: int):
        """更新剪枝中前N个模型 (最大堆，保留loss最小的N个)

        使用负loss构建最大堆，堆顶是loss最大的元素。
        当新元素loss更小时，替换堆顶。
        """
        counter = next(self._counter)
        # 使用负loss构建最大堆
        neg_loss = -loss

        if len(self.pruning_top_n) < self.top_n:
            # 堆未满，直接加入
            entry = (neg_loss, counter, epoch, self._safe_copy_state_dict(model))
            heapq.heappush(self.pruning_top_n, entry)
            logger.debug(f"加入top-n: epoch={epoch}, loss={loss:.4f}")
        elif neg_loss > self.pruning_top_n[0][0]:
            # 新loss更小（负loss更大），替换堆顶（loss最大的）
            old_entry = heapq.heapreplace(
                self.pruning_top_n,
                (neg_loss, counter, epoch, self._safe_copy_state_dict(model))
            )
            logger.debug(f"替换top-n: epoch={epoch}, loss={loss:.4f} (替换 epoch={old_entry[2]})")

    def _update_pruning_random(self, model: nn.Module, loss: float, epoch: int):
        """更新剪枝中随机采样 (蓄水池采样算法)"""
        self.pruning_sample_count += 1

        # 蓄水池采样: 以 1/n 的概率替换当前样本
        if self.pruning_random is None:
            self.pruning_random = {
                'state_dict': self._safe_copy_state_dict(model),
                'loss': loss,
                'epoch': epoch
            }
        elif random.random() < 1.0 / self.pruning_sample_count:
            self.pruning_random = {
                'state_dict': self._safe_copy_state_dict(model),
                'loss': loss,
                'epoch': epoch
            }
            logger.debug(f"更新随机采样: epoch={epoch}, loss={loss:.4f}")

    def get_teachers(self) -> List[Dict[str, Any]]:
        """获取所有教师模型

        Returns:
            教师模型列表，每个元素包含 state_dict, loss, epoch
        """
        teachers = []

        # 1. 剪枝前最优
        if self.pre_pruning_best is not None:
            teachers.append(self.pre_pruning_best)

        # 2. 剪枝中前N个 (按loss排序，neg_loss转回正loss)
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

    def get_teacher_logits(self, model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
        """获取所有教师模型的logits

        Args:
            model: 学生模型 (用于获取模型结构)
            x: 输入数据

        Returns:
            教师logits列表
        """
        teachers = self.get_teachers()
        logits_list = []

        # 创建临时模型用于推理
        temp_model = copy.deepcopy(model)
        temp_model.eval()

        with torch.no_grad():
            for teacher in teachers:
                temp_model.load_state_dict(teacher['state_dict'])
                temp_model = temp_model.to(x.device)
                logits = temp_model(x)
                logits_list.append(logits)

        return logits_list

    def _safe_copy_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """安全复制模型状态字典

        使用深拷贝避免引用问题，但只复制参数（不复制梯度）
        """
        return {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    def state_dict(self) -> Dict[str, Any]:
        """保存教师集合状态"""
        return {
            'pre_pruning_best': self.pre_pruning_best,
            'pruning_top_n': self.pruning_top_n,
            'pruning_random': self.pruning_random,
            'pruning_sample_count': self.pruning_sample_count,
            'is_pruning_phase': self.is_pruning_phase,
            'top_n': self.top_n
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """恢复教师集合状态"""
        self.pre_pruning_best = state_dict.get('pre_pruning_best')
        self.pruning_top_n = state_dict.get('pruning_top_n', [])
        self.pruning_random = state_dict.get('pruning_random')
        self.pruning_sample_count = state_dict.get('pruning_sample_count', 0)
        self.is_pruning_phase = state_dict.get('is_pruning_phase', False)
        self.top_n = state_dict.get('top_n', self.top_n)

    def __len__(self) -> int:
        """返回教师模型数量"""
        count = 0
        if self.pre_pruning_best is not None:
            count += 1
        count += len(self.pruning_top_n)
        if self.pruning_random is not None:
            count += 1
        return count

    def clear(self):
        """清空教师集合"""
        self.pre_pruning_best = None
        self.pruning_top_n.clear()
        self.pruning_random = None
        self.pruning_sample_count = 0
        self.is_pruning_phase = False
