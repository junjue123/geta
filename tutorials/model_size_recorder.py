import numpy as np
from collections import deque
from typing import Optional


class AdaptivePruningScheduler:
    def __init__(self,
                 total_ratio: float,
                 num_steps: int,
                 original_size: float,
                 original_flops: float,
                 target_size: Optional[float] = None,
                 target_flops: Optional[float] = None,
                 window_size: int = 3,
                 damping_factor: float = 10.0,
                 min_pruning_rate: float = 0.001):
        """
        自适应剪枝调度器。

        参数:
            total_ratio: 预期的总剪枝比例 (用于计算Base Schedule)
            num_steps: 预期的剪枝步数 (用于计算Base Schedule)
            original_size: 原始模型参数量 (MB 或 数量)
            original_flops: 原始模型计算量
            target_size: 目标模型参数量 (若不指定，则仅依赖total_ratio)
            target_flops: 目标模型计算量 (若不指定，则仅依赖total_ratio)
            window_size: 计算梯度的历史窗口大小 (n个epoch)
            damping_factor: 梯度阻尼系数，值越大，对快速下降的反应越敏感（剪枝越慢）
            min_pruning_rate: 最小剪枝率 (当超过预期步数仍未达标时的保底速率)
        """
        self.total_ratio = total_ratio
        self.num_steps = num_steps
        self.original_size = original_size
        self.original_flops = original_flops
        self.target_size = target_size if target_size else 0
        self.target_flops = target_flops if target_flops else 0

        self.window_size = window_size
        self.damping_factor = damping_factor
        self.min_pruning_rate = min_pruning_rate

        # 内部状态
        self.current_step = 0
        # 使用双端队列存储最近n次的大小和FLOPs，用于计算梯度
        self.history_size = deque(maxlen=window_size + 1)
        self.history_flops = deque(maxlen=window_size + 1)

        # 初始化历史记录
        self.history_size.append(original_size)
        self.history_flops.append(original_flops)

    def _get_base_ratio(self) -> float:
        """
        基于余弦下降生成当前步骤的基础剪枝率 (Base Schedule)。
        对应原来的 'fall' 模式。
        """
        # 如果超过了预设步数，返回最小保底剪枝率，保证继续剪枝
        if self.current_step >= self.num_steps:
            return self.min_pruning_rate

        # 计算进度 (0到1)
        progress = self.current_step / self.num_steps

        # 余弦下降曲线：权重从1平滑降低到0
        # 注意：这里计算的是"当前这一步"相对于"总remaining"的权重
        # 为了简化，我们模拟原函数的逻辑，生成一个单步的瞬时目标

        # 计算该步在原本平滑曲线中应承担的比例份额
        # 使用微分思想：cos曲线在当前点的斜率绝对值 * total_ratio
        # 或者更简单：直接预计算好权重的分布，取当前索引
        weight = (1 + np.cos(progress * np.pi)) / 2

        # 归一化处理比较复杂，这里简化为：
        # 假设平均每步剪 total_ratio / num_steps
        # 用 weight 调节这个平均值
        avg_ratio = self.total_ratio / self.num_steps
        base_ratio = avg_ratio * 2 * weight  # *2 是为了让积分面积近似归一

        return max(base_ratio, self.min_pruning_rate)

    def _calculate_gradient_penalty(self) -> float:
        """
        计算梯度惩罚因子。
        如果最近模型变小得太快，返回一个大于1的数来除以基础剪枝率。
        """
        if len(self.history_size) < 2:
            return 1.0  # 历史数据不足，不惩罚

        # 1. 计算 Size 的变化梯度 (归一化)
        # 过去N步的变化量 / 原始大小
        delta_size = self.history_size[0] - self.history_size[-1]
        grad_size = (delta_size / self.original_size) / len(self.history_size)

        # 2. 计算 FLOPs 的变化梯度 (归一化)
        delta_flops = self.history_flops[0] - self.history_flops[-1]
        grad_flops = (delta_flops / self.original_flops) / len(self.history_flops)

        # 取两者中变化更剧烈的一个作为主梯度
        max_grad = max(grad_size, grad_flops)

        # 避免负梯度（模型变大？）
        max_grad = max(0.0, max_grad)

        # 计算惩罚系数：梯度越大，分母越大，最终剪枝率越小
        # Penalty = 1 + alpha * Gradient
        penalty = 1 + (self.damping_factor * max_grad * 100)  # *100是为了把小数放大到敏感区间

        return penalty

    def step(self, current_size: float, current_flops: float) -> float:
        """
        执行一步调度，返回当前这一步应该执行的剪枝率 (sparsity to prune in this step)。

        返回:
            current_pruning_ratio: 本次迭代需要剪掉的比例 (float)
        """
        # 1. 检查是否达到目标 (终止条件)
        size_condition = current_size <= self.target_size
        flops_condition = current_flops <= self.target_flops

        # 如果两个目标都满足（或者未设置目标），则停止剪枝
        if (self.target_size > 0 and size_condition) and \
                (self.target_flops > 0 and flops_condition):
            return 0.0

        # 2. 更新历史记录
        self.history_size.append(current_size)
        self.history_flops.append(current_flops)

        # 3. 获取基础调度剪枝率
        base_ratio = self._get_base_ratio()

        # 4. 计算梯度惩罚
        penalty = self._calculate_gradient_penalty()

        # 5. 调整剪枝率
        adjusted_ratio = base_ratio / penalty

        # 6. 更新步数计数器
        self.current_step += 1

        return adjusted_ratio


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 初始化调度器
    scheduler = AdaptivePruningScheduler(
        total_ratio=0.7,  # 计划总共剪掉70% (仅作参考基准)
        num_steps=50,  # 计划在50步内完成
        original_size=100.0,  # 假设原模型 100MB
        original_flops=500.0,  # 假设原 FLOPs 500M
        target_size=30.0,  # 目标：压缩到 30MB
        target_flops=150.0,  # 目标：压缩到 150M FLOPs
        window_size=3,  # 观察过去3个epoch的变化
        damping_factor=5.0  # 阻尼系数
    )

    # 2. 模拟训练循环
    current_model_size = 100.0
    current_model_flops = 500.0

    print(f"{'Step':<5} | {'Base Size':<10} | {'Ratio':<10} | {'Status'}")
    print("-" * 50)

    for epoch in range(1, 70):  # 故意设置比 num_steps(50) 长，测试自动延展功能

        # 获取本轮剪枝率
        ratio = scheduler.step(current_model_size, current_model_flops)

        if ratio == 0.0:
            print(f"Epoch {epoch}: 目标已达成，停止剪枝。")
            break

        # 模拟剪枝操作 (假设模型真的变小了)
        # 在实际代码中，这里会调用 pruner.step()
        # 假设每次剪枝并不完美，有一定的随机性
        actual_prune_effect = ratio * np.random.uniform(0.8, 1.1)

        current_model_size = current_model_size * (1 - actual_prune_effect)
        current_model_flops = current_model_flops * (1 - actual_prune_effect)

        print(f"{epoch:<5} | {current_model_size:<10.2f} | {ratio:<10.4f} | Pruning...")