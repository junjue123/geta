import numpy as np


class DynamicPruner:
    def __init__(self, original_size, target_size, prune_rate_range=(0.01, 0.08), smooth_factor=1.0,
                 grad_sensitivity=0.3, grad_window=2, pid_gains=(0.5, 0.1, 0.2)):
        """
        动态剪枝器：基于模型大小进度、历史梯度反馈的自适应剪枝率调整器
        核心逻辑：通过余弦退火基础调度+梯度敏感度调节+PID反馈控制，实现平滑且精准的剪枝率动态调整
        避免单次剪枝幅度过大导致模型性能崩溃，同时确保逐步逼近目标压缩率

        参数说明：
        ----------
        original_size : int/float
            模型原始大小（单位统一即可，如 KB、MB、参数量等），作为剪枝进度计算的基准
        target_size : int/float
            模型压缩目标大小（需小于 original_size），用于计算目标稀疏度
        prune_rate_range : tuple (min_rate, max_rate)
            剪枝率的有效范围限制（默认 (0.01, 0.08)），避免剪枝率过高或过低
            - min_rate: 最小剪枝率（防止剪枝不足），需满足 0 ≤ min_rate < max_rate ≤ 1.0
            - max_rate: 最大剪枝率（防止单次剪枝过度）
        smooth_factor : float (≥0.1)
            剪枝率下降平滑系数（默认 1.0），值越大，剪枝率随进度下降越平缓
            - 用于指数衰减项，控制剪枝率的全局下降趋势
        grad_sensitivity : float (0.01~1.0)
            梯度敏感度系数（默认 0.3），值越大，历史梯度对剪枝率的影响越显著
            - 梯度反映模型大小的变化速率，用于自适应调整剪枝节奏（如梯度小则适当提高剪枝率）
        grad_window : int (≥1)
            梯度计算的历史窗口大小（默认 2），控制计算平均梯度时使用的历史步数
            - 窗口越大，梯度估计越平稳，抗噪声能力越强；窗口越小，响应越灵敏
        pid_gains : tuple (Kp, Ki, Kd)
            PID 反馈控制器的增益参数（默认 (0.5, 0.1, 0.2)），用于修正剪枝率偏差
            - Kp: 比例增益，直接响应当前稀疏度与目标的偏差（核心调节项）
            - Ki: 积分增益，累积历史偏差，消除静态误差（防止长期达不到目标）
            - Kd: 微分增益，反映偏差变化率，抑制剪枝率波动（稳定调节过程）
        """
        # 基础配置参数（带合法性校验）
        self.original_size = original_size
        self.target_size = target_size
        self.min_prune_rate, self.max_prune_rate = prune_rate_range
        self.smooth_factor = max(0.1, smooth_factor)  # 避免过小值导致剪枝率突变
        self.grad_sensitivity = np.clip(grad_sensitivity, 0.01, 1.0)  # 限制敏感度范围
        self.grad_window = max(1, int(grad_window))  # 确保窗口大小为正整数
        self.Kp, self.Ki, self.Kd = pid_gains

        # 输入合法性校验
        if self.target_size >= self.original_size:
            raise ValueError(f"目标大小({target_size})必须小于原始大小({original_size})")
        if self.min_prune_rate >= self.max_prune_rate or self.min_prune_rate < 0 or self.max_prune_rate > 1.0:
            raise ValueError(f"剪枝率范围需满足 0 ≤ min_rate < max_rate ≤ 1.0，当前输入：{prune_rate_range}")

        # 历史数据存储（用于梯度计算和PID控制）
        self.size_history = []  # 记录每一步的模型大小
        self.prune_rate_history = []  # 记录每一步的剪枝率
        self.error_history = []  # 记录每一步的稀疏度偏差（用于PID积分/微分）

        # 核心目标参数
        self.target_sparsity = 1.0 - (self.target_size / self.original_size)  # 目标稀疏度（已剪枝比例）
        self.max_single_prune = 0.1  # 单次最大剪枝率限制（安全阈值，避免过度剪枝）

    def _calc_gradient(self):
        """
        私有方法：计算模型大小变化的梯度（归一化）
        梯度反映模型大小的下降速率，用于判断前序剪枝的效果
        返回：窗口内的平均归一化梯度（[-1, 1]区间）
        """
        # 至少需要2个历史大小才能计算梯度
        if len(self.size_history) < 2:
            return 0.0

        # 计算历史梯度：(上一步大小 - 当前大小) / 上一步大小 → 反映剪枝导致的大小下降比例
        grads = []
        for i in range(1, len(self.size_history)):
            size_prev = self.size_history[i-1]
            size_curr = self.size_history[i]
            grad = (size_prev - size_curr) / size_prev  # 梯度为正表示大小下降（剪枝有效）
            grads.append(grad)

        # 取最近N步梯度的均值（N=grad_window），增强稳定性
        window_grads = grads[-self.grad_window:] if len(grads) >= self.grad_window else grads
        avg_grad = np.mean(window_grads)

        # 归一化到[-1, 1]区间（基于最大预期梯度，避免极端值影响）
        max_expected_grad = self.max_prune_rate * 1.5  # 预期最大梯度（经验值，平衡灵敏度）
        avg_grad_norm = np.clip(avg_grad / max_expected_grad, -1.0, 1.0)

        return avg_grad_norm

    def _calc_base_prune_rate(self, progress):
        """
        私有方法：计算基础剪枝率（基于余弦退火调度）
        核心逻辑：剪枝进度越高（越接近目标），基础剪枝率越低，避免后期过度剪枝
        - 余弦退火提供周期性衰减，增强适应性
        - 指数衰减确保全局下降趋势
        """
        # 余弦退火项：(1+cos(π*progress))/2 → 进度从0→1时，值从1→0（平滑衰减）
        cos_anneal = (1 + np.cos(progress * np.pi)) / 2

        # 指数衰减项：exp(-progress * smooth_factor) → 控制衰减速率
        exp_decay = np.exp(-progress * self.smooth_factor)

        # 组合计算基础剪枝率（限制在[min_rate, max_rate]范围内）
        combined_weight = cos_anneal * exp_decay
        base_rate = self.min_prune_rate + (self.max_prune_rate - self.min_prune_rate) * combined_weight

        return base_rate

    def _calc_pid_adjust(self, current_sparsity):
        """
        私有方法：PID反馈调节因子计算
        作用：根据当前稀疏度与目标的偏差，动态调整剪枝率（修正偏差）
        返回：PID调节因子（0.5~1.5区间，1.0表示无调节）
        """
        # 计算当前偏差：目标稀疏度 - 当前稀疏度（正偏差→需要更多剪枝，负偏差→剪枝过度）
        error = self.target_sparsity - current_sparsity
        self.error_history.append(error)

        # PID三部分计算
        P = self.Kp * error  # 比例项：直接响应当前偏差
        I = self.Ki * np.sum(self.error_history[-self.grad_window:])  # 积分项：累积历史偏差（窗口限制防溢出）
        D = self.Kd * (error - self.error_history[-2]) if len(self.error_history) >= 2 else 0.0  # 微分项：抑制波动

        # PID总输出归一化为调节因子（0.5~1.5，避免极端调节）
        pid_output = P + I + D
        pid_adjust = np.clip(1.0 + pid_output, 0.5, 1.5)

        return pid_adjust

    def _soft_normalize(self, x, lower=0.4, upper=1.6):
        """
        私有方法：软归一化函数（基于tanh）
        作用：将任意实数映射到指定区间，兼具平滑性和边界限制（避免硬截断导致的突变）
        """
        return lower + (upper - lower) * (np.tanh(x) + 1) / 2

    def get_current_prune_rate(self, current_model_size):
        """
        计算当前步骤的剪枝率（核心对外接口）
        参数：
            current_model_size : int/float - 当前模型的实际大小（需与original_size单位一致）
        返回：
            float - 当前步骤的剪枝率（保留6位小数，在[min_prune_rate, max_prune_rate]范围内）
        异常：
            ValueError - 当当前大小小于目标大小（无需剪枝）或大于原始大小（输入错误）时触发
        """
        # 输入合法性校验
        if current_model_size < self.target_size:
            raise ValueError(f"当前大小({current_model_size:.2f})已小于目标大小({self.target_size})，无需继续剪枝")
        if current_model_size > self.original_size:
            raise ValueError(f"当前大小({current_model_size:.2f})不能大于原始大小({self.original_size})")

        # 记录当前大小
        self.size_history.append(current_model_size)

        # 第一步强制使用最小剪枝率（探索性剪枝，避免初始剪枝幅度过大）
        if len(self.size_history) == 1:
            first_rate = self.min_prune_rate
            self.prune_rate_history.append(round(first_rate, 6))
            return round(first_rate, 6)

        # 1. 计算当前剪枝进度和稀疏度
        current_sparsity = 1.0 - (current_model_size / self.original_size)  # 当前已剪枝比例
        progress = (self.original_size - current_model_size) / (self.original_size - self.target_size)  # 进度0→1
        progress = np.clip(progress, 0.01, 1.0)  # 避免进度为0导致计算异常

        # 2. 计算基础剪枝率（余弦退火调度）
        base_rate = self._calc_base_prune_rate(progress)

        # 3. 计算梯度调节因子（基于历史大小变化率）
        avg_grad_norm = self._calc_gradient()
        grad_input = avg_grad_norm * self.grad_sensitivity * 3.0  # 放大梯度影响（经验系数）
        grad_adjust = self._soft_normalize(grad_input, lower=0.5, upper=1.5)  # 温和调节范围

        # 4. 计算PID调节因子（基于稀疏度偏差）
        pid_adjust = self._calc_pid_adjust(current_sparsity)
        pid_adjust = self._soft_normalize(pid_adjust, lower=0.6, upper=1.4)  # 平滑PID输出

        # 5. 融合所有因子得到最终剪枝率
        raw_rate = base_rate * grad_adjust * pid_adjust

        # 6. 应用安全限制（单次最大剪枝率+剪枝率范围）
        final_rate = np.clip(raw_rate, self.min_prune_rate, self.max_prune_rate)
        final_rate = min(final_rate, self.max_single_prune)  # 额外限制单次最大剪枝率

        # 7. 接近目标时强制使用最小剪枝率（精细调整，避免超调）
        if abs(current_sparsity - self.target_sparsity) < 1e-4:
            final_rate = self.min_prune_rate

        # 记录并返回剪枝率（保留6位小数）
        final_rate = round(final_rate, 6)
        self.prune_rate_history.append(final_rate)
        return final_rate

    def get_prune_status(self):
        """
        辅助方法：获取当前剪枝状态详情（用于调试和监控）
        返回：
            dict - 包含历史记录、当前状态、梯度信息等关键参数
        """
        # 无历史数据时返回初始状态
        if not self.size_history:
            return {
                "历史大小记录": "无",
                "历史剪枝率记录": "无",
                "当前大小": None,
                "当前剪枝率": None,
                "当前稀疏度": 0.0,
                "目标稀疏度": round(self.target_sparsity, 4),
                "当前进度": 0.0,
                "梯度计算窗口": self.grad_window,
                "窗口内平均梯度": 0.0,
                "是否接近目标": False
            }

        # 有历史数据时计算详细状态
        latest_size = self.size_history[-1]
        current_sparsity = 1.0 - (latest_size / self.original_size)
        progress = (self.original_size - latest_size) / (self.original_size - self.target_size)
        progress = np.clip(progress, 0.0, 1.0)
        avg_grad_norm = self._calc_gradient()

        return {
            "历史大小记录": self.size_history,
            "历史剪枝率记录": self.prune_rate_history,
            "当前大小": latest_size,
            "当前剪枝率": self.prune_rate_history[-1] if self.prune_rate_history else None,
            "当前稀疏度": round(current_sparsity, 4),
            "目标稀疏度": round(self.target_sparsity, 4),
            "当前进度": round(progress, 4),
            "梯度计算窗口": self.grad_window,
            "窗口内平均梯度": round(avg_grad_norm, 4),
            "是否接近目标": abs(current_sparsity - self.target_sparsity) < 1e-4
        }


# ------------------------------ 改进后的使用示例 ------------------------------
if __name__ == "__main__":
    print("=== 动态剪枝器仿真：基于剪枝率自动衰减模型大小 ===")
    # 初始化剪枝器：1000KB → 300KB（目标稀疏度70%）
    pruner = DynamicPruner(
        original_size=1200,  # 原始大小1000KB
        target_size=256,     # 目标大小300KB
        prune_rate_range=(0.005, 0.02),  # 剪枝率0.5%~5%（更精细的范围）
        smooth_factor=1.2,   # 剪枝率下降中等平缓
        grad_sensitivity=0.3,  # 梯度敏感度适中
        grad_window=3,       # 梯度窗口2步
        pid_gains=(0.8, 0.01, 0.3)  # PID增益（增强比例项，加快收敛）
    )

    # 仿真核心逻辑：
    # 1. 初始大小 = 原始大小
    # 2. 每一步用当前大小获取剪枝率
    # 3. 下一步大小 = 当前大小 × (1 - 剪枝率)
    # 4. 直到大小接近/达到目标大小

    current_size = pruner.original_size  # 初始大小
    max_steps = 150  # 最大仿真步数（防止无限循环）
    print(f"原始大小：{pruner.original_size}KB，目标大小：{pruner.target_size}KB")
    print(f"目标稀疏度：{pruner.target_sparsity:.4f}，剪枝率范围：{pruner.min_prune_rate:.3f}~{pruner.max_prune_rate:.3f}")
    print(f"最大仿真步数：{max_steps}\n")
    print(f"{'步骤':<4} {'当前大小(KB)':<14} {'剪枝率':<10} {'当前稀疏度':<12} {'窗口平均梯度':<14} {'进度':<6}")
    print("-" * 80)

    for step in range(1, max_steps + 1):
        try:
            # 1. 获取当前剪枝率
            prune_rate = pruner.get_current_prune_rate(current_size)
            # 2. 获取当前状态
            status = pruner.get_prune_status()
            # 3. 打印当前步骤信息
            print(f"{step:<4} {current_size:<14.2f} {prune_rate:<10.4f} {status['当前稀疏度']:<12.4f} "
                  f"{status['窗口内平均梯度']:<14.4f} {status['当前进度']:<6.4f}")
            # 4. 计算下一步模型大小（核心：基于剪枝率衰减）
            next_size = current_size * (1 - prune_rate)
            # 5. 检查是否接近目标（误差<0.5%则停止）
            if abs(next_size - pruner.target_size) / pruner.target_size < 0.005:
                print(f"\n✅ 步骤{step+1}：模型大小已接近目标！")
                print(f"下一步大小：{next_size:.2f}KB，目标大小：{pruner.target_size}KB，误差：{abs(next_size-pruner.target_size):.2f}KB")
                break
            # 6. 更新当前大小，进入下一步
            current_size = next_size
        except ValueError as e:
            print(f"\n❌ 步骤{step}：{e}")
            break

    # 最终状态汇总
    final_status = pruner.get_prune_status()
    print(f"\n=== 仿真结束 ===")
    print(f"最终模型大小：{final_status['当前大小']:.2f}KB")
    print(f"最终稀疏度：{final_status['当前稀疏度']:.4f}（目标：{final_status['目标稀疏度']:.4f}）")
    print(f"总剪枝步数：{step}")
    print(f"历史剪枝率范围：{min(pruner.prune_rate_history):.4f}~{max(pruner.prune_rate_history):.4f}")