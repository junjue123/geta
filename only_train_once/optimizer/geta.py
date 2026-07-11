import logging
import math
import os
from contextlib import contextmanager

import numpy as np
import torch
from torch.optim.optimizer import required

from only_train_once.transform import (
    TensorTransform,
    tensor_transformation_param_group,
)

from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class RCAJSController:
    """
    [DACO - RCAJS] Rate-Control Adaptive Joint Sparsity (RCAJS)
    Lagrangian KL-divergence closed-loop controller for adaptive pruning.

    论文核心:
        - 目标: min  KL(p_target || p_curr) + λ * stability_term
        - 提前停止: 当 C_curr <= C_target 时停止
        - 自适应剪枝率: r_p = r_base * exp(-β_p * Δ_p)
        - 位宽自适应: φ_b 阈值触发位宽缩减
    """

    def __init__(
        self,
        total_groups: int,
        target_sparsity: float,
        beta_p: float = 0.1,
        lambda_stability: float = 0.1,
        phi_b_threshold: float = 0.8,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.total_groups = total_groups
        self.target_sparsity = target_sparsity  # 目标稀疏度 (0~1)
        self.target_num_groups = int(total_groups * (1 - target_sparsity))  # 目标重要组数
        self.beta_p = beta_p  # 自适应速率系数
        self.lambda_stability = lambda_stability  # Lagrangian 稳定性权重
        self.phi_b_threshold = phi_b_threshold  # 位宽自适应触发阈值
        self.verbose = verbose
        self.device = device

        # 状态追踪
        self.history_sparsity = []  # 历史稀疏度记录
        self.history_delta_p = []   # 历史 Δ_p = target - current
        self.history_prune_rate = []  # 历史剪枝率
        self.current_sparsity = 0.0
        self.curr_pruning_rate = 0.05  # 初始剪枝率 (cosine 基础值)
        self.early_stopped = False

        # KL 散度追踪
        self.prev_kl_div = None
        self.kl_window_size = 5
        self.kl_history = []

        # 位宽自适应
        self.bit_layers_below_max_count = 0
        self.bit_layers_total_count = 0
        self.bit_reduction_triggered = False

    def compute_current_sparsity(self, num_pruned_groups: int) -> float:
        """计算当前稀疏度 = 已剪枝组数 / 总组数"""
        self.current_sparsity = num_pruned_groups / max(self.total_groups, 1)
        return self.current_sparsity

    def compute_delta_p(self) -> float:
        """
        计算稀疏度偏差 Δ_p = target_sparsity - current_sparsity
        正偏差 → 需要更多剪枝；负偏差 → 剪枝过度
        """
        delta_p = self.target_sparsity - self.current_sparsity
        return delta_p

    def compute_stability_term(self) -> float:
        """
        稳定性项: 基于历史稀疏度变化的波动程度
        Ψ_stability = exp(-σ(Δ_p_history))
        波动小 → Ψ 接近 1（鼓励继续剪枝）
        波动大 → Ψ 接近 0（抑制激进剪枝）
        """
        if len(self.history_delta_p) < 2:
            return 1.0
        delta_p_arr = np.array(self.history_delta_p[-self.kl_window_size:])
        std_delta = np.std(delta_p_arr)
        stability_term = math.exp(-std_delta)
        return max(0.1, stability_term)  # 下限 0.1

    def compute_kl_divergence(self, score_dist: torch.Tensor) -> float:
        """
        计算当前分数分布与均匀分布的 KL 散度
        KL(Uniform || Score) 越小 → 分布越不均匀（剪枝越激进）
        用于判断是否接近收敛
        """
        # 归一化为概率分布
        probs = torch.nn.functional.softmax(score_dist.flatten(), dim=0)
        probs = probs.cpu().numpy()
        probs = np.maximum(probs, 1e-10)  # 避免 log(0)

        # 均匀分布
        uniform = 1.0 / len(probs)
        kl_div = np.sum(probs * np.log(probs / uniform))

        # 滑动平均
        self.kl_history.append(kl_div)
        if len(self.kl_history) > self.kl_window_size:
            self.kl_history.pop(0)

        return kl_div

    def should_early_stop(self, num_active_redundant: int) -> bool:
        """
        提前停止判断: 当剩余待剪枝冗余组数 <= 0 时停止（即已无冗余组可剪）
        论文条件: C_curr <= C_target → 无更多冗余组需要剪枝时停止
        """
        if num_active_redundant <= 0:
            self.early_stopped = True
            if self.verbose:
                target_redundant = self.total_groups - self.target_num_groups
                print(f"  [RCAJS] 提前停止触发: 剩余待剪冗余={num_active_redundant} <= 0, "
                      f"已剪={self.total_groups - num_active_redundant}, 目标保留={self.target_num_groups}")
            return True
        return False

    def compute_adaptive_prune_rate(self, base_rate: float = None) -> float:
        """
        [核心] 自适应剪枝率计算 — Lagrangian 闭环控制

        公式: r_p = r_base * exp(-β_p * Δ_p) * Ψ_stability

        - Δ_p > 0 (稀疏度不足): exp(-β*Δ) < 1 → 加速剪枝
        - Δ_p < 0 (过度剪枝): exp(-β*Δ) > 1 → 减速/回退
        - Ψ_stability < 1 (波动大): 进一步抑制激进剪枝
        """
        if self.early_stopped:
            return 0.0

        delta_p = self.compute_delta_p()
        stability_term = self.compute_stability_term()

        if base_rate is None:
            # Cosine 退火基础剪枝率 (与原 DGD 一致)
            progress = min(1.0, self.current_sparsity / max(self.target_sparsity, 0.01))
            base_rate = 0.01 + 0.07 * (1 + math.cos(math.pi * progress)) / 2

        # Lagrangian 自适应修正
        adaptive_factor = math.exp(-self.beta_p * delta_p)
        self.curr_pruning_rate = base_rate * adaptive_factor * stability_term

        # 硬约束: 剪枝率上限
        self.curr_pruning_rate = np.clip(self.curr_pruning_rate, 0.001, 0.15)

        return self.curr_pruning_rate

    def update_bit_adaptive(self, bit_layers_dict: dict, max_bit: int = 16):
        """
        位宽自适应触发判断: 当 φ_b (低于 max_bit 的层比例) > φ_b_threshold 时
        触发一次 bit_reduction
        """
        if not bit_layers_dict:
            return False

        below_max_count = 0
        total_count = 0
        for layer_name, bits in bit_layers_dict.items():
            if isinstance(bits, dict) and "weight" in bits:
                if bits["weight"] < max_bit:
                    below_max_count += 1
                total_count += 1
            elif isinstance(bits, (int, float)):
                if bits < max_bit:
                    below_max_count += 1
                total_count += 1

        if total_count == 0:
            return False

        phi_b = below_max_count / total_count
        self.bit_layers_below_max_count = below_max_count
        self.bit_layers_total_count = total_count

        should_trigger = (phi_b > self.phi_b_threshold) and not self.bit_reduction_triggered
        if should_trigger:
            self.bit_reduction_triggered = True
            if self.verbose:
                print(f"  [RCAJS] 位宽自适应触发: φ_b={phi_b:.2%} > {self.phi_b_threshold} | "
                      f"{below_max_count}/{total_count} 层低于 max_bit={max_bit}")

        return should_trigger

    def step(
        self,
        num_pruned_groups: int,
        score_dist: torch.Tensor = None,
        base_rate: float = None,
    ) -> dict:
        """
        RCAJS 一步更新: 更新状态、返回自适应剪枝率

        Returns:
            dict: {
                'prune_rate': float,          # 自适应剪枝率
                'current_sparsity': float,     # 当前稀疏度
                'delta_p': float,              # 稀疏度偏差
                'stability_term': float,       # 稳定性项 Ψ
                'kl_div': float,               # KL 散度
                'early_stop': bool,            # 是否提前停止
                'bit_adaptive_trigger': bool,  # 是否触发位宽自适应
            }
        """
        # 1. 更新当前稀疏度
        self.compute_current_sparsity(num_pruned_groups)
        delta_p = self.compute_delta_p()

        # 2. 记录历史
        self.history_sparsity.append(self.current_sparsity)
        self.history_delta_p.append(delta_p)

        # 3. 计算 KL 散度（如果提供了分数分布）
        kl_div = 0.0
        if score_dist is not None:
            kl_div = self.compute_kl_divergence(score_dist)

        # 4. 提前停止检查
        target_redundant = self.total_groups - self.target_num_groups
        num_active_redundant = self.total_groups * self.target_sparsity - num_pruned_groups
        early_stop = self.should_early_stop(int(max(0, num_active_redundant)))

        # 5. 计算自适应剪枝率
        prune_rate = self.compute_adaptive_prune_rate(base_rate)
        stability_term = self.compute_stability_term()

        # 6. 记录
        self.history_prune_rate.append(prune_rate)

        return {
            "prune_rate": prune_rate,
            "current_sparsity": self.current_sparsity,
            "delta_p": delta_p,
            "stability_term": stability_term,
            "kl_div": kl_div,
            "early_stop": early_stop,
            "bit_adaptive_trigger": False,  # 由外部调用 update_bit_adaptive 设置
        }

    def get_summary(self) -> dict:
        """返回 RCAJS 控制器状态摘要"""
        return {
            "total_groups": self.total_groups,
            "target_sparsity": self.target_sparsity,
            "current_sparsity": self.current_sparsity,
            "delta_p": self.compute_delta_p(),
            "curr_pruning_rate": self.curr_pruning_rate,
            "stability_term": self.compute_stability_term(),
            "early_stopped": self.early_stopped,
            "bit_adaptive_triggered": self.bit_reduction_triggered,
            "history_len": len(self.history_sparsity),
        }


class GETA(BaseHybridSparseOptimizer):
    """
    GETA: General and Efficient Training framework that Automates
    joint structured pruning and quantization.
    """

    def __init__(
        self,
        params,
        variant="sgd",
        lr=required,
        lr_quant=1e-3,
        first_momentum=None,
        second_momentum=None,
        dampening=None,
        weight_decay=None,

        diffusion_noise_init=1e-4,
        diffusion_steps=None,

        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_steps=1,
        projection_periods=1,
        start_pruning_step=1,
        pruning_steps=1,
        pruning_periods=1,
        group_divisible=1,
        importance_score_criteria="default",
        bit_reduction=2,
        min_bit_wt=2,
        max_bit_wt=16,
        min_bit_act=2,
        max_bit_act=16,
        grad_clip_min=-1.0,
        grad_clip_max=1.0,
        verbose="False",  # if verbose="False", no information is printed
        device="cuda",
        log_dir="outputs",
    ):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_projection_step = start_projection_step
        self.projection_steps = projection_steps
        self.projection_periods = projection_periods
        self.projection_period_duration = (
            self.projection_steps // self.projection_periods
        )
        self.start_pruning_step = start_pruning_step
        self.pruning_periods = int(
            max(1, pruning_periods)
        )  # How many periods that the pruning last for.
        self.pruning_steps = pruning_steps
        self.pruning_period_duration = (
            self.pruning_steps // self.pruning_periods
        )  # How many pruning steps for each period
        self.curr_pruning_period = 0  # Track pruning period
        self.lr_quant = lr_quant
        self.bit_reduction = bit_reduction
        self.min_bit_wt = min_bit_wt  # Minimum bit width for weights
        self.max_bit_wt = max_bit_wt  # Maximum bit width for weights
        self.min_bit_act = min_bit_act  # Minimum bit width for activations
        self.max_bit_act = max_bit_act  # Maximum bit width for activations
        self.grad_clip_min = grad_clip_min
        self.grad_clip_max = grad_clip_max
        self.verbose = verbose
        self.device = device
        self.pruned_group_idxes = list()
        self.gamma = 0.0
        self.d_quant = 0.0
        self.bit_layers = {}  # Store the bit width for each layer

        self.diffusion_noise_init = diffusion_noise_init
        self.diffusion_steps = diffusion_steps if diffusion_steps is not None else pruning_steps
        self.diffusion_noise_ratio = 0.5

        self.logger.info(f"Diffusion SGLD Enabled: Init Noise={self.diffusion_noise_init}")

        if importance_score_criteria == "default":
            self.importance_score_criteria = {
                "magnitude": 0.2,
                "avg_magnitude": 0.2,
                "cosine_similarity": 0.2,
                "taylor_first_order": 0.2,
                "taylor_second_order": 0.2,
            }
        else:
            self.importance_score_criteria = importance_score_criteria
        self.logger.info("Setup GETA")
        self.logger.info(f"importance_score_criteria: {self.importance_score_criteria}")
        self.logger.info(f"start_projection_step: {start_projection_step}")
        self.logger.info(f"projection_steps: {projection_steps}")
        self.logger.info(f"projection_periods: {projection_periods}")
        self.logger.info(f"start_pruning_step: {start_pruning_step}")
        self.logger.info(f"pruning_steps: {pruning_steps}")
        self.logger.info(f"pruning_periods: {self.pruning_periods}")
        self.logger.info(f"pruning_period_duration: {self.pruning_period_duration}")

        super(GETA, self).__init__(
            params=params,
            variant=variant,
            lr=lr,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            target_group_sparsity=target_group_sparsity,
            group_divisible=group_divisible,
        )

        for param_group in self.param_groups:
            param_group["important_idxes"] = [
                i for i in range(param_group["num_groups"])
            ]
            param_group["active_redundant_idxes"] = list()
            param_group["pruned_idxes"] = list()
            param_group["importance_scores"] = dict()
            param_group["lr_quant"] = lr_quant

        self.active_num_redundant_groups = list()
        # Set up active number redundant groups for each pruning period
        groups_sum = 0
        for p in range(self.pruning_periods):
            if p == self.pruning_periods - 1:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups - groups_sum
                )
            else:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups // self.pruning_periods
                )
                groups_sum += self.active_num_redundant_groups[p]
        self.logger.info(
            f"Target redundant groups per period: {self.active_num_redundant_groups}"
        )

        # --- [DACO - RCAJS] Lagrangian KL-divergence 控制器初始化 ---
        self.rcajs = RCAJSController(
            total_groups=self.total_num_groups,
            target_sparsity=target_group_sparsity,
            beta_p=0.1,
            lambda_stability=0.1,
            phi_b_threshold=0.8,
            verbose=(verbose == "True"),
            device=device,
        )
        self.logger.info(
            f"[RCAJS] 控制器初始化: total_groups={self.total_num_groups}, "
            f"target_sparsity={target_group_sparsity}, beta_p=0.1"
        )

    @contextmanager
    def safe_open_file(self, filename, mode="a"):
        file = None
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file = open(filename, mode)
            yield file
        except IOError as e:
            self.logger.error(f"Error opening file {filename}: {e}")
        finally:
            if file is not None:
                file.close()

    def _get_diffusion_decay_factor(self):
        """计算扩散噪声的衰减系数 (Cosine Schedule: 1.0 -> 0.0)"""
        # 设定扩散开始时间 (通常与投影或剪枝开始时间对齐)
        start_step = self.start_pruning_step
        current_t = self.num_steps - start_step

        # 还没开始或已经结束 -> 不加噪
        if current_t < 0 or current_t >= self.diffusion_steps:
            return 0.0

        # 余弦退火计算
        decay = 0.5 * (1 + math.cos(math.pi * current_t / self.diffusion_steps))
        return decay

    def grad_clipping(self):
        grad_clip_min = self.grad_clip_min
        grad_clip_max = self.grad_clip_max
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = p.grad.clamp(min=grad_clip_min, max=grad_clip_max)

    def identify_redundant_groups(self):
        global_scores = torch.cat(self.global_scores, dim=0)
        curr_active_num_redundant_groups = self.active_num_redundant_groups[
            self.curr_pruning_period
        ]

        # --- [DACO - RCAJS] 使用 Lagrangian 自适应剪枝率替换固定预算 ---
        rcajs_state = self.rcajs.step(
            num_pruned_groups=len(self.pruned_group_idxes),
            score_dist=global_scores,
        )
        # 用 RCAJS 剪枝率计算本次 period 应剪枝的组数
        # rate * total_groups 得到本次总共要剪的，再减去已剪的
        rcajs_adaptive_K = int(math.ceil(rcajs_state["prune_rate"] * self.total_num_groups))
        # 与原固定预算取较小值（保守策略），避免超过 period 预算
        curr_active_num_adaptive = min(rcajs_adaptive_K, curr_active_num_redundant_groups)

        if rcajs_state["early_stop"]:
            curr_active_num_adaptive = 0
            self.logger.info(
                f"[RCAJS] Period {self.curr_pruning_period} 提前停止: "
                f"sparsity={rcajs_state['current_sparsity']:.3f}, "
                f"Δ_p={rcajs_state['delta_p']:.4f}, "
                f"Ψ={rcajs_state['stability_term']:.3f}, "
                f"KL={rcajs_state['kl_div']:.4f}"
            )

        # --- 日志输出 RCAJS 状态（每 5 个 period 或 verbose 模式） ---
        if (self.verbose == "True" or self.curr_pruning_period % 5 == 0):
            self.logger.info(
                f"[RCAJS] Period {self.curr_pruning_period}: "
                f"rate={rcajs_state['prune_rate']:.4f} "
                f"(fixed={curr_active_num_redundant_groups}, adaptive={curr_active_num_adaptive}), "
                f"sparsity={rcajs_state['current_sparsity']:.3f}/target={self.rcajs.target_sparsity:.3f}, "
                f"Δ_p={rcajs_state['delta_p']:+.4f}, "
                f"Ψ={rcajs_state['stability_term']:.3f}"
            )

        # --- 确定 top-K 冗余组 ---
        # 【Bug Fix】torch.topk K 越界时直接崩溃，加安全边界
        curr_K = len(self.pruned_group_idxes) + curr_active_num_adaptive
        actual_K = min(curr_K, len(global_scores))
        _, top_indices = torch.topk(-global_scores, actual_K)
        top_indices = top_indices.cpu().numpy()
        top_indices = np.setdiff1d(top_indices, self.pruned_group_idxes)[
            :curr_active_num_adaptive
        ].tolist()
        self.pruned_group_idxes.extend(top_indices)

        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                global_active_redundant_idx = np.intersect1d(
                    top_indices, group["global_idxes"]
                )
                group["active_redundant_idxes"] = (
                    global_active_redundant_idx - group["global_start_idx"]
                ).tolist()
                # Refine important_idx by group_divisible
                if group["num_groups"] < self.group_divisible:
                    group["active_redundant_idxes"].clear()
                    group["pruned_idxes"].clear()
                else:
                    curr_num_important_groups = len(group["important_idxes"])
                    trial_num_important_groups = curr_num_important_groups - len(
                        group["active_redundant_idxes"]
                    )
                    if (
                        trial_num_important_groups % self.group_divisible != 0
                        or trial_num_important_groups <= 0
                    ):
                        ratio = (
                            trial_num_important_groups // self.group_divisible + 1
                        )  # Add one will preserve more groups, otherwise will slim more.
                        refined_num_important_groups = None
                        if ratio <= 1 or trial_num_important_groups == 0:
                            refined_num_important_groups = max(
                                int(self.group_divisible), 1
                            )
                        else:
                            refined_num_important_groups = max(
                                int(ratio * self.group_divisible),
                                int(self.group_divisible),
                            )
                        refined_num_important_groups = min(
                            group["num_groups"], refined_num_important_groups
                        )
                        refined_num_active_redundant_groups = (
                            group["num_groups"]
                            - len(group["pruned_idxes"])
                            - refined_num_important_groups
                        )
                        self.target_num_redundant_groups += (
                            refined_num_active_redundant_groups
                            - len(group["active_redundant_idxes"])
                        )
                        group["active_redundant_idxes"] = group[
                            "active_redundant_idxes"
                        ][:refined_num_active_redundant_groups]
                group["important_idxes"] = [
                    i
                    for i in group["important_idxes"]
                    if (
                        i not in group["active_redundant_idxes"]
                        and i not in group["pruned_idxes"]
                    )
                ]

    def commit_redundant_idxes(self):
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                group["pruned_idxes"].extend(group["active_redundant_idxes"].copy())
                group["active_redundant_idxes"].clear()
                group["important_idxes"] = [
                    i
                    for i in range(group["num_groups"])
                    if i not in group["pruned_idxes"]
                ]
                group["importance_scores"].clear()

    def quantize_weight(self, param_group, target_name):
        is_quantize = False  # Check if the "target_name" layer involves quantization
        t_quant = None
        for p_name in param_group["p_names"]:
            if "d_quant_wt" in p_name:
                layer_name = ".".join(p_name.split(".")[:-1])
                if layer_name in target_name and "weight" in target_name:
                    is_quantize = True
                    quantize_layer_name = layer_name
                    break

        if not is_quantize:
            return is_quantize, None
        else:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if quantize_layer_name in p_name and "d_quant_wt" in p_name:
                    d_quant = p.data
                if quantize_layer_name in p_name and "t_quant_wt" in p_name:
                    t_quant = p.data
                if quantize_layer_name in p_name and "q_m_wt" in p_name:
                    q_m = p.data
                if quantize_layer_name in p_name and "weight" in p_name:
                    weight = p.data

            with torch.no_grad():
                quantized_weight = self._quantize_helper(
                    weight, d_quant, q_m, t_quant=t_quant
                )

            return is_quantize, quantized_weight

    def compute_gamma_d(self, param_group, active_redundant_idxes, bit_range):
        t_quant = None
        qm_list = []
        layer_name_list = []  # Store layers with quantization mapping
        prune_param_clip_list = []  # Store prunable parameter clipping values
        prune_param_grad_list = []  # Store prunable parameter gradients
        prune_param_res_list = []  # Store prunable parameter residual values
        prune_param_clip_redundant_list = []
        prune_param_res_redundant_list = []
        prune_param_grad_redundant_list = []

        ###########################################
        ####  Layers with quantization mapping ####
        ###########################################
        for p_name in param_group["p_names"]:
            if "d_quant_wt" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "d_quant_wt" in p_name:
                        d_quant = p.data
                    if "t_quant_wt" in p_name:
                        t_quant = p.data
                    if "q_m_wt" in p_name:
                        q_m = p.data
                        qm_list.append(q_m.item())
                    if "weight" in p_name:
                        weight = p.data
            for p_name, p, p_transform in zip(
                param_group["p_names"],
                param_group["params"],
                param_group["p_transform"],
            ):
                if layer_name in p_name and "weight" in p_name:
                    clipped_weight = self._clip_helper(weight, q_m, t_quant=t_quant)
                    prune_param_clip_list.append(clipped_weight.data)
                    residual_weight = self._residual_helper(
                        weight, d_quant, q_m, t_quant=t_quant
                    )
                    prune_param_res_list.append(residual_weight)
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])
                elif (
                    layer_name in p_name and p_transform != 1
                ):  # bias in quantization mapping
                    prune_param_clip_list.append(p.data)
                    prune_param_res_list.append(
                        torch.tensor([0.0]).to(p.device).expand_as(p.data)
                    )
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])

        ###########################################
        ### Layers without quantization mapping ###
        ###########################################
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if not any(layer_name in p_name for layer_name in layer_name_list):
                prune_param_clip_list.append(p.data)
                prune_param_res_list.append(
                    torch.tensor([0.0]).to(p.device).expand_as(p.data)
                )
                prune_param_grad_list.append(param_group["grad_variant"][p_name])

        # Access values at redundant indices
        for i in range(len(prune_param_clip_list)):
            prune_param_clip_redundant_list.append(
                prune_param_clip_list[i][active_redundant_idxes]
            )
            prune_param_res_redundant_list.append(
                prune_param_res_list[i][active_redundant_idxes]
            )
            prune_param_grad_redundant_list.append(
                prune_param_grad_list[i][active_redundant_idxes]
            )

        # Get flattened value with its norm
        flatten_clip = torch.cat(
            [tensor.flatten() for tensor in prune_param_clip_redundant_list]
        )
        flatten_grad = torch.cat(
            [tensor.flatten() for tensor in prune_param_grad_redundant_list]
        )
        flatten_res = torch.cat(
            [tensor.flatten() for tensor in prune_param_res_redundant_list]
        )
        flatten_clip_norm = torch.norm(flatten_clip, p=2)
        flatten_grad_norm = torch.norm(flatten_grad, p=2)
        flatten_res_norm = torch.norm(flatten_res, p=2)

        eps = 1e-8
        cosine_similarity_clip = torch.div(
            torch.dot(flatten_clip, flatten_grad),
            torch.max(flatten_clip_norm, torch.tensor(eps).to(flatten_clip.device))
            * flatten_grad_norm,
        )
        cosine_similarity_res = torch.div(
            torch.dot(flatten_res, flatten_grad),
            torch.max(flatten_res_norm, torch.tensor(eps).to(flatten_res.device))
            * flatten_grad_norm,
        )

        eta = 0.999
        zeta = 0.9
        forget_rate = 1.0
        if torch.mean(flatten_clip).item() < 1e-8:
            forget_rate = 0.0
        else:
            if torch.isinf(cosine_similarity_clip) or torch.isnan(
                cosine_similarity_clip
            ):
                self.logger.warning(
                    "cosine_similarity_clip is inf or nan, setting forget rate to 0.0"
                )
                forget_rate = 0.0
            elif cosine_similarity_clip >= 0.0 and cosine_similarity_clip <= 1.0:
                t = (
                    self.num_steps - self.start_pruning_step
                ) % self.pruning_period_duration
                forget_rate = 1.0 - (self.pruning_period_duration - t - 1.0) / (
                    self.pruning_period_duration - t
                )
            elif cosine_similarity_clip >= -1.0 and cosine_similarity_clip < 0.0:
                forget_rate = (
                    -(1 - eta)
                    * param_group["lr"]
                    * flatten_grad_norm
                    / (cosine_similarity_clip * flatten_clip_norm)
                )
            else:
                # TODO: @xiaoyi, refactor
                if self.verbose == "True":
                    self.logger.warning(
                        f"Unexpected cosine_similarity_clip value: {cosine_similarity_clip}"
                    )
                    outID = "log_info_" + str(self.start_pruning_step)
                    filename = os.path.join("outputs",f"sparsity_{self.target_group_sparsity*100}", f"{outID}.txt")
                    with self.safe_open_file(filename) as logfile:
                        logfile.write("Throw an error: cosine_similarity_clip error\n")
                        logfile.write(
                            "similarity_value: {cos_similar:^8.9e}\n".format(
                                cos_similar=cosine_similarity_clip.item()
                            )
                        )
                        logfile.write(
                            "flatten_grad_max: {flatten_grad:^8.9e}\n".format(
                                flatten_grad=torch.max(flatten_grad).item()
                            )
                        )
                        logfile.write(
                            "flatten_clip_max: {flatten_clip:^8.9e}\n".format(
                                flatten_clip=torch.max(flatten_clip).item()
                            )
                        )
                        logfile.write(
                            "flatten_grad_mean: {flatten_grad:^8.9e}\n".format(
                                flatten_grad=torch.mean(flatten_grad).item()
                            )
                        )
                        logfile.write(
                            "flatten_clip_mean: {flatten_clip:^8.9e}\n".format(
                                flatten_clip=torch.mean(flatten_clip).item()
                            )
                        )
                        # logfile.write("all_grad_max: {flatten_grad:^8.9e}\n".format(flatten_grad=torch.max(torch.Tensor(prune_param_grad_list))) )
                    self.logger.error("Error with computing cosine_similarity_clip!")
                    assert 1 == 2

        # Determine d_quant range
        bit_width_lower = bit_range[0]
        bit_width_upper = bit_range[1]
        d_quant_upper = self._d_quant_helper(
            bit_width_lower, max(np.abs(qm_list)), t_quant
        )
        d_quant_lower = self._d_quant_helper(
            bit_width_upper, max(np.abs(qm_list)), t_quant
        )

        # Safeguard mechanism for d_quant
        if cosine_similarity_res >= 0.0 or forget_rate == 0.0:
            d_quant = d_quant_upper
        else:
            d_quant = (
                -zeta
                * eta
                * param_group["lr"]
                * flatten_grad_norm
                / (forget_rate * cosine_similarity_res * flatten_res_norm)
            )
            while d_quant < d_quant_lower:  # Avoid quant step size d being too small.
                forget_rate = forget_rate * 0.8
                d_quant = d_quant / 0.8
            d_quant = min(
                d_quant_upper, d_quant
            )  # Avoid quant step size d being too large.

        return forget_rate, d_quant

    def get_bitwidth_dict(self, param_group):
        d_quant_wt = None
        q_m_wt = None
        t_quant_wt = None
        d_quant_act = None
        q_m_act = None
        t_quant_act = None
        layer_name_list = []
        bit_dict = {}
        for p_name in param_group["p_names"]:
            if "d_quant" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data
                    if "d_quant_wt" in p_name:
                        d_quant_wt = p.data
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data
                    if "q_m_act" in p_name:
                        q_m_act = p.data
                    if "d_quant_act" in p_name:
                        d_quant_act = p.data

            bit_dict[layer_name] = {}
            bit_width_wt = self._bit_width_helper(
                d_quant=d_quant_wt, q_m=q_m_wt, t_quant=t_quant_wt
            )
            bit_width_act = self._bit_width_helper(
                d_quant=d_quant_act, q_m=q_m_act, t_quant=t_quant_act
            )

            if bit_width_wt is not None:
                bit_dict[layer_name]["weight"] = round(bit_width_wt)

            if bit_width_act is not None:
                bit_dict[layer_name]["activation"] = round(bit_width_act)

        return bit_dict

    def gradient_descent_step(self, param_group):
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

    def partial_projected_gradient_descent_step_range_wt(self, param_group):
        """Apply projected gradient descent for weight quantization parameters."""
        # First part remains the same - gradient descent step
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if (
                    "d_quant_wt" in p_name
                    or "t_quant_wt" in p_name
                    or "q_m_wt" in p_name
                ):
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            # Assign different learning rate to weights and quantization params
            if "d_quant_wt" in p_name or "t_quant_wt" in p_name or "q_m_wt" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

        # Find layers in each param_group
        layer_name_list = []
        for p_name in param_group["p_names"]:
            if "d_quant_wt" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # Projection
        for layer_name in layer_name_list:
            t_quant_wt = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data

            # Calculate bounds for this layer
            d_quant_min = self._d_quant_helper(self.max_bit_wt, q_m_wt, t_quant_wt)
            d_quant_max = self._d_quant_helper(self.min_bit_wt, q_m_wt, t_quant_wt)

            # Convert bounds to scalars if they're tensors
            if isinstance(d_quant_min, torch.Tensor):
                d_quant_min = float(d_quant_min.item())
            if isinstance(d_quant_max, torch.Tensor):
                d_quant_max = float(d_quant_max.item())

            # Apply bounds to d_quant parameters
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name and "d_quant_wt" in p_name:
                    # Clip using scalar bounds
                    p.data.clamp_(min=d_quant_min, max=d_quant_max)

    def partial_projected_gradient_descent_step_range_act(self, param_group):
        # Add them as hyperparameter in hesso_quant optimizer in the future
        min_bit = self.min_bit_act
        max_bit = self.max_bit_act

        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if (
                    "d_quant_act" in p_name
                    or "t_quant_act" in p_name
                    or "q_m_act" in p_name
                ):
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )

            # Assign learning rate to activation quantization params
            if (
                "d_quant_act" in p_name
                or "t_quant_act" in p_name
                or "q_m_act" in p_name
            ):
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )

        # Find layers in each param_group
        layer_name_list = []
        for p_name in param_group["p_names"]:
            if "d_quant_act" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # Projection
        for layer_name in layer_name_list:
            t_quant_act = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data
                    if "q_m_act" in p_name:
                        q_m_act = p.data
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name and "d_quant_act" in p_name:
                    d_quant_min = self._d_quant_helper(max_bit, q_m_act, t_quant_act)
                    d_quant_max = self._d_quant_helper(min_bit, q_m_act, t_quant_act)
                    p.data = torch.clip(p.data, min=d_quant_min, max=d_quant_max)

    def partial_projected_gradient_descent_step_fix(self, param_group, bit_dict):
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            # Assign different learning rate to weights and quantization params
            if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

        for layer_name in bit_dict.keys():
            t_quant_wt = None
            t_quant_act = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data
                    if "q_m_act" in p_name:
                        q_m_act = p.data
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name and "d_quant_wt" in p_name:
                    bit_width = bit_dict[layer_name]["weight"]
                    d_quant_wt = self._d_quant_helper(bit_width, q_m_wt, t_quant_wt)
                    p.data = torch.clip(p.data, min=d_quant_wt, max=d_quant_wt)
                if layer_name in p_name and "d_quant_act" in p_name:
                    bit_width = bit_dict[layer_name]["activation"]
                    d_quant_act = self._d_quant_helper(bit_width, q_m_act, t_quant_act)
                    p.data = torch.clip(p.data, min=d_quant_act, max=d_quant_act)

    @staticmethod
    def _bit_width_helper(d_quant=None, q_m=None, t_quant=None):
        if d_quant is None:
            return None

        if t_quant is None:
            t_quant = 1.0
        # Clamp exponent to avoid overflow
        _exp = t_quant * math.log(max(abs(q_m), 1e-10))
        _exp = max(min(_exp, 50.0), -50.0)
        bit_width = (
            math.log2(math.exp(_exp) / max(abs(d_quant), 1e-10) + 1) + 1
        )

        return bit_width

    @staticmethod
    def _d_quant_helper(bit_width, q_m, t_quant):
        """Calculate d_quant, using max absolute value of q_m for uniform quantization."""
        if t_quant is None:
            t_quant = 1.0

        # Get maximum absolute value if q_m is a tensor
        if isinstance(q_m, torch.Tensor):
            q_m = torch.max(torch.abs(q_m)).item()
        else:
            q_m = abs(q_m)
        # Prevent exact zero
        q_m = max(abs(q_m), 1e-10)

        # Calculate d_quant using scalar math
        # Clamp exponent to avoid math.exp overflow
        exponent = t_quant * math.log(abs(q_m))
        exponent = max(min(exponent, 50.0), -50.0)
        # 防御: bit_width==1 时 (2**0 - 1)=0 会触发 ZeroDivisionError，
        #       钳制分母使用的有效位宽下限为 2 (与 min_bit_wt 一致)。
        eff_bw = max(bit_width, 2)
        d_quant = math.exp(exponent) / (2 ** (eff_bw - 1) - 1)
        return d_quant

    @staticmethod
    def _quantize_helper(weight, d_quant, q_m, t_quant):
        if t_quant is None:
            t_quant = 1.0
        weight_abs = torch.abs(weight)
        q_s = 0.0
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))
        input_pow = torch.exp(t_quant * torch.log(weight_abs - q_s))  # weight_abs > q_s
        output = d_quant * torch.round(input_pow.div(d_quant))
        output[weight_abs <= q_s] = 0
        output[weight_abs >= q_m] = d_quant * torch.round(range_pow.div(d_quant))
        output = torch.sign(weight) * output

        return output

    @staticmethod
    def _clip_helper(weight, q_m, t_quant):
        if t_quant is None:
            t_quant = 1.0
        weight_abs = torch.abs(weight)
        q_s = 0.0
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))
        output = torch.exp(t_quant * torch.log(weight_abs - q_s))  # weight_abs > q_s
        output[weight_abs <= q_s] = 0
        output[weight_abs >= q_m] = range_pow
        output = torch.sign(weight) * output

        return output

    @staticmethod
    def _residual_helper(weight, d_quant, q_m, t_quant):
        if t_quant is None:
            t_quant = 1.0
        weight_abs = torch.abs(weight)
        q_s = 0.0
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))
        input_pow = torch.exp(t_quant * torch.log(weight_abs - q_s))  # weight_abs > q_s
        output = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
        output[weight_abs <= q_s] = 0
        output[weight_abs >= q_m] = torch.round(range_pow.div(d_quant)) - range_pow.div(
            d_quant
        )
        output = torch.sign(weight) * output

        return output

    def log_qm_projection(self):
        """Log q_m during projection"""
        if (self.num_steps >= self.start_projection_step and 
            self.num_steps <= self.start_projection_step + self.projection_steps and
            self.num_steps % 1000 == 0):
            
            log_file = os.path.join(self.log_dir, f"projection_qm_{self.num_steps}.txt")
            with self.safe_open_file(log_file, "w") as f:
                curr_period = (self.num_steps - self.start_projection_step) // self.projection_period_duration
                f.write(f"Step: {self.num_steps}, Projection Period: {curr_period}\n")
                f.write(f"Current max_bit_wt: {self.max_bit_wt}\n\n")
                
                for group in self.param_groups:
                    for p_name, p in zip(group["p_names"], group["params"]):
                        if "q_m_wt" in p_name:
                            layer_name = ".".join(p_name.split(".")[:-1])
                            f.write(f"Layer: {layer_name}\n")
                            f.write(f"q_m stats: min={p.data.min().item():.6f}, ")
                            f.write(f"max={p.data.max().item():.6f}, ")
                            f.write(f"mean={p.data.mean().item():.6f}\n\n")

    def step(self, loss=None, closure=None):
        """
        Core function.
        """

        if closure is not None:
            _ = closure()

        self.num_steps += 1
        self.compute_grad_variant()

        # Determine the bit range projection for weights
        if (
            self.num_steps >= self.start_projection_step
            and self.num_steps <= self.start_pruning_step
            and self.start_projection_step != self.start_pruning_step
        ):
            if (
                self.num_steps - self.start_projection_step - 1
            ) % self.projection_period_duration == 0 and (
                self.num_steps - self.start_projection_step - 1
            ) != 0:
                # --- [DACO - RCAJS] 位宽自适应触发判断 ---
                # 当 φ_b > φ_b_threshold 时，触发一次额外 bit reduction
                bit_adaptive_triggered = self.rcajs.update_bit_adaptive(
                    self.bit_layers, max_bit=int(self.max_bit_wt)
                )
                if bit_adaptive_triggered:
                    self.max_bit_wt = max(self.max_bit_wt - self.bit_reduction, self.min_bit_wt)
                    self.logger.info(
                        f"[RCAJS] 位宽自适应触发: max_bit_wt -> {self.max_bit_wt}"
                    )
                else:
                    self.max_bit_wt = self.max_bit_wt - self.bit_reduction
                self.min_bit_wt = self.min_bit_wt

        # Partition groups into important and redundant groups
        if (
            self.num_steps >= self.start_pruning_step
            and self.curr_pruning_period < self.pruning_periods
            and self.pruning_period_duration != 0
        ):
            if (
                self.num_steps - self.start_pruning_step - 1
            ) % self.pruning_period_duration == 0:
                self.logger.info(
                    f"Determining important and redundant groups using saliency scores. Step={self.num_steps}"
                )
                self.commit_redundant_idxes()
                self.compute_importance_scores()
                self.identify_redundant_groups()
                self.curr_pruning_period += 1

        # Second pass to update variables
        if self.pruning_period_duration != 0:
            t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
        for group in self.param_groups:
            if not group["is_prunable"] or len(group["active_redundant_idxes"]) == 0:
                if self.num_steps <= self.start_projection_step:  # First stage
                    # self.logger.info(
                    #     f"Warmup stage: updating trainable parameters and quantization parameters using SGD. Step={self.num_steps}"
                    # )
                    self.gradient_descent_step(group)
                elif self.num_steps > self.start_pruning_step + self.pruning_steps:
                    if (
                        self.num_steps
                        == self.start_pruning_step + self.pruning_steps + 1
                    ):
                        bit_layer = self.get_bitwidth_dict(group)
                        self.bit_layers.update(bit_layer)
                    self.partial_projected_gradient_descent_step_fix(
                        group, self.bit_layers
                    )
                else:
                    self.partial_projected_gradient_descent_step_range_wt(group)
                    self.partial_projected_gradient_descent_step_range_act(group) # Uncomment this line if apply activation quantization
            elif (
                group["is_prunable"] and len(group["active_redundant_idxes"]) > 0
            ):  # Third stage
                # self.partial_projected_gradient_descent_step_range_act(group)
                # self.logger.info(
                #     f"Joint pruning and quantization stage. Step={self.num_steps}"
                # )
                decay_factor = self._get_diffusion_decay_factor()
                if decay_factor > 1e-6:
                    for p_name, p in zip(group["p_names"], group["params"]):
                        if p.grad is None: continue

                        # A. 区分学习率: 量化参数用 lr_quant, 权重用 lr
                        if "t_quant" in p_name or "q_m_wt" in p_name:
                            current_lr = group["lr_quant"]

                            noise_scale = current_lr * self.diffusion_noise_ratio * decay_factor * 0.1
                        elif "d_quant" in p_name:
                            # d_quant 是后面解析解算出来的，加噪会被覆盖，所以跳过
                            continue
                        else:
                            # 普通权重
                            current_lr = group["lr"]
                            noise_scale = current_lr * self.diffusion_noise_ratio * decay_factor

                        # B. 注入噪声
                        if noise_scale > 1e-9:
                            noise = torch.randn_like(p.data) * noise_scale
                            p.data.add_(noise)

                # Add stochastic gradient term for quantization params (d_quant excluded)
                for p_name, p, p_transform in zip(
                    group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue
                    if "t_quant_wt" in p_name or "q_m_wt" in p_name:
                        p.data.add_(
                            group["grad_variant"][p_name], alpha=-group["lr_quant"]
                        )

                # Identify redundant idxes
                active_redundant_idxes = group["active_redundant_idxes"]

                # Compute forget rate (gamma) and quant step size (d)
                gamma, d_quant = self.compute_gamma_d(
                    group, active_redundant_idxes, [self.min_bit_wt, self.max_bit_wt]
                )
                # self.logger.info(
                #     f"Forget rate (gamma): {gamma}, Quant step size (d): {d_quant}"
                # )
                self.gamma, self.d_quant = gamma, d_quant

                # Update quant step size d
                for i, (p_name, p_transform) in enumerate(
                    zip(group["p_names"], group["p_transform"])
                ):
                    if "d_quant_wt" in p_name:
                        with torch.no_grad():
                            group["params"][i].copy_(d_quant)

                for p_name, p, p_transform in zip(
                    group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue

                    # Add redundant info removal term
                    is_quantize, quantize_weight = self.quantize_weight(group, p_name)
                    if p_transform != TensorTransform.NO_PRUNE:
                        if is_quantize:
                            p.data[active_redundant_idxes] = (
                                p.data[active_redundant_idxes]
                                - gamma * quantize_weight.data[active_redundant_idxes]
                            )
                        else:
                            p.data[active_redundant_idxes] = (
                                p.data[active_redundant_idxes]
                                - gamma * p.data[active_redundant_idxes]
                            )

                    # --- [DACO - DGD] 重要组量化蒸馏引导 (论文 Eq.12) ---
                    # 论文公式: 对重要组添加 γ₁ * Φ_res 量化蒸馏损失
                    # 其中 Φ_res 是量化残差 (weight - quantized_weight)
                    # 此处实现为: 对重要组施加量化引导，使其趋向量化后的表示
                    important_idxes = group["important_idxes"]
                    if p_transform != TensorTransform.NO_PRUNE and len(important_idxes) > 0:
                        if is_quantize:
                            # 重要组量化引导系数 γ₁ = γ * 0.5（冗余组的一半）
                            gamma_guided = gamma * 0.5
                            # 量化残差: 原始 - 量化
                            quant_residual = (
                                p.data[important_idxes]
                                - quantize_weight.data[important_idxes]
                            )
                            # 安全边际 c: 残差过大时衰减
                            residual_norm = torch.norm(quant_residual, p=2)
                            safety_margin = min(1.0, 1.0 / (residual_norm.item() + 1e-6))
                            p.data[important_idxes] -= (
                                gamma_guided * safety_margin * quant_residual
                            )

                    # Add stochastic gradient term for non-quantization parameters
                    if (
                        "d_quant" not in p_name
                        and "t_quant" not in p_name
                        and "q_m" not in p_name
                    ):
                        p.data.add_(group["grad_variant"][p_name], alpha=-group["lr"])

                    # Tackle auxiliary params
                    for ng_id, offset in group["auxiliary_ngs"]:
                        active_redundant_aux_idxes = [
                            i + offset for i in active_redundant_idxes
                        ]
                        for aux_p in self.auxiliary_param_groups[ng_id]["params"]:
                            if aux_p.grad is None:
                                continue
                            aux_p.data[active_redundant_aux_idxes, ...] *= (
                                self.pruning_period_duration - t - 1.0
                            ) / (self.pruning_period_duration - t)

            self.fix_pruned_groups_as_zeros(group)

        if self.pruning_period_duration != 0:
            if self.num_steps >= self.start_pruning_step and t == self.pruning_period_duration - 1:
                self.commit_redundant_idxes()

    def compute_metrics(self):
        """Compute optimizer metrics, skipping quantization parameters."""
        self.opt_metrics.norm_params = 0.0
        self.opt_metrics.norm_important_groups = 0.0
        self.opt_metrics.norm_redundant_groups = 0.0
        self.opt_metrics.num_zero_groups = 0
        self.opt_metrics.num_important_groups = 0
        self.opt_metrics.num_redundant_groups = 0

        for group in self.param_groups:
            if not (group["is_prunable"] and not group["is_auxiliary"]):
                continue

            norm_group = None
            import_idxes = group["important_idxes"]
            redund_idxes = group["active_redundant_idxes"] + group["pruned_idxes"]

            for param, p_transform in zip(group["params"], group["p_transform"]):
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                param_transform = tensor_transformation_param_group(param.data, p_transform, group)
                if norm_group == None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2

            if norm_group is not None:  # Only process if we have valid parameters
                norm_group = torch.sqrt(norm_group)
                self.opt_metrics.num_zero_groups += torch.sum(norm_group == 0).item()
                self.opt_metrics.norm_params += torch.sum(norm_group).item()
                self.opt_metrics.norm_important_groups += torch.sum(
                    norm_group[import_idxes]
                ).item()
                self.opt_metrics.norm_redundant_groups += torch.sum(
                    norm_group[redund_idxes]
                ).item()
                self.opt_metrics.num_important_groups += len(import_idxes)
                self.opt_metrics.num_redundant_groups += len(redund_idxes)

        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / float(
            self.total_num_groups + self.safe_guard
        )

        return self.opt_metrics

    def state_dict(self, debug=False):
        """
        Return a state_dict of the optimizer for restore.
        """
        parent_state = super().state_dict() # includes param_gropups, param_data, and optimizer-specific state
        self.logger.debug(f"Parent state_dict keys: {parent_state.keys()}")
        # Add GETA quantization state
        state_dict = {"param_groups": self.param_groups}
        state_dict.update(parent_state)
        state_dict.update({
            'num_steps': self.num_steps,
            'curr_pruning_period': self.curr_pruning_period, 
            'start_pruning_step': self.start_pruning_step,
            'pruning_periods': self.pruning_periods,
            'pruning_steps': self.pruning_steps,
            'start_projection_step': self.start_projection_step,
            'projection_periods': self.projection_periods,
            'projection_steps': self.projection_steps,
            'pruning_period_duration': self.pruning_period_duration,
            'bit_layers': self.bit_layers,
            'projection_period_duration': self.projection_period_duration,
            'min_bit_wt': self.min_bit_wt,
            'max_bit_wt': self.max_bit_wt,
            'min_bit_act': self.min_bit_act,
            'max_bit_act': self.max_bit_act,
            'bit_reduction': self.bit_reduction,
            'pruned_group_indices': self.pruned_group_idxes,
        })
        self.logger.debug(f"Final state_dict keys: {state_dict.keys()}")
        return state_dict

    # def load_state_dict(self, state_dict):
    #     """Loads the optimizer state.

    #     Args:
    #         state_dict (dict): Optimizer state dictionary containing:
    #             - param_groups: List of parameter group dictionaries
    #             - parameter data and gradients
    #             - optimizer-specific state

    #     Raises:
    #         ValueError: If state_dict missing required data, contains invalid structure,
    #             or parameters don't match current model

    #     Note:
    #         Each parameter group must contain a 'param_data' field with parameter
    #         name, data and optional gradient information. Parameter shapes must match
    #         between saved state and current model.
    #     """
    #     if "param_groups" not in state_dict:
    #         raise ValueError("Missing param_groups in state_dict")
    #     saved_groups = state_dict["param_groups"]

    #     if "num_steps" not in state_dict:
    #         raise ValueError("Missing num_steps in state_dict")
    #     self.num_steps = state_dict["num_steps"]

    #     # First, collect all parameters from saved state
    #     saved_params = {}
    #     for i, group in enumerate(saved_groups):
    #         if "param_data" not in group:
    #             raise ValueError(
    #                 f"Parameter group {i} is missing required 'param_data' field"
    #             )
    #         for param_info in group["param_data"]:
    #             if not isinstance(param_info, dict) or "name" not in param_info:
    #                 raise ValueError(
    #                     "Invalid parameter info structure in saved state, check checkpoint format"
    #                 )
    #             name = param_info["name"]
    #             if name in saved_params:
    #                 raise ValueError(
    #                     f"Duplicate parameter name '{name}' in saved state"
    #                 )
    #             saved_params[name] = param_info

    #     # Update current groups
    #     for current_group, saved_group in zip(self.param_groups, saved_groups):
    #         # Update non-parameter attributes
    #         for key in saved_group:
    #             if key not in ["params", "param_data"]:
    #                 current_group[key] = saved_group[key]

    #         # Handle parameters
    #         for current_p_name, current_param in zip(
    #             current_group["p_names"], current_group["params"]
    #         ):
    #             if current_p_name not in saved_params:
    #                 raise ValueError(
    #                     f"Parameter '{current_p_name}' not found in saved state"
    #                 )

    #             saved_param_info = saved_params[current_p_name]
    #             if "data" not in saved_param_info:
    #                 raise ValueError(f"No data found for parameter '{current_p_name}'")

    #             saved_data = saved_param_info["data"]

    #             if current_param is None:
    #                 raise ValueError(f"Current parameter '{current_p_name}' is None")
    #             if saved_data is None:
    #                 raise ValueError(
    #                     f"Saved data for parameter '{current_p_name}' is None"
    #                 )

    #             # Verify shapes match
    #             if saved_data.shape != current_param.data.shape:
    #                 raise ValueError(
    #                     f"Shape mismatch for parameter '{current_p_name}': "
    #                     f"saved={saved_data.shape}, current={current_param.data.shape}"
    #                 )
    #             # Copy data
    #             current_param.data.copy_(saved_data.to(current_param.device))
    #             if "requires_grad" in saved_param_info:
    #                 current_param.requires_grad = saved_param_info["requires_grad"]

    #             # Copy gradient if it exists
    #             if "grad_variant" in saved_param_info:
    #                 grad_data = saved_param_info["grad_variant"]
    #                 if grad_data.shape != current_param.data.shape:
    #                     raise ValueError(
    #                         f"Gradient shape mismatch for '{current_p_name}': "
    #                         f"saved={grad_data.shape}, param={current_param.data.shape}"
    #                     )
    #                 if "grad_variant" not in current_group:
    #                     current_group["grad_variant"] = {}
    #                 grad_data = grad_data.to(current_param.device)
    #                 if current_p_name in current_group["grad_variant"]:
    #                     current_group["grad_variant"][current_p_name].copy_(grad_data)
    #                 else:
    #                     current_group["grad_variant"][current_p_name] = grad_data

    #     # Rebuild auxiliary param groups
    #     self.auxiliary_param_groups = {
    #         group["id"]: group
    #         for group in self.param_groups
    #         if group.get("is_auxiliary", False)
    #     }

    #     del state_dict
    def load_state_dict(self, state_dict):
        import copy

        for attr_name in state_dict:
            if attr_name != "param_groups":
                setattr(self, attr_name, state_dict[attr_name])
            else:
                prev_param_groups = state_dict[attr_name]
                for param_group in self.param_groups:
                    prev_param_group = next(
                        (_g for _g in prev_param_groups if param_group["id"] == _g["id"]),
                        None,
                    )
                    if prev_param_group is None:
                        raise Warning(f"Param group {param_group['id']} not found in previous state_dict.")
                        continue
                    for param_group_attr_name in prev_param_group:
                        if param_group_attr_name == "params":
                            for p_name, param, prev_p_name, prev_param in zip(
                                param_group["p_names"],
                                param_group["params"], 
                                prev_param_group["p_names"],
                                prev_param_group["params"],
                            ):
                                if p_name == prev_p_name:
                                    param.data.copy_(prev_param.data)
                                    if prev_param.grad is not None:
                                        param.grad.copy_(prev_param.grad)
                                else:
                                    print(f"\tParam {p_name} not found in previous state_dict.")
                        else:
                            param_group[param_group_attr_name] = copy.deepcopy(
                                prev_param_group[param_group_attr_name]
                            )

        self.auxiliary_param_groups = {
            group["id"]: group
            for group in self.param_groups
            if group.get("is_auxiliary", False)
        }

        del state_dict

    def create_checkpoint(self, model, epoch, loss):
        """Creates a standardized checkpoint dictionary.

        Returns:
            dict: A checkpoint containing model and optimizer state, plus metadata.
        """
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.state_dict(),  # Contains num_steps
            "epoch": epoch,
            "loss": loss,
        }
