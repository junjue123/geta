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
        os.makedirs(self.log_dir, exist_ok=True)  # 创建日志目录，不存在则创建
        self.logger = logging.getLogger(self.__class__.__name__)  # 初始化日志器
        self.start_projection_step = start_projection_step  # 开始投影的步骤
        self.projection_steps = projection_steps  # 投影总步骤
        self.projection_periods = projection_periods  # 投影周期数
        self.projection_period_duration = self.projection_steps // self.projection_periods  # 每个投影周期的步骤数
        self.start_pruning_step = start_pruning_step  # 开始剪枝的步骤
        self.pruning_periods = int(max(1, pruning_periods))  # 剪枝周期数(至少为1)
        self.pruning_steps = pruning_steps  # 剪枝总步骤
        self.pruning_period_duration = self.pruning_steps // self.pruning_periods  # 每个剪枝周期的步骤数
        self.curr_pruning_period = 0  # 当前剪枝周期(初始为0)
        self.lr_quant = lr_quant  # 量化学习率
        self.bit_reduction = bit_reduction  # 位减少量
        self.min_bit_wt = min_bit_wt  # 权重最小位数
        self.max_bit_wt = max_bit_wt  # 权重最大位数
        self.min_bit_act = min_bit_act  # 激活值最小位数
        self.max_bit_act = max_bit_act  # 激活值最大位数
        self.grad_clip_min = grad_clip_min  # 梯度裁剪最小值
        self.grad_clip_max = grad_clip_max  # 梯度裁剪最大值
        self.verbose = verbose  # 日志详细程度
        self.device = device  # 计算设备(cuda或cpu)
        self.pruned_group_idxes = list()  # 已剪枝组的索引列表
        self.gamma = 0.0  # 可能是某种系数
        self.d_quant = 0.0  # 可能是量化相关的参数
        self.bit_layers = {}  # 存储每个层的位数

        if importance_score_criteria == "default":
            # 默认使用多种评分标准及其权重
            self.importance_score_criteria = {
                "magnitude": 0.2,  # 权重大小
                "avg_magnitude": 0.2,  # 平均权重大小
                "cosine_similarity": 0.2,  # 余弦相似度
                "taylor_first_order": 0.2,  # 一阶泰勒展开
                "taylor_second_order": 0.2,  # 二阶泰勒展开
            }
        else:
            self.importance_score_criteria = importance_score_criteria  # 使用自定义标准
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

        # 遍历所有参数组（神经网络中通常按层或按参数类型分组）
        for param_group in self.param_groups:
            # 初始化重要组索引列表，初始时所有组都被视为重要
            # param_group["num_groups"]表示该参数组中的总组数
            param_group["important_idxes"] = [
                i for i in range(param_group["num_groups"])
            ]

            # 初始化活跃冗余组索引列表，用于记录当前周期中被标记为冗余但未剪枝的组
            param_group["active_redundant_idxes"] = list()

            # 初始化已剪枝组索引列表，记录已被剪枝的组
            param_group["pruned_idxes"] = list()

            # 初始化重要性评分字典，用于存储每组的重要性评分
            param_group["importance_scores"] = dict()

            # 为该参数组设置量化学习率
            param_group["lr_quant"] = lr_quant

        # 初始化每个剪枝周期的目标冗余组数量列表
        self.active_num_redundant_groups = list()

        # 计算并设置每个剪枝周期需要处理的冗余组数量
        groups_sum = 0  # 用于累计已分配的冗余组数量
        for p in range(self.pruning_periods):
            # 处理最后一个周期，确保总数量等于目标冗余组数量
            if p == self.pruning_periods - 1:
                # 最后一个周期分配剩余的所有冗余组
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups - groups_sum
                )
            else:
                # 前面的周期平均分配冗余组（使用整数除法）
                # 注意！！！这里看能否修改为自适应而不是平均分配
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups // self.pruning_periods
                )
                # 累计已分配的冗余组数量
                groups_sum += self.active_num_redundant_groups[p]

        # 记录每个周期的目标冗余组数量，便于调试和追踪
        self.logger.info(
            f"Target redundant groups per period: {self.active_num_redundant_groups}"
        )

    @contextmanager
    def safe_open_file(self, filename, mode="a"):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file = open(filename, mode)
            yield file
        except IOError as e:
            self.logger.error(f"Error opening file {filename}: {e}")
        finally:
            file.close()

    def grad_clipping(self):
        grad_clip_min = self.grad_clip_min
        grad_clip_max = self.grad_clip_max
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = p.grad.clamp(min=grad_clip_min, max=grad_clip_max)

    def identify_redundant_groups(self):
        """识别并标记当前剪枝周期中需要被剪枝的冗余组"""
        # 将全局评分拼接成一个张量（global_scores存储了各组的重要性评分）
        # 评分越低的组越可能被判定为冗余组
        global_scores = torch.cat(self.global_scores, dim=0)

        # 获取当前剪枝周期需要处理的冗余组数量，就是这里，可以靠外来传参修改！！！
        curr_active_num_redundant_groups = self.active_num_redundant_groups[
            self.curr_pruning_period
        ]

        # 计算当前需要考虑的总组数 = 已剪枝组数量 + 当前周期需剪枝的组数量
        curr_K = len(self.pruned_group_idxes) + curr_active_num_redundant_groups

        # 找到评分最低的curr_K个组（用负号实现topk取最小值）
        # top_indices为这些组的全局索引
        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy()  # 转换为numpy数组便于后续处理

        # 从候选组中排除已剪枝的组，并截取当前周期需要的数量
        # np.setdiff1d用于计算集合差，确保不重复剪枝
        top_indices = np.setdiff1d(top_indices, self.pruned_group_idxes)[
                      :curr_active_num_redundant_groups
                      ].tolist()

        # 将当前周期识别出的冗余组添加到已剪枝列表（准备后续剪枝操作）
        self.pruned_group_idxes.extend(top_indices)

        # 遍历所有参数组，更新各组的冗余组信息
        for group in self.param_groups:
            # 只处理可剪枝且非辅助的参数组
            if group["is_prunable"] and not group["is_auxiliary"]:
                # 找到当前组中属于全局冗余组的索引
                # np.intersect1d用于计算集合交集，找到属于当前组的冗余组
                global_active_redundant_idx = np.intersect1d(
                    top_indices, group["global_idxes"]
                )

                # 将全局索引转换为组内局部索引并存储
                # 减去group["global_start_idx"]得到组内相对位置
                group["active_redundant_idxes"] = (
                        global_active_redundant_idx - group["global_start_idx"]
                ).tolist()

                # 根据group_divisible（分组 divisor）调整重要组索引
                # 确保剪枝后剩余的组数量是group_divisible的倍数，避免硬件不兼容
                if group["num_groups"] < self.group_divisible:
                    # 如果组总数小于group_divisible，不进行剪枝
                    group["active_redundant_idxes"].clear()
                    group["pruned_idxes"].clear()
                else:
                    # 计算当前重要组数量
                    curr_num_important_groups = len(group["important_idxes"])
                    # 计算剪枝后可能的重要组数量
                    trial_num_important_groups = curr_num_important_groups - len(
                        group["active_redundant_idxes"]
                    )

                    # 检查剪枝后数量是否合法（是否为group_divisible的倍数且为正数）
                    if (
                            trial_num_important_groups % self.group_divisible != 0
                            or trial_num_important_groups <= 0
                    ):
                        # 计算调整系数，确保剩余组数量是group_divisible的倍数
                        ratio = (
                                trial_num_important_groups // self.group_divisible + 1
                        )  # +1 保留更多组，否则会剪枝更多

                        # 计算调整后的重要组数量
                        refined_num_important_groups = None
                        if ratio <= 1 or trial_num_important_groups == 0:
                            # 确保至少保留group_divisible个组
                            refined_num_important_groups = max(
                                int(self.group_divisible), 1
                            )
                        else:
                            # 按比例调整，确保不小于group_divisible
                            refined_num_important_groups = max(
                                int(ratio * self.group_divisible),
                                int(self.group_divisible),
                            )

                        # 确保调整后的数量不超过总组数
                        refined_num_important_groups = min(
                            group["num_groups"], refined_num_important_groups
                        )

                        # 计算调整后需要的冗余组数量
                        refined_num_active_redundant_groups = (
                                group["num_groups"]
                                - len(group["pruned_idxes"])
                                - refined_num_important_groups
                        )

                        # 更新全局目标冗余组数量，以反映本次调整
                        self.target_num_redundant_groups += (
                                refined_num_active_redundant_groups
                                - len(group["active_redundant_idxes"])
                        )

                        # 调整当前组的活跃冗余组列表
                        group["active_redundant_idxes"] = group[
                                                              "active_redundant_idxes"
                                                          ][:refined_num_active_redundant_groups]

                # 更新当前组的重要组索引：排除已标记为冗余和已剪枝的组
                group["important_idxes"] = [
                    i
                    for i in group["important_idxes"]
                    if (
                            i not in group["active_redundant_idxes"]  # 不在当前冗余组中
                            and i not in group["pruned_idxes"]  # 不在已剪枝组中
                    )
                ]

    def commit_redundant_idxes(self):
        """
        提交当前周期标记的冗余组，将其正式标记为已剪枝状态
        并更新各组的重要组索引和相关状态
        """
        # 遍历所有参数组
        for group in self.param_groups:
            # 只处理可剪枝且非辅助的参数组
            if group["is_prunable"] and not group["is_auxiliary"]:
                # 将当前活跃的冗余组（待剪枝）复制到已剪枝列表中
                # 使用copy()确保原始列表修改不影响已提交的记录
                group["pruned_idxes"].extend(group["active_redundant_idxes"].copy())

                # 清空活跃冗余组列表，为下一个周期做准备
                group["active_redundant_idxes"].clear()

                # 重新计算重要组索引：所有未被剪枝的组都是重要组
                # 遍历组内所有索引，筛选出不在已剪枝列表中的索引
                group["important_idxes"] = [
                    i
                    for i in range(group["num_groups"])
                    if i not in group["pruned_idxes"]
                ]

                # 清空当前组的重要性评分字典
                # 因为剪枝后组的状态已改变，旧的评分不再适用
                group["importance_scores"].clear()

    def quantize_weight(self, param_group, target_name):
        """
        对指定参数组中目标层的权重进行量化处理

        参数:
            param_group: 包含待处理参数的参数组
            target_name: 目标层的名称，用于定位需要量化的权重

        返回:
            is_quantize: 布尔值，表示是否进行了量化操作
            quantized_weight: 量化后的权重（如果进行了量化），否则为None
        """
        # 标记是否需要对目标层进行量化
        is_quantize = False  # 检查"target_name"层是否需要量化
        t_quant = None  # 存储量化阈值参数

        # 遍历参数组中的所有参数名称，查找是否存在量化相关参数
        for p_name in param_group["p_names"]:
            # 检查是否存在权重量化的方向参数(d_quant_wt)
            if "d_quant_wt" in p_name:
                # 提取层名称（去掉参数后缀部分）
                layer_name = ".".join(p_name.split(".")[:-1])

                # 确认该层是目标层且包含权重
                if layer_name in target_name and "weight" in target_name:
                    is_quantize = True  # 标记需要量化
                    quantize_layer_name = layer_name  # 记录需要量化的层名称
                    break  # 找到目标后退出循环

        # 如果不需要量化，直接返回
        if not is_quantize:
            return is_quantize, None
        else:
            # 遍历参数组中的参数，收集量化所需的各项参数
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if quantize_layer_name in p_name and "d_quant_wt" in p_name:
                    # 量化方向参数（可能用于控制量化精度调整方向）
                    d_quant = p.data
                if quantize_layer_name in p_name and "t_quant_wt" in p_name:
                    # 量化阈值参数（用于确定量化范围）
                    t_quant = p.data
                if quantize_layer_name in p_name and "q_m_wt" in p_name:
                    # 量化缩放因子（用于将权重缩放到量化范围）
                    q_m = p.data
                if quantize_layer_name in p_name and "weight" in p_name:
                    # 原始权重数据（需要被量化的对象）
                    weight = p.data

            # 禁用梯度计算，进行量化操作（量化是确定性操作，不需要梯度）
            with torch.no_grad():
                # 调用量化辅助函数执行实际的量化计算
                quantized_weight = self._quantize_helper(
                    weight, d_quant, q_m, t_quant=t_quant
                )

            # 返回量化结果
            return is_quantize, quantized_weight

    def compute_gamma_d(self, param_group, active_redundant_idxes, bit_range):
        """
        计算剪枝的遗忘率(forget_rate)和量化步长(d_quant)

        参数:
            param_group: 参数组，包含模型参数、参数名称等信息
            active_redundant_idxes: 活跃的冗余索引，用于标识需要剪枝的参数位置
            bit_range: 量化位宽范围，格式为[下限, 上限]

        返回:
            forget_rate: 遗忘率，控制剪枝强度
            d_quant: 量化步长，控制参数量化精度
        """
        t_quant = None  # 存储量化阈值
        qm_list = []  # 存储量化映射系数
        # 存储有量化映射的层名称
        layer_name_list = []
        # 存储可剪枝参数的裁剪值
        prune_param_clip_list = []
        # 存储可剪枝参数的梯度
        prune_param_grad_list = []
        # 存储可剪枝参数的残差值
        prune_param_res_list = []
        # 存储冗余索引处的裁剪值
        prune_param_clip_redundant_list = []
        # 存储冗余索引处的残差值
        prune_param_res_redundant_list = []
        # 存储冗余索引处的梯度
        prune_param_grad_redundant_list = []

        ###########################################
        ####  处理带有量化映射的层  ####
        ###########################################
        # 收集所有包含量化映射的层名称
        for p_name in param_group["p_names"]:
            # 筛选出包含"d_quant_wt"的参数名（量化相关参数）
            if "d_quant_wt" not in p_name:
                continue
            # 提取层名称（去掉参数名的最后一部分）
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # 处理每个带有量化映射的层
        for layer_name in layer_name_list:
            # 提取该层的相关参数（量化参数、权重等）
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "d_quant_wt" in p_name:
                        d_quant = p.data  # 量化步长参数
                    if "t_quant_wt" in p_name:
                        t_quant = p.data  # 量化阈值参数
                    if "q_m_wt" in p_name:
                        q_m = p.data  # 量化映射系数
                        qm_list.append(q_m.item())
                    if "weight" in p_name:
                        weight = p.data  # 层权重

            # 处理该层的可剪枝参数
            for p_name, p, p_transform in zip(
                    param_group["p_names"],
                    param_group["params"],
                    param_group["p_transform"],
            ):
                # 处理权重参数
                if layer_name in p_name and "weight" in p_name:
                    # 对权重进行裁剪
                    clipped_weight = self._clip_helper(weight, q_m, t_quant=t_quant)
                    prune_param_clip_list.append(clipped_weight.data)
                    # 计算权重残差（原始值与量化后的值之差）
                    residual_weight = self._residual_helper(
                        weight, d_quant, q_m, t_quant=t_quant
                    )
                    prune_param_res_list.append(residual_weight)
                    # 记录权重梯度
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])
                # 处理偏置参数（非权重的可量化参数）
                elif (
                        layer_name in p_name and p_transform != 1
                ):  # p_transform != 1 表示是偏置参数
                    prune_param_clip_list.append(p.data)
                    # 偏置没有残差，用0填充
                    prune_param_res_list.append(
                        torch.tensor([0.0]).to(p.device).expand_as(p.data)
                    )
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])

        ###########################################
        ### 处理没有量化映射的层 ###
        ###########################################
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            # 筛选出不包含在任何量化层中的参数
            if not any(layer_name in p_name for layer_name in layer_name_list):
                prune_param_clip_list.append(p.data)
                # 没有量化映射的参数残差为0
                prune_param_res_list.append(
                    torch.tensor([0.0]).to(p.device).expand_as(p.data)
                )
                prune_param_grad_list.append(param_group["grad_variant"][p_name])

        # 提取冗余索引处的参数值（需要剪枝的位置）
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

        # 将所有冗余参数展平并拼接
        flatten_clip = torch.cat(
            [tensor.flatten() for tensor in prune_param_clip_redundant_list]
        )
        flatten_grad = torch.cat(
            [tensor.flatten() for tensor in prune_param_grad_redundant_list]
        )
        flatten_res = torch.cat(
            [tensor.flatten() for tensor in prune_param_res_redundant_list]
        )

        # 计算各展平张量的L2范数
        flatten_clip_norm = torch.norm(flatten_clip, p=2)
        flatten_grad_norm = torch.norm(flatten_grad, p=2)
        flatten_res_norm = torch.norm(flatten_res, p=2)

        # 计算余弦相似度（用于衡量参数与梯度的方向一致性）
        eps = 1e-8  # 防止除零错误
        # 裁剪值与梯度的余弦相似度
        cosine_similarity_clip = torch.div(
            torch.dot(flatten_clip, flatten_grad),  # 点积
            # 范数乘积（加安全机制防止除零）
            torch.max(flatten_clip_norm, torch.tensor(eps).to(flatten_clip.device))
            * flatten_grad_norm,
        )
        # 残差值与梯度的余弦相似度
        cosine_similarity_res = torch.div(
            torch.dot(flatten_res, flatten_grad),
            torch.max(flatten_res_norm, torch.tensor(eps).to(flatten_res.device))
            * flatten_grad_norm,
        )

        # 超参数：控制量化和剪枝的强度
        eta = 0.999
        zeta = 0.9

        # 计算遗忘率(forget_rate)
        if torch.mean(flatten_clip).item() < 1e-8:
            # 裁剪值过小，不需要剪枝
            forget_rate = 0.0
        else:
            # 处理异常值
            if torch.isinf(cosine_similarity_clip) or torch.isnan(cosine_similarity_clip):
                self.logger.warning("cosine_similarity_clip is inf or nan, setting forget rate to 0.0")
                forget_rate = 0.0
            # 余弦相似度在[0,1]之间（方向一致）
            elif cosine_similarity_clip >= 0.0 and cosine_similarity_clip <= 1.0:
                # 根据当前剪枝周期位置计算遗忘率
                t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
                forget_rate = 1.0 - (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
            # 余弦相似度在[-1,0)之间（方向相反）
            elif cosine_similarity_clip >= -1.0 and cosine_similarity_clip < 0.0:
                # 根据梯度和裁剪值计算遗忘率
                forget_rate = (
                        -(1 - eta)
                        * param_group["lr"]
                        * flatten_grad_norm
                        / (cosine_similarity_clip * flatten_clip_norm)
                )
            else:
                # 处理未预期的余弦相似度值（错误情况）
                if self.verbose == "True":
                    self.logger.warning(f"Unexpected cosine_similarity_clip value: {cosine_similarity_clip}")
                    outID = "log_info_" + str(self.start_pruning_step)
                    # 保存错误日志
                    filename = os.path.join("outputs", f"sparsity_{self.target_group_sparsity * 100}", f"{outID}.txt")
                    with self.safe_open_file(filename) as logfile:
                        logfile.write("Throw an error: cosine_similarity_clip error\n")
                        logfile.write("similarity_value: {cos_similar:^8.9e}\n".format(
                            cos_similar=cosine_similarity_clip.item()
                        ))
                        logfile.write("flatten_grad_max: {flatten_grad:^8.9e}\n".format(
                            flatten_grad=torch.max(flatten_grad).item()
                        ))
                        logfile.write("flatten_clip_max: {flatten_clip:^8.9e}\n".format(
                            flatten_clip=torch.max(flatten_clip).item()
                        ))
                        logfile.write("flatten_grad_mean: {flatten_grad:^8.9e}\n".format(
                            flatten_grad=torch.mean(flatten_grad).item()
                        ))
                        logfile.write("flatten_clip_mean: {flatten_clip:^8.9e}\n".format(
                            flatten_clip=torch.mean(flatten_clip).item()
                        ))
                    self.logger.error("Error with computing cosine_similarity_clip!")
                # 触发断言错误，终止程序
                assert 1 == 2

        # 确定量化步长(d_quant)的范围
        bit_width_lower = bit_range[0]  # 位宽下限
        bit_width_upper = bit_range[1]  # 位宽上限
        # 根据位宽计算量化步长的上下限
        d_quant_upper = self._d_quant_helper(
            bit_width_lower, max(np.abs(qm_list)), t_quant
        )
        d_quant_lower = self._d_quant_helper(
            bit_width_upper, max(np.abs(qm_list)), t_quant
        )

        # 量化步长的安全机制
        if cosine_similarity_res >= 0.0 or forget_rate == 0.0:
            # 残差与梯度方向一致或不剪枝时，使用最大量化步长
            d_quant = d_quant_upper
        else:
            # 根据梯度和残差计算量化步长
            d_quant = (
                    -zeta
                    * eta
                    * param_group["lr"]
                    * flatten_grad_norm
                    / (forget_rate * cosine_similarity_res * flatten_res_norm)
            )
            # 确保量化步长不小于下限（避免过小）
            while d_quant < d_quant_lower:
                forget_rate = forget_rate * 0.8
                d_quant = d_quant / 0.8
            # 确保量化步长不大于上限（避免过大）
            d_quant = min(d_quant_upper, d_quant)

        # 详细日志记录（如果启用）
        if self.verbose == "True":
            filename = os.path.join("outputs", f"sparsity_{self.target_group_sparsity * 100}", f"{outID}.txt")
            with self.safe_open_file(filename) as logfile:
                content = "Step: {num_step:^11s} Layer_name: {name:^30s} clip_max: {clip_max:^8.5e} res_max: {res_max:^8.5e} grad_max: {grad_max:^8.5e} clip_grad: {clip_grad:^8.5e} res_grad: {res_grad:^8.5e} flatten_clip_norm: {flatten_clip_norm:^8.5e} flatten_res_norm: {flatten_res_norm:^8.5e} flatten_grad_norm: {flatten_grad_norm:^8.5e} cos(gamma): {angle_gamma:^8.5e} cos(d):{angle_d:^8.5e} forget_rate: {gamma:^8.5e} d_quant: {d_quant:^8.5e} \n".format(
                    num_step=str(self.num_steps),
                    name=layer_name_list[0] if layer_name_list else "N/A",
                    clip_max=torch.max(flatten_clip).item(),
                    res_max=torch.max(flatten_res).item(),
                    grad_max=torch.max(flatten_grad).item(),
                    clip_grad=torch.dot(flatten_clip, flatten_grad).item(),
                    res_grad=torch.dot(flatten_res, flatten_grad).item(),
                    flatten_clip_norm=flatten_clip_norm,
                    flatten_res_norm=flatten_res_norm,
                    flatten_grad_norm=flatten_grad_norm,
                    angle_gamma=cosine_similarity_clip,
                    angle_d=cosine_similarity_res,
                    gamma=forget_rate,
                    d_quant=d_quant,
                )
                logfile.write(content)

        return forget_rate, d_quant

    def get_bitwidth_dict(self, param_group):
        """
        获取各层的权重和激活函数的位宽信息，并以字典形式返回

        参数:
            param_group: 包含参数名称和参数数据的字典，结构应包含
                        "p_names": 参数名称列表
                        "params": 对应的参数数据列表

        返回:
            bit_dict: 字典，键为层名称，值为包含该层权重和激活位宽的字典
        """
        # 初始化量化相关参数变量
        d_quant_wt = None  # 权重的量化步长参数
        q_m_wt = None  # 权重的量化比例参数
        t_quant_wt = None  # 权重的量化类型参数
        d_quant_act = None  # 激活函数的量化步长参数
        q_m_act = None  # 激活函数的量化比例参数
        t_quant_act = None  # 激活函数的量化类型参数

        layer_name_list = []  # 存储所有层名称的列表
        bit_dict = {}  # 最终要返回的位宽信息字典

        # 第一遍遍历：收集所有包含"d_quant"的参数对应的层名称
        for p_name in param_group["p_names"]:
            # 只处理包含"d_quant"的参数名（量化相关参数）
            if "d_quant" not in p_name:
                continue

            # 从参数名中提取层名称（去掉最后一部分）
            # 例如，若p_name是"layer1.0.d_quant_wt"，则层名称是"layer1.0"
            layer_name = ".".join(p_name.split(".")[:-1])

            # 确保层名称不重复添加
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # 第二遍遍历：为每个层提取量化参数并计算位宽
        for layer_name in layer_name_list:
            # 遍历所有参数，找到当前层对应的量化参数
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                # 只处理属于当前层的参数
                if layer_name in p_name:
                    # 根据参数名判断参数类型并赋值
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data  # 权重量化类型
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data  # 权重量化比例
                    if "d_quant_wt" in p_name:
                        d_quant_wt = p.data  # 权重量化步长
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data  # 激活量化类型
                    if "q_m_act" in p_name:
                        q_m_act = p.data  # 激活量化比例
                    if "d_quant_act" in p_name:
                        d_quant_act = p.data  # 激活量化步长

            # 为当前层创建一个条目
            bit_dict[layer_name] = {}

            # 计算权重的位宽
            bit_width_wt = self._bit_width_helper(
                d_quant=d_quant_wt, q_m=q_m_wt, t_quant=t_quant_wt
            )

            # 计算激活函数的位宽
            bit_width_act = self._bit_width_helper(
                d_quant=d_quant_act, q_m=q_m_act, t_quant=t_quant_act
            )

            # 将计算得到的位宽四舍五入后存入字典（只存储有效结果）
            if bit_width_wt is not None:
                bit_dict[layer_name]["weight"] = round(bit_width_wt)

            if bit_width_act is not None:
                bit_dict[layer_name]["activation"] = round(bit_width_act)

        # 返回包含所有层位宽信息的字典
        return bit_dict

    def gradient_descent_step(self, param_group):
        """
        执行梯度下降步骤，更新参数组中的参数

        参数:
            param_group: 包含参数信息的字典，应包含以下键:
                        - "p_names": 参数名称列表
                        - "params": 参数对象列表（与p_names对应）
                        - "grad_variant": 存储参数梯度信息的字典
                        - "weight_decay": 权重衰减系数（可选）
                        - "variant": 优化器类型（如"adamw"）
                        - "lr": 主学习率
                        - "lr_quant": 量化相关参数的学习率
        """
        # 遍历参数组中的所有参数名称和对应参数
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            # 如果当前参数不在梯度信息字典中，则跳过该参数
            if p_name not in param_group["grad_variant"]:
                continue

            # 处理权重衰减（仅当使用AdamW优化器且设置了权重衰减系数时）
            if (
                    param_group["weight_decay"] is not None  # 检查是否设置了权重衰减
                    and param_group["variant"] == "adamw"  # 检查是否为AdamW优化器
            ):
                # 判断是否为量化相关参数（d_quant/t_quant/q_m）
                if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                    # 对量化参数应用权重衰减，使用量化学习率lr_quant
                    # 等价于: p.data = p.data - param_group["lr_quant"] * (param_group["weight_decay"] * p.data)
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    # 对普通参数应用权重衰减，使用普通学习率lr
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr"]
                    )

            # 应用梯度更新（参数更新的核心步骤）
            # 判断是否为量化相关参数
            if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                # 使用量化学习率lr_quant更新量化参数
                # 等价于: p.data = p.data - param_group["lr_quant"] * param_group["grad_variant"][p_name]
                p.data.add_(
                    param_group["grad_variant"][p_name],  # 梯度值
                    alpha=-param_group["lr_quant"]  # 学习率（负号表示梯度下降）
                )
            else:
                # 使用普通学习率lr更新普通参数
                p.data.add_(
                    param_group["grad_variant"][p_name],  # 梯度值
                    alpha=-param_group["lr"]  # 学习率（负号表示梯度下降）
                )

    def partial_projected_gradient_descent_step_range_wt(self, param_group):
        """
        对权重量化参数应用部分投影梯度下降（带范围约束的梯度下降）
        不仅执行常规梯度更新，还会将权重量化相关参数投影到特定范围内
        """
        # 第一部分：常规梯度下降步骤（与之前的梯度更新逻辑相同）
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            # 跳过没有梯度信息的参数
            if p_name not in param_group["grad_variant"]:
                continue

            # 处理AdamW优化器的权重衰减
            if (
                    param_group["weight_decay"] is not None  # 检查是否设置了权重衰减
                    and param_group["variant"] == "adamw"  # 检查是否为AdamW优化器
            ):
                # 对权重量化相关参数应用权重衰减（使用量化学习率）
                if (
                        "d_quant_wt" in p_name  # 权重的量化步长参数
                        or "t_quant_wt" in p_name  # 权重的量化类型参数
                        or "q_m_wt" in p_name  # 权重的量化比例参数
                ):
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],  # 使用量化学习率
                    )
                else:
                    # 对普通参数应用权重衰减（使用普通学习率）
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr"]  # 使用普通学习率
                    )

            # 根据参数类型分配不同的学习率进行梯度更新
            if "d_quant_wt" in p_name or "t_quant_wt" in p_name or "q_m_wt" in p_name:
                # 权重量化参数使用量化学习率更新
                p.data.add_(
                    param_group["grad_variant"][p_name],  # 梯度值
                    alpha=-param_group["lr_quant"]  # 量化学习率（负号表示梯度下降）
                )
            else:
                # 普通参数使用常规学习率更新
                p.data.add_(
                    param_group["grad_variant"][p_name],  # 梯度值
                    alpha=-param_group["lr"]  # 普通学习率
                )

        # 第二部分：找出参数组中所有包含权重量化参数的层
        layer_name_list = []
        for p_name in param_group["p_names"]:
            # 只关注权重的量化步长参数，以此确定相关层
            if "d_quant_wt" not in p_name:
                continue

            # 从参数名中提取层名称（去掉最后一部分）
            # 例如："layer1.0.d_quant_wt" -> 层名称为"layer1.0"
            layer_name = ".".join(p_name.split(".")[:-1])

            # 确保层名称不重复添加
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # 第三部分：对权重量化参数应用投影（范围约束）
        for layer_name in layer_name_list:
            # 初始化当前层的量化类型参数
            t_quant_wt = None
            # 遍历参数找到当前层的量化相关参数
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data  # 权重的量化类型参数
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data  # 权重的量化比例参数

            # 计算当前层d_quant_wt的约束范围（上下界）
            # 通过辅助函数计算最大/最小位宽对应的d_quant值
            d_quant_min = self._d_quant_helper(self.max_bit_wt, q_m_wt, t_quant_wt)
            d_quant_max = self._d_quant_helper(self.min_bit_wt, q_m_wt, t_quant_wt)

            # 如果边界是张量，将其转换为标量（便于后续clamp操作）
            if isinstance(d_quant_min, torch.Tensor):
                d_quant_min = float(d_quant_min.item())
            if isinstance(d_quant_max, torch.Tensor):
                d_quant_max = float(d_quant_max.item())

            # 对当前层的d_quant_wt参数应用范围约束（投影操作）
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                # 找到当前层的权重量化步长参数
                if layer_name in p_name and "d_quant_wt" in p_name:
                    # 将参数值限制在[d_quant_min, d_quant_max]范围内
                    # clamp_是原地操作，直接修改张量数据
                    p.data.clamp_(min=d_quant_min, max=d_quant_max)

    def partial_projected_gradient_descent_step_range_act(self, param_group):
        """
        对激活函数量化参数应用部分投影梯度下降（带范围约束的梯度更新）
        专门针对激活函数的量化参数进行优化，并确保其在合理范围内
        """
        # 设置激活函数量化的位宽范围（未来可作为HessoQuant优化器的超参数）
        min_bit = self.min_bit_act  # 激活函数量化的最小位宽
        max_bit = self.max_bit_act  # 激活函数量化的最大位宽

        # 第一部分：对激活函数量化参数执行梯度下降更新
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            # 跳过没有梯度信息的参数
            if p_name not in param_group["grad_variant"]:
                continue

            # 处理AdamW优化器的权重衰减（仅针对激活量化参数）
            if (
                    param_group["weight_decay"] is not None  # 检查是否设置了权重衰减
                    and param_group["variant"] == "adamw"  # 检查是否为AdamW优化器
            ):
                # 仅对激活函数的量化参数应用权重衰减
                if (
                        "d_quant_act" in p_name  # 激活的量化步长参数
                        or "t_quant_act" in p_name  # 激活的量化类型参数
                        or "q_m_act" in p_name  # 激活的量化比例参数
                ):
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],  # 使用量化学习率
                    )

            # 为激活函数量化参数分配学习率并更新
            if (
                    "d_quant_act" in p_name
                    or "t_quant_act" in p_name
                    or "q_m_act" in p_name
            ):
                # 激活量化参数使用量化学习率更新
                p.data.add_(
                    param_group["grad_variant"][p_name],  # 梯度值
                    alpha=-param_group["lr_quant"]  # 量化学习率（负号表示梯度下降）
                )

        # 第二部分：找出参数组中所有包含激活量化参数的层
        layer_name_list = []
        for p_name in param_group["p_names"]:
            # 只关注激活的量化步长参数，以此确定相关层
            if "d_quant_act" not in p_name:
                continue

            # 从参数名中提取层名称（去掉最后一部分）
            # 例如："layer1.0.d_quant_act" -> 层名称为"layer1.0"
            layer_name = ".".join(p_name.split(".")[:-1])

            # 确保层名称不重复添加
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # 第三部分：对激活量化参数应用投影（范围约束）
        for layer_name in layer_name_list:
            # 初始化当前层的激活量化参数
            t_quant_act = None
            # 遍历参数找到当前层的激活量化相关参数
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data  # 激活的量化类型参数
                    if "q_m_act" in p_name:
                        q_m_act = p.data  # 激活的量化比例参数

            # 对当前层的d_quant_act参数应用范围约束
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                # 找到当前层的激活量化步长参数
                if layer_name in p_name and "d_quant_act" in p_name:
                    # 计算激活量化步长的约束范围（上下界）
                    d_quant_min = self._d_quant_helper(max_bit, q_m_act, t_quant_act)
                    d_quant_max = self._d_quant_helper(min_bit, q_m_act, t_quant_act)

                    # 将参数值限制在[d_quant_min, d_quant_max]范围内
                    # 使用torch.clip进行裁剪（与clamp_功能类似，返回新张量）
                    p.data = torch.clip(p.data, min=d_quant_min, max=d_quant_max)

    def partial_projected_gradient_descent_step_fix(self, param_group, bit_dict):
        """
        对量化参数应用固定位宽的部分投影梯度下降
        根据预定义的位宽字典(bit_dict)将量化参数固定在精确值上，而非范围内

        参数:
            param_group: 包含参数信息的字典
            bit_dict: 位宽信息字典，键为层名称，值包含"weight"和"activation"的位宽
        """
        # 第一部分：常规梯度下降步骤（与之前逻辑一致）
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            # 跳过没有梯度信息的参数
            if p_name not in param_group["grad_variant"]:
                continue

            # 处理AdamW优化器的权重衰减
            if (
                    param_group["weight_decay"] is not None  # 检查是否设置了权重衰减
                    and param_group["variant"] == "adamw"  # 检查是否为AdamW优化器
            ):
                # 对量化相关参数应用权重衰减（使用量化学习率）
                if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    # 对普通参数应用权重衰减（使用普通学习率）
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr"]
                    )

            # 根据参数类型分配不同学习率进行梯度更新
            if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                # 量化参数使用量化学习率更新
                p.data.add_(
                    param_group["grad_variant"][p_name],  # 梯度值
                    alpha=-param_group["lr_quant"]  # 量化学习率
                )
            else:
                # 普通参数使用常规学习率更新
                p.data.add_(
                    param_group["grad_variant"][p_name],  # 梯度值
                    alpha=-param_group["lr"]  # 普通学习率
                )

        # 第二部分：根据固定位宽对量化参数进行精确投影
        # 遍历bit_dict中所有层
        for layer_name in bit_dict.keys():
            # 初始化当前层的量化参数
            t_quant_wt = None  # 权重量化类型
            t_quant_act = None  # 激活量化类型
            # 遍历参数找到当前层的量化相关参数
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data  # 权重量化类型参数
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data  # 权重量化比例参数
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data  # 激活量化类型参数
                    if "q_m_act" in p_name:
                        q_m_act = p.data  # 激活量化比例参数

            # 对当前层的量化步长参数应用固定值约束
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                # 处理权重量化步长参数
                if layer_name in p_name and "d_quant_wt" in p_name:
                    # 从bit_dict获取当前层权重的固定位宽
                    bit_width = bit_dict[layer_name]["weight"]
                    # 根据固定位宽计算精确的d_quant_wt值
                    d_quant_wt = self._d_quant_helper(bit_width, q_m_wt, t_quant_wt)
                    # 将参数值固定为计算出的精确值（上下界相同）
                    p.data = torch.clip(p.data, min=d_quant_wt, max=d_quant_wt)

                # 处理激活量化步长参数
                if layer_name in p_name and "d_quant_act" in p_name:
                    # 从bit_dict获取当前层激活的固定位宽
                    bit_width = bit_dict[layer_name]["activation"]
                    # 根据固定位宽计算精确的d_quant_act值
                    d_quant_act = self._d_quant_helper(bit_width, q_m_act, t_quant_act)
                    # 将参数值固定为计算出的精确值（上下界相同）
                    p.data = torch.clip(p.data, min=d_quant_act, max=d_quant_act)

    @staticmethod
    def _bit_width_helper(d_quant=None, q_m=None, t_quant=None):
        """
        计算量化参数对应的位宽（bit width）辅助函数
        根据量化步长、比例因子和量化类型计算所需的位宽

        参数:
            d_quant: 量化步长参数（必选）
            q_m: 量化比例因子参数（可选）
            t_quant: 量化类型参数（可选）

        返回:
            bit_width: 计算得到的位宽值；若d_quant为None则返回None
        """
        # 如果量化步长参数d_quant未提供，则无法计算位宽，返回None
        if d_quant is None:
            return None

        # 如果量化类型参数t_quant未提供，默认设为1.0
        if t_quant is None:
            t_quant = 1.0

        # 位宽计算公式：
        # 1. 先计算 exp(t_quant * ln(abs(q_m)))，等价于 abs(q_m)的t_quant次方
        # 2. 除以量化步长的绝对值 abs(d_quant)
        # 3. 加1后取以2为底的对数 log2(...)
        # 4. 最后加1得到最终位宽
        bit_width = (
                math.log2(math.exp(t_quant * math.log(abs(q_m))) / abs(d_quant) + 1) + 1
        )

        # 返回计算得到的位宽值
        return bit_width

    @staticmethod
    def _d_quant_helper(bit_width, q_m, t_quant):
        """
        计算量化步长参数d_quant的辅助函数
        用于均匀量化，基于位宽、比例因子和量化类型计算量化步长

        参数:
            bit_width: 目标量化位宽（整数）
            q_m: 量化比例因子（可以是张量或标量）
            t_quant: 量化类型参数

        返回:
            d_quant: 计算得到的量化步长值
        """
        # 如果量化类型参数t_quant未提供，默认设为1.0
        if t_quant is None:
            t_quant = 1.0

        # 处理q_m为张量的情况：取其绝对值的最大值并转换为标量
        if isinstance(q_m, torch.Tensor):
            q_m = torch.max(torch.abs(q_m)).item()  # 取张量所有元素绝对值的最大值
        else:
            q_m = abs(q_m)  # 若是标量则直接取绝对值

        # 防止q_m为零（避免后续除法出现问题）
        # 取q_m绝对值和极小值1e-10中的较大者
        q_m = max(abs(q_m), 1e-10)

        # 使用标量数学计算量化步长d_quant
        # 公式解析：
        # 1. math.exp(t_quant * math.log(abs(q_m))) 等价于 abs(q_m)的t_quant次方
        # 2. 分母(2 **(bit_width - 1) - 1) 表示位宽对应的量化等级范围
        # 整体表示将q_m的动态范围均匀分配到(2^(bit_width-1)-1)个量化等级上
        d_quant = math.exp(t_quant * math.log(abs(q_m))) / (2 ** (bit_width - 1) - 1)

        return d_quant

    @staticmethod
    def _quantize_helper(weight, d_quant, q_m, t_quant):
        """
        对权重进行量化处理的辅助函数
        实现了基于指数变换的量化操作，支持自定义量化范围和步长

        参数:
            weight: 待量化的权重张量
            d_quant: 量化步长参数（由_d_quant_helper计算得到）
            q_m: 量化范围的上限参数
            t_quant: 量化类型参数，控制非线性变换的程度

        返回:
            output: 量化后的权重张量
        """
        # 如果量化类型参数t_quant未提供，默认设为1.0（线性量化）
        if t_quant is None:
            t_quant = 1.0

        # 计算权重的绝对值（用于后续的幅度量化）
        weight_abs = torch.abs(weight)

        # 量化偏移量，此处固定为0.0（对称量化）
        q_s = 0.0

        # 计算量化范围上限的指数变换值
        # 等价于 (abs(q_m - q_s))^t_quant，当q_s=0时即 |q_m|^t_quant
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))

        # 计算输入权重的指数变换值
        # 等价于 (weight_abs - q_s)^t_quant，当q_s=0时即 weight_abs^t_quant
        # 注：假设weight_abs > q_s（即权重绝对值大于偏移量）
        input_pow = torch.exp(t_quant * torch.log(weight_abs - q_s))

        # 核心量化操作：
        # 1. 将输入的指数变换值除以量化步长d_quant
        # 2. 四舍五入到最近的整数（得到量化等级）
        # 3. 再乘以量化步长得到量化后的值
        output = d_quant * torch.round(input_pow.div(d_quant))

        # 处理低于量化下限的情况：将绝对值小于等于q_s的权重置为0
        output[weight_abs <= q_s] = 0

        # 处理超出量化上限的情况：将绝对值大于等于q_m的权重
        # 量化到最大量化等级（防止溢出）
        output[weight_abs >= q_m] = d_quant * torch.round(range_pow.div(d_quant))

        # 恢复原始权重的符号（因为之前只处理了绝对值）
        output = torch.sign(weight) * output

        return output

    @staticmethod
    def _clip_helper(weight, q_m, t_quant):
        """
        对权重进行基于指数变换的裁剪辅助函数
        按指定量化范围和变换规则，将权重值约束在目标范围内（非量化，仅裁剪）

        参数:
            weight: 待裁剪的权重张量（原始权重数据）
            q_m: 裁剪范围的上限参数（量化相关的范围阈值）
            t_quant: 量化类型参数，控制非线性变换的程度（默认1.0为线性变换）

        返回:
            output: 裁剪并经变换处理后的权重张量
        """
        # 若未提供量化类型参数t_quant，默认设为1.0（此时为线性变换）
        if t_quant is None:
            t_quant = 1.0

        # 计算权重的绝对值（后续针对幅度进行裁剪处理）
        weight_abs = torch.abs(weight)

        # 量化偏移量，此处固定为0.0（采用对称裁剪方式）
        q_s = 0.0

        # 计算裁剪范围上限的指数变换值
        # 等价于 (abs(q_m - q_s))^t_quant，因q_s=0，即 |q_m|^t_quant
        # 用于后续对超上限的权重进行统一约束
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))

        # 对权重绝对值进行指数变换：(weight_abs - q_s)^t_quant，因q_s=0，即 weight_abs^t_quant
        # 注：假设输入满足weight_abs > q_s（即权重绝对值大于偏移量）
        output = torch.exp(t_quant * torch.log(weight_abs - q_s))

        # 处理低于下限的权重：绝对值小于等于q_s的部分置为0
        output[weight_abs <= q_s] = 0

        # 处理超出上限的权重：绝对值大于等于q_m的部分，统一设为range_pow（上限的变换值）
        output[weight_abs >= q_m] = range_pow

        # 恢复原始权重的符号（因之前仅对绝对值处理，需保留符号信息）
        output = torch.sign(weight) * output

        return output

    @staticmethod
    def _residual_helper(weight, d_quant, q_m, t_quant):
        """
        计算权重量化残差的辅助函数
        量化残差指"权重经指数变换后与量化值的偏差"，用于量化误差分析或补偿

        参数:
            weight: 待计算残差的原始权重张量
            d_quant: 量化步长（由_d_quant_helper计算得到）
            q_m: 量化范围上限参数
            t_quant: 量化类型参数，控制非线性变换程度（默认1.0为线性）

        返回:
            output: 计算得到的量化残差张量（与输入weight同形状）
        """
        # 若未提供量化类型参数t_quant，默认设为1.0（线性变换场景）
        if t_quant is None:
            t_quant = 1.0

        # 计算权重的绝对值（残差计算先关注幅度，后续恢复符号）
        weight_abs = torch.abs(weight)

        # 量化偏移量，固定为0.0（对称量化场景，与量化/裁剪函数保持一致）
        q_s = 0.0

        # 计算量化范围上限的指数变换值：(abs(q_m - q_s))^t_quant，因q_s=0即|q_m|^t_quant
        # 用于处理超出上限的权重残差
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))

        # 对权重绝对值做指数变换：(weight_abs - q_s)^t_quant，因q_s=0即weight_abs^t_quant
        # 注：假设输入满足weight_abs > q_s（即权重绝对值大于偏移量）
        input_pow = torch.exp(t_quant * torch.log(weight_abs - q_s))

        # 核心残差计算：（四舍五入量化值 - 原始变换值/量化步长）
        # 即"量化后的整数等级"与"原始值对应的浮点等级"的差值
        output = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)

        # 处理低于量化下限的权重：绝对值<=q_s时，残差置0（此类权重量化后为0，无偏差）
        output[weight_abs <= q_s] = 0

        # 处理超出量化上限的权重：按上限的变换值计算残差
        # 逻辑同正常范围：（上限量化值 - 上限变换值/量化步长）
        output[weight_abs >= q_m] = torch.round(range_pow.div(d_quant)) - range_pow.div(d_quant)

        # 恢复残差的符号：与原始权重符号一致（因之前仅基于绝对值计算）
        output = torch.sign(weight) * output

        return output

    def log_qm_projection(self):
        """Log q_m during projection
        该方法的作用是在模型投影（projection）阶段记录q_m参数的相关信息
        q_m通常是量化过程中的一个参数，用于表示量化范围或缩放因子
        """
        # 检查当前训练步数是否在投影阶段内，并且是否是1000的倍数（控制日志记录频率）
        # self.num_steps: 当前训练步数
        # self.start_projection_step: 开始投影的步数
        # self.projection_steps: 投影阶段的总步数
        if (self.num_steps >= self.start_projection_step and
                self.num_steps <= self.start_projection_step + self.projection_steps and
                self.num_steps % 1000 == 0):

            # 构建日志文件路径，包含当前步数信息，便于区分不同阶段的日志
            log_file = os.path.join(self.log_dir, f"projection_qm_{self.num_steps}.txt")

            # 使用安全的文件打开方式写入日志（self.safe_open_file可能处理了文件权限等问题）
            with self.safe_open_file(log_file, "w") as f:
                # 计算当前处于哪个投影周期
                # self.projection_period_duration: 每个投影周期的持续步数
                curr_period = (self.num_steps - self.start_projection_step) // self.projection_period_duration

                # 写入基本信息：当前步数和投影周期
                f.write(f"Step: {self.num_steps}, Projection Period: {curr_period}\n")
                # 写入当前的max_bit_wt值（可能是与量化位宽相关的参数）
                f.write(f"Current max_bit_wt: {self.max_bit_wt}\n\n")

                # 遍历模型的参数组（param_groups，通常按层或按类型组织）
                for group in self.param_groups:
                    # 同时遍历参数名称和参数本身（zip将名称和参数对应起来）
                    for p_name, p in zip(group["p_names"], group["params"]):
                        # 筛选出名称中包含"q_m_wt"的参数（这些是我们需要记录的q_m相关参数）
                        if "q_m_wt" in p_name:
                            # 从参数名称中提取层名称（去掉最后一部分的参数名）
                            layer_name = ".".join(p_name.split(".")[:-1])
                            f.write(f"Layer: {layer_name}\n")

                            # 记录q_m参数的统计信息：最小值、最大值和平均值
                            # p.data: 获取参数的数值部分（排除梯度等信息）
                            # .item(): 将PyTorch张量转换为Python标量
                            # :.6f: 保留6位小数
                            f.write(f"q_m stats: min={p.data.min().item():.6f}, ")
                            f.write(f"max={p.data.max().item():.6f}, ")
                            f.write(f"mean={p.data.mean().item():.6f}\n\n")

    def step(self, loss=None, closure=None):
        """
        核心训练步骤函数，负责模型参数更新、量化投影和剪枝操作的协调执行
        """

        # 如果提供了闭包函数（通常用于计算损失和梯度），则执行闭包
        if closure is not None:
            _ = closure()

        # 训练步数加1，记录当前是第几步训练
        self.num_steps += 1
        # 计算梯度变体（可能是自定义的梯度计算或调整）
        self.compute_grad_variant()

        # 确定权重的位范围投影（量化相关）
        # 条件：当前步数在投影阶段内，且投影开始步与剪枝开始步不同
        if (
                self.num_steps >= self.start_projection_step
                and self.num_steps <= self.start_pruning_step
                and self.start_projection_step != self.start_pruning_step
        ):
            # 每经过一个投影周期（除了第一个周期），调整最大比特宽度
            if (
                    (self.num_steps - self.start_projection_step - 1) % self.projection_period_duration == 0
                    and (self.num_steps - self.start_projection_step - 1) != 0
            ):
                # 减小最大比特宽度（量化精度可能随训练进程降低）
                self.max_bit_wt = self.max_bit_wt - self.bit_reduction
                # 最小比特宽度保持不变
                self.min_bit_wt = self.min_bit_wt

        # 将参数组划分为重要组和冗余组（剪枝相关）
        # 条件：当前步数在剪枝阶段内，且未完成所有剪枝周期，且剪枝周期有效
        if (
                self.num_steps >= self.start_pruning_step
                and self.curr_pruning_period < self.pruning_periods
                and self.pruning_period_duration != 0
        ):
            # 每经过一个剪枝周期，执行一次重要性评估和冗余组识别
            if (
                    (self.num_steps - self.start_pruning_step - 1) % self.pruning_period_duration == 0
            ):
                self.logger.info(
                    f"使用显著性分数确定重要组和冗余组。步骤={self.num_steps}"
                )
                # 提交冗余索引（确认之前标记的冗余参数）
                self.commit_redundant_idxes()
                # 计算参数重要性分数
                self.compute_importance_scores()
                # 识别冗余组（标记需要剪枝的参数）
                self.identify_redundant_groups()
                # 剪枝周期计数加1
                self.curr_pruning_period += 1

        # 第二阶段：更新变量（计算当前剪枝周期内的相对步数）
        if self.pruning_period_duration != 0:
            t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration

        # 遍历所有参数组，根据不同阶段和参数类型执行相应的更新策略
        for group in self.param_groups:
            # 处理不可剪枝的参数组，或没有活跃冗余索引的参数组
            if not group["is_prunable"] or len(group["active_redundant_idxes"]) == 0:
                # 第一阶段：热身阶段（投影开始前）
                if self.num_steps <= self.start_projection_step:
                    # 使用SGD更新可训练参数和量化参数
                    self.gradient_descent_step(group)

                # 剪枝完成后阶段
                elif self.num_steps > self.start_pruning_step + self.pruning_steps:
                    # 记录最终的比特宽度配置
                    if (
                            self.num_steps == self.start_pruning_step + self.pruning_steps + 1
                    ):
                        bit_layer = self.get_bitwidth_dict(group)
                        self.bit_layers.update(bit_layer)
                    # 使用固定的投影梯度下降更新
                    self.partial_projected_gradient_descent_step_fix(
                        group, self.bit_layers
                    )

                # 投影阶段（未到剪枝完成）
                else:
                    # 使用基于范围的部分投影梯度下降更新权重
                    self.partial_projected_gradient_descent_step_range_wt(group)
                    # 如果需要对激活进行量化，取消下一行注释
                    # self.partial_projected_gradient_descent_step_range_act(group)

            # 第三阶段：处理可剪枝且有活跃冗余索引的参数组（联合剪枝和量化）
            elif (
                    group["is_prunable"] and len(group["active_redundant_idxes"]) > 0
            ):
                # 对量化参数（t_quant_wt和q_m_wt）应用随机梯度更新
                for p_name, p, p_transform in zip(
                        group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue
                    # 仅更新量化相关参数
                    if "t_quant_wt" in p_name or "q_m_wt" in p_name:
                        p.data.add_(
                            group["grad_variant"][p_name], alpha=-group["lr_quant"]
                        )

                # 获取当前活跃的冗余索引（需要剪枝的参数位置）
                active_redundant_idxes = group["active_redundant_idxes"]

                # 计算遗忘率(gamma)和量化步长(d_quant)
                gamma, d_quant = self.compute_gamma_d(
                    group, active_redundant_idxes, [self.min_bit_wt, self.max_bit_wt]
                )
                # 保存当前的gamma和d_quant值
                self.gamma, self.d_quant = gamma, d_quant

                # 更新量化步长d_quant
                for i, (p_name, p_transform) in enumerate(
                        zip(group["p_names"], group["p_transform"])
                ):
                    if "d_quant_wt" in p_name:
                        # 不计算梯度的方式更新参数
                        with torch.no_grad():
                            group["params"][i].copy_(d_quant)

                # 处理其他参数更新
                for p_name, p, p_transform in zip(
                        group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue

                    # 对冗余参数应用信息移除项（剪枝核心操作）
                    is_quantize, quantize_weight = self.quantize_weight(group, p_name)
                    if p_transform != TensorTransform.NO_PRUNE:
                        if is_quantize:
                            # 对量化权重的冗余部分应用遗忘率
                            p.data[active_redundant_idxes] = (
                                    p.data[active_redundant_idxes]
                                    - gamma * quantize_weight.data[active_redundant_idxes]
                            )
                        else:
                            # 对非量化权重的冗余部分应用遗忘率
                            p.data[active_redundant_idxes] = (
                                    p.data[active_redundant_idxes]
                                    - gamma * p.data[active_redundant_idxes]
                            )

                    # 对非量化参数应用随机梯度更新
                    if (
                            "d_quant" not in p_name
                            and "t_quant" not in p_name
                            and "q_m" not in p_name
                    ):
                        p.data.add_(group["grad_variant"][p_name], alpha=-group["lr"])

                    # 处理辅助参数（可能是与主参数相关的额外参数）
                    for ng_id, offset in group["auxiliary_ngs"]:
                        # 计算辅助参数的冗余索引（基于主参数的冗余索引偏移）
                        active_redundant_aux_idxes = [
                            i + offset for i in active_redundant_idxes
                        ]
                        for aux_p in self.auxiliary_param_groups[ng_id]["params"]:
                            if aux_p.grad is None:
                                continue
                            # 逐步衰减辅助参数的冗余部分
                            aux_p.data[active_redundant_aux_idxes, ...] *= (
                                                                                   self.pruning_period_duration - t - 1.0
                                                                           ) / (self.pruning_period_duration - t)

            # 将已剪枝的参数组固定为零（确保剪枝效果）
            self.fix_pruned_groups_as_zeros(group)

        # 在每个剪枝周期结束时，提交冗余索引（确认剪枝）
        if self.pruning_period_duration != 0:
            if self.num_steps >= self.start_pruning_step and t == self.pruning_period_duration - 1:
                self.commit_redundant_idxes()

    def compute_metrics(self):
        """Compute optimizer metrics, skipping quantization parameters.
        计算优化器指标，跳过量化参数。
        主要用于计算与模型参数组相关的各种归一化指标和稀疏性指标，
        区分重要参数组和冗余参数组的统计信息。
        """
        # 初始化优化器的各项指标为初始值
        self.opt_metrics.norm_params = 0.0  # 所有参数组的范数总和
        self.opt_metrics.norm_important_groups = 0.0  # 重要参数组的范数总和
        self.opt_metrics.norm_redundant_groups = 0.0  # 冗余参数组的范数总和
        self.opt_metrics.num_zero_groups = 0  # 范数为零的参数组数量
        self.opt_metrics.num_important_groups = 0  # 重要参数组的总数
        self.opt_metrics.num_redundant_groups = 0  # 冗余参数组的总数

        # 遍历所有参数组(param_groups)
        for group in self.param_groups:
            # 跳过不可剪枝的参数组和辅助参数组
            # 只处理可剪枝且非辅助的参数组
            if not (group["is_prunable"] and not group["is_auxiliary"]):
                continue

            # 初始化参数组范数为None
            norm_group = None
            # 获取重要参数的索引和冗余参数的索引
            import_idxes = group["important_idxes"]  # 重要参数索引列表
            redund_idxes = group["active_redundant_idxes"] + group["pruned_idxes"]  # 冗余参数索引列表（活跃冗余+已剪枝）

            # 遍历当前参数组中的参数及其对应的变换方式
            for param, p_transform in zip(group["params"], group["p_transform"]):
                # 跳过不需要剪枝的参数
                if p_transform == TensorTransform.NO_PRUNE:
                    continue

                # 对参数数据应用指定的变换
                param_transform = tensor_transformation_param_group(param.data, p_transform, group)

                # 计算参数变换后的L2范数的平方，并累加到当前参数组的范数中
                # 按维度1计算范数（每个组内参数的范数）
                if norm_group is None:
                    # 首次初始化，直接赋值
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    # 后续累加
                    norm_group += torch.norm(param_transform, dim=1) ** 2

            # 只有当存在有效参数时才继续处理
            if norm_group is not None:
                # 计算最终的L2范数（开平方）
                norm_group = torch.sqrt(norm_group)

                # 统计范数为零的参数组数量
                self.opt_metrics.num_zero_groups += torch.sum(norm_group == 0).item()

                # 累加所有参数组的范数总和
                self.opt_metrics.norm_params += torch.sum(norm_group).item()

                # 累加重要参数组的范数总和
                self.opt_metrics.norm_important_groups += torch.sum(
                    norm_group[import_idxes]
                ).item()

                # 累加冗余参数组的范数总和
                self.opt_metrics.norm_redundant_groups += torch.sum(
                    norm_group[redund_idxes]
                ).item()

                # 统计重要参数组的数量
                self.opt_metrics.num_important_groups += len(import_idxes)

                # 统计冗余参数组的数量
                self.opt_metrics.num_redundant_groups += len(redund_idxes)

        # 计算参数组的稀疏性：范数为零的参数组占总参数组的比例
        # 加入safe_guard是为了避免除零错误
        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / float(
            self.total_num_groups + self.safe_guard
        )

        # 返回计算得到的所有指标
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
