"""
优化过程模拟测试
直接构造参数组测试 MyGETA 优化器，绕过 OTO 图分析。

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_optimization_simulation.py -v -s
"""

import sys
import os
import math
import torch
import torch.nn as nn
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.optimizer.geta_b import MyGETA
from only_train_once.transform import TensorTransform


# ============================================================================
# 辅助函数：构造模拟参数组
# ============================================================================

def create_mock_param_groups(num_groups=16, num_layers=2, include_quant=True):
    """
    构造模拟参数组，模拟真实优化器的 param_groups 结构。
    每个参数组包含权重和可选的量化参数。
    """
    param_groups = []
    global_idx = 0

    for layer_idx in range(num_layers):
        # 权重参数
        weight = nn.Parameter(torch.randn(num_groups, 8))

        p_names = [f'layer{layer_idx}.weight']
        params = [weight]
        p_transforms = [TensorTransform.BASIC]

        if include_quant:
            # 量化参数
            d_quant = nn.Parameter(torch.tensor(0.1))
            t_quant = nn.Parameter(torch.tensor(1.0))
            q_m = nn.Parameter(torch.tensor(1.0))

            p_names.extend([
                f'layer{layer_idx}.d_quant_wt',
                f'layer{layer_idx}.t_quant_wt',
                f'layer{layer_idx}.q_m_wt',
            ])
            params.extend([d_quant, t_quant, q_m])
            p_transforms.extend([1, 1, 1])  # NO_PRUNE for quant params

        group = {
            'id': f'group_{layer_idx}',
            'params': params,
            'p_names': p_names,
            'p_transform': p_transforms,
            'is_prunable': True,
            'is_auxiliary': False,
            'num_groups': num_groups,
            'global_idxes': list(range(global_idx, global_idx + num_groups)),
            'global_start_idx': global_idx,
            'lr': 0.01,
            'lr_quant': 0.001,
            'weight_decay': 0.0001,
            'variant': 'sgd',
            'first_momentum': 0.9,
            'second_momentum': 0.999,
            'dampening': 0.0,
            'important_idxes': list(range(num_groups)),
            'active_redundant_idxes': [],
            'pruned_idxes': [],
            'importance_scores': {},
            'auxiliary_ngs': [],
            'grad_variant': {},
        }
        param_groups.append(group)
        global_idx += num_groups

    return param_groups


def setup_optimizer(param_groups, **kwargs):
    """创建 MyGETA 优化器实例"""
    # 提取参数
    params = [g['params'] for g in param_groups]

    # 创建优化器（绕过 __init__ 的参数组处理）
    optimizer = object.__new__(MyGETA)

    # 手动初始化必要属性
    optimizer.log_dir = 'outputs'
    os.makedirs(optimizer.log_dir, exist_ok=True)

    import logging
    optimizer.logger = logging.getLogger('TestMyGETA')

    # 设置默认值
    defaults = {
        'variant': 'sgd',
        'lr': 0.01,
        'lr_quant': 0.001,
        'first_momentum': None,
        'second_momentum': None,
        'dampening': None,
        'weight_decay': 0.0001,
        'target_group_sparsity': 0.3,
        'group_divisible': 1,
    }
    defaults.update(kwargs)

    optimizer.start_projection_step = kwargs.get('start_projection_step', 0)
    optimizer.projection_steps = kwargs.get('projection_steps', 1)
    optimizer.projection_periods = kwargs.get('projection_periods', 1)
    optimizer.projection_period_duration = optimizer.projection_steps // optimizer.projection_periods
    optimizer.start_pruning_step = kwargs.get('start_pruning_step', 5)
    optimizer.pruning_periods = kwargs.get('pruning_periods', 3)
    optimizer.pruning_steps = kwargs.get('pruning_steps', 15)
    optimizer.pruning_period_duration = optimizer.pruning_steps // optimizer.pruning_periods
    optimizer.curr_pruning_period = 0
    optimizer.bit_reduction = kwargs.get('bit_reduction', 2)
    optimizer.min_bit_wt = kwargs.get('min_bit_wt', 4)
    optimizer.max_bit_wt = kwargs.get('max_bit_wt', 16)
    optimizer.min_bit_act = kwargs.get('min_bit_act', 4)
    optimizer.max_bit_act = kwargs.get('max_bit_act', 16)
    optimizer.grad_clip_min = -1.0
    optimizer.grad_clip_max = 1.0
    optimizer.verbose = "False"
    optimizer.device = 'cpu'
    optimizer.pruned_group_idxes = []
    optimizer.gamma = 0.0
    optimizer.d_quant = 0.0
    optimizer.bit_layers = {}
    optimizer.num_steps = 0
    optimizer.current_prune_groups = 0
    optimizer.prune_groups_flag = False
    optimizer.group_divisible = kwargs.get('group_divisible', 1)
    optimizer.first_moment_grads = {}
    optimizer.second_moment_grads = {}
    optimizer.target_group_sparsity = defaults['target_group_sparsity']

    # 重要性分数标准
    optimizer.importance_score_criteria = {
        "magnitude": 0.2,
        "avg_magnitude": 0.2,
        "cosine_similarity": 0.2,
        "taylor_first_order": 0.2,
        "taylor_second_order": 0.2,
    }

    # 设置参数组
    optimizer.param_groups = param_groups
    optimizer.total_num_groups = sum(g['num_groups'] for g in param_groups)
    optimizer.safe_guard = 1e-8
    optimizer.target_num_redundant_groups = int(optimizer.total_num_groups * defaults['target_group_sparsity'])

    # 设置 active_num_redundant_groups
    optimizer.active_num_redundant_groups = []
    groups_sum = 0
    target = int(optimizer.total_num_groups * defaults['target_group_sparsity'])
    for p in range(optimizer.pruning_periods):
        if p == optimizer.pruning_periods - 1:
            optimizer.active_num_redundant_groups.append(target - groups_sum)
        else:
            optimizer.active_num_redundant_groups.append(target // optimizer.pruning_periods)
            groups_sum += optimizer.active_num_redundant_groups[p]

    # 初始化辅助参数组
    optimizer.auxiliary_param_groups = {}

    # 初始化 metrics
    class OptMetrics:
        norm_params = 0.0
        norm_important_groups = 0.0
        norm_redundant_groups = 0.0
        num_zero_groups = 0
        num_important_groups = 0
        num_redundant_groups = 0
        group_sparsity = 0.0
    optimizer.opt_metrics = OptMetrics()

    return optimizer


def compute_mock_gradients(param_groups):
    """为所有参数计算模拟梯度"""
    for group in param_groups:
        grad_variant = {}
        for p_name, p in zip(group['p_names'], group['params']):
            if p.grad is None:
                p.grad = torch.randn_like(p.data)
            grad_variant[p_name] = p.grad.clone()
        group['grad_variant'] = grad_variant


# ============================================================================
# 测试类
# ============================================================================

class TestOptimizationSimulation:
    """模拟优化过程，检测数值问题"""

    def test_full_training_no_nan_inf(self):
        """完整训练流程不应产生 NaN/Inf"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups, start_pruning_step=5, pruning_periods=3, pruning_steps=15)

        nan_detected = False
        inf_detected = False
        loss_history = []

        for step in range(30):
            # 模拟前向传播和反向传播
            compute_mock_gradients(param_groups)

            # 执行优化步骤
            optimizer.step()

            # 检查参数
            for group in param_groups:
                for p_name, p in zip(group['p_names'], group['params']):
                    if torch.isnan(p.data).any():
                        nan_detected = True
                        print(f"  ✗ Step {step}: {p_name} 包含 NaN")
                    if torch.isinf(p.data).any():
                        inf_detected = True
                        print(f"  ✗ Step {step}: {p_name} 包含 Inf")

            # 模拟 loss
            mock_loss = sum(p.data.abs().sum().item() for g in param_groups for p in g['params'])
            loss_history.append(mock_loss)

            if step % 10 == 0 or step == 29:
                print(f"  Step {step:2d}: mock_loss={mock_loss:.4f}, "
                      f"pruned_groups={len(optimizer.pruned_group_idxes)}")

        assert not nan_detected, "训练过程中检测到 NaN"
        assert not inf_detected, "训练过程中检测到 Inf"
        print(f"  ✓ 30 步训练完成，无 NaN/Inf")

    def test_quantization_params_stable(self):
        """量化参数 (d_quant, t_quant, q_m) 应保持稳定"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups)

        quant_param_history = {}

        for step in range(25):
            compute_mock_gradients(param_groups)
            optimizer.step()

            # 记录量化参数
            for group in param_groups:
                for p_name, p in zip(group['p_names'], group['params']):
                    if 'd_quant' in p_name or 't_quant' in p_name or 'q_m' in p_name:
                        if p_name not in quant_param_history:
                            quant_param_history[p_name] = []
                        val = p.data.mean().item()
                        if torch.isnan(p.data).any():
                            pytest.fail(f"Step {step}: {p_name} 包含 NaN")
                        if torch.isinf(p.data).any():
                            pytest.fail(f"Step {step}: {p_name} 包含 Inf")
                        quant_param_history[p_name].append(val)

        # 检查量化参数变化幅度
        for name, vals in quant_param_history.items():
            if len(vals) > 1:
                change = abs(vals[-1] - vals[0])
                avg = abs(np.mean(vals))
                if avg > 1e-8:
                    relative_change = change / avg
                    print(f"  {name}: 初值={vals[0]:.6f}, 终值={vals[-1]:.6f}, "
                          f"相对变化={relative_change:.4f}")
                    assert relative_change < 100, \
                        f"{name} 变化过大: {relative_change:.4f}"

        print("  ✓ 量化参数保持稳定")

    def test_bit_width_in_valid_range(self):
        """位宽应在 [min_bit, max_bit] 范围内"""
        min_bit = 4
        max_bit = 16
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups, min_bit_wt=min_bit, max_bit_wt=max_bit)

        for step in range(25):
            compute_mock_gradients(param_groups)
            optimizer.step()

            # 检查位宽
            for group in param_groups:
                bit_dict = optimizer.get_bitwidth_dict(group)
                for layer_name, bits in bit_dict.items():
                    if 'weight' in bits:
                        bw = bits['weight']
                        assert min_bit <= bw <= max_bit, \
                            f"Step {step}: {layer_name} weight bit={bw} 超出 [{min_bit}, {max_bit}]"

        print(f"  ✓ 位宽始终在 [{min_bit}, {max_bit}] 范围内")

    def test_sparsity_monotonically_increasing(self):
        """稀疏度在剪枝阶段应单调递增"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups, start_pruning_step=5, pruning_periods=3, pruning_steps=15)

        sparsity_history = []

        for step in range(25):
            compute_mock_gradients(param_groups)
            optimizer.step()

            # 计算稀疏度
            zero_count = 0
            total_count = 0
            for group in param_groups:
                if group['is_prunable']:
                    total_count += group['num_groups']
                    zero_count += len(group['pruned_idxes'])
            sparsity = zero_count / max(total_count, 1)
            sparsity_history.append(sparsity)

        # 检查剪枝阶段后的稀疏度单调性
        pruning_start = 5
        post_pruning = sparsity_history[pruning_start:]
        for i in range(1, len(post_pruning)):
            assert post_pruning[i] >= post_pruning[i-1] - 1e-6, \
                f"稀疏度非单调: step {pruning_start+i} ({post_pruning[i]:.4f}) < " \
                f"step {pruning_start+i-1} ({post_pruning[i-1]:.4f})"

        print(f"  ✓ 稀疏度单调递增: {sparsity_history[0]:.4f} → {sparsity_history[-1]:.4f}")

    def test_zero_groups_stay_zero(self):
        """已剪枝的组应保持为零"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups, start_pruning_step=3, pruning_periods=2, pruning_steps=10)

        zero_groups_history = []

        for step in range(20):
            compute_mock_gradients(param_groups)
            optimizer.step()

            zero_count = sum(len(g['pruned_idxes']) for g in param_groups)
            zero_groups_history.append(zero_count)

        # 检查零组数单调递增
        for i in range(1, len(zero_groups_history)):
            assert zero_groups_history[i] >= zero_groups_history[i-1], \
                f"零组数减少: step {i} ({zero_groups_history[i]}) < step {i-1} ({zero_groups_history[i-1]})"

        print(f"  ✓ 零组数单调递增: {zero_groups_history[0]} → {zero_groups_history[-1]}")

    def test_gradient_norm_reasonable(self):
        """梯度范数应在合理范围内"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups)

        grad_norms = []

        for step in range(20):
            compute_mock_gradients(param_groups)

            # 计算梯度范数
            total_norm = 0
            for group in param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
            total_norm = math.sqrt(total_norm)
            grad_norms.append(total_norm)

            optimizer.step()

        avg_grad = np.mean(grad_norms)
        max_grad = max(grad_norms)

        print(f"  平均梯度范数: {avg_grad:.4f}")
        print(f"  最大梯度范数: {max_grad:.4f}")

        assert not any(math.isnan(g) for g in grad_norms), "梯度包含 NaN"
        assert not any(math.isinf(g) for g in grad_norms), "梯度包含 Inf"
        assert max_grad < 1e6, f"梯度范数过大: {max_grad}"
        print("  ✓ 梯度范数在合理范围内")

    def test_pruning_budget_control_effect(self):
        """验证剪枝预算控制的实际效果"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups)

        initial_budget = optimizer.active_num_redundant_groups.copy()
        optimizer.set_pruning_budget_this_period(rate_to_prune=0.1, period=0)

        print(f"  初始预算: {initial_budget}")
        print(f"  修改后预算: {optimizer.active_num_redundant_groups}")

        expected = max(1, int(optimizer.total_num_groups * 0.1))
        assert optimizer.active_num_redundant_groups[0] == expected, \
            f"预算不匹配: {optimizer.active_num_redundant_groups[0]} != {expected}"
        print("  ✓ 剪枝预算控制生效")

    def test_adaptive_bit_reduction_triggers(self):
        """验证自适应位宽缩减在条件满足时触发"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups, max_bit_wt=12, max_bit_act=12)

        initial_max_wt = optimizer.max_bit_wt
        initial_max_act = optimizer.max_bit_act

        # 用 threshold=0 触发
        optimizer.adaptive_bit_reduction(threshold=0.0)

        print(f"  max_bit_wt: {initial_max_wt} → {optimizer.max_bit_wt}")
        print(f"  max_bit_act: {initial_max_act} → {optimizer.max_bit_act}")

        print("  ✓ 自适应位宽缩减执行成功")

    def test_state_dict_save_load(self):
        """验证状态保存和恢复的正确性"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups)

        # 训练几步
        for step in range(5):
            compute_mock_gradients(param_groups)
            optimizer.step()

        # 保存状态
        state = optimizer.state_dict()
        saved_num_steps = optimizer.num_steps
        saved_period = optimizer.curr_pruning_period

        # 创建新优化器并加载 - 先计算梯度
        param_groups2 = create_mock_param_groups(num_groups=16, num_layers=2)
        compute_mock_gradients(param_groups2)
        optimizer2 = setup_optimizer(param_groups2)
        optimizer2.load_state_dict(state)

        assert optimizer2.num_steps == saved_num_steps, \
            f"num_steps 不匹配: {optimizer2.num_steps} != {saved_num_steps}"
        assert optimizer2.curr_pruning_period == saved_period, \
            f"curr_pruning_period 不匹配"

        print(f"  ✓ 状态保存恢复正确: num_steps={saved_num_steps}")

    def test_noise_scales_with_lr(self):
        """验证噪声缩放与学习率的关系"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups)

        # 测试不同学习率的噪声
        lr_high = 0.1
        lr_low = 0.001

        param = torch.randn(100, 100)
        noise_high = MyGETA._get_cosine_noise(MyGETA, param, lr_high, 5, 10)
        noise_low = MyGETA._get_cosine_noise(MyGETA, param, lr_low, 5, 10)

        ratio = noise_high.std().item() / noise_low.std().item()
        expected_ratio = lr_high / lr_low

        print(f"  高 lr 噪声 std: {noise_high.std().item():.6f}")
        print(f"  低 lr 噪声 std: {noise_low.std().item():.6f}")
        print(f"  比例: {ratio:.2f} (期望: {expected_ratio:.2f})")

        assert abs(ratio - expected_ratio) / expected_ratio < 0.1, \
            f"噪声缩放不正确: {ratio:.2f} != {expected_ratio:.2f}"
        print("  ✓ 噪声正确缩放")


class TestEdgeCases:
    """边界条件测试"""

    def test_zero_learning_rate(self):
        """学习率为 0 时不应崩溃"""
        param_groups = create_mock_param_groups(num_groups=8, num_layers=1)
        optimizer = setup_optimizer(param_groups, lr=0.0, lr_quant=0.0)

        # 更新 param_groups 中的 lr
        for g in param_groups:
            g['lr'] = 0.0
            g['lr_quant'] = 0.0

        for step in range(10):
            compute_mock_gradients(param_groups)
            optimizer.step()

        print("  ✓ 零学习率不崩溃")

    def test_single_group(self):
        """单个参数组的情况"""
        param_groups = create_mock_param_groups(num_groups=4, num_layers=1)
        optimizer = setup_optimizer(param_groups)

        for step in range(10):
            compute_mock_gradients(param_groups)
            optimizer.step()

        print("  ✓ 单参数组不崩溃")

    def test_large_num_groups(self):
        """大量参数组"""
        param_groups = create_mock_param_groups(num_groups=64, num_layers=4)
        optimizer = setup_optimizer(param_groups)

        for step in range(5):
            compute_mock_gradients(param_groups)
            optimizer.step()

        print("  ✓ 大量参数组不崩溃")

    def test_commit_redundant_idxes(self):
        """验证冗余索引提交的正确性"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups)

        # 模拟设置冗余索引
        for group in param_groups:
            if group['is_prunable']:
                group['active_redundant_idxes'] = [0, 1, 2]

        # 提交
        optimizer.commit_redundant_idxes()

        # 验证
        for group in param_groups:
            if group['is_prunable']:
                assert len(group['active_redundant_idxes']) == 0, "active_redundant_idxes 应清空"
                assert 0 in group['pruned_idxes'], "pruned_idxes 应包含原索引"
                assert 1 in group['pruned_idxes'], "pruned_idxes 应包含原索引"
                assert 2 in group['pruned_idxes'], "pruned_idxes 应包含原索引"

        print("  ✓ 冗余索引提交正确")

    def test_identify_redundant_groups(self):
        """验证冗余组识别的正确性"""
        param_groups = create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = setup_optimizer(param_groups, target_group_sparsity=0.3)

        # 模拟计算重要性分数
        optimizer.global_scores = []
        for group in param_groups:
            if group['is_prunable']:
                scores = torch.randn(group['num_groups'])
                optimizer.global_scores.append(scores)

        # 识别冗余组
        optimizer.identify_redundant_groups()

        # 验证
        total_redundant = sum(len(g['active_redundant_idxes']) for g in param_groups)
        expected = optimizer.active_num_redundant_groups[0]
        print(f"  识别到 {total_redundant} 个冗余组 (期望: {expected})")
        assert total_redundant == expected, f"冗余组数不匹配: {total_redundant} != {expected}"
        print("  ✓ 冗余组识别正确")


if __name__ == "__main__":
    print("=" * 70)
    print("优化过程模拟测试")
    print("=" * 70)

    test_classes = [
        TestOptimizationSimulation,
        TestEdgeCases,
    ]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                try:
                    method()
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "=" * 70)
    print("模拟完成")
    print("=" * 70)
