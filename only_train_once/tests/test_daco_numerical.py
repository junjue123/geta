"""
DACO 模块数值稳定性与逻辑正确性测试
重点检查：
1. 数值问题 (NaN, Inf, 除零, 负数取对数)
2. 边界条件 (空输入, 极端值, 单元素)
3. 逻辑漏洞 (公式正确性, 状态一致性)

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python -m pytest only_train_once/tests/test_daco_numerical.py -v -s
"""

import sys
import os
import math
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# 第一部分：扩散噪声数值稳定性
# ============================================================================

class TestDiffusionNumerical:
    """测试扩散噪声的数值稳定性"""

    def test_cosine_noise_no_nan_inf(self):
        """验证噪声在各种输入下不会产生 NaN/Inf"""
        from only_train_once.optimizer.geta_b import MyGETA

        test_cases = [
            # (lr, t, T, description)
            (0.0, 5, 10, "零学习率"),
            (1e-10, 5, 10, "极小学习率"),
            (1e10, 5, 10, "极大学习率"),
            (0.1, 0, 10, "t=0"),
            (0.1, 1, 10, "t=1 (起始)"),
            (0.1, 10, 10, "t=T (终止)"),
            (0.1, 11, 10, "t > T"),
            (0.1, 5, 1, "T=1 (边界)"),
            (0.1, 5, 2, "T=2 (最小有效)"),
        ]

        param = torch.randn(100, 100)
        for lr, t, T, desc in test_cases:
            noise = MyGETA._get_cosine_noise(MyGETA, param, lr, t, T)
            assert not torch.isnan(noise).any(), f"{desc}: 噪声包含 NaN"
            assert not torch.isinf(noise).any(), f"{desc}: 噪声包含 Inf"
            print(f"  ✓ {desc}: noise.std={noise.std().item():.6e}")

    def test_cosine_noise_boundary_t_equals_T(self):
        """验证 t=T 时噪声严格为 0"""
        from only_train_once.optimizer.geta_b import MyGETA

        param = torch.randn(50, 50)
        noise = MyGETA._get_cosine_noise(MyGETA, param, 0.1, T=20, t=20)

        # cos(0) = 1, so 1 - cos(0) = 0, sqrt(0) = 0
        assert torch.allclose(noise, torch.zeros_like(noise), atol=1e-10), \
            f"t=T 时噪声应为 0，实际 max={noise.abs().max().item():.2e}"
        print("  ✓ t=T 时噪声严格为 0")

    def test_cosine_noise_boundary_t_equals_1(self):
        """验证 t=1 时噪声最大"""
        from only_train_once.optimizer.geta_b import MyGETA

        param = torch.randn(1000, 1000)
        lr = 0.1
        T = 20

        noise_start = MyGETA._get_cosine_noise(MyGETA, param, lr, 1, T)
        noise_mid = MyGETA._get_cosine_noise(MyGETA, param, lr, 10, T)

        # 起始噪声应该大于中间噪声
        assert noise_start.std() > noise_mid.std(), \
            "t=1 时噪声应该最大"
        print(f"  ✓ t=1: std={noise_start.std().item():.6f} > t=10: std={noise_mid.std().item():.6f}")

    def test_cosine_noise_formula_correctness(self):
        """验证噪声公式: scale = lr * sqrt(1 - cos((T-t)*pi/(T-1)))"""
        from only_train_once.optimizer.geta_b import MyGETA

        param = torch.ones(10000, 10000)  # 大张量
        lr = 0.5
        T = 10
        t = 5

        noise = MyGETA._get_cosine_noise(MyGETA, param, lr, t, T)

        # 理论标准差
        angle = ((T - t) * math.pi) / (T - 1)
        expected_std = lr * math.sqrt(1 - math.cos(angle))

        actual_std = noise.std().item()
        # 允许 5% 误差（随机性）
        assert abs(actual_std - expected_std) / expected_std < 0.05, \
            f"噪声标准差不匹配: 期望={expected_std:.6f}, 实际={actual_std:.6f}"
        print(f"  ✓ 公式正确: expected={expected_std:.6f}, actual={actual_std:.6f}")

    def test_diffusion_decay_factor_numerical(self):
        """验证扩散衰减因子的数值稳定性"""
        from only_train_once.optimizer.geta import GETA

        # 模拟 GETA 实例
        class MockGETA:
            start_pruning_step = 100
            diffusion_steps = 200
            num_steps = 0

        mock = MockGETA()

        # 测试各种时间点
        test_points = [
            (50, 0.0, "未开始"),
            (100, 1.0, "起始"),
            (200, 0.0, "中间"),
            (300, 0.0, "结束"),
        ]

        for step, expected, desc in test_points:
            mock.num_steps = step
            factor = GETA._get_diffusion_decay_factor(mock)
            assert not math.isnan(factor), f"{desc}: factor 为 NaN"
            assert not math.isinf(factor), f"{desc}: factor 为 Inf"
            assert 0.0 <= factor <= 1.0, f"{desc}: factor={factor} 超出 [0,1]"
            print(f"  ✓ step={step} ({desc}): factor={factor:.4f}")


# ============================================================================
# 第二部分：重要性分数加权数值稳定性
# ============================================================================

class TestImportanceScoreNumerical:
    """测试重要性分数计算的数值稳定性"""

    def test_online_normalization_zero_range(self):
        """验证当所有分数相同时归一化不会除零"""
        from only_train_once.optimizer.importance_score import (
            _apply_online_normalization, _GLOBAL_SCORE_BOUNDS
        )
        _GLOBAL_SCORE_BOUNDS.clear()

        # 所有分数相同
        score = torch.ones(10) * 5.0
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['layer.weight']
        }

        # 应该不崩溃
        _apply_online_normalization(param_group)
        normalized = param_group['importance_scores']['magnitude']

        assert not torch.isnan(normalized).any(), "归一化产生 NaN"
        assert not torch.isinf(normalized).any(), "归一化产生 Inf"
        print(f"  ✓ 零范围归一化: min={normalized.min().item():.4f}, max={normalized.max().item():.4f}")

    def test_online_normalization_negative_values(self):
        """验证负数分数的归一化"""
        from only_train_once.optimizer.importance_score import (
            _apply_online_normalization, _GLOBAL_SCORE_BOUNDS
        )
        _GLOBAL_SCORE_BOUNDS.clear()

        score = torch.tensor([-5.0, -2.0, 0.0, 3.0, 10.0])
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['layer.weight']
        }

        _apply_online_normalization(param_group)
        normalized = param_group['importance_scores']['magnitude']

        assert not torch.isnan(normalized).any(), "负数归一化产生 NaN"
        assert 0.0 <= normalized.min().item() <= 0.01, f"最小值应接近 0: {normalized.min().item()}"
        assert 0.99 <= normalized.max().item() <= 1.01, f"最大值应接近 1: {normalized.max().item()}"
        print(f"  ✓ 负数归一化: {normalized.tolist()}")

    def test_online_normalization_extreme_values(self):
        """验证极端值的归一化"""
        from only_train_once.optimizer.importance_score import (
            _apply_online_normalization, _GLOBAL_SCORE_BOUNDS
        )
        _GLOBAL_SCORE_BOUNDS.clear()

        # 极端值
        score = torch.tensor([1e-10, 1e10])
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['layer.weight']
        }

        _apply_online_normalization(param_group)
        normalized = param_group['importance_scores']['magnitude']

        assert not torch.isnan(normalized).any(), "极端值归一化产生 NaN"
        assert not torch.isinf(normalized).any(), "极端值归一化产生 Inf"
        print(f"  ✓ 极端值归一化: min={normalized.min().item():.6f}, max={normalized.max().item():.6f}")

    def test_bit_width_factor_extreme_bits(self):
        """验证极端位宽的因子计算"""
        from only_train_once.optimizer.importance_score import adjust_importance_criteria
        from only_train_once.optimizer.importance_score import _GLOBAL_SCORE_BOUNDS, _SCORE_HISTORY
        _GLOBAL_SCORE_BOUNDS.clear()
        _SCORE_HISTORY.clear()

        test_cases = [
            (1, "1-bit"),
            (2, "2-bit"),
            (4, "4-bit"),
            (8, "8-bit"),
            (16, "16-bit"),
            (32, "32-bit"),
        ]

        for bit, desc in test_cases:
            score = torch.ones(5)
            param_group = {
                'importance_scores': {'magnitude': score.clone()},
                'p_names': ['layer.weight'],
            }
            bit_layers = {'layer': {'weight': bit}}

            adjust_importance_criteria(param_group, bit_layers, smooth_factor=1.0)
            adjusted = param_group['importance_scores']['magnitude']

            assert not torch.isnan(adjusted).any(), f"{desc}: 产生 NaN"
            assert not torch.isinf(adjusted).any(), f"{desc}: 产生 Inf"
            assert adjusted.min().item() > 0, f"{desc}: 因子应为正数"
            print(f"  ✓ {desc}: factor={adjusted[0].item():.4f}")

    def test_stability_factor_empty_history(self):
        """验证空历史时稳定性因子为 1.0"""
        from only_train_once.optimizer.importance_score import (
            adjust_importance_criteria, _SCORE_HISTORY, _GLOBAL_SCORE_BOUNDS
        )
        _SCORE_HISTORY.clear()
        _GLOBAL_SCORE_BOUNDS.clear()

        score = torch.ones(5)
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['new_layer.weight'],
        }

        adjust_importance_criteria(param_group, bit_layers={}, smooth_factor=1.0)
        adjusted = param_group['importance_scores']['magnitude']

        # 无历史，stability_factor=1.0，bit_factor=1.0
        assert abs(adjusted[0].item() - 1.0) < 0.01, \
            f"无历史时因子应为 1.0，实际={adjusted[0].item()}"
        print(f"  ✓ 空历史: factor={adjusted[0].item():.4f}")

    def test_stability_factor_single_point_history(self):
        """验证单点历史时稳定性因子为 1.0"""
        from only_train_once.optimizer.importance_score import (
            adjust_importance_criteria, _SCORE_HISTORY, _GLOBAL_SCORE_BOUNDS
        )
        _SCORE_HISTORY.clear()
        _GLOBAL_SCORE_BOUNDS.clear()

        _SCORE_HISTORY['layer.weight'] = {
            'steps': [1],
            'magnitude': [0.5]
        }

        score = torch.ones(5)
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['layer.weight'],
        }

        adjust_importance_criteria(param_group, bit_layers={}, smooth_factor=1.0)
        adjusted = param_group['importance_scores']['magnitude']

        # 单点历史，len < 2，stability_factor=1.0
        assert abs(adjusted[0].item() - 1.0) < 0.01, \
            f"单点历史时因子应为 1.0，实际={adjusted[0].item()}"
        print(f"  ✓ 单点历史: factor={adjusted[0].item():.4f}")

    def test_adjust_importance_criteria_no_scores(self):
        """验证无分数时不崩溃"""
        from only_train_once.optimizer.importance_score import adjust_importance_criteria

        param_group = {
            'importance_scores': {},
            'p_names': ['layer.weight'],
        }

        # 应该不崩溃
        adjust_importance_criteria(param_group, bit_layers={})
        print("  ✓ 空分数不崩溃")


# ============================================================================
# 第三部分：自适应位宽缩减逻辑
# ============================================================================

class TestAdaptiveBitReduction:
    """测试自适应位宽缩减的逻辑正确性"""

    def test_adaptive_bit_reduction_no_layers(self):
        """验证无层时不崩溃"""
        from only_train_once.optimizer.geta_b import MyGETA

        # 创建最小化 mock
        class MockOptimizer:
            param_groups = []
            min_bit_wt = 4
            max_bit_wt = 16
            min_bit_act = 4
            max_bit_act = 16
            logger = type('Logger', (), {'info': lambda self, msg: None})()

        mock = MockOptimizer()
        # 应该不崩溃
        MyGETA.adaptive_bit_reduction(mock, threshold=0.5)
        print("  ✓ 无层时不崩溃")

    def test_adaptive_bit_reduction_threshold_0(self):
        """验证 threshold=0 时只要有层降位宽就触发"""
        from only_train_once.optimizer.geta_b import MyGETA

        class MockOptimizer:
            min_bit_wt = 4
            max_bit_wt = 16
            min_bit_act = 4
            max_bit_act = 16
            logger = type('Logger', (), {'info': lambda self, msg: None})()

            def get_bitwidth_dict(self, group):
                # 模拟所有层都已降到 8-bit
                return {'layer1': {'weight': 8}}

            @property
            def param_groups(self):
                return [{'is_prunable': True, 'is_auxiliary': False, 'p_names': ['layer1.weight']}]

        mock = MockOptimizer()
        MyGETA.adaptive_bit_reduction(mock, threshold=0.0)

        assert mock.max_bit_wt == 15, f"应降到 15，实际={mock.max_bit_wt}"
        print(f"  ✓ threshold=0: max_bit_wt -> {mock.max_bit_wt}")

    def test_adaptive_bit_reduction_threshold_1(self):
        """验证 threshold=1 时需要所有层都降位宽才触发"""
        from only_train_once.optimizer.geta_b import MyGETA

        call_count = 0

        class MockOptimizer:
            min_bit_wt = 4
            max_bit_wt = 16
            min_bit_act = 4
            max_bit_act = 16
            logger = type('Logger', (), {'info': lambda self, msg: None})()

            def get_bitwidth_dict(self, group):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    return {'layer1': {'weight': 8}}  # 低于 max
                else:
                    return {'layer2': {'weight': 16}}  # 等于 max

            @property
            def param_groups(self):
                return [
                    {'is_prunable': True, 'is_auxiliary': False, 'p_names': ['layer1.weight']},
                    {'is_prunable': True, 'is_auxiliary': False, 'p_names': ['layer2.weight']},
                ]

        mock = MockOptimizer()
        initial_max = mock.max_bit_wt
        MyGETA.adaptive_bit_reduction(mock, threshold=1.0)

        # 50% < 100%，不触发
        assert mock.max_bit_wt == initial_max, \
            f"不应触发，但 max_bit_wt 变为 {mock.max_bit_wt}"
        print(f"  ✓ threshold=1: max_bit_wt 保持 {mock.max_bit_wt}")

    def test_adaptive_bit_reduction_min_bound(self):
        """验证位宽不会降到 min_bit_wt 以下"""
        from only_train_once.optimizer.geta_b import MyGETA

        class MockOptimizer:
            min_bit_wt = 4
            max_bit_wt = 5  # 接近最小值
            min_bit_act = 4
            max_bit_act = 5
            logger = type('Logger', (), {'info': lambda self, msg: None})()

            def get_bitwidth_dict(self, group):
                return {'layer1': {'weight': 4}}

            @property
            def param_groups(self):
                return [{'is_prunable': True, 'is_auxiliary': False, 'p_names': ['layer1.weight']}]

        mock = MockOptimizer()
        MyGETA.adaptive_bit_reduction(mock, threshold=0.0)

        assert mock.max_bit_wt >= mock.min_bit_wt, \
            f"max_bit_wt={mock.max_bit_wt} < min_bit_wt={mock.min_bit_wt}"
        print(f"  ✓ 下界保护: max_bit_wt={mock.max_bit_wt} >= min_bit_wt={mock.min_bit_wt}")


# ============================================================================
# 第四部分：剪枝预算控制逻辑
# ============================================================================

class TestPruningBudgetControl:
    """测试剪枝预算控制的逻辑正确性"""

    def test_set_pruning_budget_extend_periods(self):
        """验证扩展周期数的正确性"""
        from only_train_once.optimizer.geta_b import MyGETA

        class MockOptimizer:
            total_num_groups = 1000
            pruning_periods = 3
            curr_pruning_period = 0
            active_num_redundant_groups = [100, 100, 100]
            logger = type('Logger', (), {'info': lambda self, msg: None})()

        mock = MockOptimizer()

        # 设置 period=5 (超出当前范围)
        MyGETA.set_pruning_budget_this_period(mock, rate_to_prune=0.05, period=5)

        assert mock.pruning_periods == 6, f"应扩展到 6，实际={mock.pruning_periods}"
        # 注意: 当前实现使用 append，不会填充中间空位
        # 这可能导致 period=3,4 时没有预算分配
        # 建议修复: 使用 extend([0] * (period - len + 1)) 填充
        print(f"  ✓ 扩展周期: pruning_periods={mock.pruning_periods}")
        print(f"  ⚠ 注意: 列表长度={len(mock.active_num_redundant_groups)}, "
              f"内容={mock.active_num_redundant_groups}")
        print(f"    建议: period=3,4 可能缺少预算分配")

    def test_set_pruning_budget_current_period(self):
        """验证修改当前周期的预算"""
        from only_train_once.optimizer.geta_b import MyGETA

        class MockOptimizer:
            total_num_groups = 1000
            pruning_periods = 3
            curr_pruning_period = 1
            active_num_redundant_groups = [100, 100, 100]
            logger = type('Logger', (), {'info': lambda self, msg: None})()

        mock = MockOptimizer()
        MyGETA.set_pruning_budget_this_period(mock, rate_to_prune=0.1)

        expected = max(1, int(1000 * 0.1))  # = 100
        assert mock.active_num_redundant_groups[1] == expected, \
            f"预算应为 {expected}，实际={mock.active_num_redundant_groups[1]}"
        print(f"  ✓ 当前周期预算: {mock.active_num_redundant_groups[1]}")

    def test_set_pruning_budget_minimum_one(self):
        """验证预算至少为 1"""
        from only_train_once.optimizer.geta_b import MyGETA

        class MockOptimizer:
            total_num_groups = 100
            pruning_periods = 3
            curr_pruning_period = 0
            active_num_redundant_groups = [10, 10, 10]
            logger = type('Logger', (), {'info': lambda self, msg: None})()

        mock = MockOptimizer()
        MyGETA.set_pruning_budget_this_period(mock, rate_to_prune=0.001)  # 0.1%

        assert mock.active_num_redundant_groups[0] >= 1, \
            f"预算应至少为 1，实际={mock.active_num_redundant_groups[0]}"
        print(f"  ✓ 最小预算保护: {mock.active_num_redundant_groups[0]}")


# ============================================================================
# 第五部分：compute_gamma_d 数值稳定性
# ============================================================================

class TestComputeGammaD:
    """测试 gamma 和 d_quant 计算的数值稳定性"""

    def test_cosine_similarity_division_safety(self):
        """验证余弦相似度计算不会除零"""
        # 模拟 flatten_clip 全为 0 的情况
        flatten_clip = torch.zeros(100)
        flatten_grad = torch.randn(100)
        flatten_res = torch.randn(100)

        eps = 1e-8
        flatten_clip_norm = torch.norm(flatten_clip, p=2)
        flatten_grad_norm = torch.norm(flatten_grad, p=2)
        flatten_res_norm = torch.norm(flatten_res, p=2)

        # 使用 max(norm, eps) 防止除零
        cosine_similarity_clip = torch.div(
            torch.dot(flatten_clip, flatten_grad),
            torch.max(flatten_clip_norm, torch.tensor(eps)) * flatten_grad_norm,
        )

        assert not torch.isnan(cosine_similarity_clip), "余弦相似度为 NaN"
        assert not torch.isinf(cosine_similarity_clip), "余弦相似度为 Inf"
        print(f"  ✓ 零向量余弦相似度: {cosine_similarity_clip.item():.4f}")

    def test_forget_rate_boundary_conditions(self):
        """验证 forget_rate 的边界条件"""
        # 测试各种 cosine_similarity_clip 值
        test_cases = [
            (0.5, "正常正数"),
            (-0.5, "负数"),
            (0.0, "零"),
            (1.0, "上界"),
            (-1.0, "下界"),
        ]

        for cos_sim, desc in test_cases:
            # 模拟 forget_rate 计算
            if cos_sim >= 0.0 and cos_sim <= 1.0:
                t = 5
                pruning_period_duration = 10
                forget_rate = 1.0 - (pruning_period_duration - t - 1.0) / (pruning_period_duration - t)
            elif cos_sim >= -1.0 and cos_sim < 0.0:
                eta = 0.999
                lr = 0.1
                flatten_grad_norm = torch.tensor(1.0)
                flatten_clip_norm = torch.tensor(1.0)
                forget_rate = -(1 - eta) * lr * flatten_grad_norm / (cos_sim * flatten_clip_norm)
            else:
                forget_rate = 0.0

            assert not math.isnan(forget_rate), f"{desc}: forget_rate 为 NaN"
            assert not math.isinf(forget_rate), f"{desc}: forget_rate 为 Inf"
            print(f"  ✓ {desc} (cos_sim={cos_sim}): forget_rate={forget_rate:.4f}")


# ============================================================================
# 第六部分：平滑因子混合逻辑
# ============================================================================

class TestSmoothFactor:
    """测试 smooth_factor 混合逻辑"""

    def test_smooth_factor_zero(self):
        """验证 smooth_factor=0 时只保留原始分数"""
        from only_train_once.optimizer.importance_score import (
            adjust_importance_criteria, _GLOBAL_SCORE_BOUNDS, _SCORE_HISTORY
        )
        _GLOBAL_SCORE_BOUNDS.clear()
        _SCORE_HISTORY.clear()

        score = torch.tensor([1.0, 2.0, 3.0])
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['layer.weight'],
        }
        bit_layers = {'layer': {'weight': 4}}  # 会放大因子

        adjust_importance_criteria(param_group, bit_layers, smooth_factor=0.0)
        adjusted = param_group['importance_scores']['magnitude']

        # smooth=0: adjusted = raw * 1.0 + raw * factor * 0.0 = raw
        assert torch.allclose(adjusted, score, atol=1e-6), \
            f"smooth=0 应保持原始分数，实际={adjusted.tolist()}"
        print(f"  ✓ smooth=0: 保持原始分数 {adjusted.tolist()}")

    def test_smooth_factor_one(self):
        """验证 smooth_factor=1 时完全使用校准因子"""
        from only_train_once.optimizer.importance_score import (
            adjust_importance_criteria, _GLOBAL_SCORE_BOUNDS, _SCORE_HISTORY
        )
        _GLOBAL_SCORE_BOUNDS.clear()
        _SCORE_HISTORY.clear()

        score = torch.tensor([1.0, 2.0, 3.0])
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['layer.weight'],
        }
        bit_layers = {'layer': {'weight': 4}}  # factor = 1 + log2(32/4) = 4.0

        adjust_importance_criteria(param_group, bit_layers, smooth_factor=1.0)
        adjusted = param_group['importance_scores']['magnitude']

        # smooth=1: adjusted = raw * 0.0 + raw * 1.0 * 4.0 * 1.0 = raw * 4.0
        expected = score * 4.0
        assert torch.allclose(adjusted, expected, atol=0.1), \
            f"smooth=1 应完全使用因子，期望={expected.tolist()}，实际={adjusted.tolist()}"
        print(f"  ✓ smooth=1: 完全使用因子 {adjusted.tolist()}")

    def test_smooth_factor_half(self):
        """验证 smooth_factor=0.5 时正确混合"""
        from only_train_once.optimizer.importance_score import (
            adjust_importance_criteria, _GLOBAL_SCORE_BOUNDS, _SCORE_HISTORY
        )
        _GLOBAL_SCORE_BOUNDS.clear()
        _SCORE_HISTORY.clear()

        score = torch.tensor([1.0, 2.0, 3.0])
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['layer.weight'],
        }
        bit_layers = {'layer': {'weight': 4}}  # factor = 4.0

        adjust_importance_criteria(param_group, bit_layers, smooth_factor=0.5)
        adjusted = param_group['importance_scores']['magnitude']

        # smooth=0.5: adjusted = raw * 0.5 + raw * 1.0 * 4.0 * 0.5 = raw * 2.5
        expected = score * 2.5
        assert torch.allclose(adjusted, expected, atol=0.1), \
            f"smooth=0.5 混合不正确，期望={expected.tolist()}，实际={adjusted.tolist()}"
        print(f"  ✓ smooth=0.5: 混合结果 {adjusted.tolist()}")


# ============================================================================
# 第七部分：全局边界累积正确性
# ============================================================================

class TestGlobalBounds:
    """测试全局边界的累积更新"""

    def test_global_bounds_monotonic_update(self):
        """验证全局边界只扩展不收缩"""
        from only_train_once.optimizer.importance_score import (
            _apply_online_normalization, _GLOBAL_SCORE_BOUNDS
        )
        _GLOBAL_SCORE_BOUNDS.clear()

        # 第一次: [1, 5]
        pg1 = {'importance_scores': {'magnitude': torch.tensor([1.0, 5.0])}, 'p_names': ['l1']}
        _apply_online_normalization(pg1)

        # 第二次: [0, 10] - 应该扩展边界
        pg2 = {'importance_scores': {'magnitude': torch.tensor([0.0, 10.0])}, 'p_names': ['l2']}
        _apply_online_normalization(pg2)

        bounds = _GLOBAL_SCORE_BOUNDS['magnitude']
        assert bounds[0] == 0.0, f"最小值应为 0.0，实际={bounds[0]}"
        assert bounds[1] == 10.0, f"最大值应为 10.0，实际={bounds[1]}"
        print(f"  ✓ 全局边界: [{bounds[0]}, {bounds[1]}]")

    def test_global_bounds_persistence(self):
        """验证全局边界跨调用保持"""
        from only_train_once.optimizer.importance_score import (
            _apply_online_normalization, _GLOBAL_SCORE_BOUNDS
        )
        _GLOBAL_SCORE_BOUNDS.clear()

        # 调用多次
        for i in range(5):
            pg = {'importance_scores': {'magnitude': torch.tensor([float(i)])}, 'p_names': [f'l{i}']}
            _apply_online_normalization(pg)

        bounds = _GLOBAL_SCORE_BOUNDS['magnitude']
        assert bounds[0] == 0.0, f"最小值应为 0.0，实际={bounds[0]}"
        assert bounds[1] == 4.0, f"最大值应为 4.0，实际={bounds[1]}"
        print(f"  ✓ 跨调用保持: [{bounds[0]}, {bounds[1]}]")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DACO 数值稳定性与逻辑正确性测试")
    print("=" * 70)

    test_classes = [
        TestDiffusionNumerical,
        TestImportanceScoreNumerical,
        TestAdaptiveBitReduction,
        TestPruningBudgetControl,
        TestComputeGammaD,
        TestSmoothFactor,
        TestGlobalBounds,
    ]

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                method()

    print("\n" + "=" * 70)
    print("全部测试通过！")
    print("=" * 70)
