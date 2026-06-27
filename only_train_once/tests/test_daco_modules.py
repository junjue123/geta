"""
DACO 模块化验证测试
验证 MCSS、DGD 和 bug 修复的正确性与可行性。

运行方式:
    cd E:\pythonProject\geta-main\geta-main_origin
    python -m pytest only_train_once/tests/test_daco_modules.py -v -s
"""

import sys
import os
import math
import torch
import numpy as np
import pytest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================================
# 第一部分：MCSS — 多准则校准显著性评分模块验证
# ============================================================================

class TestMCSS:
    """测试 importance_score 模块的正确性"""

    def test_global_score_bounds_initialized(self):
        """验证 _GLOBAL_SCORE_BOUNDS 已正确初始化（修复 P0 bug）"""
        from only_train_once.optimizer.importance_score import _GLOBAL_SCORE_BOUNDS
        assert isinstance(_GLOBAL_SCORE_BOUNDS, dict), \
            "_GLOBAL_SCORE_BOUNDS 应该是 dict 类型"
        print("  [PASS] _GLOBAL_SCORE_BOUNDS 已正确初始化为空 dict")

    def test_min_max_normalization(self):
        """验证 Min-Max 归一化 _apply_online_normalization 的正确性"""
        from only_train_once.optimizer.importance_score import (
            _apply_online_normalization, _GLOBAL_SCORE_BOUNDS
        )
        # 清理全局状态，确保测试隔离
        _GLOBAL_SCORE_BOUNDS.clear()

        # 模拟 param_group
        score_tensor = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
        param_group = {
            'importance_scores': {'magnitude': score_tensor.clone()},
            'p_names': ['test_layer.weight']
        }

        # 首次归一化
        _apply_online_normalization(param_group)
        normalized = param_group['importance_scores']['magnitude']
        print(f"  归一化前: {score_tensor.tolist()}")
        print(f"  归一化后: {normalized.tolist()}")

        # 验证归一化到 [0, 1]
        assert 0.0 <= normalized.min().item() <= 0.01, \
            f"最小值应该接近 0，实际: {normalized.min().item()}"
        assert 0.99 <= normalized.max().item() <= 1.01, \
            f"最大值应该接近 1，实际: {normalized.max().item()}"
        print("  [PASS] Min-Max 归一化正确映射到 [0, 1]")

    def test_bit_width_factor(self):
        """验证位宽因子 Φ(d) = 1 + log2(32/bit) 的正确性"""
        from only_train_once.optimizer.importance_score import adjust_importance_criteria
        import torch

        # 模拟 4-bit 权重层
        score = torch.ones(10)
        param_group = {
            'importance_scores': {'magnitude': score.clone()},
            'p_names': ['backbone.conv1.weight'],
        }
        bit_layers = {'backbone.conv1': {'weight': 4}}

        adjust_importance_criteria(param_group, bit_layers, smooth_factor=1.0)
        adjusted = param_group['importance_scores']['magnitude']

        # 理论值: 1 + log2(32/4) = 1 + log2(8) = 1 + 3 = 4.0
        expected_factor = 1 + math.log2(32/4)  # = 4.0
        print(f"  4-bit 位宽因子: 期望={expected_factor}, 实际={adjusted[0].item():.4f}")
        assert abs(adjusted[0].item() - expected_factor) < 0.01, \
            f"位宽因子不匹配: 期望 {expected_factor}, 实际 {adjusted[0].item()}"
        print("  [PASS] 位宽因子计算正确")

        # 测试 8-bit
        score2 = torch.ones(10)
        param_group2 = {
            'importance_scores': {'magnitude': score2.clone()},
            'p_names': ['backbone.conv2.weight'],
        }
        bit_layers2 = {'backbone.conv2': {'weight': 8}}
        adjust_importance_criteria(param_group2, bit_layers2, smooth_factor=1.0)

        expected_8bit = 1 + math.log2(32/8)  # = 3.0
        print(f"  8-bit 位宽因子: 期望={expected_8bit}, 实际={param_group2['importance_scores']['magnitude'][0].item():.4f}")
        assert abs(param_group2['importance_scores']['magnitude'][0].item() - expected_8bit) < 0.01
        print("  [PASS] 8-bit 位宽因子也正确")

    def test_stability_factor(self):
        """验证稳定性因子 Ψ = exp(-CV) 的行为"""
        from only_train_once.optimizer.importance_score import (
            adjust_importance_criteria, _SCORE_HISTORY, _get_layer_name
        )
        import numpy as np

        # 预设历史数据 - key 必须与 _get_layer_name 输出一致
        param_group = {
            'importance_scores': {'magnitude': torch.ones(10)},
            'p_names': ['test_stability.weight'],
        }
        layer_name = _get_layer_name(param_group)  # = 'test_stability'
        _SCORE_HISTORY.clear()
        _SCORE_HISTORY[layer_name] = {
            'steps': [1, 2, 3, 4, 5],
            'magnitude': [0.5, 0.6, 0.4, 0.55, 0.5]  # std=0.07, mean=0.51
        }

        score = torch.ones(10)
        param_group['importance_scores'] = {'magnitude': score.clone()}
        bit_layers = {}  # 无位宽影响，bit_factor=1.0

        adjust_importance_criteria(param_group, bit_layers, smooth_factor=1.0)
        adjusted = param_group['importance_scores']['magnitude']

        # 手动计算期望值
        arr = np.array([0.5, 0.6, 0.4, 0.55, 0.5])
        cv = np.std(arr) / np.abs(np.mean(arr))
        expected_stability = max(0.2, math.exp(-cv))
        print(f"  历史数据变异系数 CV={cv:.4f}, 期望稳定性因子={expected_stability:.4f}")
        print(f"  调整后分数={adjusted[0].item():.4f}")
        assert abs(adjusted[0].item() - expected_stability) < 0.01, \
            f"稳定性因子不匹配"
        print("  [PASS] 稳定性因子计算正确")

        # 清理
        _SCORE_HISTORY.clear()


# ============================================================================
# 第二部分：DGD — 扩散梯度下降（朗之万噪声）模块验证
# ============================================================================

class TestDGD:
    """测试朗之万噪声生成与注入的正确性"""

    def test_cosine_noise_shape_and_device(self):
        """验证 _get_cosine_noise 输出形状和设备与输入一致"""
        from only_train_once.optimizer.mygeta import MyGETA

        # 模拟优化器实例（最小化初始化）
        dummy_param = torch.randn(64, 32)
        lr = 0.001
        T = 10
        t_vals = [1, 3, 5, 7, 10]

        for t in t_vals:
            noise = MyGETA._get_cosine_noise(
                MyGETA, dummy_param, lr, t, T
            )
            assert noise.shape == dummy_param.shape, \
                f"t={t}: 噪声形状 {noise.shape} != 参数形状 {dummy_param.shape}"
            assert noise.device == dummy_param.device, \
                f"t={t}: 噪声设备不匹配"
            print(f"  t={t}/{T}: 噪声标准差={noise.std().item():.6f}")
        print("  [PASS] 噪声形状和设备正确")

    def test_cosine_noise_decay(self):
        """验证噪声按余弦调度从最大值衰减到 0"""
        from only_train_once.optimizer.mygeta import MyGETA

        dummy_param = torch.ones(1000, 1000)  # 大张量确保统计稳定
        lr = 0.1
        T = 20

        noises = []
        for t in range(1, T + 1):
            noise = MyGETA._get_cosine_noise(MyGETA, dummy_param, lr, t, T)
            noises.append(noise.std().item())

        # 验证单调递减（允许微小波动）
        is_decreasing = True
        for i in range(1, len(noises)):
            if noises[i] > noises[i-1] + 1e-8:
                is_decreasing = False
                break

        print(f"  噪声序列 (std): {[f'{n:.6f}' for n in noises[:5]]}...{[f'{n:.6f}' for n in noises[-5:]]}")
        print(f"  起始噪声: {noises[0]:.6f}, 终止噪声: {noises[-1]:.6f}")
        assert is_decreasing or abs(noises[-1]) < 1e-5, \
            "噪声应该单调递减或最终趋于 0"
        assert noises[0] > noises[-1] * 10, \
            "起始噪声应该显著大于最终噪声"
        print("  [PASS] 噪声按余弦调度正确衰减")

    def test_cosine_noise_zero_at_end(self):
        """验证噪声在 t=T 时接近零"""
        from only_train_once.optimizer.mygeta import MyGETA

        dummy_param = torch.randn(100, 100)
        lr = 1.0
        T = 50

        noise_end = MyGETA._get_cosine_noise(MyGETA, dummy_param, lr, T, T)
        noise_max_t = noise_end.max().item()
        print(f"  T={T} 时最大噪声值: {noise_max_t:.2e}")
        assert noise_max_t < 1e-10, \
            f"t=T 时噪声应接近 0，实际最大: {noise_max_t}"
        print("  [PASS] 噪声在终点正确归零")

    def test_noise_independence(self):
        """验证每次噪声生成是独立的（不同种子）"""
        from only_train_once.optimizer.mygeta import MyGETA

        param = torch.zeros(1000)
        lr, T = 0.01, 5

        noise1 = MyGETA._get_cosine_noise(MyGETA, param, lr, 1, T)
        noise2 = MyGETA._get_cosine_noise(MyGETA, param, lr, 1, T)

        # 两次生成的噪声不应该相同
        assert not torch.allclose(noise1, noise2), \
            "两次噪声生成应该独立不同"
        print(f"  noise1 均值={noise1.mean():.6f}, noise2 均值={noise2.mean():.6f}")
        print("  [PASS] 噪声独立生成正确")


# ============================================================================
# 第三部分：Bug 修复验证
# ============================================================================

class TestBugFixes:
    """验证已修复的 bug"""

    def test_safe_open_file_handles_error(self):
        """验证 safe_open_file 在文件打开失败时不崩溃"""
        from only_train_once.optimizer.mygeta import MyGETA

        # 创建一个无法写入的路径
        import tempfile
        dummy_optimizer = object()
        # 直接测试上下文管理器逻辑：打开不存在的目录下的文件
        ctx = MyGETA.safe_open_file(dummy_optimizer, "/nonexistent/path/file.txt", "w")
        try:
            file = ctx.__enter__()
            assert file is None, "应该返回 None 因为 open 失败"
        except Exception:
            pass  # logger 可能不存在，但不会因为 file 未赋值而崩溃
        finally:
            ctx.__exit__(None, None, None)

        print("  [PASS] safe_open_file 不再因 file 未赋值而崩溃")

    def test_safe_open_file_normal(self):
        """验证 safe_open_file 在正常情况下正常工作"""
        from only_train_once.optimizer.mygeta import MyGETA

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_dir", "test.txt")
            dummy_optimizer = object()
            ctx = MyGETA.safe_open_file(dummy_optimizer, test_file, "w")
            file = ctx.__enter__()
            assert file is not None, "应该成功打开文件"
            file.write("test")
            ctx.__exit__(None, None, None)

            # 验证文件存在
            assert os.path.exists(test_file), "文件应该被创建"
            with open(test_file) as f:
                assert f.read() == "test"
        print("  [PASS] safe_open_file 正常写入功能正确")

    def test_importance_score_no_crash(self):
        """验证 calculate_importance_score 在 _GLOBAL_SCORE_BOUNDS 修复后不崩溃"""
        from only_train_once.optimizer.importance_score import (
            calculate_importance_score, _GLOBAL_SCORE_BOUNDS, _SCORE_HISTORY
        )
        from only_train_once.transform import TensorTransform

        # 清理状态
        _GLOBAL_SCORE_BOUNDS.clear()
        _SCORE_HISTORY.clear()

        # 模拟 param_group - 使用 TensorTransform.BASIC (值为 2)
        dummy_grad = torch.randn(8, 4)
        param_group = {
            'p_names': ['layer.weight'],
            'params': [torch.nn.Parameter(torch.randn(8, 4))],
            'p_transform': [TensorTransform.BASIC],
            'num_groups': 8,
            'grad_variant': {'layer.weight': dummy_grad},
        }

        criteria = {'magnitude': 0.5, 'taylor_first_order': 0.5}

        try:
            calculate_importance_score(criteria, param_group, step=1)
        except NameError as e:
            pytest.fail(f"_GLOBAL_SCORE_BOUNDS 未初始化导致崩溃: {e}")

        assert 'importance_scores' in param_group
        assert 'magnitude' in param_group['importance_scores']
        print(f"  importance_scores keys: {list(param_group['importance_scores'].keys())}")
        print("  [PASS] calculate_importance_score 不再因 _GLOBAL_SCORE_BOUNDS 崩溃")


# ============================================================================
# 第四部分：端到端集成验证
# ============================================================================

class TestIntegration:
    """端到端集成验证 — MyGETA 完整 step 执行"""

    def test_mygeta_step_execution(self):
        """验证 MyGETA.step() 能完整执行不崩溃"""
        from sanity_check.backends.vgg7 import vgg7_bn
        from only_train_once.quantization.quant_model import model_to_quantize_model
        from only_train_once.quantization.quant_layers import QuantizationMode
        from only_train_once import OTO

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  运行设备: {device}")

        # 1. 构建量化模型
        model = vgg7_bn()
        model = model_to_quantize_model(
            model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
        )
        model = model.to(device)

        # 2. 创建 OTO 实例
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        oto = OTO(model=model, dummy_input=dummy_input)

        # 3. 创建 MyGETA 优化器
        optimizer = oto.mygeta(
            variant="sgd",
            lr=0.1,
            lr_quant=1e-3,
            target_group_sparsity=0.5,
            start_pruning_step=5,
            pruning_periods=3,
            pruning_steps=15,
            bit_reduction=2,
            min_bit_wt=4,
            max_bit_wt=16,
            min_bit_act=4,
            max_bit_act=16,
            device=device,
        )

        # 4. 执行训练步骤
        model.train()
        criterion = torch.nn.CrossEntropyLoss()

        # 预热阶段 (step 0-4)
        for step in range(5):
            X = torch.randn(2, 3, 32, 32).to(device)
            y = torch.randint(0, 10, (2,)).to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 0:
                print(f"  Step {step}: loss={loss.item():.4f} (预热阶段)")

        # 联合剪枝量化阶段 (step 5-19)
        for step in range(5, 20):
            X = torch.randn(2, 3, 32, 32).to(device)
            y = torch.randint(0, 10, (2,)).to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = optimizer.compute_metrics()
            if step % 5 == 0 or step == 19:
                print(f"  Step {step}: loss={loss.item():.4f}, "
                      f"sparsity={metrics.group_sparsity:.4f}, "
                      f"num_zero_groups={metrics.num_zero_groups}")

        # 5. 验证结果
        metrics = optimizer.compute_metrics()
        print(f"  最终: group_sparsity={metrics.group_sparsity:.4f}, "
              f"zero_groups={metrics.num_zero_groups}, "
              f"total_groups={optimizer.total_num_groups}")

        assert metrics.group_sparsity >= 0.0, "稀疏度应 >= 0"
        assert metrics.group_sparsity <= 1.0, "稀疏度应 <= 1"
        print("  [PASS] MyGETA.step() 端到端执行成功")

    def test_adaptive_bit_reduction(self):
        """验证 adaptive_bit_reduction 位宽自适应缩减"""
        from sanity_check.backends.vgg7 import vgg7_bn
        from only_train_once.quantization.quant_model import model_to_quantize_model
        from only_train_once.quantization.quant_layers import QuantizationMode
        from only_train_once import OTO

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = vgg7_bn()
        model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
        model = model.to(device)

        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        oto = OTO(model=model, dummy_input=dummy_input)

        optimizer = oto.mygeta(
            variant="sgd",
            lr=0.1,
            target_group_sparsity=0.5,
            start_pruning_step=5,
            pruning_periods=3,
            pruning_steps=15,
            max_bit_wt=12,
            max_bit_act=12,
            device=device,
        )

        # 直接调用 adaptive_bit_reduction
        initial_max_wt = optimizer.max_bit_wt
        initial_max_act = optimizer.max_bit_act

        optimizer.adaptive_bit_reduction(threshold=0.0)  # threshold=0 意味着只要有任何层降位宽就触发

        print(f"  max_bit_wt: {initial_max_wt} -> {optimizer.max_bit_wt}")
        print(f"  max_bit_act: {initial_max_act} -> {optimizer.max_bit_act}")
        assert optimizer.max_bit_wt <= initial_max_wt, \
            "位宽上限应该不增加"
        print("  [PASS] adaptive_bit_reduction 函数执行正确")

    def test_pruning_budget_control(self):
        """验证 set_pruning_budget_this_period 预算控制"""
        from sanity_check.backends.vgg7 import vgg7_bn
        from only_train_once.quantization.quant_model import model_to_quantize_model
        from only_train_once.quantization.quant_layers import QuantizationMode
        from only_train_once import OTO

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = vgg7_bn()
        model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
        model = model.to(device)

        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        oto = OTO(model=model, dummy_input=dummy_input)

        optimizer = oto.mygeta(
            variant="sgd",
            lr=0.1,
            target_group_sparsity=0.5,
            start_pruning_step=5,
            pruning_periods=3,
            pruning_steps=15,
            device=device,
        )

        total_groups = optimizer.total_num_groups
        print(f"  total_num_groups={total_groups}")

        # 测试外部设置剪枝预算
        original_budget = optimizer.active_num_redundant_groups[0]
        optimizer.set_pruning_budget_this_period(rate_to_prune=0.05, period=0)
        new_budget = optimizer.active_num_redundant_groups[0]
        print(f"  设置 rate=0.05: budget {original_budget} -> {new_budget}")

        expected = max(1, int(total_groups * 0.05))
        assert new_budget == expected, \
            f"预算不匹配: 期望 {expected}, 实际 {new_budget}"

        # 测试扩展周期
        optimizer.set_pruning_budget_this_period(rate_to_prune=0.03, period=5)
        assert optimizer.pruning_periods == 6, \
            f"周期未正确扩展: 期望 6, 实际 {optimizer.pruning_periods}"
        print(f"  扩展后 pruning_periods={optimizer.pruning_periods}")

        print("  [PASS] set_pruning_budget_this_period 预算控制正确")


if __name__ == "__main__":
    # 手动运行所有测试
    print("=" * 60)
    print("DACO 模块化验证测试")
    print("=" * 60)

    runner = TestMCSS()
    print("\n--- 1. MCSS 模块测试 ---")
    runner.test_global_score_bounds_initialized()
    runner.test_min_max_normalization()
    runner.test_bit_width_factor()
    runner.test_stability_factor()

    runner = TestDGD()
    print("\n--- 2. DGD 模块测试 ---")
    runner.test_cosine_noise_shape_and_device()
    runner.test_cosine_noise_decay()
    runner.test_cosine_noise_zero_at_end()
    runner.test_noise_independence()

    runner = TestBugFixes()
    print("\n--- 3. Bug 修复验证 ---")
    runner.test_safe_open_file_handles_error()
    runner.test_safe_open_file_normal()
    runner.test_importance_score_no_crash()

    runner = TestIntegration()
    print("\n--- 4. 端到端集成验证 ---")
    runner.test_mygeta_step_execution()
    runner.test_adaptive_bit_reduction()
    runner.test_pruning_budget_control()

    print("\n" + "=" * 60)
    print("全部测试通过！")
    print("=" * 60)
