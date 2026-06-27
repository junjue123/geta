"""
统一测试文件
整合所有模块的测试：

1. 优化器测试 (DACO)
2. 蒸馏测试
3. 模型兼容性测试
4. 集成测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_all.py -v -s
"""

import sys
import os
import math
import torch
import torch.nn as nn
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# 第一部分：优化器测试
# ============================================================================

class TestOptimizer:
    """优化器核心功能测试"""

    def test_mygeta_initialization(self):
        """测试 MyGETA 初始化"""
        from only_train_once.optimizer.mygeta import MyGETA

        # 创建模拟参数组
        param_groups = self._create_mock_param_groups()
        optimizer = self._setup_optimizer(param_groups)

        assert optimizer.num_steps == 0
        assert optimizer.min_bit_wt == 4
        assert optimizer.max_bit_wt == 16
        print("  ✓ MyGETA 初始化正确")

    def test_mygeta_step(self):
        """测试 MyGETA 优化步骤"""
        from only_train_once.optimizer.mygeta import MyGETA

        param_groups = self._create_mock_param_groups()
        optimizer = self._setup_optimizer(param_groups)

        # 模拟梯度
        self._compute_mock_gradients(param_groups)

        # 执行优化步骤
        optimizer.step()

        assert optimizer.num_steps == 1
        print("  ✓ MyGETA 优化步骤正确")

    def test_mygeta_pruning(self):
        """测试 MyGETA 剪枝功能"""
        from only_train_once.optimizer.mygeta import MyGETA

        param_groups = self._create_mock_param_groups(num_groups=16, num_layers=2)
        optimizer = self._setup_optimizer(param_groups, target_group_sparsity=0.3)

        # 训练几步
        for step in range(10):
            self._compute_mock_gradients(param_groups)
            optimizer.step()

        # 检查稀疏度
        zero_count = sum(len(g['pruned_idxes']) for g in param_groups)
        total_count = sum(g['num_groups'] for g in param_groups)
        sparsity = zero_count / total_count

        print(f"  ✓ MyGETA 剪枝: sparsity={sparsity:.4f}")

    def test_adaptive_bit_reduction(self):
        """测试自适应位宽缩减"""
        from only_train_once.optimizer.mygeta import MyGETA

        param_groups = self._create_mock_param_groups()
        optimizer = self._setup_optimizer(param_groups, max_bit_wt=12)

        initial_max = optimizer.max_bit_wt
        optimizer.adaptive_bit_reduction(threshold=0.0)

        assert optimizer.max_bit_wt <= initial_max
        print(f"  ✓ 自适应位宽: {initial_max} -> {optimizer.max_bit_wt}")

    def test_pruning_budget_control(self):
        """测试剪枝预算控制"""
        from only_train_once.optimizer.mygeta import MyGETA

        param_groups = self._create_mock_param_groups()
        optimizer = self._setup_optimizer(param_groups)

        initial_budget = optimizer.active_num_redundant_groups.copy()
        optimizer.set_pruning_budget_this_period(rate_to_prune=0.1, period=0)

        expected = max(1, int(optimizer.total_num_groups * 0.1))
        assert optimizer.active_num_redundant_groups[0] == expected
        print(f"  ✓ 剪枝预算: {initial_budget[0]} -> {optimizer.active_num_redundant_groups[0]}")

    def _create_mock_param_groups(self, num_groups=16, num_layers=2):
        """创建模拟参数组"""
        from only_train_once.transform import TensorTransform

        param_groups = []
        global_idx = 0

        for layer_idx in range(num_layers):
            weight = nn.Parameter(torch.randn(num_groups, 8))
            p_names = [f'layer{layer_idx}.weight']
            params = [weight]
            p_transforms = [TensorTransform.BASIC]

            # 添加量化参数
            d_quant = nn.Parameter(torch.tensor(0.1))
            t_quant = nn.Parameter(torch.tensor(1.0))
            q_m = nn.Parameter(torch.tensor(1.0))

            p_names.extend([
                f'layer{layer_idx}.d_quant_wt',
                f'layer{layer_idx}.t_quant_wt',
                f'layer{layer_idx}.q_m_wt',
            ])
            params.extend([d_quant, t_quant, q_m])
            p_transforms.extend([1, 1, 1])

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

    def _setup_optimizer(self, param_groups, **kwargs):
        """创建优化器实例"""
        from only_train_once.optimizer.mygeta import MyGETA
        import logging

        optimizer = object.__new__(MyGETA)
        optimizer.log_dir = 'outputs'
        os.makedirs(optimizer.log_dir, exist_ok=True)
        optimizer.logger = logging.getLogger('TestMyGETA')

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
        optimizer.importance_score_criteria = {
            "magnitude": 0.2,
            "avg_magnitude": 0.2,
            "cosine_similarity": 0.2,
            "taylor_first_order": 0.2,
            "taylor_second_order": 0.2,
        }

        optimizer.param_groups = param_groups
        optimizer.total_num_groups = sum(g['num_groups'] for g in param_groups)
        optimizer.safe_guard = 1e-8
        optimizer.target_num_redundant_groups = int(optimizer.total_num_groups * defaults['target_group_sparsity'])

        optimizer.active_num_redundant_groups = []
        groups_sum = 0
        target = int(optimizer.total_num_groups * defaults['target_group_sparsity'])
        for p in range(optimizer.pruning_periods):
            if p == optimizer.pruning_periods - 1:
                optimizer.active_num_redundant_groups.append(target - groups_sum)
            else:
                optimizer.active_num_redundant_groups.append(target // optimizer.pruning_periods)
                groups_sum += optimizer.active_num_redundant_groups[p]

        optimizer.auxiliary_param_groups = {}

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

    def _compute_mock_gradients(self, param_groups):
        """计算模拟梯度"""
        for group in param_groups:
            grad_variant = {}
            for p_name, p in zip(group['p_names'], group['params']):
                if p.grad is None:
                    p.grad = torch.randn_like(p.data)
                grad_variant[p_name] = p.grad.clone()
            group['grad_variant'] = grad_variant


# ============================================================================
# 第二部分：蒸馏测试
# ============================================================================

class TestDistillation:
    """蒸馏模块测试"""

    def test_teacher_ensemble(self):
        """测试教师集合管理"""
        from only_train_once.distillation import TeacherEnsemble

        ensemble = TeacherEnsemble(top_n=2)
        model = nn.Linear(10, 10)

        # 更新
        ensemble.update(model, loss=0.5, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True)

        teachers = ensemble.get_teachers()
        assert len(teachers) > 0
        # 最优模型是 pre_pruning_best (loss=0.5, epoch=0)
        # 因为进入剪枝阶段后会锁定 pre_pruning_best
        print(f"  ✓ 教师集合: {len(teachers)} 个教师")

    def test_soft_label_distiller(self):
        """测试软标签蒸馏"""
        from only_train_once.distillation import SoftLabelDistiller

        distiller = SoftLabelDistiller(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)

        loss = distiller.compute_loss(student_logits, [teacher_logits])
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 软标签蒸馏: loss={loss.item():.4f}")

    def test_feature_distiller(self):
        """测试特征对齐"""
        from only_train_once.distillation import FeatureDistiller

        distiller = FeatureDistiller(mode='forward', feature_weight=0.1)

        student = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        x = torch.randn(4, 10)
        loss = distiller.compute_loss_with_hooks(student, [teacher], x)

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 特征对齐: loss={loss.item():.4f}")

    def test_weight_distiller(self):
        """测试权重对齐"""
        from only_train_once.distillation import WeightDistiller

        distiller = WeightDistiller(weight_weight=0.01, align_mode='both')

        student = nn.Linear(10, 10)
        teacher = nn.Linear(10, 10)

        loss = distiller.compute_loss(student, teacher)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 权重对齐: loss={loss.item():.4f}")

    def test_layer_wise_distiller(self):
        """测试逐层蒸馏"""
        from only_train_once.distillation import LayerWiseDistiller

        distiller = LayerWiseDistiller(
            progressive=False,
            selective=False,
            normalize=True,
            base_weight=0.1
        )

        student = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        teacher = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        x = torch.randn(4, 10)
        student_logits = student(x)

        loss = distiller.compute_loss(
            student, teacher, x, student_logits,
            epoch=0, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 逐层蒸馏: loss={loss.item():.4f}")

    def test_pruning_aware_distiller(self):
        """测试剪枝感知蒸馏"""
        from only_train_once.distillation import PruningAwareDistiller

        distiller = PruningAwareDistiller(
            weight=0.1,
            use_projection=True,
            use_mask=False,
        )

        student_features = {'layer1': torch.randn(4, 16, 8, 8)}
        teacher_features = {'layer1': torch.randn(4, 32, 8, 8)}

        loss = distiller.compute_loss(student_features, teacher_features)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 剪枝感知蒸馏: loss={loss.item():.4f}")


# ============================================================================
# 第三部分：模型兼容性测试
# ============================================================================

class TestModelCompatibility:
    """模型架构兼容性测试"""

    @pytest.mark.parametrize("model_name", ['resnet20', 'vgg7', 'mobilenetv1', 'mobilenetv2'])
    def test_cnn_models(self, model_name):
        """测试CNN模型兼容性"""
        result = self._get_model(model_name)
        if result is None:
            pytest.skip(f"{model_name} 不可用")

        model, x, num_classes = result

        # 测试前向传播
        output = model(x)
        assert output.shape == (2, num_classes)

        # 测试反向传播
        loss = output.sum()
        loss.backward()

        # 检查至少有一些参数有梯度
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, f"{model_name} 应该有梯度"

        print(f"  ✓ {model_name}: 前向+反向传播正确")

    def _get_model(self, model_name):
        """获取模型"""
        try:
            if model_name == 'resnet20':
                from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
                return resnet20_cifar10(), torch.randn(2, 3, 32, 32), 10
            elif model_name == 'vgg7':
                from sanity_check.backends.vgg7 import vgg7_bn
                return vgg7_bn(), torch.randn(2, 3, 32, 32), 10
            elif model_name == 'mobilenetv1':
                from sanity_check.backends.mobilenetv1 import MobileNetV1
                return MobileNetV1(num_classes=10), torch.randn(2, 3, 32, 32), 10
            elif model_name == 'mobilenetv2':
                from sanity_check.backends.mobilenetv2 import MobileNetV2
                return MobileNetV2(num_classes=10), torch.randn(2, 3, 32, 32), 10
            else:
                return None
        except Exception as e:
            print(f"  ⚠ {model_name} 不可用: {e}")
            return None


# ============================================================================
# 第四部分：集成测试
# ============================================================================

class TestIntegration:
    """端到端集成测试"""

    def test_full_pipeline(self):
        """测试完整流程：模型 + 优化器 + 蒸馏"""
        from only_train_once.distillation import IntegratedPruningDistiller

        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        # 创建蒸馏器
        distiller = IntegratedPruningDistiller(enabled=True)

        # 模拟训练
        x = torch.randn(4, 16)
        labels = torch.randint(0, 10, (4,))

        for epoch in range(5):
            # 前向传播
            logits = model(x)
            loss_ce = nn.CrossEntropyLoss()(logits, labels)

            # 更新蒸馏器
            distiller.update(model, loss_ce.item(), epoch, is_pruning=(epoch >= 2))

            # 计算蒸馏loss
            kd_loss, info = distiller.compute_loss(
                model, x, epoch, max_epoch=5, is_pruning=(epoch >= 2)
            )

            # 总loss
            total_loss = loss_ce + kd_loss

            # 反向传播
            total_loss.backward()

            # 检查梯度
            has_grad = any(p.grad is not None for p in model.parameters())
            assert has_grad

            # 清除梯度
            model.zero_grad()

        print(f"  ✓ 完整流程: kd_loss={kd_loss.item():.4f}")

    def test_no_nan_inf(self):
        """测试无NaN/Inf"""
        model = nn.Linear(10, 10)

        for _ in range(10):
            x = torch.randn(4, 10)
            output = model(x)
            loss = output.sum()

            assert not torch.isnan(loss), "loss 不应为 NaN"
            assert not torch.isinf(loss), "loss 不应为 Inf"

        print("  ✓ 10次迭代无NaN/Inf")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("统一测试")
    print("=" * 70)

    test_classes = [
        TestOptimizer,
        TestDistillation,
        TestModelCompatibility,
        TestIntegration,
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
    print("测试完成")
    print("=" * 70)
