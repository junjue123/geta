"""
蒸馏核心模块测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_distillation_core.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.distillation import (
    TeacherEnsemble,
    SoftLabelDistiller,
    FeatureDistiller,
    WeightDistiller,
    Distiller,
)


# ============================================================================
# 测试模型
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# TeacherEnsemble 测试
# ============================================================================

class TestTeacherEnsemble:
    """测试教师集合管理"""

    def test_initialization(self):
        """测试初始化"""
        ensemble = TeacherEnsemble(top_n=3)
        assert len(ensemble) == 0
        assert ensemble.pre_pruning_best is None
        print("  ✓ 初始化正确")

    def test_update_pre_pruning(self):
        """测试剪枝前更新"""
        ensemble = TeacherEnsemble(top_n=3)
        model = nn.Linear(10, 10)

        ensemble.update(model, loss=1.0, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.5, epoch=1, is_pruning=False)

        assert ensemble.pre_pruning_best is not None
        assert ensemble.pre_pruning_best['loss'] == 0.5
        print("  ✓ 剪枝前最优正确")

    def test_update_pruning(self):
        """测试剪枝中更新"""
        ensemble = TeacherEnsemble(top_n=3)
        model = nn.Linear(10, 10)

        # 进入剪枝阶段
        ensemble.update(model, loss=0.5, epoch=0, is_pruning=True)
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True)
        ensemble.update(model, loss=0.8, epoch=2, is_pruning=True)

        teachers = ensemble.get_teachers()
        assert len(teachers) > 0
        print(f"  ✓ 剪枝中更新: {len(teachers)} 个教师")

    def test_get_teachers(self):
        """测试获取教师列表"""
        ensemble = TeacherEnsemble(top_n=2)
        model = nn.Linear(10, 10)

        ensemble.update(model, loss=0.5, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True)

        teachers = ensemble.get_teachers()
        assert len(teachers) >= 2
        print(f"  ✓ 获取教师: {len(teachers)} 个")

    def test_state_dict(self):
        """测试状态保存恢复"""
        ensemble = TeacherEnsemble(top_n=2)
        model = nn.Linear(10, 10)

        ensemble.update(model, loss=0.5, epoch=0)
        state = ensemble.state_dict()

        ensemble2 = TeacherEnsemble()
        ensemble2.load_state_dict(state)

        assert len(ensemble2) == len(ensemble)
        print("  ✓ 状态保存恢复正确")


# ============================================================================
# SoftLabelDistiller 测试
# ============================================================================

class TestSoftLabelDistiller:
    """测试软标签蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        distiller = SoftLabelDistiller(temperature=4.0, alpha=0.5)
        assert distiller.temperature == 4.0
        assert distiller.alpha == 0.5
        print("  ✓ 初始化正确")

    def test_compute_loss(self):
        """测试loss计算"""
        distiller = SoftLabelDistiller(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        loss = distiller.compute_loss(student_logits, [teacher_logits], labels)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 软标签loss: {loss.item():.4f}")

    def test_forward_reverse_mode(self):
        """测试前向/反向模式"""
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)

        for mode in ['forward', 'reverse']:
            distiller = SoftLabelDistiller(mode=mode)
            loss = distiller.compute_loss(student_logits, [teacher_logits])
            assert loss.item() >= 0
            print(f"  ✓ {mode} 模式: {loss.item():.4f}")

    def test_temperature_annealing(self):
        """测试温度退火"""
        distiller = SoftLabelDistiller(temperature=10.0)

        temp_0 = distiller.get_temperature(0, 100)
        temp_50 = distiller.get_temperature(50, 100)
        temp_100 = distiller.get_temperature(100, 100)

        assert temp_0 >= temp_50 >= temp_100
        print(f"  ✓ 温度退火: {temp_0:.1f} -> {temp_50:.1f} -> {temp_100:.1f}")


# ============================================================================
# FeatureDistiller 测试
# ============================================================================

class TestFeatureDistiller:
    """测试特征对齐"""

    def test_initialization(self):
        """测试初始化"""
        distiller = FeatureDistiller(mode='forward', weight=0.1)
        assert distiller.mode == 'forward'
        assert distiller.weight == 0.1
        print("  ✓ 初始化正确")

    def test_compute_loss(self):
        """测试loss计算"""
        distiller = FeatureDistiller(
            mode='forward',
            weight=0.1,
            progressive=False,
            adaptive_temp=False
        )

        student = SimpleCNN()
        teacher = SimpleCNN()
        x = torch.randn(2, 3, 8, 8)

        loss = distiller.compute_loss(student, teacher, x, epoch=0, max_epoch=100)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 特征对齐loss: {loss.item():.4f}")

    def test_progressive_layers(self):
        """测试渐进式层解锁"""
        distiller = FeatureDistiller(progressive=True)

        all_layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6']

        # 早期：只解锁浅层
        early_layers = distiller._get_progressive_layers(all_layers, epoch=0, max_epoch=100)
        assert len(early_layers) <= 2

        # 后期：解锁所有层
        late_layers = distiller._get_progressive_layers(all_layers, epoch=90, max_epoch=100)
        assert len(late_layers) == 6

        print(f"  ✓ 渐进式: 早期={len(early_layers)}层, 后期={len(late_layers)}层")

    def test_adaptive_temperature(self):
        """测试自适应温度"""
        distiller = FeatureDistiller(adaptive_temp=True)

        # 浅层低温
        temp_0 = distiller._get_adaptive_temperature(0, 3, base_temp=4.0)
        # 深层高温
        temp_2 = distiller._get_adaptive_temperature(2, 3, base_temp=4.0)

        assert temp_0 < temp_2
        print(f"  ✓ 自适应温度: 浅层={temp_0:.1f}, 深层={temp_2:.1f}")

    def test_projection(self):
        """测试投影头"""
        distiller = FeatureDistiller(use_projection=True)

        # 测试不同维度
        proj = distiller._get_projection('layer1', 16, 32, torch.device('cpu'))
        assert proj.weight.shape == (32, 16)
        print("  ✓ 投影头正确")


# ============================================================================
# WeightDistiller 测试
# ============================================================================

class TestWeightDistiller:
    """测试权重对齐"""

    def test_initialization(self):
        """测试初始化"""
        distiller = WeightDistiller(weight=0.01, align_mode='both')
        assert distiller.weight == 0.01
        assert distiller.align_mode == 'both'
        print("  ✓ 初始化正确")

    def test_compute_svd(self):
        """测试SVD计算"""
        # 使用完整SVD
        distiller = WeightDistiller(use_lowrank=False)

        weight = torch.randn(32, 16)
        sv = distiller._compute_svd(weight)

        assert sv is not None
        assert len(sv) == min(weight.shape)
        print(f"  ✓ SVD计算: {len(sv)} 个奇异值")

    def test_compute_loss(self):
        """测试loss计算"""
        distiller = WeightDistiller(weight=0.01, align_mode='both')

        student = SimpleCNN()
        teacher = SimpleCNN()

        loss = distiller.compute_loss(student, teacher)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 权重对齐loss: {loss.item():.4f}")

    def test_align_modes(self):
        """测试不同对齐模式"""
        student = SimpleCNN()
        teacher = SimpleCNN()

        for mode in ['max', 'avg', 'both']:
            distiller = WeightDistiller(align_mode=mode)
            loss = distiller.compute_loss(student, teacher)
            assert loss.item() >= 0
            print(f"  ✓ {mode} 模式: {loss.item():.4f}")

    def test_svd_statistics(self):
        """测试奇异值统计"""
        distiller = WeightDistiller()
        model = SimpleCNN()

        stats = distiller.get_svd_statistics(model)
        assert len(stats) > 0
        print(f"  ✓ 奇异值统计: {len(stats)} 层")


# ============================================================================
# Distiller 统一接口测试
# ============================================================================

class TestDistiller:
    """测试统一蒸馏器"""

    def test_initialization(self):
        """测试初始化"""
        distiller = Distiller(
            temperature=4.0,
            alpha=0.5,
            feature_weight=0.1,
            weight_weight=0.01
        )
        assert distiller.soft_label is not None
        assert distiller.feature is not None
        assert distiller.weight is not None
        assert distiller.teacher_ensemble is not None
        print("  ✓ 初始化正确")

    def test_update_teacher(self):
        """测试更新教师"""
        distiller = Distiller()
        model = SimpleCNN()

        distiller.update_teacher(model, loss=0.5, epoch=0, is_pruning=False)
        distiller.update_teacher(model, loss=0.3, epoch=1, is_pruning=True)

        teachers = distiller.teacher_ensemble.get_teachers()
        assert len(teachers) > 0
        print(f"  ✓ 更新教师: {len(teachers)} 个")

    def test_compute_loss(self):
        """测试计算loss"""
        distiller = Distiller(
            temperature=4.0,
            alpha=0.5,
            feature_weight=0.1,
            weight_weight=0.01,
            progressive=False,
            adaptive_temp=False
        )

        student = SimpleCNN()
        teacher = SimpleCNN()

        # 更新教师
        distiller.update_teacher(teacher, loss=0.5, epoch=0)

        # 计算loss
        x = torch.randn(2, 3, 8, 8)
        student_logits = student(x)
        labels = torch.randint(0, 10, (2,))

        loss, loss_dict = distiller.compute_loss(
            student, x, student_logits, labels,
            epoch=0, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert 'soft_label' in loss_dict
        assert 'feature' in loss_dict
        assert 'weight' in loss_dict
        print(f"  ✓ 统一loss: {loss.item():.4f}, 组成: {loss_dict}")

    def test_backward_pass(self):
        """测试反向传播"""
        distiller = Distiller(
            feature_weight=0.1,
            weight_weight=0.01,
            progressive=False,
            adaptive_temp=False
        )

        student = SimpleCNN()
        teacher = SimpleCNN()

        distiller.update_teacher(teacher, loss=0.5, epoch=0)

        x = torch.randn(2, 3, 8, 8)
        student_logits = student(x)

        loss, _ = distiller.compute_loss(
            student, x, student_logits,
            labels=torch.randint(0, 10, (2,)),
            epoch=0, max_epoch=100
        )

        loss.backward()

        has_grad = any(p.grad is not None for p in student.parameters())
        assert has_grad
        print("  ✓ 反向传播正确")

    def test_no_nan_inf(self):
        """测试无NaN/Inf"""
        distiller = Distiller(
            progressive=False,
            adaptive_temp=False
        )

        student = SimpleCNN()
        teacher = SimpleCNN()

        distiller.update_teacher(teacher, loss=0.5, epoch=0)

        for _ in range(10):
            x = torch.randn(2, 3, 8, 8)
            student_logits = student(x)

            loss, _ = distiller.compute_loss(
                student, x, student_logits,
                labels=torch.randint(0, 10, (2,)),
                epoch=0, max_epoch=100
            )

            assert not torch.isnan(loss), "loss 不应为 NaN"
            assert not torch.isinf(loss), "loss 不应为 Inf"

        print("  ✓ 10次迭代无NaN/Inf")


if __name__ == "__main__":
    print("=" * 70)
    print("蒸馏核心模块测试")
    print("=" * 70)

    test_classes = [
        TestTeacherEnsemble,
        TestSoftLabelDistiller,
        TestFeatureDistiller,
        TestWeightDistiller,
        TestDistiller,
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
