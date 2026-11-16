"""
优化蒸馏模块测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_optimized_distiller.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.distillation import (
    OnlineDistiller,
    SelfDistiller,
    FeatureReweighter,
    DistributionMatcher,
    ArchitectureAwareDistiller,
    CachedDistiller,
    OptimizedDistiller,
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
# OnlineDistiller 测试
# ============================================================================

class TestOnlineDistiller:
    """测试在线蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        distiller = OnlineDistiller(alpha=0.5, beta=0.1, temperature=4.0)
        assert distiller.alpha == 0.5
        assert distiller.beta == 0.1
        assert distiller.temperature == 4.0
        print("  ✓ 初始化正确")

    def test_compute_mutual_loss(self):
        """测试互相蒸馏loss"""
        distiller = OnlineDistiller()

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        s_loss, t_loss = distiller.compute_mutual_loss(
            student_logits, teacher_logits, labels
        )

        assert s_loss.dim() == 0
        assert t_loss.dim() == 0
        assert s_loss.item() >= 0
        assert t_loss.item() >= 0
        print(f"  ✓ 互相蒸馏: student={s_loss.item():.4f}, teacher={t_loss.item():.4f}")


# ============================================================================
# SelfDistiller 测试
# ============================================================================

class TestSelfDistiller:
    """测试自蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        distiller = SelfDistiller(weight=0.1)
        assert distiller.weight == 0.1
        print("  ✓ 初始化正确")

    def test_compute_loss(self):
        """测试自蒸馏loss"""
        distiller = SelfDistiller(weight=0.1)

        # 模拟多层特征
        features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
            'layer3': torch.randn(4, 64, 2, 2),
        }
        layer_order = ['layer1', 'layer2', 'layer3']

        loss = distiller.compute_loss(features, layer_order)

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 自蒸馏loss: {loss.item():.4f}")

    def test_different_dimensions(self):
        """测试不同维度的层"""
        distiller = SelfDistiller(weight=0.1)

        features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
        }
        layer_order = ['layer1', 'layer2']

        loss = distiller.compute_loss(features, layer_order)

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 不同维度自蒸馏: {loss.item():.4f}")

    def test_single_layer(self):
        """测试单层（应返回0）"""
        distiller = SelfDistiller()

        features = {'layer1': torch.randn(4, 16, 8, 8)}
        layer_order = ['layer1']

        loss = distiller.compute_loss(features, layer_order)
        assert loss.item() == 0.0
        print("  ✓ 单层返回0")


# ============================================================================
# FeatureReweighter 测试
# ============================================================================

class TestFeatureReweighter:
    """测试特征重加权"""

    def test_initialization(self):
        """测试初始化"""
        reweighter = FeatureReweighter(weight=0.1, attention_type='spatial')
        assert reweighter.weight == 0.1
        assert reweighter.attention_type == 'spatial'
        print("  ✓ 初始化正确")

    def test_compute_attention_spatial(self):
        """测试空间注意力"""
        reweighter = FeatureReweighter(attention_type='spatial')

        features = torch.randn(4, 16, 8, 8)
        attention = reweighter._compute_attention(features)

        assert attention.shape == (4, 1, 8, 8)
        assert attention.min() >= 0
        assert attention.max() <= 1
        print(f"  ✓ 空间注意力: shape={attention.shape}")

    def test_compute_attention_channel(self):
        """测试通道注意力"""
        reweighter = FeatureReweighter(attention_type='channel')

        features = torch.randn(4, 16, 8, 8)
        attention = reweighter._compute_attention(features)

        assert attention.shape == (4, 16, 1, 1)
        print(f"  ✓ 通道注意力: shape={attention.shape}")

    def test_compute_loss(self):
        """测试重加权loss"""
        reweighter = FeatureReweighter(weight=0.1, attention_type='spatial')

        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }

        loss = reweighter.compute_loss(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 重加权loss: {loss.item():.4f}")


# ============================================================================
# DistributionMatcher 测试
# ============================================================================

class TestDistributionMatcher:
    """测试分布匹配"""

    def test_initialization(self):
        """测试初始化"""
        matcher = DistributionMatcher(weight=0.1, kernel='rbf', bandwidth=1.0)
        assert matcher.weight == 0.1
        assert matcher.kernel == 'rbf'
        assert matcher.bandwidth == 1.0
        print("  ✓ 初始化正确")

    def test_rbf_kernel(self):
        """测试RBF核"""
        matcher = DistributionMatcher(kernel='rbf')

        x = torch.randn(4, 10)
        y = torch.randn(4, 10)

        kernel = matcher._rbf_kernel(x, y)

        assert kernel.shape == (4, 4)
        assert kernel.min() >= 0
        assert kernel.max() <= 1
        print(f"  ✓ RBF核: shape={kernel.shape}")

    def test_compute_mmd(self):
        """测试MMD计算"""
        matcher = DistributionMatcher(kernel='rbf')

        x = torch.randn(10, 16)
        y = torch.randn(10, 16)

        mmd = matcher._compute_mmd(x, y)

        assert mmd.dim() == 0
        assert mmd.item() >= 0
        print(f"  ✓ MMD: {mmd.item():.4f}")

    def test_compute_loss(self):
        """测试分布匹配loss"""
        matcher = DistributionMatcher(weight=0.1)

        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }

        loss = matcher.compute_loss(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 分布匹配loss: {loss.item():.4f}")


# ============================================================================
# ArchitectureAwareDistiller 测试
# ============================================================================

class TestArchitectureAwareDistiller:
    """测试架构感知蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        distiller = ArchitectureAwareDistiller(default_weight=0.1)
        assert distiller.default_weight == 0.1
        print("  ✓ 初始化正确")

    def test_detect_architecture(self):
        """测试架构检测"""
        distiller = ArchitectureAwareDistiller()

        # CNN
        cnn = SimpleCNN()
        arch = distiller.detect_architecture(cnn)
        assert arch == 'cnn'
        print(f"  ✓ CNN检测: {arch}")

    def test_compute_loss_cnn(self):
        """测试CNN架构loss"""
        distiller = ArchitectureAwareDistiller()

        student = SimpleCNN()
        teacher = SimpleCNN()

        student_features = {
            'features.0': torch.randn(4, 16, 8, 8),
            'features.4': torch.randn(4, 32, 4, 4),
        }
        teacher_features = {
            'features.0': torch.randn(4, 16, 8, 8),
            'features.4': torch.randn(4, 32, 4, 4),
        }

        loss = distiller.compute_loss(
            student, teacher, student_features, teacher_features
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ CNN架构loss: {loss.item():.4f}")


# ============================================================================
# CachedDistiller 测试
# ============================================================================

class TestCachedDistiller:
    """测试缓存优化"""

    def test_initialization(self):
        """测试初始化"""
        distiller = CachedDistiller(max_cache_size=10, cache_mode='epoch')
        assert distiller.max_cache_size == 10
        assert distiller.cache_mode == 'epoch'
        print("  ✓ 初始化正确")

    def test_get_cached_features(self):
        """测试缓存特征"""
        distiller = CachedDistiller(max_cache_size=5, cache_mode='epoch')

        # 第一次计算
        features1 = distiller.get_cached_features(
            'epoch_0',
            lambda: {'layer1': torch.randn(4, 16, 8, 8)}
        )

        # 第二次获取（应使用缓存）
        features2 = distiller.get_cached_features(
            'epoch_0',
            lambda: {'layer1': torch.randn(4, 16, 8, 8)}
        )

        # 应该是同一个对象
        assert features1 is features2
        print("  ✓ 缓存命中")

    def test_cache_size_limit(self):
        """测试缓存大小限制"""
        distiller = CachedDistiller(max_cache_size=3, cache_mode='epoch')

        # 添加5个缓存
        for i in range(5):
            distiller.get_cached_features(
                f'epoch_{i}',
                lambda: {'layer1': torch.randn(4, 16, 8, 8)}
            )

        stats = distiller.get_cache_stats()
        assert stats['feature_cache_size'] == 3  # 只保留最近3个
        print(f"  ✓ 缓存大小限制: {stats}")

    def test_clear_cache(self):
        """测试清除缓存"""
        distiller = CachedDistiller()

        distiller.get_cached_features(
            'epoch_0',
            lambda: {'layer1': torch.randn(4, 16, 8, 8)}
        )

        distiller.clear_cache()

        stats = distiller.get_cache_stats()
        assert stats['feature_cache_size'] == 0
        print("  ✓ 清除缓存正确")

    def test_cache_disabled(self):
        """测试禁用缓存"""
        distiller = CachedDistiller(cache_mode='none')

        features1 = distiller.get_cached_features(
            'epoch_0',
            lambda: {'layer1': torch.randn(4, 16, 8, 8)}
        )
        features2 = distiller.get_cached_features(
            'epoch_0',
            lambda: {'layer1': torch.randn(4, 16, 8, 8)}
        )

        # 禁用缓存时应该是不同对象
        assert features1 is not features2
        print("  ✓ 禁用缓存正确")


# ============================================================================
# OptimizedDistiller 测试
# ============================================================================

class TestOptimizedDistiller:
    """测试组合优化蒸馏器"""

    def test_initialization(self):
        """测试初始化"""
        distiller = OptimizedDistiller(
            use_online=False,
            use_self=True,
            use_reweight=True,
            use_distribution=False,
            use_architecture=True,
            use_cache=True,
        )
        assert distiller.self_distiller is not None
        assert distiller.reweighter is not None
        assert distiller.architecture is not None
        assert distiller.cache is not None
        print("  ✓ 初始化正确")

    def test_compute_loss(self):
        """测试组合loss"""
        distiller = OptimizedDistiller(
            use_online=False,
            use_self=True,
            use_reweight=True,
            use_distribution=False,
            use_architecture=True,
            use_cache=False,
        )

        student = SimpleCNN()
        teacher = SimpleCNN()

        x = torch.randn(4, 3, 8, 8)
        student_logits = student(x)
        teacher_logits = teacher(x)

        student_features = {
            'features.0': torch.randn(4, 16, 8, 8),
            'features.4': torch.randn(4, 32, 4, 4),
        }
        teacher_features = {
            'features.0': torch.randn(4, 16, 8, 8),
            'features.4': torch.randn(4, 32, 4, 4),
        }

        labels = torch.randint(0, 10, (4,))

        loss, loss_dict = distiller.compute_loss(
            student, teacher,
            student_features, teacher_features,
            student_logits, teacher_logits,
            labels=labels,
            layer_order=['features.0', 'features.4']
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert 'total' in loss_dict
        print(f"  ✓ 组合loss: {loss.item():.4f}, 组成: {loss_dict}")

    def test_backward_pass(self):
        """测试反向传播"""
        distiller = OptimizedDistiller(
            use_online=False,
            use_self=True,
            use_reweight=True,
            use_architecture=True,
            use_cache=False,
        )

        student = SimpleCNN()
        teacher = SimpleCNN()

        x = torch.randn(4, 3, 8, 8)
        student_logits = student(x)
        teacher_logits = teacher(x)

        # 使用模型前向传播获取特征（确保梯度连接）
        student_features = {}
        teacher_features = {}

        def make_hook(d, name):
            def hook(module, input, output):
                d[name] = output
            return hook

        # 注册钩子
        s_hooks = []
        t_hooks = []
        for name, module in student.named_modules():
            if isinstance(module, nn.Conv2d):
                s_hooks.append(module.register_forward_hook(make_hook(student_features, name)))
        for name, module in teacher.named_modules():
            if isinstance(module, nn.Conv2d):
                t_hooks.append(module.register_forward_hook(make_hook(teacher_features, name)))

        # 前向传播
        student_logits = student(x)
        with torch.no_grad():
            teacher_logits = teacher(x)

        # 清理钩子
        for h in s_hooks + t_hooks:
            h.remove()

        loss, _ = distiller.compute_loss(
            student, teacher,
            student_features, teacher_features,
            student_logits, teacher_logits,
            labels=torch.randint(0, 10, (4,)),
            layer_order=list(student_features.keys())
        )

        loss.backward()

        has_grad = any(p.grad is not None for p in student.parameters())
        assert has_grad
        print("  ✓ 反向传播正确")

    def test_no_nan_inf(self):
        """测试无NaN/Inf"""
        distiller = OptimizedDistiller(
            use_online=False,
            use_self=True,
            use_reweight=True,
            use_architecture=True,
            use_cache=False,
        )

        student = SimpleCNN()
        teacher = SimpleCNN()

        for _ in range(10):
            x = torch.randn(4, 3, 8, 8)
            student_logits = student(x)
            teacher_logits = teacher(x)

            student_features = {
                'features.0': torch.randn(4, 16, 8, 8),
            }
            teacher_features = {
                'features.0': torch.randn(4, 16, 8, 8),
            }

            loss, _ = distiller.compute_loss(
                student, teacher,
                student_features, teacher_features,
                student_logits, teacher_logits,
                labels=torch.randint(0, 10, (4,)),
                layer_order=['features.0']
            )

            assert not torch.isnan(loss), "loss 不应为 NaN"
            assert not torch.isinf(loss), "loss 不应为 Inf"

        print("  ✓ 10次迭代无NaN/Inf")


if __name__ == "__main__":
    print("=" * 70)
    print("优化蒸馏模块测试")
    print("=" * 70)

    test_classes = [
        TestOnlineDistiller,
        TestSelfDistiller,
        TestFeatureReweighter,
        TestDistributionMatcher,
        TestArchitectureAwareDistiller,
        TestCachedDistiller,
        TestOptimizedDistiller,
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
