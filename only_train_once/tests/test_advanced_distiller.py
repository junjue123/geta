"""
高级蒸馏技术测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_advanced_distiller.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.distillation import (
    DynamicLayerWeighting,
    AttentionDistiller,
    TemperatureAnnealer,
    SampleCurriculum,
    AdapterDistiller,
    AdvancedDistiller,
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
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================================================================
# DynamicLayerWeighting 测试
# ============================================================================

class TestDynamicLayerWeighting:
    """测试动态层权重"""

    def test_initialization(self):
        """测试初始化"""
        dlw = DynamicLayerWeighting(temperature=5.0, update_freq=10, momentum=0.9)
        assert dlw.temperature == 5.0
        assert dlw.update_freq == 10
        assert dlw.momentum == 0.9
        print("  ✓ 初始化正确")

    def test_compute_weights(self):
        """测试权重计算"""
        dlw = DynamicLayerWeighting(temperature=5.0, update_freq=1)

        # 模拟特征
        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
        }

        weights = dlw.compute_weights(student_features, teacher_features)

        assert len(weights) > 0
        assert all(0 <= w <= 1 for w in weights.values())
        print(f"  ✓ 权重计算: {weights}")

    def test_update_frequency(self):
        """测试更新频率"""
        dlw = DynamicLayerWeighting(update_freq=5)

        student_features = {'layer1': torch.randn(4, 16, 8, 8)}
        teacher_features = {'layer1': torch.randn(4, 16, 8, 8)}

        # 前4步不更新
        for _ in range(4):
            weights = dlw.compute_weights(student_features, teacher_features)
            assert len(weights) == 0 or dlw._step % 5 != 0

        # 第5步更新
        weights = dlw.compute_weights(student_features, teacher_features)
        assert len(weights) > 0
        print("  ✓ 更新频率正确")

    def test_momentum(self):
        """测试动量更新"""
        dlw = DynamicLayerWeighting(update_freq=1, momentum=0.5)

        student_features = {'layer1': torch.randn(4, 16, 8, 8)}
        teacher_features = {'layer1': torch.randn(4, 16, 8, 8)}

        # 多次更新
        for _ in range(5):
            weights = dlw.compute_weights(student_features, teacher_features)

        # 权重应该稳定
        assert len(weights) > 0
        print(f"  ✓ 动量更新: {weights}")


# ============================================================================
# AttentionDistiller 测试
# ============================================================================

class TestAttentionDistiller:
    """测试注意力对齐"""

    def test_initialization(self):
        """测试初始化"""
        ad = AttentionDistiller(weight=0.1, normalize=True)
        assert ad.weight == 0.1
        assert ad.normalize is True
        print("  ✓ 初始化正确")

    def test_compute_attention_4d(self):
        """测试4D特征图的注意力计算"""
        ad = AttentionDistiller()

        feature_map = torch.randn(4, 16, 8, 8)
        attention = ad._compute_attention(feature_map)

        assert attention.shape == (4, 1, 8, 8)
        assert attention.min() >= 0
        assert attention.max() <= 1
        print(f"  ✓ 4D注意力: shape={attention.shape}")

    def test_compute_attention_2d(self):
        """测试2D特征的注意力计算"""
        ad = AttentionDistiller()

        feature_map = torch.randn(4, 64)
        attention = ad._compute_attention(feature_map)

        assert attention.shape == (4, 1)
        print(f"  ✓ 2D注意力: shape={attention.shape}")

    def test_compute_loss(self):
        """测试loss计算"""
        ad = AttentionDistiller(weight=0.1)

        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
        }

        loss = ad.compute_loss(student_features, teacher_features)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 注意力loss: {loss.item():.4f}")

    def test_normalize_disabled(self):
        """测试禁用归一化"""
        ad = AttentionDistiller(normalize=False)

        feature_map = torch.randn(4, 16, 8, 8)
        attention = ad._compute_attention(feature_map)

        # 不归一化时值可能超出 [0, 1]
        print(f"  ✓ 禁用归一化: min={attention.min():.4f}, max={attention.max():.4f}")


# ============================================================================
# TemperatureAnnealer 测试
# ============================================================================

class TestTemperatureAnnealer:
    """测试温度退火"""

    def test_initialization(self):
        """测试初始化"""
        ta = TemperatureAnnealer(init_temp=10.0, min_temp=2.0, anneal_type='cosine')
        assert ta.init_temp == 10.0
        assert ta.min_temp == 2.0
        assert ta.anneal_type == 'cosine'
        print("  ✓ 初始化正确")

    def test_linear_anneal(self):
        """测试线性退火"""
        ta = TemperatureAnnealer(init_temp=10.0, min_temp=2.0, anneal_type='linear')

        temp_0 = ta.get_temperature(0, 100)
        temp_50 = ta.get_temperature(50, 100)
        temp_100 = ta.get_temperature(100, 100)

        assert temp_0 == 10.0
        assert abs(temp_50 - 6.0) < 0.1
        assert temp_100 == 2.0
        print(f"  ✓ 线性退火: {temp_0:.1f} -> {temp_50:.1f} -> {temp_100:.1f}")

    def test_cosine_anneal(self):
        """测试余弦退火"""
        ta = TemperatureAnnealer(init_temp=10.0, min_temp=2.0, anneal_type='cosine')

        temp_0 = ta.get_temperature(0, 100)
        temp_50 = ta.get_temperature(50, 100)
        temp_100 = ta.get_temperature(100, 100)

        assert abs(temp_0 - 10.0) < 0.01
        assert 5.0 <= temp_50 <= 7.0  # 余弦中间值
        assert abs(temp_100 - 2.0) < 0.01
        print(f"  ✓ 余弦退火: {temp_0:.1f} -> {temp_50:.1f} -> {temp_100:.1f}")

    def test_exponential_anneal(self):
        """测试指数退火"""
        ta = TemperatureAnnealer(init_temp=10.0, min_temp=2.0, anneal_type='exponential')

        temp_0 = ta.get_temperature(0, 100)
        temp_50 = ta.get_temperature(50, 100)
        temp_100 = ta.get_temperature(100, 100)

        assert temp_0 == 10.0
        assert temp_50 < temp_0
        assert temp_100 >= 2.0
        print(f"  ✓ 指数退火: {temp_0:.1f} -> {temp_50:.1f} -> {temp_100:.1f}")

    def test_min_temp_bound(self):
        """测试最小温度边界"""
        ta = TemperatureAnnealer(init_temp=10.0, min_temp=2.0, anneal_type='linear')

        # 即使在最后，温度也不低于 min_temp
        temp = ta.get_temperature(200, 100)
        assert temp >= 2.0
        print(f"  ✓ 最小温度边界: {temp:.1f}")


# ============================================================================
# SampleCurriculum 测试
# ============================================================================

class TestSampleCurriculum:
    """测试样本级课程蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        sc = SampleCurriculum(warmup_epochs=5, difficulty_threshold=0.5)
        assert sc.warmup_epochs == 5
        assert sc.difficulty_threshold == 0.5
        print("  ✓ 初始化正确")

    def test_get_sample_mask(self):
        """测试样本掩码"""
        sc = SampleCurriculum(warmup_epochs=5, difficulty_threshold=0.5)

        student_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        mask = sc.get_sample_mask(student_logits, labels, epoch=10)

        assert mask.shape == (8,)
        assert all(m in [0.0, 1.0] for m in mask.tolist())
        assert mask.sum() > 0  # 至少有一些样本被选中
        print(f"  ✓ 样本掩码: {mask.sum().item():.0f}/{len(mask)} 被选中")

    def test_warmup_phase(self):
        """测试预热阶段"""
        sc = SampleCurriculum(warmup_epochs=5, difficulty_threshold=0.5)

        student_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        # 预热阶段：阈值更低
        mask_early = sc.get_sample_mask(student_logits, labels, epoch=0)
        mask_late = sc.get_sample_mask(student_logits, labels, epoch=10)

        # 预热阶段应该选中更少的样本
        print(f"  ✓ 预热阶段: 早期={mask_early.sum().item():.0f}, 后期={mask_late.sum().item():.0f}")

    def test_compute_weighted_loss(self):
        """测试加权loss"""
        sc = SampleCurriculum(warmup_epochs=5, difficulty_threshold=0.5)

        losses = torch.randn(8)
        student_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        weighted_loss = sc.compute_weighted_loss(losses, student_logits, labels, epoch=10)

        assert weighted_loss.dim() == 0
        print(f"  ✓ 加权loss: {weighted_loss.item():.4f}")

    def test_all_samples_selected(self):
        """测试所有样本都被选中"""
        sc = SampleCurriculum(warmup_epochs=0, difficulty_threshold=1.0)

        # 低置信度 -> 所有样本都困难
        student_logits = torch.ones(8, 10) * 0.1
        labels = torch.randint(0, 10, (8,))

        mask = sc.get_sample_mask(student_logits, labels, epoch=10)

        # 所有样本都应该被选中
        assert mask.sum() == 8
        print("  ✓ 所有样本被选中")


# ============================================================================
# AdapterDistiller 测试
# ============================================================================

class TestAdapterDistiller:
    """测试适配器蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        student_dims = {'layer1': 16, 'layer2': 32}
        teacher_dims = {'layer1': 16, 'layer2': 64}

        adapter = AdapterDistiller(student_dims, teacher_dims, adapter_hidden=64)

        assert 'layer2' in adapter.adapters  # 维度不同需要适配
        assert 'layer1' not in adapter.adapters  # 维度相同不需要适配
        print("  ✓ 初始化正确")

    def test_forward(self):
        """测试前向传播"""
        student_dims = {'layer1': 16, 'layer2': 32}
        teacher_dims = {'layer1': 16, 'layer2': 64}

        adapter = AdapterDistiller(student_dims, teacher_dims, adapter_hidden=64)

        student_features = {
            'layer1': torch.randn(4, 16),
            'layer2': torch.randn(4, 32),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16),
            'layer2': torch.randn(4, 64),
        }

        loss = adapter(student_features, teacher_features)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 前向loss: {loss.item():.4f}")

    def test_get_adapted_features(self):
        """测试获取适配特征"""
        student_dims = {'layer1': 32}
        teacher_dims = {'layer1': 64}

        adapter = AdapterDistiller(student_dims, teacher_dims, adapter_hidden=64)

        student_features = {'layer1': torch.randn(4, 32)}
        adapted = adapter.get_adapted_features(student_features)

        assert 'layer1' in adapted
        assert adapted['layer1'].shape[-1] == 64  # 映射到 Teacher 维度
        print(f"  ✓ 适配特征: {adapted['layer1'].shape}")

    def test_backward(self):
        """测试反向传播"""
        student_dims = {'layer1': 32}
        teacher_dims = {'layer1': 64}

        adapter = AdapterDistiller(student_dims, teacher_dims, adapter_hidden=64)

        student_features = {'layer1': torch.randn(4, 32)}
        teacher_features = {'layer1': torch.randn(4, 64)}

        loss = adapter(student_features, teacher_features)
        loss.backward()

        # 检查适配器有梯度
        has_grad = any(p.grad is not None for p in adapter.parameters())
        assert has_grad
        print("  ✓ 反向传播正确")


# ============================================================================
# AdvancedDistiller 测试
# ============================================================================

class TestAdvancedDistiller:
    """测试高级蒸馏器"""

    def test_initialization(self):
        """测试初始化"""
        ad = AdvancedDistiller(
            use_dynamic_weight=True,
            use_attention=True,
            use_temperature_anneal=True,
            use_curriculum=True,
        )
        assert ad.dynamic_weight is not None
        assert ad.attention_distiller is not None
        assert ad.temp_annealer is not None
        assert ad.curriculum is not None
        print("  ✓ 初始化正确")

    def test_compute_loss(self):
        """测试组合loss计算"""
        ad = AdvancedDistiller(
            use_dynamic_weight=True,
            use_attention=True,
            use_temperature_anneal=True,
            use_curriculum=True,
            base_weight=0.1,
        )

        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16, 8, 8),
            'layer2': torch.randn(4, 32, 4, 4),
        }
        student_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        loss, loss_dict = ad.compute_loss(
            student_features, teacher_features,
            student_logits, labels,
            epoch=50, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert 'total' in loss_dict
        print(f"  ✓ 组合loss: {loss.item():.4f}, 组成: {loss_dict}")

    def test_get_temperature(self):
        """测试获取温度"""
        ad = AdvancedDistiller(use_temperature_anneal=True)

        temp = ad.get_temperature(epoch=50, max_epoch=100)
        assert isinstance(temp, float)
        assert temp >= 2.0
        print(f"  ✓ 当前温度: {temp:.1f}")

    def test_init_adapter(self):
        """测试初始化适配器"""
        ad = AdvancedDistiller(use_adapter=True)

        student_dims = {'layer1': 16, 'layer2': 32}
        teacher_dims = {'layer1': 16, 'layer2': 64}

        ad.init_adapter(student_dims, teacher_dims)
        assert ad.adapter is not None
        print("  ✓ 适配器初始化正确")

    def test_no_optional_components(self):
        """测试禁用所有可选组件"""
        ad = AdvancedDistiller(
            use_dynamic_weight=False,
            use_attention=False,
            use_temperature_anneal=False,
            use_curriculum=False,
        )

        student_features = {'layer1': torch.randn(4, 16, 8, 8)}
        teacher_features = {'layer1': torch.randn(4, 16, 8, 8)}
        student_logits = torch.randn(4, 10)

        loss, loss_dict = ad.compute_loss(
            student_features, teacher_features,
            student_logits, labels=None,
            epoch=0, max_epoch=100
        )

        assert loss.dim() == 0
        print(f"  ✓ 禁用可选组件: loss={loss.item():.4f}")

    def test_backward_pass(self):
        """测试反向传播"""
        ad = AdvancedDistiller(
            use_dynamic_weight=True,
            use_attention=True,
        )

        # 创建可训练模型
        student = SimpleCNN()
        teacher = SimpleCNN()

        x = torch.randn(2, 3, 8, 8)

        # 注册钩子获取特征
        student_features = {}
        teacher_features = {}

        def make_hook(d, name):
            def hook(module, input, output):
                d[name] = output
            return hook

        student.features[0].register_forward_hook(make_hook(student_features, 'layer1'))
        teacher.features[0].register_forward_hook(make_hook(teacher_features, 'layer1'))

        student_logits = student(x)
        teacher(x)

        loss, _ = ad.compute_loss(
            student_features, teacher_features,
            student_logits, labels=None,
            epoch=0, max_epoch=100
        )

        loss.backward()

        has_grad = any(p.grad is not None for p in student.parameters())
        assert has_grad
        print("  ✓ 反向传播正确")


if __name__ == "__main__":
    print("=" * 70)
    print("高级蒸馏技术测试")
    print("=" * 70)

    test_classes = [
        TestDynamicLayerWeighting,
        TestAttentionDistiller,
        TestTemperatureAnnealer,
        TestSampleCurriculum,
        TestAdapterDistiller,
        TestAdvancedDistiller,
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
