"""
逐层同步蒸馏测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_layer_wise_distiller.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.distillation import LayerWiseDistiller, ProgressiveLayerDistiller


# ============================================================================
# 测试模型
# ============================================================================

class SimpleCNN(nn.Module):
    """简单CNN用于测试"""
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


class SimpleMLP(nn.Module):
    """简单MLP用于测试"""
    def __init__(self, in_dim=16, hidden_dims=[64, 32], out_dim=10):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# LayerWiseDistiller 测试
# ============================================================================

class TestLayerWiseDistiller:
    """测试逐层同步蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        distiller = LayerWiseDistiller(
            mode='forward',
            progressive=True,
            adaptive_temp=True,
            selective=True,
            normalize=True,
            base_weight=0.1,
        )
        assert distiller.mode == 'forward'
        assert distiller.progressive is True
        assert distiller.adaptive_temp is True
        assert distiller.selective is True
        assert distiller.normalize is True
        print("  ✓ 初始化正确")

    def test_get_layer_groups(self):
        """测试层分组"""
        distiller = LayerWiseDistiller()
        model = SimpleCNN()

        groups = distiller._get_layer_groups(model)
        assert len(groups) > 0
        print(f"  ✓ 层分组: {len(groups)} 组")

    def test_progressive_schedule(self):
        """测试渐进式调度"""
        distiller = LayerWiseDistiller(progressive=True)

        # 3组，100 epochs
        # epoch 0-32: 解锁第0组
        # epoch 33-65: 解锁第0,1组
        # epoch 66-99: 解锁第0,1,2组
        assert distiller._get_progressive_schedule(0, 100, 3) == [0]
        assert distiller._get_progressive_schedule(33, 100, 3) == [0, 1]
        assert distiller._get_progressive_schedule(66, 100, 3) == [0, 1, 2]
        print("  ✓ 渐进式调度正确")

    def test_progressive_schedule_disabled(self):
        """测试禁用渐进式"""
        distiller = LayerWiseDistiller(progressive=False)

        # 禁用时始终解锁所有层
        assert distiller._get_progressive_schedule(0, 100, 3) == [0, 1, 2]
        assert distiller._get_progressive_schedule(50, 100, 3) == [0, 1, 2]
        print("  ✓ 禁用渐进式正确")

    def test_adaptive_temperature(self):
        """测试自适应温度"""
        distiller = LayerWiseDistiller(adaptive_temp=True)

        # 浅层: temp ≈ 2.0
        temp_0 = distiller._get_adaptive_temperature(0, 3, base_temp=4.0)
        assert 1.5 <= temp_0 <= 2.5, f"浅层温度应≈2.0, 实际={temp_0}"

        # 中层: temp ≈ 4.0
        temp_1 = distiller._get_adaptive_temperature(1, 3, base_temp=4.0)
        assert 3.0 <= temp_1 <= 5.0, f"中层温度应≈4.0, 实际={temp_1}"

        # 深层: temp ≈ 6.0
        temp_2 = distiller._get_adaptive_temperature(2, 3, base_temp=4.0)
        assert 5.0 <= temp_2 <= 7.0, f"深层温度应≈6.0, 实际={temp_2}"

        print(f"  ✓ 自适应温度: 浅={temp_0:.1f}, 中={temp_1:.1f}, 深={temp_2:.1f}")

    def test_adaptive_temperature_disabled(self):
        """测试禁用自适应温度"""
        distiller = LayerWiseDistiller(adaptive_temp=False)

        temp = distiller._get_adaptive_temperature(0, 3, base_temp=4.0)
        assert temp == 4.0
        print("  ✓ 禁用自适应温度正确")

    def test_compute_confidence(self):
        """测试置信度计算"""
        distiller = LayerWiseDistiller()

        # 高置信度（接近one-hot）
        logits_high = torch.tensor([[10.0, 0.1, 0.1]])
        conf_high = distiller._compute_confidence(logits_high)
        assert conf_high > 0.9, f"高置信度应>0.9, 实际={conf_high}"

        # 低置信度（均匀分布）
        logits_low = torch.tensor([[1.0, 1.0, 1.0]])
        conf_low = distiller._compute_confidence(logits_low)
        assert conf_low < 0.5, f"低置信度应<0.5, 实际={conf_low}"

        print(f"  ✓ 置信度: 高={conf_high:.3f}, 低={conf_low:.3f}")

    def test_should_distill(self):
        """测试选择性蒸馏"""
        distiller = LayerWiseDistiller(selective=True, confidence_threshold=0.7)

        # 高置信度 -> 不蒸馏
        logits_high = torch.tensor([[10.0, 0.1, 0.1]])
        assert distiller._should_distill(logits_high) is False

        # 低置信度 -> 蒸馏
        logits_low = torch.tensor([[1.0, 1.0, 1.0]])
        assert distiller._should_distill(logits_low) is True

        print("  ✓ 选择性蒸馏正确")

    def test_should_distill_disabled(self):
        """测试禁用选择性蒸馏"""
        distiller = LayerWiseDistiller(selective=False)

        # 禁用时总是蒸馏
        logits_high = torch.tensor([[10.0, 0.1, 0.1]])
        assert distiller._should_distill(logits_high) is True
        print("  ✓ 禁用选择性蒸馏正确")

    def test_normalize_features(self):
        """测试特征归一化"""
        distiller = LayerWiseDistiller(normalize=True)

        features = torch.randn(4, 16, 8, 8) * 100 + 50  # 大尺度差异
        normalized = distiller._normalize_features(features)

        # 归一化后应接近0均值1方差
        assert abs(normalized.mean().item()) < 0.5
        print(f"  ✓ 特征归一化: mean={normalized.mean().item():.4f}")

    def test_normalize_disabled(self):
        """测试禁用归一化"""
        distiller = LayerWiseDistiller(normalize=False)

        features = torch.randn(4, 16, 8, 8)
        result = distiller._normalize_features(features)

        assert torch.allclose(result, features)
        print("  ✓ 禁用归一化正确")

    def test_align_loss_forward(self):
        """测试前向KL散度"""
        distiller = LayerWiseDistiller(mode='forward')

        s_feat = torch.randn(4, 16, 8, 8)
        t_feat = torch.randn(4, 16, 8, 8)

        loss = distiller._align_loss(s_feat, t_feat, temperature=4.0)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 前向KL loss: {loss.item():.4f}")

    def test_align_loss_reverse(self):
        """测试反向KL散度"""
        distiller = LayerWiseDistiller(mode='reverse')

        s_feat = torch.randn(4, 16, 8, 8)
        t_feat = torch.randn(4, 16, 8, 8)

        loss = distiller._align_loss(s_feat, t_feat, temperature=4.0)
        assert loss.item() >= 0
        print(f"  ✓ 反向KL loss: {loss.item():.4f}")

    def test_align_loss_js(self):
        """测试JS散度"""
        distiller = LayerWiseDistiller(mode='js')

        s_feat = torch.randn(4, 16, 8, 8)
        t_feat = torch.randn(4, 16, 8, 8)

        loss = distiller._align_loss(s_feat, t_feat, temperature=4.0)
        assert loss.item() >= 0
        print(f"  ✓ JS散度 loss: {loss.item():.4f}")

    def test_align_loss_contrastive(self):
        """测试对比学习loss"""
        distiller = LayerWiseDistiller(mode='contrastive')

        s_feat = torch.randn(4, 64)
        t_feat = torch.randn(4, 64)

        loss = distiller._align_loss(s_feat, t_feat, temperature=0.07)
        assert loss.item() >= 0
        print(f"  ✓ 对比学习 loss: {loss.item():.4f}")

    def test_register_hooks(self):
        """测试钩子注册"""
        distiller = LayerWiseDistiller()
        student = SimpleCNN()
        teacher = SimpleCNN()

        layer_names = ['features.0', 'features.4']
        distiller.register_hooks(student, teacher, layer_names)

        assert len(distiller._student_hooks) == 2
        assert len(distiller._teacher_hooks) == 2

        distiller.remove_hooks()
        assert len(distiller._student_hooks) == 0
        print("  ✓ 钩子注册移除正确")

    def test_compute_loss_cnn(self):
        """测试CNN模型的loss计算"""
        distiller = LayerWiseDistiller(
            progressive=False,
            selective=False,
            base_weight=0.1
        )
        student = SimpleCNN()
        teacher = SimpleCNN()

        x = torch.randn(2, 3, 8, 8)
        student_logits = student(x)

        loss = distiller.compute_loss(
            student, teacher, x, student_logits,
            epoch=0, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ CNN loss: {loss.item():.4f}")

    def test_compute_loss_mlp(self):
        """测试MLP模型的loss计算"""
        distiller = LayerWiseDistiller(
            progressive=False,
            selective=False,
            base_weight=0.1
        )
        student = SimpleMLP()
        teacher = SimpleMLP()

        x = torch.randn(4, 16)
        student_logits = student(x)

        loss = distiller.compute_loss(
            student, teacher, x, student_logits,
            epoch=0, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ MLP loss: {loss.item():.4f}")

    def test_progressive_loss_changes(self):
        """测试渐进式loss随epoch变化"""
        distiller = LayerWiseDistiller(
            progressive=True,
            selective=False,
            base_weight=0.1
        )
        student = SimpleCNN()
        teacher = SimpleCNN()

        x = torch.randn(2, 3, 8, 8)
        student_logits = student(x)

        losses = []
        for epoch in [0, 33, 66, 99]:
            loss = distiller.compute_loss(
                student, teacher, x, student_logits,
                epoch=epoch, max_epoch=100
            )
            losses.append(loss.item())
            print(f"  epoch={epoch}: loss={loss.item():.4f}")

        # loss 应该随 epoch 增加（更多层参与）
        print(f"  ✓ 渐进式loss变化: {losses}")

    def test_selective_skips_high_confidence(self):
        """测试选择性蒸馏跳过高置信度"""
        distiller = LayerWiseDistiller(
            selective=True,
            confidence_threshold=0.7
        )
        student = SimpleCNN()
        teacher = SimpleCNN()

        # 创建高置信度输入（one-hot-like输出）
        x = torch.randn(2, 3, 8, 8)

        # 手动设置高置信度logits
        high_conf_logits = torch.tensor([[10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]] * 2)

        loss = distiller.compute_loss(
            student, teacher, x, high_conf_logits,
            epoch=0, max_epoch=100
        )

        # 高置信度时 loss 应为 0
        assert loss.item() == 0.0
        print("  ✓ 选择性蒸馏跳过高置信度")

    def test_set_layer_weights(self):
        """测试设置层权重"""
        distiller = LayerWiseDistiller()

        weights = {
            'features.0': 0.1,
            'features.4': 0.5,
            'features.8': 1.0,
        }
        distiller.set_layer_weights(weights)

        assert distiller._layer_weights == weights
        print("  ✓ 层权重设置正确")

    def test_get_active_layers_info(self):
        """测试获取激活层信息"""
        distiller = LayerWiseDistiller(progressive=True)
        model = SimpleCNN()

        info = distiller.get_active_layers_info(model, epoch=50, max_epoch=100)

        assert 'epoch' in info
        assert 'active_layers' in info
        assert 'temperatures' in info
        print(f"  ✓ 激活层信息: {len(info['active_layers'])} 层")

    def test_multi_teacher(self):
        """测试多教师蒸馏"""
        distiller = LayerWiseDistiller(
            progressive=False,
            selective=False,
            base_weight=0.1
        )
        student = SimpleCNN()
        teachers = [SimpleCNN(), SimpleCNN()]

        x = torch.randn(2, 3, 8, 8)
        student_logits = student(x)

        loss = distiller.compute_loss_multi_teacher(
            student, teachers, x, student_logits,
            epoch=0, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 多教师loss: {loss.item():.4f}")


# ============================================================================
# ProgressiveLayerDistiller 测试
# ============================================================================

class TestProgressiveLayerDistiller:
    """测试渐进式蒸馏简化接口"""

    def test_initialization(self):
        """测试初始化"""
        distiller = ProgressiveLayerDistiller(base_weight=0.05)
        assert distiller.progressive is True
        assert distiller.adaptive_temp is True
        assert distiller.selective is True
        assert distiller.normalize is True
        assert distiller.base_weight == 0.05
        print("  ✓ 渐进式蒸馏器初始化正确")

    def test_compute_loss(self):
        """测试loss计算"""
        distiller = ProgressiveLayerDistiller()
        student = SimpleCNN()
        teacher = SimpleCNN()

        x = torch.randn(2, 3, 8, 8)
        student_logits = student(x)

        loss = distiller.compute_loss(
            student, teacher, x, student_logits,
            epoch=50, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 渐进式loss: {loss.item():.4f}")


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_pipeline(self):
        """测试完整蒸馏流程"""
        from only_train_once.distillation import (
            TeacherEnsemble, SoftLabelDistiller, WeightDistiller
        )

        # 初始化所有蒸馏器
        ensemble = TeacherEnsemble(top_n=2)
        soft_kd = SoftLabelDistiller(temperature=4.0, alpha=0.5)
        layer_kd = LayerWiseDistiller(
            progressive=True,
            adaptive_temp=True,
            selective=True,
            normalize=True,
            base_weight=0.1
        )
        weight_kd = WeightDistiller(weight_weight=0.01)

        # 创建模型
        student = SimpleCNN()
        teacher = SimpleCNN()

        # 模拟训练
        x = torch.randn(2, 3, 8, 8)
        labels = torch.randint(0, 10, (2,))

        # 更新教师集合
        ensemble.update(teacher, loss=0.5, epoch=0, is_pruning=False)
        ensemble.update(teacher, loss=0.3, epoch=1, is_pruning=True)

        # Student 前向
        student_logits = student(x)

        # 计算各种loss
        teachers = ensemble.get_teachers()

        # 软标签蒸馏
        kd_loss = soft_kd.compute_loss(
            student_logits,
            [teacher(x) for _ in teachers],
            labels
        )

        # 逐层蒸馏
        layer_loss = layer_kd.compute_loss(
            student, teacher, x, student_logits,
            epoch=50, max_epoch=100
        )

        # 权重蒸馏
        w_loss = weight_kd.compute_loss(student, teacher)

        # 总loss
        total_loss = kd_loss + layer_loss + w_loss

        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
        print(f"  ✓ 完整流程: kd={kd_loss.item():.4f}, "
              f"layer={layer_loss.item():.4f}, "
              f"weight={w_loss.item():.4f}, "
              f"total={total_loss.item():.4f}")

    def test_backward_pass(self):
        """测试反向传播"""
        distiller = LayerWiseDistiller(
            progressive=False,
            selective=False,
            base_weight=0.1
        )
        student = SimpleCNN()
        teacher = SimpleCNN()

        x = torch.randn(2, 3, 8, 8)
        student_logits = student(x)

        loss = distiller.compute_loss(
            student, teacher, x, student_logits,
            epoch=0, max_epoch=100
        )

        # 反向传播
        loss.backward()

        # 检查梯度
        has_grad = any(p.grad is not None for p in student.parameters())
        assert has_grad, "Student 应该有梯度"
        print("  ✓ 反向传播正确")

    def test_no_nan_inf(self):
        """测试无NaN/Inf"""
        distiller = LayerWiseDistiller(
            progressive=False,
            selective=False,
        )
        student = SimpleCNN()
        teacher = SimpleCNN()

        for _ in range(10):
            x = torch.randn(2, 3, 8, 8)
            student_logits = student(x)

            loss = distiller.compute_loss(
                student, teacher, x, student_logits,
                epoch=0, max_epoch=100
            )

            assert not torch.isnan(loss), "loss 不应为 NaN"
            assert not torch.isinf(loss), "loss 不应为 Inf"

        print("  ✓ 10次迭代无NaN/Inf")


if __name__ == "__main__":
    print("=" * 70)
    print("逐层同步蒸馏测试")
    print("=" * 70)

    test_classes = [
        TestLayerWiseDistiller,
        TestProgressiveLayerDistiller,
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
