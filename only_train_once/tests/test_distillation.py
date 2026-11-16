"""
蒸馏模块单元测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_distillation.py -v -s
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
    WeightDistiller
)


# ============================================================================
# 简单测试模型
# ============================================================================

class SimpleModel(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ConvModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 2 * 2, num_classes)  # 8x8 input -> 2x2 after two max_pool2d

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)  # 8x8 -> 4x4
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)  # 4x4 -> 2x2
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# TeacherEnsemble 测试
# ============================================================================

class TestTeacherEnsemble:
    """测试教师模型集合管理"""

    def test_initialization(self):
        """测试初始化"""
        ensemble = TeacherEnsemble(top_n=3)
        assert len(ensemble) == 0
        assert ensemble.pre_pruning_best is None
        assert len(ensemble.pruning_top_n) == 0
        assert ensemble.pruning_random is None
        print("  ✓ 初始化正确")

    def test_update_pre_pruning(self):
        """测试剪枝前更新"""
        ensemble = TeacherEnsemble(top_n=3)
        model = SimpleModel()

        # 更新多次
        ensemble.update(model, loss=1.0, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.5, epoch=1, is_pruning=False)
        ensemble.update(model, loss=0.8, epoch=2, is_pruning=False)

        # 应该保留loss最小的
        assert ensemble.pre_pruning_best is not None
        assert ensemble.pre_pruning_best['loss'] == 0.5
        assert ensemble.pre_pruning_best['epoch'] == 1
        print("  ✓ 剪枝前最优模型正确")

    def test_update_pruning_top_n(self):
        """测试剪枝中前N个更新"""
        ensemble = TeacherEnsemble(top_n=3)
        model = SimpleModel()

        # 进入剪枝阶段
        ensemble.update(model, loss=0.5, epoch=0, is_pruning=True)
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True)
        ensemble.update(model, loss=0.8, epoch=2, is_pruning=True)
        ensemble.update(model, loss=0.2, epoch=3, is_pruning=True)
        ensemble.update(model, loss=0.6, epoch=4, is_pruning=True)

        # 应该保留loss最小的3个
        assert len(ensemble.pruning_top_n) == 3
        # 堆中存储的是 neg_loss，需要转回正loss
        losses = [-entry[0] for entry in ensemble.pruning_top_n]
        # 验证保留的是最小的3个
        assert sorted(losses) == [0.2, 0.3, 0.5]
        # 验证没有保留较大的loss
        assert 0.8 not in losses
        assert 0.6 not in losses
        print(f"  ✓ 剪枝中前N个正确: {sorted(losses)}")

    def test_update_pruning_random(self):
        """测试剪枝中随机采样"""
        ensemble = TeacherEnsemble(top_n=3)
        model = SimpleModel()

        # 进入剪枝阶段
        for i in range(10):
            ensemble.update(model, loss=float(i), epoch=i, is_pruning=True)

        # 应该有随机采样
        assert ensemble.pruning_random is not None
        assert ensemble.pruning_sample_count == 10
        print("  ✓ 剪枝中随机采样正确")

    def test_get_teachers(self):
        """测试获取教师列表"""
        ensemble = TeacherEnsemble(top_n=2)
        model = SimpleModel()

        # 剪枝前
        ensemble.update(model, loss=0.5, epoch=0, is_pruning=False)

        # 剪枝中
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True)
        ensemble.update(model, loss=0.8, epoch=2, is_pruning=True)
        ensemble.update(model, loss=0.2, epoch=3, is_pruning=True)

        teachers = ensemble.get_teachers()
        # 1 (pre-pruning) + 2 (top-n) + 1 (random)
        assert len(teachers) >= 3
        print(f"  ✓ 获取教师列表: {len(teachers)} 个教师")

    def test_state_dict(self):
        """测试状态保存恢复"""
        ensemble = TeacherEnsemble(top_n=2)
        model = SimpleModel()

        # 更新
        ensemble.update(model, loss=0.5, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True)

        # 保存状态
        state = ensemble.state_dict()

        # 创建新集合并恢复
        ensemble2 = TeacherEnsemble()
        ensemble2.load_state_dict(state)

        assert len(ensemble2) == len(ensemble)
        assert ensemble2.pre_pruning_best['loss'] == 0.5
        print("  ✓ 状态保存恢复正确")

    def test_clear(self):
        """测试清空"""
        ensemble = TeacherEnsemble(top_n=2)
        model = SimpleModel()

        ensemble.update(model, loss=0.5, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True)

        ensemble.clear()

        assert len(ensemble) == 0
        assert ensemble.pre_pruning_best is None
        print("  ✓ 清空正确")


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

    def test_compute_loss_single_teacher(self):
        """测试单教师loss计算"""
        distiller = SoftLabelDistiller(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)

        loss = distiller.compute_loss(student_logits, [teacher_logits])
        assert loss.dim() == 0  # 标量
        assert loss.item() >= 0
        print(f"  ✓ 单教师loss: {loss.item():.4f}")

    def test_compute_loss_multi_teacher(self):
        """测试多教师loss计算"""
        distiller = SoftLabelDistiller(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(4, 10)
        teacher_logits_list = [torch.randn(4, 10) for _ in range(3)]

        loss = distiller.compute_loss(student_logits, teacher_logits_list)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 多教师loss: {loss.item():.4f}")

    def test_compute_loss_with_labels(self):
        """测试带标签的loss计算"""
        distiller = SoftLabelDistiller(temperature=4.0, alpha=0.5)

        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        loss = distiller.compute_loss(student_logits, [teacher_logits], labels)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 带标签loss: {loss.item():.4f}")

    def test_different_modes(self):
        """测试不同KD模式"""
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)

        for mode in ['forward', 'reverse', 'js']:
            distiller = SoftLabelDistiller(kd_mode=mode)
            loss = distiller.compute_loss(student_logits, [teacher_logits])
            assert loss.item() >= 0
            print(f"  ✓ {mode} 模式loss: {loss.item():.4f}")

    def test_empty_teachers(self):
        """测试空教师列表"""
        distiller = SoftLabelDistiller()
        student_logits = torch.randn(4, 10)

        loss = distiller.compute_loss(student_logits, [])
        assert loss.item() == 0.0
        print("  ✓ 空教师返回0")


# ============================================================================
# FeatureDistiller 测试
# ============================================================================

class TestFeatureDistiller:
    """测试层输出分布对齐"""

    def test_initialization(self):
        """测试初始化"""
        distiller = FeatureDistiller(mode='forward', feature_weight=0.1)
        assert distiller.mode == 'forward'
        assert distiller.feature_weight == 0.1
        print("  ✓ 初始化正确")

    def test_register_hooks(self):
        """测试钩子注册"""
        distiller = FeatureDistiller()
        student = SimpleModel()
        teacher = SimpleModel()

        distiller.register_hooks(student, teacher, layer_names=['fc1', 'fc2'])
        assert len(distiller._student_hooks) > 0
        assert len(distiller._teacher_hooks) > 0

        distiller.remove_hooks()
        assert len(distiller._student_hooks) == 0
        print("  ✓ 钩子注册移除正确")

    def test_auto_select_layers(self):
        """测试自动选择层"""
        distiller = FeatureDistiller()
        student = SimpleModel()
        teacher = SimpleModel()

        layers = distiller._auto_select_layers(student, teacher)
        assert len(layers) > 0
        print(f"  ✓ 自动选择层: {layers}")

    def test_compute_loss(self):
        """测试loss计算"""
        distiller = FeatureDistiller(feature_weight=0.1)
        student = SimpleModel()
        teacher = SimpleModel()

        x = torch.randn(4, 16)
        distiller.register_hooks(student, teacher, layer_names=['fc1', 'fc2'])

        # 前向传播
        with torch.no_grad():
            teacher(x)
        student(x)

        loss = distiller.compute_loss()
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 特征对齐loss: {loss.item():.4f}")

        distiller.remove_hooks()

    def test_different_modes(self):
        """测试不同模式"""
        student = SimpleModel()
        teacher = SimpleModel()
        x = torch.randn(4, 16)

        for mode in ['forward', 'reverse', 'js']:
            distiller = FeatureDistiller(mode=mode)
            loss = distiller.compute_loss_with_hooks(student, [teacher], x)
            assert loss.item() >= 0
            print(f"  ✓ {mode} 模式loss: {loss.item():.4f}")

    def test_conv_model(self):
        """测试卷积模型"""
        distiller = FeatureDistiller()
        student = ConvModel()
        teacher = ConvModel()

        x = torch.randn(2, 3, 8, 8)
        loss = distiller.compute_loss_with_hooks(student, [teacher], x)
        assert loss.item() >= 0
        print(f"  ✓ 卷积模型loss: {loss.item():.4f}")


# ============================================================================
# WeightDistiller 测试
# ============================================================================

class TestWeightDistiller:
    """测试权重特征对齐"""

    def test_initialization(self):
        """测试初始化"""
        distiller = WeightDistiller(weight_weight=0.01, align_mode='both')
        assert distiller.weight_weight == 0.01
        assert distiller.align_mode == 'both'
        print("  ✓ 初始化正确")

    def test_compute_svd(self):
        """测试SVD计算"""
        distiller = WeightDistiller(svd_method='full')
        weight = torch.randn(32, 16)

        sv = distiller._compute_svd(weight)
        assert sv is not None
        assert len(sv) == min(weight.shape)
        assert sv[0] >= sv[-1]  # 奇异值降序
        print(f"  ✓ SVD计算正确: {len(sv)} 个奇异值")

    def test_compute_svd_lowrank(self):
        """测试随机化SVD"""
        distiller = WeightDistiller(svd_method='lowrank', lowrank_k=5)
        weight = torch.randn(64, 32)

        sv = distiller._compute_svd(weight)
        assert sv is not None
        assert len(sv) == 5
        print(f"  ✓ 随机化SVD正确: {len(sv)} 个奇异值")

    def test_compute_loss(self):
        """测试loss计算"""
        distiller = WeightDistiller(weight_weight=0.01, align_mode='both')
        student = SimpleModel()
        teacher = SimpleModel()

        loss = distiller.compute_loss(student, teacher)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 权重对齐loss: {loss.item():.4f}")

    def test_different_align_modes(self):
        """测试不同对齐模式"""
        student = SimpleModel()
        teacher = SimpleModel()

        for mode in ['max', 'avg', 'both']:
            distiller = WeightDistiller(align_mode=mode)
            loss = distiller.compute_loss(student, teacher)
            assert loss.item() >= 0
            print(f"  ✓ {mode} 模式loss: {loss.item():.4f}")

    def test_cache_teachers(self):
        """测试教师缓存"""
        distiller = WeightDistiller(cache_teachers=True)
        student = SimpleModel()
        teacher = SimpleModel()

        # 第一次计算
        loss1 = distiller.compute_loss(student, teacher, teacher_id=0)

        # 第二次计算（应使用缓存）
        loss2 = distiller.compute_loss(student, teacher, teacher_id=0)

        # loss应该相同（因为教师没变），使用宽松阈值
        assert abs(loss1.item() - loss2.item()) < 1e-4
        assert 0 in distiller._teacher_sv_cache
        print("  ✓ 教师缓存正确")

    def test_get_svd_statistics(self):
        """测试奇异值统计"""
        distiller = WeightDistiller()
        model = SimpleModel()

        stats = distiller.get_svd_statistics(model)
        assert len(stats) > 0
        for name, stat in stats.items():
            assert 'max_sv' in stat
            assert 'mean_sv' in stat
            assert 'num_sv' in stat
        print(f"  ✓ 奇异值统计: {len(stats)} 层")

    def test_clear_cache(self):
        """测试清除缓存"""
        distiller = WeightDistiller(cache_teachers=True)
        student = SimpleModel()
        teacher = SimpleModel()

        distiller.compute_loss(student, teacher, teacher_id=0)
        assert len(distiller._teacher_sv_cache) > 0

        distiller.clear_cache()
        assert len(distiller._teacher_sv_cache) == 0
        print("  ✓ 清除缓存正确")


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_distillation_pipeline(self):
        """测试完整蒸馏流程"""
        # 初始化
        ensemble = TeacherEnsemble(top_n=2)
        soft_distiller = SoftLabelDistiller(temperature=4.0, alpha=0.5)
        feature_distiller = FeatureDistiller(feature_weight=0.1)
        weight_distiller = WeightDistiller(weight_weight=0.01)

        # 创建模型
        student = SimpleModel()
        teacher = SimpleModel()

        # 模拟训练
        x = torch.randn(4, 16)
        labels = torch.randint(0, 10, (4,))

        # 更新教师集合
        ensemble.update(teacher, loss=0.5, epoch=0, is_pruning=False)
        ensemble.update(teacher, loss=0.3, epoch=1, is_pruning=True)

        # 计算蒸馏loss
        student_logits = student(x)
        teachers = ensemble.get_teachers()

        # 软标签蒸馏
        kd_loss = soft_distiller.compute_loss(
            student_logits,
            [teacher(x) for _ in teachers],
            labels
        )

        # 特征对齐
        feat_loss = feature_distiller.compute_loss_with_hooks(
            student, [teacher], x
        )

        # 权重对齐
        w_loss = weight_distiller.compute_loss(student, teacher)

        # 总loss
        total_loss = kd_loss + feat_loss + w_loss

        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
        print(f"  ✓ 完整蒸馏流程: kd={kd_loss.item():.4f}, "
              f"feat={feat_loss.item():.4f}, weight={w_loss.item():.4f}, "
              f"total={total_loss.item():.4f}")

    def test_device_consistency(self):
        """测试设备一致性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA不可用")

        device = 'cuda'
        ensemble = TeacherEnsemble(top_n=2)
        soft_distiller = SoftLabelDistiller()

        student = SimpleModel().to(device)
        teacher = SimpleModel().to(device)
        x = torch.randn(4, 16).to(device)

        ensemble.update(teacher, loss=0.5, epoch=0, is_pruning=False)

        student_logits = student(x)
        kd_loss = soft_distiller.compute_loss(student_logits, [teacher(x)])

        assert kd_loss.device.type == device
        print("  ✓ 设备一致性正确")


if __name__ == "__main__":
    print("=" * 70)
    print("蒸馏模块测试")
    print("=" * 70)

    test_classes = [
        TestTeacherEnsemble,
        TestSoftLabelDistiller,
        TestFeatureDistiller,
        TestWeightDistiller,
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
