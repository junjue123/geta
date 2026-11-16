"""
剪枝感知蒸馏测试

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_pruning_aware_distiller.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.distillation import (
    PruningAwareDistiller,
    PruningAwareTeacherEnsemble,
    PruningStageScheduler,
    IntegratedPruningDistiller,
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
# PruningAwareDistiller 测试
# ============================================================================

class TestPruningAwareDistiller:
    """测试剪枝感知蒸馏"""

    def test_initialization(self):
        """测试初始化"""
        distiller = PruningAwareDistiller(
            weight=0.1,
            use_projection=True,
            use_mask=True,
            use_importance=True,
        )
        assert distiller.weight == 0.1
        assert distiller.use_projection is True
        assert distiller.use_mask is True
        assert distiller.use_importance is True
        print("  ✓ 初始化正确")

    def test_update_pruning_info(self):
        """测试更新剪枝信息"""
        distiller = PruningAwareDistiller()

        param_groups = [
            {
                'p_names': ['features.0.weight'],
                'num_groups': 16,
                'pruned_idxes': [0, 1, 2],
                'is_prunable': True,
            }
        ]

        distiller.update_pruning_info(
            pruned_groups=[0, 1, 2],
            param_groups=param_groups
        )

        assert 'features.0' in distiller._pruning_masks
        mask = distiller._pruning_masks['features.0']
        assert mask[0] == 0.0  # 已剪枝
        assert mask[3] == 1.0  # 存活
        print(f"  ✓ 剪枝掩码: {mask.tolist()}")

    def test_apply_mask(self):
        """测试应用掩码"""
        distiller = PruningAwareDistiller(use_mask=True)

        features = torch.randn(4, 8, 4, 4)
        mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])

        masked = distiller._apply_mask(features, mask, dim=1)

        # 检查已剪枝通道被置零
        assert masked[:, 2, :, :].sum() == 0
        assert masked[:, 3, :, :].sum() == 0
        # 检查存活通道保持不变
        assert not masked[:, 0, :, :].sum() == 0
        print("  ✓ 掩码应用正确")

    def test_compute_loss_same_dim(self):
        """测试相同维度的loss计算"""
        distiller = PruningAwareDistiller(use_projection=False, use_mask=False)

        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }

        loss = distiller.compute_loss(student_features, teacher_features)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 相同维度loss: {loss.item():.4f}")

    def test_compute_loss_different_dim(self):
        """测试不同维度的loss计算（使用投影头）"""
        distiller = PruningAwareDistiller(use_projection=True, use_mask=False)

        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),  # 16通道
        }
        teacher_features = {
            'layer1': torch.randn(4, 32, 8, 8),  # 32通道
        }

        loss = distiller.compute_loss(student_features, teacher_features)
        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ 不同维度loss: {loss.item():.4f}")

    def test_compute_loss_with_mask(self):
        """测试带掩码的loss计算"""
        distiller = PruningAwareDistiller(use_mask=True)

        # 模拟剪枝
        distiller._pruning_masks = {
            'layer1': torch.tensor([1, 1, 0, 0, 1, 1, 0, 0,
                                     1, 1, 0, 0, 1, 1, 0, 0])
        }

        student_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }
        teacher_features = {
            'layer1': torch.randn(4, 16, 8, 8),
        }

        loss = distiller.compute_loss(student_features, teacher_features)
        assert loss.dim() == 0
        print(f"  ✓ 带掩码loss: {loss.item():.4f}")

    def test_get_projection_parameters(self):
        """测试获取投影头参数"""
        distiller = PruningAwareDistiller(use_projection=True)

        # 触发创建投影头
        student_features = {'layer1': torch.randn(4, 16, 8, 8)}
        teacher_features = {'layer1': torch.randn(4, 32, 8, 8)}
        distiller.compute_loss(student_features, teacher_features)

        params = distiller.get_projection_parameters()
        assert len(params) > 0
        print(f"  ✓ 投影头参数: {len(params)} 个")


# ============================================================================
# PruningAwareTeacherEnsemble 测试
# ============================================================================

class TestPruningAwareTeacherEnsemble:
    """测试剪枝感知教师集合"""

    def test_initialization(self):
        """测试初始化"""
        ensemble = PruningAwareTeacherEnsemble(max_history=10)
        assert ensemble.max_history == 10
        assert len(ensemble._history) == 0
        assert ensemble._best_model is None
        print("  ✓ 初始化正确")

    def test_update_and_get_teachers(self):
        """测试更新和获取教师"""
        ensemble = PruningAwareTeacherEnsemble()
        model = SimpleCNN()

        # 更新多次
        ensemble.update(model, loss=1.0, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.5, epoch=1, is_pruning=False)
        ensemble.update(model, loss=0.8, epoch=2, is_pruning=True)

        teachers = ensemble.get_teachers(is_pruning=False)

        assert len(teachers) > 0
        # 最优模型应该是loss=0.5的
        assert teachers[0]['type'] == 'best'
        assert teachers[0]['loss'] == 0.5
        print(f"  ✓ 教师数量: {len(teachers)}")

    def test_just_pruned(self):
        """测试剪枝后模型"""
        ensemble = PruningAwareTeacherEnsemble()
        model = SimpleCNN()

        ensemble.update(model, loss=0.5, epoch=0, is_pruning=False)
        ensemble.update(model, loss=0.3, epoch=1, is_pruning=True, just_pruned=True)

        teachers = ensemble.get_teachers(is_pruning=True)

        # 应该包含最优和剪枝后模型
        types = [t['type'] for t in teachers]
        assert 'best' in types
        assert 'last_pruned' in types
        print(f"  ✓ 教师类型: {types}")

    def test_state_dict(self):
        """测试状态保存恢复"""
        ensemble = PruningAwareTeacherEnsemble()
        model = SimpleCNN()

        ensemble.update(model, loss=0.5, epoch=0)

        state = ensemble.state_dict()
        ensemble2 = PruningAwareTeacherEnsemble()
        ensemble2.load_state_dict(state)

        assert ensemble2._best_model is not None
        assert ensemble2._best_model[1] == 0.5
        print("  ✓ 状态保存恢复正确")


# ============================================================================
# PruningStageScheduler 测试
# ============================================================================

class TestPruningStageScheduler:
    """测试剪枝阶段调度器"""

    def test_initialization(self):
        """测试初始化"""
        scheduler = PruningStageScheduler()
        assert scheduler.warmup_ratio == 0.2
        assert scheduler.refine_ratio == 0.2
        print("  ✓ 初始化正确")

    def test_warmup_stage(self):
        """测试预热阶段"""
        scheduler = PruningStageScheduler()

        config = scheduler.get_stage_config(epoch=10, max_epoch=100, is_pruning=False)

        assert config['stage'] == 'warmup'
        assert config['weight'] == 0.05
        assert config['temperature'] == 6.0
        assert config['use_projection'] is False
        print(f"  ✓ 预热阶段: {config}")

    def test_pruning_stage(self):
        """测试剪枝阶段"""
        scheduler = PruningStageScheduler()

        config = scheduler.get_stage_config(epoch=50, max_epoch=100, is_pruning=True)

        assert config['stage'] == 'pruning'
        assert config['weight'] == 0.1
        assert config['temperature'] == 4.0
        assert config['use_projection'] is True
        print(f"  ✓ 剪枝阶段: {config}")

    def test_refine_stage(self):
        """测试精炼阶段"""
        scheduler = PruningStageScheduler()

        config = scheduler.get_stage_config(epoch=90, max_epoch=100, is_pruning=False)

        assert config['stage'] == 'refine'
        assert config['weight'] == 0.2
        assert config['temperature'] == 2.0
        assert config['use_mask'] is True
        print(f"  ✓ 精炼阶段: {config}")

    def test_stage_transitions(self):
        """测试阶段转换"""
        scheduler = PruningStageScheduler()

        stages = []
        for epoch in range(0, 100, 10):
            config = scheduler.get_stage_config(epoch, 100, True)
            stages.append(config['stage'])

        assert stages[0] == 'warmup'
        assert stages[5] == 'pruning'
        assert stages[9] == 'refine'
        print(f"  ✓ 阶段转换: {stages}")


# ============================================================================
# IntegratedPruningDistiller 测试
# ============================================================================

class TestIntegratedPruningDistiller:
    """测试集成剪枝蒸馏器"""

    def test_initialization(self):
        """测试初始化"""
        distiller = IntegratedPruningDistiller(enabled=True, top_n_teachers=3)
        assert distiller.enabled is True
        print("  ✓ 初始化正确")

    def test_disabled(self):
        """测试禁用蒸馏"""
        distiller = IntegratedPruningDistiller(enabled=False)

        model = SimpleCNN()
        x = torch.randn(2, 3, 8, 8)

        loss, info = distiller.compute_loss(model, x, epoch=0, max_epoch=100)
        assert loss.item() == 0.0
        assert info == {}
        print("  ✓ 禁用蒸馏返回0")

    def test_update(self):
        """测试更新"""
        distiller = IntegratedPruningDistiller(enabled=True)
        model = SimpleCNN()

        # 更新
        distiller.update(model, loss=0.5, epoch=0, is_pruning=False)
        distiller.update(model, loss=0.3, epoch=1, is_pruning=True, just_pruned=True)

        # 检查教师集合
        teachers = distiller.teacher_ensemble.get_teachers(is_pruning=True)
        assert len(teachers) > 0
        print(f"  ✓ 更新成功: {len(teachers)} 个教师")

    def test_compute_loss(self):
        """测试计算loss"""
        distiller = IntegratedPruningDistiller(enabled=True)
        model = SimpleCNN()

        # 更新教师
        distiller.update(model, loss=0.5, epoch=0, is_pruning=False)

        # 计算loss
        x = torch.randn(2, 3, 8, 8)
        loss, info = distiller.compute_loss(
            model, x, epoch=50, max_epoch=100, is_pruning=True
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        assert 'stage' in info
        print(f"  ✓ 计算loss: {loss.item():.4f}, info: {info}")

    def test_state_dict(self):
        """测试状态保存恢复"""
        distiller = IntegratedPruningDistiller(enabled=True)
        model = SimpleCNN()

        distiller.update(model, loss=0.5, epoch=0)

        state = distiller.state_dict()
        distiller2 = IntegratedPruningDistiller(enabled=True)
        distiller2.load_state_dict(state)

        teachers = distiller2.teacher_ensemble.get_teachers()
        assert len(teachers) > 0
        print("  ✓ 状态保存恢复正确")

    def test_backward_pass(self):
        """测试反向传播"""
        distiller = IntegratedPruningDistiller(enabled=True)
        model = SimpleCNN()

        distiller.update(model, loss=0.5, epoch=0)

        x = torch.randn(2, 3, 8, 8)
        loss, _ = distiller.compute_loss(model, x, epoch=50, max_epoch=100)

        loss.backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
        print("  ✓ 反向传播正确")


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_pipeline(self):
        """测试完整流程"""
        distiller = IntegratedPruningDistiller(enabled=True)
        model = SimpleCNN()

        # 模拟训练过程
        for epoch in range(20):
            x = torch.randn(2, 3, 8, 8)
            logits = model(x)
            loss = torch.nn.CrossEntropyLoss()(logits, torch.randint(0, 10, (2,)))

            # 更新蒸馏器
            is_pruning = epoch >= 10
            just_pruned = epoch == 10
            distiller.update(model, loss.item(), epoch, is_pruning, just_pruned)

            # 计算蒸馏loss
            kd_loss, info = distiller.compute_loss(
                model, x, epoch, max_epoch=20, is_pruning=is_pruning
            )

            if epoch % 5 == 0:
                print(f"  epoch={epoch}: kd_loss={kd_loss.item():.4f}, "
                      f"stage={info.get('stage', 'N/A')}")

        print("  ✓ 完整流程执行成功")

    def test_no_nan_inf(self):
        """测试无NaN/Inf"""
        distiller = IntegratedPruningDistiller(enabled=True)
        model = SimpleCNN()

        distiller.update(model, loss=0.5, epoch=0)

        for _ in range(10):
            x = torch.randn(2, 3, 8, 8)
            loss, _ = distiller.compute_loss(model, x, epoch=50, max_epoch=100)

            assert not torch.isnan(loss), "loss 不应为 NaN"
            assert not torch.isinf(loss), "loss 不应为 Inf"

        print("  ✓ 10次迭代无NaN/Inf")


if __name__ == "__main__":
    print("=" * 70)
    print("剪枝感知蒸馏测试")
    print("=" * 70)

    test_classes = [
        TestPruningAwareDistiller,
        TestPruningAwareTeacherEnsemble,
        TestPruningStageScheduler,
        TestIntegratedPruningDistiller,
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
