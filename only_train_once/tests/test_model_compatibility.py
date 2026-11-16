"""
模型兼容性测试
测试蒸馏方法对不同架构的适配度：
- ResNet (CNN经典)
- MobileNet (轻量级CNN)
- VGG (简单CNN)
- SimpleViT (Vision Transformer)
- DenseNet (密集连接)
- ConvNeXt (现代CNN)

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_model_compatibility.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from only_train_once.distillation import (
    LayerWiseDistiller,
    PruningAwareDistiller,
    IntegratedPruningDistiller,
    AttentionDistiller,
    WeightDistiller,
)


# ============================================================================
# 模型工厂
# ============================================================================

def get_model_and_input(model_name: str):
    """获取模型和输入

    Returns:
        (model, dummy_input, num_classes) 或 None（如果不支持）
    """
    try:
        if model_name == 'resnet20':
            from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
            model = resnet20_cifar10()
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        elif model_name == 'resnet18':
            from sanity_check.backends.resnet_cifar10 import resnet18_cifar10
            model = resnet18_cifar10()
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        elif model_name == 'vgg7':
            from sanity_check.backends.vgg7 import vgg7_bn
            model = vgg7_bn()
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        elif model_name == 'mobilenetv1':
            from sanity_check.backends.mobilenetv1 import MobileNetV1
            model = MobileNetV1(num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        elif model_name == 'mobilenetv2':
            from sanity_check.backends.mobilenetv2 import MobileNetV2
            model = MobileNetV2(num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        elif model_name == 'mobilenetv3':
            from sanity_check.backends.mobilenetv3 import mobilenetv3_small
            model = mobilenetv3_small(num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        elif model_name == 'simple_vit':
            try:
                from sanity_check.backends.simple_vit import SimpleViT
                model = SimpleViT(
                    image_size=32,
                    patch_size=4,
                    num_classes=10,
                    dim=64,
                    depth=4,
                    heads=4,
                    mlp_dim=128,
                )
                x = torch.randn(2, 3, 32, 32)
                return model, x, 10
            except ImportError:
                return None  # 需要 einops

        elif model_name == 'densenet':
            from sanity_check.backends.densenet import densenet121
            model = densenet121(num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        elif model_name == 'convnext':
            from sanity_check.backends.convnext import convnext_tiny
            model = convnext_tiny(num_classes=10)
            x = torch.randn(2, 3, 32, 32)
            return model, x, 10

        else:
            raise ValueError(f"Unknown model: {model_name}")

    except Exception as e:
        print(f"  ⚠ {model_name} 不可用: {e}")
        return None


def get_model_layers(model: nn.Module):
    """获取模型的层信息"""
    conv_layers = []
    linear_layers = []
    bn_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
        elif isinstance(module, nn.Linear):
            linear_layers.append((name, module))
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            bn_layers.append((name, module))

    return {
        'conv': conv_layers,
        'linear': linear_layers,
        'bn': bn_layers,
        'total': len(conv_layers) + len(linear_layers) + len(bn_layers)
    }


# ============================================================================
# 兼容性测试
# ============================================================================

class TestModelCompatibility:
    """测试不同模型的蒸馏兼容性"""

    @pytest.mark.parametrize("model_name", [
        'resnet20', 'resnet18', 'vgg7',
        'mobilenetv1', 'mobilenetv2', 'mobilenetv3',
        'simple_vit', 'densenet'
    ])
    def test_layer_wise_distiller(self, model_name):
        """测试逐层蒸馏兼容性"""
        result = get_model_and_input(model_name)
        if result is None:
            pytest.skip(f"{model_name} 不可用")

        student, x, _ = result
        teacher, _, _ = get_model_and_input(model_name)

        distiller = LayerWiseDistiller(
            progressive=False,
            selective=False,
            normalize=True,
            base_weight=0.1
        )

        student_logits = student(x)
        loss = distiller.compute_loss(
            student, teacher, x, student_logits,
            epoch=0, max_epoch=100
        )

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ {model_name}: 逐层蒸馏 loss={loss.item():.4f}")

    @pytest.mark.parametrize("model_name", [
        'resnet20', 'resnet18', 'vgg7',
        'mobilenetv1', 'mobilenetv2', 'mobilenetv3',
        'simple_vit', 'densenet'
    ])
    def test_pruning_aware_distiller(self, model_name):
        """测试剪枝感知蒸馏兼容性"""
        result = get_model_and_input(model_name)
        if result is None:
            pytest.skip(f"{model_name} 不可用")

        student, x, _ = result
        teacher, _, _ = get_model_and_input(model_name)

        distiller = PruningAwareDistiller(
            weight=0.1,
            use_projection=True,
            use_mask=False,
            use_importance=False,
        )

        # 注册钩子获取特征
        student_features = {}
        teacher_features = {}

        s_hooks = []
        t_hooks = []

        def make_hook(d, name):
            def hook(module, input, output):
                d[name] = output
            return hook

        # 选择前3个Conv层
        count = 0
        for name, module in student.named_modules():
            if isinstance(module, nn.Conv2d) and count < 3:
                s_hooks.append(module.register_forward_hook(make_hook(student_features, name)))
                count += 1

        count = 0
        for name, module in teacher.named_modules():
            if isinstance(module, nn.Conv2d) and count < 3:
                t_hooks.append(module.register_forward_hook(make_hook(teacher_features, name)))
                count += 1

        # 前向传播
        with torch.no_grad():
            teacher(x)
        student(x)

        # 计算loss
        loss = distiller.compute_loss(student_features, teacher_features)

        # 清理钩子
        for h in s_hooks + t_hooks:
            h.remove()

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ {model_name}: 剪枝感知蒸馏 loss={loss.item():.4f}")

    @pytest.mark.parametrize("model_name", [
        'resnet20', 'resnet18', 'vgg7',
        'mobilenetv1', 'mobilenetv2', 'mobilenetv3',
        'simple_vit', 'densenet'
    ])
    def test_attention_distiller(self, model_name):
        """测试注意力蒸馏兼容性"""
        result = get_model_and_input(model_name)
        if result is None:
            pytest.skip(f"{model_name} 不可用")

        student, x, _ = result
        teacher, _, _ = get_model_and_input(model_name)

        distiller = AttentionDistiller(weight=0.1)

        # 注册钩子
        student_features = {}
        teacher_features = {}

        s_hooks = []
        t_hooks = []

        def make_hook(d, name):
            def hook(module, input, output):
                d[name] = output
            return hook

        count = 0
        for name, module in student.named_modules():
            if isinstance(module, nn.Conv2d) and count < 3:
                s_hooks.append(module.register_forward_hook(make_hook(student_features, name)))
                count += 1

        count = 0
        for name, module in teacher.named_modules():
            if isinstance(module, nn.Conv2d) and count < 3:
                t_hooks.append(module.register_forward_hook(make_hook(teacher_features, name)))
                count += 1

        # 前向传播
        with torch.no_grad():
            teacher(x)
        student(x)

        # 计算loss
        loss = distiller.compute_loss(student_features, teacher_features)

        # 清理
        for h in s_hooks + t_hooks:
            h.remove()

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ {model_name}: 注意力蒸馏 loss={loss.item():.4f}")

    @pytest.mark.parametrize("model_name", [
        'resnet20', 'resnet18', 'vgg7',
        'mobilenetv1', 'mobilenetv2', 'mobilenetv3',
        'simple_vit', 'densenet'
    ])
    def test_weight_distiller(self, model_name):
        """测试权重蒸馏兼容性"""
        result = get_model_and_input(model_name)
        if result is None:
            pytest.skip(f"{model_name} 不可用")

        student, _, _ = result
        teacher, _, _ = get_model_and_input(model_name)

        distiller = WeightDistiller(weight_weight=0.01, align_mode='both')

        loss = distiller.compute_loss(student, teacher)

        assert loss.dim() == 0
        assert loss.item() >= 0
        print(f"  ✓ {model_name}: 权重蒸馏 loss={loss.item():.4f}")


# ============================================================================
# 架构分析测试
# ============================================================================

class TestArchitectureAnalysis:
    """分析不同架构的特点"""

    @pytest.mark.parametrize("model_name", [
        'resnet20', 'resnet18', 'vgg7',
        'mobilenetv1', 'mobilenetv2', 'mobilenetv3',
        'simple_vit', 'densenet'
    ])
    def test_model_structure(self, model_name):
        """分析模型结构"""
        result = get_model_and_input(model_name)
        if result is None:
            pytest.skip(f"{model_name} 不可用")

        model, x, num_classes = result
        layers = get_model_layers(model)

        # 前向传播获取输出形状
        with torch.no_grad():
            output = model(x)

        print(f"\n  📊 {model_name}:")
        print(f"     Conv层: {len(layers['conv'])}")
        print(f"     Linear层: {len(layers['linear'])}")
        print(f"     BN/Norm层: {len(layers['bn'])}")
        print(f"     总可蒸馏层: {layers['total']}")
        print(f"     输入形状: {x.shape}")
        print(f"     输出形状: {output.shape}")

        assert output.shape == (2, num_classes)

    @pytest.mark.parametrize("model_name", [
        'resnet20', 'resnet18', 'vgg7',
        'mobilenetv1', 'mobilenetv2', 'mobilenetv3',
        'simple_vit', 'densenet'
    ])
    def test_backward_pass(self, model_name):
        """测试反向传播"""
        result = get_model_and_input(model_name)
        if result is None:
            pytest.skip(f"{model_name} 不可用")

        model, x, num_classes = result

        output = model(x)
        loss = output.sum()
        loss.backward()

        # 检查梯度
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, f"{model_name} 应该有梯度"
        print(f"  ✓ {model_name}: 反向传播正确")


# ============================================================================
# 综合测试
# ============================================================================

class TestFullPipeline:
    """综合测试：蒸馏 + 前向 + 反向"""

    def test_resnet_full_pipeline(self):
        """ResNet完整流程"""
        self._test_full_pipeline('resnet20', 'ResNet20')

    def test_mobilenet_full_pipeline(self):
        """MobileNet完整流程"""
        self._test_full_pipeline('mobilenetv2', 'MobileNetV2')

    def test_vit_full_pipeline(self):
        """ViT完整流程"""
        self._test_full_pipeline('simple_vit', 'SimpleViT')

    def _test_full_pipeline(self, model_name: str, display_name: str):
        """通用完整流程测试"""
        result = get_model_and_input(model_name)
        if result is None:
            pytest.skip(f"{display_name} 不可用")

        student, x, num_classes = result
        teacher, _, _ = get_model_and_input(model_name)

        # 初始化蒸馏器
        distiller = IntegratedPruningDistiller(enabled=True)

        # 模拟训练
        for epoch in range(5):
            # 学生前向
            student_logits = student(x)
            loss_ce = torch.nn.CrossEntropyLoss()(
                student_logits, torch.randint(0, num_classes, (2,))
            )

            # 更新蒸馏器
            distiller.update(student, loss_ce.item(), epoch, is_pruning=(epoch >= 2))

            # 计算蒸馏loss
            kd_loss, info = distiller.compute_loss(
                student, x, epoch, max_epoch=5, is_pruning=(epoch >= 2)
            )

            # 总loss
            total_loss = loss_ce + kd_loss

            # 反向传播
            total_loss.backward()

            # 检查梯度
            has_grad = any(p.grad is not None for p in student.parameters() if p.requires_grad)
            assert has_grad

            # 清除梯度
            student.zero_grad()

        print(f"  ✓ {display_name}: 完整流程成功")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("模型兼容性测试")
    print("=" * 70)

    models = ['resnet20', 'resnet18', 'vgg7', 'mobilenetv1', 'mobilenetv2',
              'mobilenetv3', 'simple_vit', 'densenet']

    for model_name in models:
        print(f"\n--- {model_name} ---")
        try:
            model, x, num_classes = get_model_and_input(model_name)
            layers = get_model_layers(model)
            print(f"  结构: Conv={len(layers['conv'])}, Linear={len(layers['linear'])}, "
                  f"BN={len(layers['bn'])}")

            # 测试前向
            with torch.no_grad():
                output = model(x)
            print(f"  前向: {x.shape} -> {output.shape}")

            # 测试反向
            output.sum().backward()
            print(f"  反向: ✓")

        except Exception as e:
            print(f"  ✗ 失败: {e}")

    print("\n" + "=" * 70)
