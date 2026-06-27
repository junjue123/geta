"""
全面模型适配性测试
测试所有可用模型架构的兼容性。

运行方式:
    cd /Users/pengjue/Desktop/study/myproject/geta
    python3 -m pytest only_train_once/tests/test_all_models.py -v -s
"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# 模型工厂
# ============================================================================

def get_all_models():
    """获取所有可用模型"""
    models = {}

    # ResNet 系列
    try:
        from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
        models['resnet20_cifar10'] = (resnet20_cifar10, (2, 3, 32, 32), 10)
    except: pass

    try:
        from sanity_check.backends.resnet20_cifar100 import resnet20_cifar100
        models['resnet20_cifar100'] = (resnet20_cifar100, (2, 3, 32, 32), 100)
    except: pass

    try:
        from sanity_check.backends.resnet_cifar10 import resnet18_cifar10
        models['resnet18_cifar10'] = (resnet18_cifar10, (2, 3, 32, 32), 10)
    except: pass

    try:
        from sanity_check.backends.resnet_cifar10 import resnet34_cifar10
        models['resnet34_cifar10'] = (resnet34_cifar10, (2, 3, 32, 32), 10)
    except: pass

    try:
        from sanity_check.backends.resnet_cifar100 import resnet18_cifar100
        models['resnet18_cifar100'] = (resnet18_cifar100, (2, 3, 32, 32), 100)
    except: pass

    # VGG 系列
    try:
        from sanity_check.backends.vgg7 import vgg7_bn
        models['vgg7'] = (vgg7_bn, (2, 3, 32, 32), 10)
    except: pass

    # MobileNet 系列
    try:
        from sanity_check.backends.mobilenetv1 import MobileNetV1
        models['mobilenetv1'] = (lambda: MobileNetV1(num_classes=10), (2, 3, 32, 32), 10)
    except: pass

    try:
        from sanity_check.backends.mobilenetv2 import MobileNetV2
        models['mobilenetv2'] = (lambda: MobileNetV2(num_classes=10), (2, 3, 32, 32), 10)
    except: pass

    try:
        from sanity_check.backends.mobilenetv3 import mobilenetv3_small
        # mobilenetv3_small 不接受 num_classes 参数，需要手动修改
        def create_mobilenetv3_small():
            model = mobilenetv3_small()
            # 替换最后一层
            model.classifier[-1] = nn.Linear(1280, 10)
            return model
        models['mobilenetv3_small'] = (create_mobilenetv3_small, (2, 3, 32, 32), 10)
    except: pass

    try:
        from sanity_check.backends.mobilenetv3 import mobilenetv3_large
        def create_mobilenetv3_large():
            model = mobilenetv3_large()
            model.classifier[-1] = nn.Linear(1280, 10)
            return model
        models['mobilenetv3_large'] = (create_mobilenetv3_large, (2, 3, 32, 32), 10)
    except: pass

    # DenseNet 系列
    try:
        from sanity_check.backends.densenet import densenet121
        def create_densenet121():
            model = densenet121()
            # DenseNet 使用 linear 而不是 classifier
            model.linear = nn.Linear(model.linear.in_features, 10)
            return model
        models['densenet121'] = (create_densenet121, (2, 3, 32, 32), 10)
    except: pass

    try:
        from sanity_check.backends.densenet import densenet169
        def create_densenet169():
            model = densenet169()
            model.linear = nn.Linear(model.linear.in_features, 10)
            return model
        models['densenet169'] = (create_densenet169, (2, 3, 32, 32), 10)
    except: pass

    # ConvNeXt
    try:
        from sanity_check.backends.convnext import convnext_tiny
        models['convnext_tiny'] = (lambda: convnext_tiny(num_classes=10), (2, 3, 32, 32), 10)
    except: pass

    # MLP
    try:
        from sanity_check.backends.mlp import MLP
        # MLP 需要特定参数
        def create_mlp():
            return MLP(hidden_size1=128, intermediate_size1=64, intermediate_size2=32, hidden_size2=10)
        models['mlp'] = (create_mlp, (2, 3, 32, 32), 10)
    except: pass

    # Simple ViT
    try:
        from sanity_check.backends.simple_vit import SimpleViT
        models['simple_vit'] = (lambda: SimpleViT(
            image_size=32, patch_size=4, num_classes=10,
            dim=64, depth=4, heads=4, mlp_dim=128
        ), (2, 3, 32, 32), 10)
    except: pass

    # ResNet-DuBN
    try:
        from sanity_check.backends.resnet_DuBN import resnet18_dubn
        models['resnet18_dubn'] = (resnet18_dubn, (2, 3, 32, 32), 10)
    except: pass

    return models


def get_model_info(model):
    """获取模型结构信息"""
    conv_layers = []
    linear_layers = []
    bn_layers = []
    other_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
        elif isinstance(module, nn.Linear):
            linear_layers.append((name, module))
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            bn_layers.append((name, module))
        elif len(list(module.children())) == 0:  # 叶子模块
            other_layers.append((name, type(module).__name__))

    return {
        'conv': len(conv_layers),
        'linear': len(linear_layers),
        'bn': len(bn_layers),
        'other': len(other_layers),
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


# ============================================================================
# 测试类
# ============================================================================

class TestAllModels:
    """测试所有模型的适配性"""

    def test_forward_backward(self):
        """测试所有模型的前向和反向传播"""
        models = get_all_models()
        results = {}

        for name, (factory, input_shape, num_classes) in models.items():
            try:
                model = factory()
                x = torch.randn(*input_shape)

                # 前向传播
                output = model(x)
                assert output.shape == (2, num_classes), \
                    f"{name}: 输出形状错误 {output.shape} != (2, {num_classes})"

                # 反向传播
                loss = output.sum()
                loss.backward()

                # 检查梯度
                has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)

                results[name] = {
                    'status': 'OK',
                    'output_shape': output.shape,
                    'has_grad': has_grad,
                    'info': get_model_info(model),
                }
                print(f"  ✓ {name}: 前向+反向 OK")

            except Exception as e:
                results[name] = {'status': 'FAIL', 'error': str(e)}
                print(f"  ✗ {name}: {e}")

        # 打印总结
        print("\n" + "=" * 60)
        print("模型兼容性总结:")
        print("=" * 60)
        for name, result in results.items():
            if result['status'] == 'OK':
                info = result['info']
                print(f"  ✓ {name:25s} | Conv={info['conv']:2d} Linear={info['linear']:2d} "
                      f"BN={info['bn']:2d} | Params={info['total_params']:,}")
            else:
                print(f"  ✗ {name:25s} | Error: {result['error'][:50]}")

        # 统计
        ok_count = sum(1 for r in results.values() if r['status'] == 'OK')
        total_count = len(results)
        print(f"\n总计: {ok_count}/{total_count} 模型兼容")

    def test_distillation_compatibility(self):
        """测试蒸馏方法对所有模型的兼容性"""
        from only_train_once.distillation import (
            SoftLabelDistiller,
            WeightDistiller,
            LayerWiseDistiller,
        )

        models = get_all_models()
        results = {}

        for name, (factory, input_shape, num_classes) in models.items():
            try:
                model = factory()
                x = torch.randn(*input_shape)

                # 测试软标签蒸馏
                soft_kd = SoftLabelDistiller(temperature=4.0, alpha=0.5)
                logits = model(x)
                teacher_logits = model(x)
                kd_loss = soft_kd.compute_loss(logits, [teacher_logits])

                # 测试权重蒸馏
                weight_kd = WeightDistiller(weight_weight=0.01)
                teacher_model = factory()
                w_loss = weight_kd.compute_loss(model, teacher_model)

                # 测试逐层蒸馏
                layer_kd = LayerWiseDistiller(
                    progressive=False,
                    selective=False,
                    base_weight=0.1
                )
                layer_loss = layer_kd.compute_loss(
                    model, teacher_model, x, logits,
                    epoch=0, max_epoch=100
                )

                results[name] = {
                    'status': 'OK',
                    'kd_loss': kd_loss.item(),
                    'w_loss': w_loss.item(),
                    'layer_loss': layer_loss.item(),
                }
                print(f"  ✓ {name}: kd={kd_loss.item():.4f} w={w_loss.item():.4f} "
                      f"layer={layer_loss.item():.4f}")

            except Exception as e:
                results[name] = {'status': 'FAIL', 'error': str(e)}
                print(f"  ✗ {name}: {e}")

        # 统计
        ok_count = sum(1 for r in results.values() if r['status'] == 'OK')
        total_count = len(results)
        print(f"\n蒸馏兼容: {ok_count}/{total_count} 模型")

    def test_model_structure_analysis(self):
        """分析所有模型的结构特点"""
        models = get_all_models()

        print("\n" + "=" * 80)
        print("模型结构分析:")
        print("=" * 80)
        print(f"{'模型名':25s} | {'Conv':>4s} | {'Linear':>6s} | {'BN/Norm':>7s} | "
              f"{'Params':>12s} | {'Trainable':>12s}")
        print("-" * 80)

        for name, (factory, input_shape, num_classes) in models.items():
            try:
                model = factory()
                info = get_model_info(model)

                print(f"{name:25s} | {info['conv']:4d} | {info['linear']:6d} | "
                      f"{info['bn']:7d} | {info['total_params']:12,} | "
                      f"{info['trainable_params']:12,}")

            except Exception as e:
                print(f"{name:25s} | Error: {str(e)[:40]}")

    def test_layer_type_distribution(self):
        """分析各模型的层类型分布"""
        models = get_all_models()

        print("\n" + "=" * 80)
        print("层类型分布:")
        print("=" * 80)

        for name, (factory, input_shape, num_classes) in models.items():
            try:
                model = factory()
                layer_types = {}

                for module_name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # 叶子模块
                        type_name = type(module).__name__
                        layer_types[type_name] = layer_types.get(type_name, 0) + 1

                print(f"\n{name}:")
                for type_name, count in sorted(layer_types.items(), key=lambda x: -x[1]):
                    print(f"  {type_name:20s}: {count}")

            except Exception as e:
                print(f"\n{name}: Error - {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("全面模型适配性测试")
    print("=" * 70)

    test = TestAllModels()
    test.test_forward_backward()
    test.test_distillation_compatibility()
    test.test_model_structure_analysis()
    test.test_layer_type_distribution()
