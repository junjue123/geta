"""
GETA 多模型 GPU 轻量化验证
用 GPU 跑 ViT / MobileNetV2 / ConvNeXt 等模型，1步训练验证。
"""
import os, sys, time, traceback
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

PASSED, FAILED = [], []

def check(name, func):
    print(f"\n{'='*60}\n[CHECK] {name}\n{'='*60}", flush=True)
    t0 = time.time()
    torch.cuda.empty_cache() if device=="cuda" else None
    try:
        func()
        print(f"\n>>> [PASS] {name}  ({time.time()-t0:.1f}s)", flush=True)
        PASSED.append(name)
    except Exception as e:
        print(f"\n>>> [FAIL] {name}  ({time.time()-t0:.1f}s)\n    {e}", flush=True)
        traceback.print_exc()
        FAILED.append((name, str(e)[:100]))

def run_one_step(build_model_fn, dummy_shape, mark_unprunable=None, use_geta=True):
    """通用: 构建模型 → 量化 → OTO → 优化器 → 1步训练 → 蒸馏验证"""
    from only_train_once import OTO
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once.distillation import SoftLabelDistiller

    model = build_model_fn()
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(device)
    dummy = torch.rand(*dummy_shape).to(device)

    oto = OTO(model=model, dummy_input=dummy)
    if mark_unprunable:
        oto.mark_unprunable_by_param_names(mark_unprunable)

    if use_geta:
        opt = oto.geta(variant="sgd", lr=1e-3, lr_quant=1e-3,
            target_group_sparsity=0.3, start_projection_step=0, projection_periods=1,
            projection_steps=1, start_pruning_step=1, pruning_periods=1, pruning_steps=1,
            bit_reduction=2, min_bit_wt=4, max_bit_wt=16)
    else:
        opt = oto.mygeta(variant="sgd", lr=0.01, lr_quant=1e-3,
            target_group_sparsity=0.3, start_pruning_step=1, pruning_periods=1,
            pruning_steps=1, bit_reduction=2, min_bit_wt=4, max_bit_wt=16, device=device)

    model.train()
    X = torch.randn(2, *dummy_shape[1:]).to(device)
    y = torch.randint(0, 10, (2,)).to(device)
    loss = nn.CrossEntropyLoss()(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  loss={loss.item():.4f}", flush=True)
    assert not torch.isnan(loss), "loss is NaN"

    # 蒸馏验证
    model.eval()
    with torch.no_grad():
        s_out = model(X)
    kd = SoftLabelDistiller(temperature=4.0, alpha=0.5).compute_loss(
        s_out, [torch.randn_like(s_out)], y)
    print(f"  kd={kd.item():.4f}", flush=True)
    assert not torch.isnan(kd), "kd_loss is NaN"

# 1. SimpleViT
def test_simple_vit():
    from sanity_check.backends.simple_vit import simpleViT_cifar10
    run_one_step(simpleViT_cifar10, (1,3,32,32),
                 mark_unprunable=["to_patch_embedding.2.weight"])

# 2. HF-ViT (4层小ViT)
def test_hf_vit():
    from sanity_check.backends.simple_vit import SimpleViT
    run_one_step(lambda: SimpleViT(image_size=32,patch_size=4,num_classes=10,
                 dim=128,depth=4,heads=4,mlp_dim=64), (1,3,32,32),
                 mark_unprunable=["to_patch_embedding.2.weight"])

# 3. MobileNetV2
def test_mobilenetv2():
    from sanity_check.backends.mobilenetv2 import mobilenetv2_cifar10
    run_one_step(mobilenetv2_cifar10, (1,3,32,32))

# 4. ConvNeXt-Tiny
def test_convnext():
    from sanity_check.backends.convnext import convnext_tiny
    run_one_step(lambda: convnext_tiny(num_classes=10), (1,3,32,32))

# 5. VGG7 + MyGETA
def test_vgg7_mygeta():
    from sanity_check.backends.vgg7 import vgg7_bn
    run_one_step(vgg7_bn, (1,3,32,32), use_geta=False)

# 6. ResNet20 (GPU加速版)
def test_resnet20():
    from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
    run_one_step(resnet20_cifar10, (1,3,32,32))

if __name__ == "__main__":
    print("="*60, flush=True)
    print("GETA 多模型 GPU 验证 (1步训练)", flush=True)
    print("="*60, flush=True)

    check("1. SimpleViT + GETA", test_simple_vit)
    check("2. HF-ViT(4层) + GETA", test_hf_vit)
    check("3. MobileNetV2 + GETA", test_mobilenetv2)
    check("4. ConvNeXt-Tiny + GETA", test_convnext)
    check("5. VGG7 + MyGETA", test_vgg7_mygeta)
    check("6. ResNet20 + GETA", test_resnet20)

    print("\n" + "="*60, flush=True)
    print(f"通过: {len(PASSED)}/{len(PASSED)+len(FAILED)}", flush=True)
    for p in PASSED: print(f"  [PASS] {p}", flush=True)
    if FAILED:
        print(f"失败: {len(FAILED)}/{len(PASSED)+len(FAILED)}", flush=True)
        for name, err in FAILED:
            print(f"  [FAIL] {name}\n         {err}", flush=True)
    print("="*60, flush=True)
