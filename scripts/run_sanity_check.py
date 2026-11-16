"""
GETA 框架综合 Sanity Check (CPU 模式)
用小模型 + 随机数据跑几个前向/反向步骤，验证各模块代码能跑通。
"""
import os
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

PASSED = []
FAILED = []

def check(name, func):
    print(f"\n{'='*60}\n[CHECK] {name}\n{'='*60}", flush=True)
    t0 = time.time()
    try:
        func()
        print(f"\n>>> [PASS] {name}  ({time.time()-t0:.1f}s)", flush=True)
        PASSED.append(name)
    except Exception as e:
        print(f"\n>>> [FAIL] {name}  ({time.time()-t0:.1f}s)\n    {e}", flush=True)
        traceback.print_exc()
        FAILED.append((name, str(e)))

# 1. VGG7 + 量化 + GETA 优化器
def test_vgg7_geta():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO

    model = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    optimizer = oto.geta(variant="sgd", lr=1e-3, lr_quant=1e-3,
        target_group_sparsity=0.5, start_projection_step=0, projection_periods=2,
        projection_steps=3, start_pruning_step=3, pruning_periods=2, pruning_steps=3,
        bit_reduction=2, min_bit_wt=4, max_bit_wt=16)
    model.train(); criterion = nn.CrossEntropyLoss()
    for step in range(6):
        X = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
        loss = criterion(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 3 == 0 or step == 5:
            m = optimizer.compute_metrics()
            print(f"  step {step}: loss={loss.item():.4f}, sparsity={m.group_sparsity:.4f}", flush=True)
    m = optimizer.compute_metrics()
    print(f"  最终: sparsity={m.group_sparsity:.4f}, zero_groups={m.num_zero_groups}", flush=True)
    assert 0.0 <= m.group_sparsity <= 1.0

# 2. SimpleViT + HESSO 优化器 (仅剪枝)
def test_simplevit_hesso():
    from sanity_check.backends.simple_vit import simpleViT_cifar10
    from only_train_once import OTO
    model = simpleViT_cifar10().to(device)
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    oto.mark_unprunable_by_param_names(["to_patch_embedding.2.weight"])
    optimizer = oto.hesso(variant="sgd", lr=0.01, weight_decay=1e-4,
        target_group_sparsity=0.3, start_pruning_step=2, pruning_periods=2, pruning_steps=3)
    model.train(); criterion = nn.CrossEntropyLoss()
    for step in range(5):
        X = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
        loss = criterion(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 2 == 0 or step == 4:
            m = optimizer.compute_metrics()
            print(f"  step {step}: loss={loss.item():.4f}, sparsity={m.group_sparsity:.4f}", flush=True)
    m = optimizer.compute_metrics()
    print(f"  最终: sparsity={m.group_sparsity:.4f}", flush=True)
    assert 0.0 <= m.group_sparsity <= 1.0

# 3. MyGETA 优化器 (daco 新增)
def test_mygeta():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO
    model = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    optimizer = oto.mygeta(variant="sgd", lr=0.1, lr_quant=1e-3,
        target_group_sparsity=0.5, start_pruning_step=3, pruning_periods=2,
        pruning_steps=5, bit_reduction=2, min_bit_wt=4, max_bit_wt=16, device=device)
    model.train(); criterion = nn.CrossEntropyLoss()
    for step in range(6):
        X = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
        loss = criterion(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 3 == 0 or step == 5:
            m = optimizer.compute_metrics()
            print(f"  step {step}: loss={loss.item():.4f}, sparsity={m.group_sparsity:.4f}", flush=True)
    m = optimizer.compute_metrics()
    print(f"  最终: sparsity={m.group_sparsity:.4f}, zero_groups={m.num_zero_groups}", flush=True)
    assert 0.0 <= m.group_sparsity <= 1.0

# 4. 知识蒸馏模块 (daco 新增)
def test_distillation():
    from only_train_once.distillation import (
        TeacherEnsemble, SoftLabelDistiller, FeatureDistiller,
        WeightDistiller, Distiller)
    from sanity_check.backends.vgg7 import vgg7_bn
    model = vgg7_bn().to(device); teacher = vgg7_bn().to(device); teacher.eval()

    # 4.1 TeacherEnsemble
    ensemble = TeacherEnsemble(top_n=3)
    for ep in range(5):
        ensemble.update(model, 1.0/(ep+1), ep, is_pruning=(ep >= 2))
    assert ensemble.pre_pruning_best is not None
    print(f"  TeacherEnsemble: best_loss={ensemble.pre_pruning_best['loss']:.4f}, top_n={len(ensemble.pruning_top_n)}", flush=True)

    # 4.2 SoftLabelDistiller
    sl = SoftLabelDistiller(temperature=4.0, alpha=0.5)
    s_logits = torch.randn(4, 10); t_logits = torch.randn(4, 10); labels = torch.randint(0, 10, (4,))
    loss_kd = sl.compute_loss(s_logits, [t_logits], labels)
    print(f"  SoftLabelDistiller: kd_loss={loss_kd.item():.4f}", flush=True)
    assert loss_kd.item() > 0

    # 4.3 FeatureDistiller
    feat = FeatureDistiller(mode='forward', weight=0.1)
    s_feats = {}; t_feats = {}
    h1 = model.features[0].register_forward_hook(lambda m,i,o: s_feats.update(out=o))
    h2 = teacher.features[0].register_forward_hook(lambda m,i,o: t_feats.update(out=o))
    X = torch.randn(2, 3, 32, 32); model(X); teacher(X)
    feat_loss = feat.compute_loss(s_feats['out'], t_feats['out'])
    print(f"  FeatureDistiller: feat_loss={feat_loss.item():.4f}", flush=True)
    h1.remove(); h2.remove()

    # 4.4 WeightDistiller
    wt = WeightDistiller()
    wt_loss = wt.compute_loss(model.features[0].weight, teacher.features[0].weight)
    print(f"  WeightDistiller: wt_loss={wt_loss.item():.4f}", flush=True)

    # 4.5 Distiller 统一接口
    distiller = Distiller(temperature=4.0, alpha=0.5)
    print(f"  Distiller 统一接口创建成功", flush=True)

# 5. 部署模块 (daco 新增)
def test_deployment():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once.deployment import extract_model_info, get_kernel_config_for_layer
    model = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    model_info = extract_model_info(model, dummy_input)
    print(f"  模型信息提取成功: type={type(model_info).__name__}", flush=True)
    config = get_kernel_config_for_layer(model_info, 0)
    print(f"  核函数配置获取成功: type={type(config).__name__}", flush=True)

# 6. 子网构建与压缩导出
def test_construct_subnet():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once import OTO
    model = vgg7_bn().to(device)
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)
    optimizer = oto.hesso(variant="sgd", lr=0.01, target_group_sparsity=0.3,
        start_pruning_step=2, pruning_periods=2, pruning_steps=3)
    criterion = nn.CrossEntropyLoss(); model.train()
    for step in range(5):
        X = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
        loss = criterion(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "sanity_test")
    os.makedirs(out_dir, exist_ok=True)
    oto.construct_subnet(out_dir=out_dir)
    compressed = oto.compressed_model_path
    print(f"  压缩模型路径: {compressed}", flush=True)
    assert compressed is not None and os.path.exists(compressed), "压缩模型未生成"
    cm = torch.load(compressed, map_location=device, weights_only=False).to(device)
    cm.eval()
    with torch.no_grad(): out = cm(dummy_input)
    print(f"  压缩模型前向输出: shape={out.shape}", flush=True)
    assert out.shape == (1, 10)

if __name__ == "__main__":
    print("="*60, flush=True)
    print("GETA 框架综合 Sanity Check (CPU)", flush=True)
    print("="*60, flush=True)
    print(f"PyTorch: {torch.__version__}  设备: {device}", flush=True)

    check("1. VGG7 + 量化 + GETA 优化器", test_vgg7_geta)
    check("2. SimpleViT + HESSO 优化器", test_simplevit_hesso)
    check("3. MyGETA 优化器 (daco)", test_mygeta)
    check("4. 知识蒸馏模块 (daco)", test_distillation)
    check("5. 部署模块 (daco)", test_deployment)
    check("6. 子网构建与压缩导出", test_construct_subnet)

    print("\n" + "="*60, flush=True)
    print(f"通过: {len(PASSED)}/{len(PASSED)+len(FAILED)}", flush=True)
    for p in PASSED: print(f"  [PASS] {p}", flush=True)
    if FAILED:
        print(f"失败: {len(FAILED)}/{len(PASSED)+len(FAILED)}", flush=True)
        for name, err in FAILED:
            print(f"  [FAIL] {name}\n         {err}", flush=True)
    print("="*60, flush=True)
