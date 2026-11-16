"""
GETA 框架综合轻量化验证测试
用最简单的模型结构，做 1-2 步训练，把剪枝+量化+蒸馏全流程验证一遍。

运行: CUDA_VISIBLE_DEVICES="" python run_lightweight_test.py
"""
import os, sys, time, traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

PASSED, FAILED = [], []

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


# ============================================================================
# 1. VGG7 + GETA 量化剪枝 (一步训练)
# ============================================================================
def test_geta_quant_prune():
    """验证: 量化模型 → GETA优化器 → 前向+反向一步 → 不NaN"""
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO

    model = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    dummy = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model, dummy_input=dummy)
    opt = oto.geta(variant="sgd", lr=1e-3, lr_quant=1e-3,
        target_group_sparsity=0.5, start_projection_step=0, projection_periods=1,
        projection_steps=2, start_pruning_step=2, pruning_periods=1, pruning_steps=2,
        bit_reduction=2, min_bit_wt=4, max_bit_wt=16)

    model.train()
    X = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
    loss = nn.CrossEntropyLoss()(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
    m = opt.compute_metrics()
    print(f"  step0: loss={loss.item():.4f}, sparsity={m.group_sparsity:.4f}", flush=True)

    # 第二步
    loss2 = nn.CrossEntropyLoss()(model(X), y)
    opt.zero_grad(); loss2.backward(); opt.step()
    m2 = opt.compute_metrics()
    print(f"  step1: loss={loss2.item():.4f}, sparsity={m2.group_sparsity:.4f}", flush=True)
    assert not torch.isnan(loss2), "loss 变 NaN"


# ============================================================================
# 2. VGG7 + MyGETA 量化剪枝 (一步训练)
# ============================================================================
def test_mygeta_quant_prune():
    """验证: 量化模型 → MyGETA优化器 → 前向+反向一步 → 不NaN"""
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO

    model = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    dummy = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model, dummy_input=dummy)
    opt = oto.mygeta(variant="sgd", lr=0.01, lr_quant=1e-3,
        target_group_sparsity=0.5, start_pruning_step=2, pruning_periods=1,
        pruning_steps=2, bit_reduction=2, min_bit_wt=4, max_bit_wt=16, device=device)

    model.train()
    X = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
    loss = nn.CrossEntropyLoss()(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
    m = opt.compute_metrics()
    print(f"  step0: loss={loss.item():.4f}, sparsity={m.group_sparsity:.4f}", flush=True)
    assert not torch.isnan(loss), "loss is NaN"


# ============================================================================
# 3. 知识蒸馏全流程 (VGG7 student + teacher)
# ============================================================================
def test_distillation_full():
    """验证: TeacherEnsemble + SoftLabel + Feature + Weight + Distiller"""
    from only_train_once.distillation import (
        TeacherEnsemble, SoftLabelDistiller, FeatureDistiller,
        WeightDistiller, Distiller)
    from sanity_check.backends.vgg7 import vgg7_bn

    student = vgg7_bn(); teacher = vgg7_bn(); teacher.eval()

    # 3.1 TeacherEnsemble — 模拟5个epoch
    ens = TeacherEnsemble(top_n=3)
    for ep in range(5):
        ens.update(student, 1.0/(ep+1), ep, is_pruning=(ep >= 2))
    print(f"  TeacherEnsemble: best_loss={ens.pre_pruning_best['loss']:.4f}, top_n={len(ens.pruning_top_n)}", flush=True)

    # 3.2 SoftLabelDistiller
    sl = SoftLabelDistiller(temperature=4.0, alpha=0.5)
    s_log = torch.randn(4, 10); t_log = torch.randn(4, 10); labels = torch.randint(0, 10, (4,))
    kd_loss = sl.compute_loss(s_log, [t_log], labels)
    print(f"  SoftLabel: kd_loss={kd_loss.item():.4f}", flush=True)
    assert kd_loss.item() > 0

    # 3.3 FeatureDistiller (传模型+输入)
    fd = FeatureDistiller(mode='forward', weight=0.1)
    X = torch.randn(2, 3, 32, 32)
    feat_loss = fd.compute_loss(student, teacher, X, epoch=0, max_epoch=100)
    print(f"  Feature: feat_loss={feat_loss.item():.4f}", flush=True)

    # 3.4 WeightDistiller (传模型)
    wt = WeightDistiller()
    wt_loss = wt.compute_loss(student, teacher)
    print(f"  Weight: wt_loss={wt_loss.item():.4f}", flush=True)

    # 3.5 Distiller 统一接口
    d = Distiller(temperature=4.0, alpha=0.5)
    print(f"  Distiller 统一接口: OK", flush=True)


# ============================================================================
# 4. 部署模块 (模型信息提取 + 核函数配置 + 导出)
# ============================================================================
def test_deployment_full():
    """验证: extract_model_info → get_kernel_config → export_kernel_config"""
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.deployment import extract_model_info, get_kernel_config_for_layer, export_kernel_config
    import tempfile, json

    model = vgg7_bn()
    mi = extract_model_info(model)
    print(f"  extract_model_info: {len(mi.layers)} layers", flush=True)

    first_key = list(mi.layers.keys())[0]
    cfg = get_kernel_config_for_layer(mi.layers[first_key])
    print(f"  get_kernel_config: {list(cfg.keys())}", flush=True)

    tmp = tempfile.mktemp(suffix='.json')
    export_kernel_config(mi, tmp)
    with open(tmp) as f: data = json.load(f)
    print(f"  export_kernel_config: keys={list(data.keys())[:5]}", flush=True)
    os.unlink(tmp)


# ============================================================================
# 5. 子网构建与压缩导出 (剪枝 → 导出 → 前向验证)
# ============================================================================
def test_construct_subnet():
    """验证: random_set_zero → construct_subnet → 加载压缩模型 → 前向"""
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once import OTO

    model = vgg7_bn()
    dummy = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model, dummy_input=dummy)
    oto.random_set_zero_groups(target_group_sparsity=0.3)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "lightweight_test")
    os.makedirs(out_dir, exist_ok=True)
    oto.construct_subnet(out_dir=out_dir)
    compressed = oto.compressed_model_path
    print(f"  压缩模型: {compressed}", flush=True)
    assert compressed and os.path.exists(compressed)

    cm = torch.load(compressed, map_location='cpu', weights_only=False)
    cm.eval()
    with torch.no_grad(): out = cm(dummy)
    print(f"  压缩模型前向: shape={out.shape}", flush=True)
    assert out.shape == (1, 10)

    orig = sum(p.numel() for p in model.parameters())
    comp = sum(p.numel() for p in cm.parameters())
    print(f"  参数: {orig} → {comp} (压缩 {1-comp/orig:.1%})", flush=True)


# ============================================================================
# 6. 蒸馏 + 量化联合 (student量化 + teacher不量化 → KD loss)
# ============================================================================
def test_distillation_with_quantization():
    """验证: 量化student + 原始teacher → 软标签蒸馏 + 特征蒸馏"""
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once.distillation import SoftLabelDistiller, FeatureDistiller

    student = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    teacher = vgg7_bn(); teacher.eval()

    X = torch.randn(2, 3, 32, 32)
    s_out = student(X); t_out = teacher(X)

    # 软标签蒸馏
    sl = SoftLabelDistiller(temperature=4.0, alpha=0.5)
    labels = torch.randint(0, 10, (2,))
    kd_loss = sl.compute_loss(s_out, [t_out], labels)
    print(f"  量化student KD loss={kd_loss.item():.4f}", flush=True)
    assert not torch.isnan(kd_loss)

    # 特征蒸馏
    fd = FeatureDistiller(mode='forward', weight=0.1)
    feat_loss = fd.compute_loss(student, teacher, X, epoch=0, max_epoch=100)
    print(f"  量化student feat_loss={feat_loss.item():.4f}", flush=True)
    assert not torch.isnan(feat_loss)


# ============================================================================
# 7. 完整流程: 量化训练1步 → 蒸馏 → 剪枝导出
# ============================================================================
def test_full_pipeline():
    """端到端: GETA量化训练1步 → 蒸馏loss → 非量化模型剪枝导出 → 前向验证"""
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO
    from only_train_once.distillation import SoftLabelDistiller

    # 1) 量化训练1步 (验证量化训练不NaN)
    q_model = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    teacher = vgg7_bn(); teacher.eval()
    dummy = torch.rand(1, 3, 32, 32)
    oto_q = OTO(model=q_model, dummy_input=dummy)
    opt = oto_q.geta(variant="sgd", lr=1e-3, lr_quant=1e-3,
        target_group_sparsity=0.3, start_projection_step=0, projection_periods=1,
        projection_steps=1, start_pruning_step=1, pruning_periods=1, pruning_steps=1,
        bit_reduction=2, min_bit_wt=4, max_bit_wt=16)
    q_model.train()
    X = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
    loss = nn.CrossEntropyLoss()(q_model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  量化训练1步: loss={loss.item():.4f}", flush=True)

    # 2) 蒸馏 loss (量化student → 原始teacher)
    q_model.eval()
    with torch.no_grad():
        s_out = q_model(X); t_out = teacher(X)
    sl = SoftLabelDistiller(temperature=4.0, alpha=0.5)
    kd_loss = sl.compute_loss(s_out, [t_out], y)
    print(f"  蒸馏loss: kd={kd_loss.item():.4f}", flush=True)

    # 3) 非量化模型剪枝导出 (量化模型导出通道不匹配是已知限制)
    model = vgg7_bn()
    oto = OTO(model=model, dummy_input=dummy)
    oto.random_set_zero_groups(target_group_sparsity=0.3)
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "full_pipeline_test")
    os.makedirs(out_dir, exist_ok=True)
    oto.construct_subnet(out_dir=out_dir)
    compressed = oto.compressed_model_path
    cm = torch.load(compressed, map_location='cpu', weights_only=False)
    cm.eval()
    with torch.no_grad(): out = cm(dummy)
    print(f"  压缩模型前向: shape={out.shape}", flush=True)
    assert out.shape == (1, 10)
    assert not torch.isnan(out).any(), "输出含NaN"


if __name__ == "__main__":
    print("="*60, flush=True)
    print("GETA 框架综合轻量化验证 (CPU, VGG7, 1-2步训练)", flush=True)
    print("="*60, flush=True)
    print(f"PyTorch: {torch.__version__}  设备: {device}", flush=True)

    check("1. GETA 量化剪枝 (1步)", test_geta_quant_prune)
    check("2. MyGETA 量化剪枝 (1步)", test_mygeta_quant_prune)
    check("3. 知识蒸馏全流程", test_distillation_full)
    check("4. 部署模块", test_deployment_full)
    check("5. 子网构建与压缩导出", test_construct_subnet)
    check("6. 蒸馏+量化联合", test_distillation_with_quantization)
    check("7. 完整流程 (量化→蒸馏→剪枝导出)", test_full_pipeline)

    print("\n" + "="*60, flush=True)
    print(f"通过: {len(PASSED)}/{len(PASSED)+len(FAILED)}", flush=True)
    for p in PASSED: print(f"  [PASS] {p}", flush=True)
    if FAILED:
        print(f"失败: {len(FAILED)}/{len(PASSED)+len(FAILED)}", flush=True)
        for name, err in FAILED:
            print(f"  [FAIL] {name}\n         {err}", flush=True)
    print("="*60, flush=True)
