"""
GETA 全功能验证测试 (GPU)
用 VGG7 跑一遍所有功能模块: 量化训练 + 5种蒸馏 + 部署 + 压缩导出 + 完整流程
"""
import os, sys, time, traceback
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

PASSED, FAILED = [], []
def check(name, func):
    print(f"\n[CHECK] {name}", flush=True)
    t0=time.time()
    try:
        func(); print(f">>> [PASS] {name} ({time.time()-t0:.1f}s)", flush=True); PASSED.append(name)
    except Exception as e:
        print(f">>> [FAIL] {name} ({time.time()-t0:.1f}s) {str(e)[:80]}", flush=True)
        traceback.print_exc(); FAILED.append((name, str(e)[:80]))

# ── 1. GETA 量化剪枝训练 ──
def test_geta_train():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO
    m = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    dummy = torch.rand(1,3,32,32).to(device)
    oto = OTO(model=m, dummy_input=dummy)
    opt = oto.geta(variant="sgd", lr=1e-3, lr_quant=1e-3, target_group_sparsity=0.3,
        start_projection_step=0, projection_periods=1, projection_steps=1,
        start_pruning_step=1, pruning_periods=1, pruning_steps=1, bit_reduction=2, min_bit_wt=4, max_bit_wt=16)
    m.train(); X=torch.randn(2,3,32,32).to(device); y=torch.randint(0,10,(2,)).to(device)
    loss=nn.CrossEntropyLoss()(m(X),y); opt.zero_grad(); loss.backward(); opt.step()
    print(f"  loss={loss.item():.4f}", flush=True); assert not torch.isnan(loss)

# ── 2. MyGETA 量化剪枝训练 ──
def test_mygeta_train():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO
    m = model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    dummy = torch.rand(1,3,32,32).to(device)
    oto = OTO(model=m, dummy_input=dummy)
    opt = oto.mygeta(variant="sgd", lr=0.01, lr_quant=1e-3, target_group_sparsity=0.3,
        start_pruning_step=1, pruning_periods=1, pruning_steps=1, bit_reduction=2, min_bit_wt=4, max_bit_wt=16, device=device)
    m.train(); X=torch.randn(2,3,32,32).to(device); y=torch.randint(0,10,(2,)).to(device)
    loss=nn.CrossEntropyLoss()(m(X),y); opt.zero_grad(); loss.backward(); opt.step()
    print(f"  loss={loss.item():.4f}", flush=True); assert not torch.isnan(loss)

# ── 3. 软标签蒸馏 (SoftLabelDistiller) ──
def test_soft_label():
    from only_train_once.distillation import SoftLabelDistiller
    sl = SoftLabelDistiller(temperature=4.0, alpha=0.5)
    s=torch.randn(4,10).to(device); t=torch.randn(4,10).to(device); y=torch.randint(0,10,(4,)).to(device)
    loss = sl.compute_loss(s,[t],y)
    print(f"  kd_loss={loss.item():.4f}", flush=True); assert loss.item()>0 and not torch.isnan(loss)

# ── 4. 特征蒸馏 (FeatureDistiller) ──
def test_feature_distill():
    from only_train_once.distillation import FeatureDistiller
    from sanity_check.backends.vgg7 import vgg7_bn
    s=vgg7_bn().to(device); t=vgg7_bn().to(device); t.eval()
    fd = FeatureDistiller(mode='forward', weight=0.1)
    X=torch.randn(2,3,32,32).to(device)
    loss = fd.compute_loss(s, t, X, epoch=0, max_epoch=100)
    print(f"  feat_loss={loss.item():.4f}", flush=True); assert not torch.isnan(loss)

# ── 5. 权重蒸馏 (WeightDistiller) ──
def test_weight_distill():
    from only_train_once.distillation import WeightDistiller
    from sanity_check.backends.vgg7 import vgg7_bn
    s=vgg7_bn().to(device); t=vgg7_bn().to(device)
    wt = WeightDistiller()
    loss = wt.compute_loss(s, t)
    print(f"  wt_loss={loss.item():.4f}", flush=True); assert not torch.isnan(loss)

# ── 6. TeacherEnsemble (教师集合管理) ──
def test_teacher_ensemble():
    from only_train_once.distillation import TeacherEnsemble
    from sanity_check.backends.vgg7 import vgg7_bn
    m=vgg7_bn().to(device)
    ens = TeacherEnsemble(top_n=3)
    for ep in range(5): ens.update(m, 1.0/(ep+1), ep, is_pruning=(ep>=2))
    print(f"  best_loss={ens.pre_pruning_best['loss']:.4f} top_n={len(ens.pruning_top_n)}", flush=True)
    assert ens.pre_pruning_best is not None

# ── 7. Distiller 统一接口 ──
def test_distiller_unified():
    from only_train_once.distillation import Distiller
    d = Distiller(temperature=4.0, alpha=0.5, feature_weight=0.1, weight_weight=0.01)
    print(f"  soft_label={type(d.soft_label).__name__} feature={type(d.feature).__name__} weight={type(d.weight).__name__}", flush=True)

# ── 8. 部署: 模型信息提取 ──
def test_deployment_info():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.deployment import extract_model_info, get_kernel_config_for_layer, export_kernel_config
    import tempfile, json
    m = vgg7_bn()
    mi = extract_model_info(m)
    print(f"  layers={len(mi.layers)}", flush=True)
    k0 = list(mi.layers.keys())[0]
    cfg = get_kernel_config_for_layer(mi.layers[k0])
    print(f"  kernel_config keys={list(cfg.keys())[:5]}", flush=True)
    tmp = tempfile.mktemp(suffix='.json')
    export_kernel_config(mi, tmp)
    with open(tmp) as f: data=json.load(f)
    os.unlink(tmp)
    print(f"  export keys={list(data.keys())[:5]}", flush=True)

# ── 9. 子网构建与压缩导出 ──
def test_compress_export():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once import OTO
    m=vgg7_bn().to(device); dummy=torch.rand(1,3,32,32).to(device)
    oto=OTO(model=m, dummy_input=dummy)
    oto.random_set_zero_groups(target_group_sparsity=0.3)
    out_dir=os.path.join(os.path.dirname(__file__),"outputs","full_func_test")
    os.makedirs(out_dir, exist_ok=True)
    oto.construct_subnet(out_dir=out_dir)
    cm=torch.load(oto.compressed_model_path, map_location=device, weights_only=False).to(device)
    cm.eval()
    with torch.no_grad(): out=cm(dummy)
    orig=sum(p.numel() for p in m.parameters()); comp=sum(p.numel() for p in cm.parameters())
    print(f"  {orig}→{comp} ({1-comp/orig:.1%}) out={out.shape}", flush=True)
    assert out.shape==(1,10) and not torch.isnan(out).any()

# ── 10. 蒸馏+量化联合 ──
def test_quant_distill():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once.distillation import SoftLabelDistiller
    s=model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    t=vgg7_bn().to(device); t.eval()
    X=torch.randn(2,3,32,32).to(device); y=torch.randint(0,10,(2,)).to(device)
    with torch.no_grad(): so=s(X); to=t(X)
    kd=SoftLabelDistiller(temperature=4.0,alpha=0.5).compute_loss(so,[to],y)
    print(f"  quant_kd={kd.item():.4f}", flush=True); assert not torch.isnan(kd)

# ── 11. 完整流程: 量化训练→蒸馏→剪枝导出 ──
def test_full_pipeline():
    from sanity_check.backends.vgg7 import vgg7_bn
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once import OTO
    from only_train_once.distillation import SoftLabelDistiller
    # 量化训练1步
    qm=model_to_quantize_model(vgg7_bn(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    teacher=vgg7_bn().to(device); teacher.eval()
    dummy=torch.rand(1,3,32,32).to(device)
    oto=OTO(model=qm, dummy_input=dummy)
    opt=oto.geta(variant="sgd",lr=1e-3,lr_quant=1e-3,target_group_sparsity=0.3,
        start_projection_step=0,projection_periods=1,projection_steps=1,
        start_pruning_step=1,pruning_periods=1,pruning_steps=1,bit_reduction=2,min_bit_wt=4,max_bit_wt=16)
    qm.train(); X=torch.randn(2,3,32,32).to(device); y=torch.randint(0,10,(2,)).to(device)
    loss=nn.CrossEntropyLoss()(qm(X),y); opt.zero_grad(); loss.backward(); opt.step()
    print(f"  train_loss={loss.item():.4f}", flush=True)
    # 蒸馏
    qm.eval()
    with torch.no_grad(): so=qm(X); to=teacher(X)
    kd=SoftLabelDistiller(temperature=4.0,alpha=0.5).compute_loss(so,[to],y)
    print(f"  kd={kd.item():.4f}", flush=True)
    # 剪枝导出(非量化模型)
    m2=vgg7_bn().to(device); oto2=OTO(model=m2,dummy_input=dummy)
    oto2.random_set_zero_groups(target_group_sparsity=0.3)
    out_dir=os.path.join(os.path.dirname(__file__),"outputs","pipeline_test")
    os.makedirs(out_dir,exist_ok=True)
    oto2.construct_subnet(out_dir=out_dir)
    cm=torch.load(oto2.compressed_model_path,map_location=device,weights_only=False).to(device)
    cm.eval()
    with torch.no_grad(): out=cm(dummy)
    print(f"  compressed_out={out.shape}", flush=True)
    assert out.shape==(1,10) and not torch.isnan(out).any()

if __name__=="__main__":
    print("="*60, flush=True)
    print("GETA 全功能验证 (GPU)", flush=True)
    print("="*60, flush=True)
    check("1.  GETA量化剪枝训练", test_geta_train)
    check("2.  MyGETA量化剪枝训练", test_mygeta_train)
    check("3.  软标签蒸馏", test_soft_label)
    check("4.  特征蒸馏", test_feature_distill)
    check("5.  权重蒸馏", test_weight_distill)
    check("6.  TeacherEnsemble", test_teacher_ensemble)
    check("7.  Distiller统一接口", test_distiller_unified)
    check("8.  部署:模型信息提取", test_deployment_info)
    check("9.  子网构建与压缩导出", test_compress_export)
    check("10. 蒸馏+量化联合", test_quant_distill)
    check("11. 完整流程", test_full_pipeline)
    print(f"\n通过:{len(PASSED)}/{len(PASSED)+len(FAILED)}", flush=True)
    for n,_ in FAILED: print(f"  FAIL: {n}", flush=True)
