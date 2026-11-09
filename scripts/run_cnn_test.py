"""只测 CNN 模型 (跳过 ViT, 图构建太慢)"""
import os, sys, time, traceback
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})", flush=True)

PASSED, FAILED = [], []
def check(name, func):
    print(f"\n[CHECK] {name}", flush=True)
    t0=time.time()
    try:
        func()
        print(f">>> [PASS] {name} ({time.time()-t0:.1f}s)", flush=True)
        PASSED.append(name)
    except Exception as e:
        print(f">>> [FAIL] {name} ({time.time()-t0:.1f}s) {str(e)[:80]}", flush=True)
        FAILED.append((name, str(e)[:80]))

def run(model, dummy_shape, use_geta=True, mark=None):
    from only_train_once import OTO
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    from only_train_once.distillation import SoftLabelDistiller
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(device)
    dummy = torch.rand(*dummy_shape).to(device)
    oto = OTO(model=model, dummy_input=dummy)
    if mark: oto.mark_unprunable_by_param_names(mark)
    if use_geta:
        opt = oto.geta(variant="sgd", lr=1e-3, lr_quant=1e-3, target_group_sparsity=0.3,
            start_projection_step=0, projection_periods=1, projection_steps=1,
            start_pruning_step=1, pruning_periods=1, pruning_steps=1,
            bit_reduction=2, min_bit_wt=4, max_bit_wt=16)
    else:
        opt = oto.mygeta(variant="sgd", lr=0.01, lr_quant=1e-3, target_group_sparsity=0.3,
            start_pruning_step=1, pruning_periods=1, pruning_steps=1,
            bit_reduction=2, min_bit_wt=4, max_bit_wt=16, device=device)
    model.train()
    X=torch.randn(2,*dummy_shape[1:]).to(device); y=torch.randint(0,10,(2,)).to(device)
    loss=nn.CrossEntropyLoss()(model(X),y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  loss={loss.item():.4f}", flush=True)
    assert not torch.isnan(loss)
    model.eval()
    with torch.no_grad(): s=model(X)
    kd=SoftLabelDistiller(temperature=4.0,alpha=0.5).compute_loss(s,[torch.randn_like(s)],y)
    print(f"  kd={kd.item():.4f}", flush=True)
    assert not torch.isnan(kd)

# MobileNetV2
def t1():
    from sanity_check.backends.mobilenetv2 import mobilenetv2_cifar10
    run(mobilenetv2_cifar10(), (1,3,32,32))
# ConvNeXt-Tiny
def t2():
    from sanity_check.backends.convnext import convnext_tiny
    run(convnext_tiny(num_classes=10), (1,3,32,32))
# VGG7 + MyGETA
def t3():
    from sanity_check.backends.vgg7 import vgg7_bn
    run(vgg7_bn(), (1,3,32,32), use_geta=False)
# ResNet20
def t4():
    from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
    run(resnet20_cifar10(), (1,3,32,32))
# VGG7 (复测确认GPU)
def t5():
    from sanity_check.backends.vgg7 import vgg7_bn
    run(vgg7_bn(), (1,3,32,32))

check("1. MobileNetV2", t1)
check("2. ConvNeXt-Tiny", t2)
check("3. VGG7+MyGETA", t3)
check("4. ResNet20", t4)
check("5. VGG7(GETA复测)", t5)

print(f"\n通过:{len(PASSED)}/{len(PASSED)+len(FAILED)}", flush=True)
for n,_ in FAILED: print(f"  FAIL: {n}", flush=True)
