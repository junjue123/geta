#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""优化前后数值一致性验证: 固定 seed + 固定数据顺序, 记录逐 step loss。

用于确认 quant_layers 的性能优化(pow/where/masked_fill)不改变训练轨迹。
"""
import os, sys, json, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--out", required=True)
args = ap.parse_args()

SCRIPT = r"E:/pythonProject/geta-main/geta-main_origin"
sys.path.insert(0, SCRIPT)
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
from run_daco_vs_baseline import _FastCIFARLocal

DEV = "cuda"


def set_seed(s=1234):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


set_seed()
# shuffle=False + 固定顺序 -> 两次运行的输入完全相同
ds = _FastCIFARLocal(os.path.join(SCRIPT, "datasets/cifar10"), train=True, augment=False)
loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0,
                    pin_memory=True, drop_last=True)

set_seed()
m = resnet20_cifar10()
m = model_to_quantize_model(m, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(DEV)
dummy = torch.rand(1, 3, 32, 32).to(DEV)

opt = OTO(model=m, dummy_input=dummy).geta(
    variant="adam", lr=1e-3, lr_quant=1e-3, weight_decay=1e-4,
    target_group_sparsity=0.5,
    start_projection_step=0, projection_periods=1, projection_steps=1,
    start_pruning_step=10 ** 9, pruning_periods=1, pruning_steps=1,
    bit_reduction=1, min_bit_wt=4, max_bit_wt=16, min_bit_act=2, max_bit_act=16,
    importance_score_criteria="default", verbose="False", device=DEV,
    log_dir="temp/consistency",
)

crit = nn.CrossEntropyLoss()
m.train()
losses = []
it = 0
for X, y in loader:
    if it >= args.steps:
        break
    X, y = X.to(DEV), y.to(DEV)
    out = m(X)
    loss = crit(out, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    it += 1

# 同时记录最终量化参数与权重, 确认状态一致
snap = {}
for n, p in m.named_parameters():
    if any(k in n for k in ("d_quant", "q_m", "t_quant")):
        snap[n] = float(p.detach().item())
wsum = sum(float(p.detach().abs().sum().item()) for n, p in m.named_parameters()
           if "quant" not in n)

json.dump({"losses": losses, "quant_params": snap, "weight_l1": wsum},
          open(args.out, "w"), indent=1)
print(f"steps={it} first_loss={losses[0]:.6f} last_loss={losses[-1]:.6f} weight_l1={wsum:.8f}")
print(f"-> {args.out}")
