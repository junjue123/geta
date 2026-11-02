#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分段计时: 定位训练慢在哪 (数据 / forward / backward / opt.step)"""
import os, sys, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=None)
ap.add_argument("--quant", default="1", help="1=量化模型, 0=纯浮点模型")
ap.add_argument("--variant", default="adam")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--bs", type=int, default=128)
args = ap.parse_args()

SCRIPT = r"E:/pythonProject/geta-main/geta-main_origin"
ROOT = os.path.abspath(args.root or SCRIPT)
sys.path.insert(0, ROOT)
print(f"[ENV] root={ROOT} quant={args.quant} variant={args.variant}")

import numpy as np, torch, torch.nn as nn, pickle
from torch.utils.data import DataLoader
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10

torch.manual_seed(0)
DEV = "cuda"
torch.backends.cudnn.benchmark = True

# ---- 数据 (与 run_daco_vs_baseline 的 fast 管道一致) ----
sys.path.insert(0, SCRIPT)
from run_daco_vs_baseline import _FastCIFARLocal
ds = _FastCIFARLocal(os.path.join(SCRIPT, "datasets/cifar10"), train=True, augment=True)
loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0,
                    pin_memory=False, drop_last=True)

m = resnet20_cifar10()
if args.quant == "1":
    m = model_to_quantize_model(m, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
m = m.to(DEV)
dummy = torch.rand(1, 3, 32, 32).to(DEV)

oto = OTO(model=m, dummy_input=dummy)
if args.quant == "1":
    opt = oto.geta(variant=args.variant, lr=1e-3, lr_quant=1e-3, weight_decay=1e-4,
                   target_group_sparsity=0.5, start_projection_step=10**9,
                   start_pruning_step=10**9, bit_reduction=1, min_bit_wt=4,
                   max_bit_wt=16, min_bit_act=2, max_bit_act=16,
                   importance_score_criteria="default", verbose="False",
                   device=DEV, log_dir="temp/profile")
else:
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

crit = nn.CrossEntropyLoss()
m.train()

t_data = t_fwd = t_bwd = t_step = 0.0
it = 0
torch.cuda.synchronize()
t_prev = time.time()
for X, y in loader:
    if it >= args.steps:
        break
    torch.cuda.synchronize(); t0 = time.time()
    X, y = X.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
    torch.cuda.synchronize(); t1 = time.time()
    out = m(X)
    torch.cuda.synchronize(); t2 = time.time()
    loss = crit(out, y)
    opt.zero_grad(); loss.backward()
    torch.cuda.synchronize(); t3 = time.time()
    opt.step()
    torch.cuda.synchronize(); t4 = time.time()
    t_data += t1 - t0; t_fwd += t2 - t1
    t_bwd += t3 - t2; t_step += t4 - t3
    it += 1

tot = t_data + t_fwd + t_bwd + t_step
print(f"\nsteps={it} bs={args.bs} quant={args.quant} variant={args.variant}")
print(f"  data    {t_data/it*1000:7.1f} ms/step  ({t_data/tot*100:4.1f}%)")
print(f"  forward {t_fwd/it*1000:7.1f} ms/step  ({t_fwd/tot*100:4.1f}%)")
print(f"  backward{t_bwd/it*1000:7.1f} ms/step  ({t_bwd/tot*100:4.1f}%)")
print(f"  opt.step{t_step/it*1000:7.1f} ms/step  ({t_step/tot*100:4.1f}%)")
print(f"  TOTAL   {tot/it*1000:7.1f} ms/step  -> {tot/it*390:.0f} s/epoch(390 steps)")
