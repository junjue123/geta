#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针: 打印第一次 backward 后各参数梯度量级, 定位 1e9 梯度来源"""
import os, sys
ROOT = "E:/pythonProject/geta-main/geta-main_origin"
sys.path.insert(0, ROOT); os.chdir(ROOT)

import torch, torch.nn as nn, numpy as np, pickle

from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

torch.manual_seed(0)
DEV = "cuda"
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10

m = model_to_quantize_model(resnet20_cifar10(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(DEV)

base = "datasets/cifar10/cifar-10-batches-py"
with open(os.path.join(base, "data_batch_1"), "rb") as f:
    d = pickle.load(f, encoding="latin1")
X = d["data"][:128].reshape(-1,3,32,32).astype(np.float32)/255.
mean = np.array([0.4914,0.4822,0.4465],dtype=np.float32).reshape(1,3,1,1)
std  = np.array([0.2470,0.2435,0.2616],dtype=np.float32).reshape(1,3,1,1)
X = torch.from_numpy((X-mean)/std).to(DEV)
y = torch.from_numpy(np.array(d["labels"][:128],dtype=np.int64)).to(DEV)

out = m(X)
loss = nn.CrossEntropyLoss()(out, y)
print(f"forward: out.absmean={out.abs().mean().item():.4e} loss={loss.item():.4f}")
loss.backward()

print(f"\n{'param':<40}{'shape':<20}{'grad_absmax':>14}{'grad_absmean':>14}")
rows=[]
for n,p in m.named_parameters():
    if p.grad is None:
        rows.append((n, tuple(p.shape), -1.0, -1.0, 0.0))
        continue
    rows.append((n, tuple(p.shape), p.grad.abs().max().item(), p.grad.abs().mean().item(), p.data.abs().max().item()))
# 按梯度大小排序, 只看最大的 15 个
rows_ok=[r for r in rows if r[2]>=0]
rows_ok.sort(key=lambda r:-r[2])
for n,s,gmax,gmean,wmax in rows_ok[:15]:
    print(f"{n:<40}{str(s):<20}{gmax:>14.4e}{gmean:>14.4e}  w={wmax:.3e}")
print("\n--- 量化参数梯度 ---")
for n,s,gmax,gmean,wmax in rows_ok:
    if "quant" in n:
        print(f"{n:<40}{str(s):<20}{gmax:>14.4e}{gmean:>14.4e}  w={wmax:.3e}")
print("\n--- 无梯度参数 ---")
for n,s,gmax,gmean,wmax in rows:
    if gmax<0: print("  ", n)
