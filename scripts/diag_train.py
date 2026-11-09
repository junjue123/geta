#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 量化 ResNet-20 为何 loss 不降"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from train import get_dataloaders, build_model

device = torch.device("cuda")
print("=== 诊断: 量化 ResNet-20 训练 ===")

tl, vl = get_dataloaders("datasets/cifar10", batch_size=128, num_workers=0)
model, dummy = build_model("resnet20", device)
criterion = nn.CrossEntropyLoss()

# 测试1: 量化模型 forward 输出分布
model.eval()
X, y = next(iter(tl))
X, y = X.to(device), y.to(device)
with torch.no_grad():
    out = model(X)
print(f"forward 输出: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
print(f"  各样本输出方差: {out.var(dim=1).mean():.6f} (若~0 说明输出退化)")
print(f"  预测分布: {out.argmax(1).bincount(minlength=10).tolist()}")

# 测试2: 普通SGD 训量化模型 3 batch, 看 loss 是否降
print("\n=== 测试2: 普通SGD 训量化模型 ===")
opt_sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
model.train()
batch_iter = iter(tl)
for step in range(6):
    X, y = next(batch_iter)
    X, y = X.to(device), y.to(device)
    out = model(X)
    loss = criterion(out, y)
    opt_sgd.zero_grad()
    loss.backward()
    opt_sgd.step()
    acc = (out.argmax(1) == y).float().mean()
    print(f"  step{step}: loss={loss.item():.4f} acc={acc:.4f}")

# 测试3: GETA warmup 更新是否生效
print("\n=== 测试3: GETA warmup 阶段 ===")
from only_train_once import OTO
model2, dummy2 = build_model("resnet20", device)
oto = OTO(model=model2, dummy_input=dummy2.to(device))
opt_geta = oto.geta(
    variant="sgd", lr=0.1, lr_quant=1e-3, weight_decay=1e-4,
    target_group_sparsity=0.5, start_projection_step=99999,
    start_pruning_step=99999, pruning_steps=1, pruning_periods=1,
    bit_reduction=1, min_bit_wt=4, max_bit_wt=16, min_bit_act=2, max_bit_act=16,
    importance_score_criteria="default", verbose="False",
    device=str(device), log_dir="outputs/diag")
# 检查 grad_variant 是否含权重
print(f"  param_groups 数: {len(opt_geta.param_groups)}")
for gi, g in enumerate(opt_geta.param_groups):
    names = g.get("p_names", [])
    gv = g.get("grad_variant", {})
    print(f"  group{gi}: is_prunable={g.get('is_prunable')} "
          f"params={len(g['params'])} grad_variant_keys={len(gv)}")
    if names:
        in_gv = [n for n in names if n in gv]
        not_gv = [n for n in names if n not in gv]
        print(f"    在grad_variant: {len(in_gv)}/{len(names)}")
        if not_gv[:3]:
            print(f"    不在grad_variant(前3): {not_gv[:3]}")

# 训3 batch看loss
model2.train()
batch_iter = iter(tl)
for step in range(6):
    X, y = next(batch_iter)
    X, y = X.to(device), y.to(device)
    out = model2(X)
    loss = criterion(out, y)
    opt_geta.zero_grad()
    loss.backward()
    opt_geta.step()
    acc = (out.argmax(1) == y).float().mean()
    print(f"  GETA step{step}: loss={loss.item():.4f} acc={acc:.4f}")
