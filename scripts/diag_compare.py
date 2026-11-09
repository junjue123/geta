#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比: run_daco_cifar10 原版配置 vs 我的配置, 3 epoch 真实数据看 loss"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from train import get_dataloaders, build_model, configure_training_schedule, collect_metrics, BASELINE_OVERRIDE, PAPER_DEFAULTS
from only_train_once import OTO

device = torch.device("cuda")
tl, vl = get_dataloaders("datasets/cifar10", batch_size=128, num_workers=0)
steps = len(tl)
criterion = nn.CrossEntropyLoss()

def run_3epoch(tag, pruning_periods, warmup_ep):
    print(f"\n=== {tag}: pruning_periods={pruning_periods}, warmup={warmup_ep} ===")
    model, dummy = build_model("resnet20", device)
    oto = OTO(model=model, dummy_input=dummy.to(device))
    # 原版 run_daco_cifar10 的参数 (pruning_periods=5)
    opt = oto.geta(
        variant="sgd", lr=0.1, lr_quant=1e-3, weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0, projection_periods=1, projection_steps=1,
        start_pruning_step=0, pruning_periods=pruning_periods, pruning_steps=1,
        bit_reduction=1, min_bit_wt=4, max_bit_wt=16, min_bit_act=2, max_bit_act=16,
        importance_score_criteria="default", verbose="False",
        device=str(device), log_dir=f"outputs/diag_{tag}")
    # warmup 阶段调度
    configure_training_schedule(opt, warmup_ep, 3, steps)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(warmup_ep+3)*steps, eta_min=1e-3)

    for ep in range(warmup_ep + 3):
        phase = "warmup" if ep < warmup_ep else "joint"
        model.train()
        losses = []
        bi = iter(tl)
        for b in range(steps):
            X, y = next(bi)
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            losses.append(loss.item())
        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in vl:
                X, y = X.to(device), y.to(device)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        import numpy as np
        print(f"  Ep{ep+1}/{warmup_ep+3} [{phase}] loss={np.mean(losses):.4f} val_acc={correct/total:.4f}")

# 测试1: 原版配置 pruning_periods=5, warmup=3
run_3epoch("orig_pp5", pruning_periods=5, warmup_ep=3)
