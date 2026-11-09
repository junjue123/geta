#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断2: 纯 ResNet-20 (不量化) 能否正常训练"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from train import get_dataloaders
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10

device = torch.device("cuda")
print("=== 纯 ResNet-20 (不量化) 普通SGD ===")
tl, vl = get_dataloaders("datasets/cifar10", batch_size=128, num_workers=0)
model = resnet20_cifar10().to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

model.train()
bi = iter(tl)
for step in range(10):
    X, y = next(bi)
    X, y = X.to(device), y.to(device)
    out = model(X)
    loss = criterion(out, y)
    opt.zero_grad(); loss.backward(); opt.step()
    acc = (out.argmax(1) == y).float().mean()
    if step % 2 == 0:
        print(f"  step{step}: loss={loss.item():.4f} acc={acc:.4f}")

# 1 epoch 后评估
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X, y in vl:
        X, y = X.to(device), y.to(device)
        out = model(X)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
print(f"  10 step 后 val_acc={correct/total:.4f}")
