#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化层健康检查 —— 定位"量化层失效"到底崩在哪一步

思路: 跑 N 个真实训练 step, 每步打印量化参数(d_quant/q_m/t)的统计和
量化后权重的统计, 找出参数发散/输出归零的临界 step。

用法:
  python diag_quant_health.py --steps 30
  python diag_quant_health.py --steps 30 --root E:/pythonProject/geta-main/geta_origin_main
"""

import os
import sys
import argparse

p = argparse.ArgumentParser()
p.add_argument("--root", default=None, help="框架根目录 (默认当前脚本目录)")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--lr-quant", type=float, default=1e-3)
p.add_argument("--variant", default="adam")
p.add_argument("--mode", default="WA", choices=["WA", "W"], help="WA=权重+激活, W=仅权重")
p.add_argument("--data", default="E:/pythonProject/geta-main/geta-main_origin/datasets/cifar10")
args = p.parse_args()

ROOT = os.path.abspath(args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
print(f"[ENV] framework root = {ROOT}")

import torch
import torch.nn as nn
import numpy as np

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode, QuantizeConv2d, QuantizeLinear

torch.manual_seed(0)
np.random.seed(0)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 数据 ----------------
def get_loader(n_batch=8, bs=128):
    import pickle
    base = os.path.join(args.data, "cifar-10-batches-py")
    with open(os.path.join(base, "data_batch_1"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    X = d["data"][: n_batch * bs].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std
    y = np.array(d["labels"][: n_batch * bs], dtype=np.int64)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return [(X[i:i + bs], y[i:i + bs]) for i in range(0, len(X), bs)]


# ---------------- 量化参数快照 ----------------
def snap_quant_params(model, tag):
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, (QuantizeConv2d, QuantizeLinear)):
            continue
        r = {"layer": name}
        for pname in ["d_quant_wt", "q_m_wt", "t_quant_wt", "d_quant_act", "q_m_act", "t_quant_act"]:
            if hasattr(m, pname):
                r[pname] = float(getattr(m, pname).detach().item())
        # 量化后权重统计
        with torch.no_grad():
            wq = m.quantize_weight(m.weight.detach())
        r["wq_absmean"] = float(wq.abs().mean().item())
        r["wq_zero_frac"] = float((wq == 0).float().mean().item())
        r["wq_max"] = float(wq.abs().max().item())
        r["w_absmax"] = float(m.weight.detach().abs().max().item())
        rows.append(r)
    return rows


def print_snap(rows, step):
    print(f"\n===== step {step} =====")
    hdr = f"{'layer':<28}{'d_wt':>10}{'q_m_wt':>10}{'t_wt':>8}{'d_act':>10}{'q_m_act':>10}{'w_absmax':>10}{'wq_absmean':>12}{'wq_zero%':>10}"
    print(hdr)
    for r in rows:
        print(f"{r['layer']:<28}"
              f"{r.get('d_quant_wt', float('nan')):>10.3e}"
              f"{r.get('q_m_wt', float('nan')):>10.3e}"
              f"{r.get('t_quant_wt', float('nan')):>8.3f}"
              f"{r.get('d_quant_act', float('nan')):>10.3e}"
              f"{r.get('q_m_act', float('nan')):>10.3e}"
              f"{r['w_absmax']:>10.3e}"
              f"{r['wq_absmean']:>12.3e}"
              f"{r['wq_zero_frac'] * 100:>9.1f}%")


# ---------------- 主流程 ----------------
def main():
    from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10

    model = resnet20_cifar10()
    qm = QuantizationMode.WEIGHT_AND_ACTIVATION if args.mode == "WA" else QuantizationMode.WEIGHT_ONLY
    model = model_to_quantize_model(model, quant_mode=qm)
    model = model.to(DEV)

    dummy = torch.rand(1, 3, 32, 32).to(DEV)

    # 初始 forward 健康度
    model.eval()
    with torch.no_grad():
        out0 = model(dummy)
    print(f"[INIT FORWARD] out.absmean={out0.abs().mean().item():.6e} out.absmax={out0.abs().max().item():.6e}")
    rows = snap_quant_params(model, "init")
    print_snap(rows, 0)

    oto = OTO(model=model, dummy_input=dummy)
    opt = oto.geta(
        variant=args.variant, lr=args.lr, lr_quant=args.lr_quant, weight_decay=1e-4,
        target_group_sparsity=0.5, start_projection_step=10 ** 9,
        start_pruning_step=10 ** 9,  # 关闭剪枝, 只看量化训练本身
        bit_reduction=1, min_bit_wt=4, max_bit_wt=16, min_bit_act=2, max_bit_act=16,
        importance_score_criteria="default", verbose="False", device=DEV,
        log_dir="temp/diag_quant",
    )

    loader = get_loader(n_batch=4)
    crit = nn.CrossEntropyLoss()
    model.train()

    it = 0
    for X, y in loader:
        if it >= args.steps:
            break
        X, y = X.to(DEV), y.to(DEV)
        out = model(X)
        loss = crit(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        it += 1
        if it % 5 == 0 or it == 1:
            with torch.no_grad():
                o = model(X[:32])
            print(f"\n[step {it}] loss={loss.item():.4f} "
                  f"out.absmean={o.abs().mean().item():.4e} "
                  f"out.absmax={o.abs().max().item():.4e} "
                  f"pred_uniq={o.argmax(1).unique().numel()}")
            print_snap(snap_quant_params(model, str(it)), it)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
