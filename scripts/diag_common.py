#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1/E2 诊断脚本的共享工具层: 数据读取 / 模型构建 / 权重加载 / 评估。

约定
----
- 本仓库的 CIFAR-10 以原生 pickle 存于 datasets/cifar10/cifar-10-batches-py,
  不依赖 torchvision 的数据集下载逻辑(离线环境可用)。
- checkpoint 有两种格式(见 train.py):
    * 续训格式: {"epoch", "model", "scheduler", ...}
    * 最终格式: {"model_state_dict", "val_acc", "history"}
  本模块统一用 _state_dict_of() 取出 state_dict。
- 训练时存的是 **量化模型**(model_to_quantize_model)的 state_dict。两条使用路径:

    (1) **评估精度** → 必须 load_quant_model(), 即用量化模型加载完整 state_dict。
        实测: 同一 ckpt, 量化模型 val_acc=48.6%(与 history 的 47.6% 一致), 而
        直接用浮点模型 + 同名 conv/bn 参数组合只有 11.3%(随机水平), 且 conv 输出
        逐块放大到 1e10。原因是量化路径的 BN 统计与浮点激活尺度不一致 ——
        浮点模型的 BN running_var 与真实激活方差差若干数量级, 归一化失效。
    (2) **参数空间度量**(E1 的距离/谱量) → 用 load_plain_weights() 取原生参数即可,
        因为度量只作用在权重张量上, 不经过前向。
  踩过的坑: 不要用浮点模型的 val_acc 当基准, 会误判成"权重没加载上"。
"""

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
DATA_DIR = ROOT / "datasets" / "cifar10" / "cifar-10-batches-py"
CKPT_DIR = ROOT / "outputs" / "framework_compare" / "checkpoints"

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
# 注意: 必须与 configs/compare_resnet20.yaml 的 data.normalization.std 一致
# (torchvision 的 CIFAR-10 统计量), 不是经典 AlexNet 的 [0.2470, 0.2435, 0.2616]。
# 用错会导致同一 checkpoint 精度从 47.6% 掉到 11%(随机水平), 这是踩过的坑。
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)


# ===========================================================================
# 数据
# ===========================================================================
def load_cifar10_split(train: bool, n: int = None, offset: int = 0):
    """读取 CIFAR-10 的一个 split, 返回 (x_uint8[N,3,32,32], y[N])。

    train=True 时按 data_batch_1..5 拼接; train=False 读 test_batch。
    返回的是**未归一化**的 uint8 张量, 便于后续做增广(增广应在归一化之前)。
    """
    if train:
        files = [DATA_DIR / f"data_batch_{i}" for i in range(1, 6)]
    else:
        files = [DATA_DIR / "test_batch"]

    xs, ys = [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        xs.append(d["data"])
        ys.append(d["labels"])
    x = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).astype(np.uint8)
    y = np.concatenate(ys, axis=0).astype(np.int64)

    if offset:
        x, y = x[offset:], y[offset:]
    if n is not None:
        x, y = x[:n], y[:n]
    return torch.from_numpy(x), torch.from_numpy(y)


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,3,32,32] -> 归一化 float32 [N,3,32,32]"""
    x = x_uint8.to(torch.float32).div(255.0).numpy()
    x = (x - CIFAR_MEAN) / CIFAR_STD
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


# ===========================================================================
# 模型
# ===========================================================================
def build_plain(model_name: str = "resnet20"):
    """构建未量化的浮点模型(E1 的度量只在浮点参数上做)。"""
    if model_name == "resnet20":
        from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
        return resnet20_cifar10()
    if model_name == "vgg7":
        from sanity_check.backends.vgg7 import vgg7_bn
        return vgg7_bn()
    raise ValueError(f"unknown model: {model_name}")


def build_quant(model_name: str = "resnet20", quant_mode: str = "weight_and_activation"):
    """构建量化模型(E2 里要跑真实部署路径时用)。"""
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    m = build_plain(model_name)
    qm = (QuantizationMode.WEIGHT_ONLY if quant_mode == "weight_only"
          else QuantizationMode.WEIGHT_AND_ACTIVATION)
    return model_to_quantize_model(m, quant_mode=qm)


def _state_dict_of(ckpt_path):
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    for k in ("model", "model_state_dict", "state_dict"):
        if k in ck:
            return ck[k], ck
    return ck, ck


_PLAIN_SUFFIXES = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")


def load_plain_weights(model, ckpt_path, verbose: bool = False):
    """把 checkpoint(可能是量化模型的 state_dict)里的原生参数灌进浮点模型。

    量化模型里 conv/bn 仍是 nn.Conv2d / nn.BatchNorm2d 的子模块(外面套了量化
    壳), 所以 conv1.weight 这类 key 直接对得上; 量化缓存(d_quant_wt 等)被丢弃。
    """
    sd, meta = _state_dict_of(ckpt_path)
    plain = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.rsplit(".", 1)[-1] in _PLAIN_SUFFIXES:
            plain[k] = v
    missing, unexpected = model.load_state_dict(plain, strict=False)
    # 过滤掉量化缓存造成的 "unexpected", 只留真正缺的原生参数
    unexpected_real = [k for k in unexpected if k.rsplit(".", 1)[-1] in _PLAIN_SUFFIXES]
    if verbose:
        print(f"  loaded {len(plain)} tensors from {Path(ckpt_path).name}")
        if missing:
            print(f"  missing: {missing}")
        if unexpected_real:
            print(f"  unexpected(plain): {unexpected_real}")
    if len([k for k in missing if k.rsplit('.', 1)[-1] in _PLAIN_SUFFIXES]) > 0:
        raise RuntimeError(f"权重没对上, 缺: {missing}")
    return meta


# ===========================================================================
# 评估
# ===========================================================================
def load_quant_model(ckpt_path, model_name="resnet20", device="cuda",
                     quant_mode="weight_and_activation", verbose=True):
    """加载可评估的量化模型(唯一能给出正确精度的路径, 见模块 docstring)。

    返回 (model, meta)。meta 里通常含 history(可用来核对 val_acc)。
    """
    sd, meta = _state_dict_of(ckpt_path)
    model = build_quant(model_name, quant_mode=quant_mode).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"量化模型权重不匹配: missing={missing[:3]}({len(missing)}) "
            f"unexpected={unexpected[:3]}({len(unexpected)})")
    model.eval()
    if verbose:
        print(f"  [quant] loaded {Path(ckpt_path).name} "
              f"({sum(1 for _ in model.parameters())} param tensors)")
    return model, meta


@torch.no_grad()
def evaluate(model, x, y, device="cuda", batch_size=256, return_logits=False):
    """返回 top1 准确率; return_logits=True 时额外返回 logits 与预测。"""
    model.eval()
    correct = 0
    total = 0
    logits_all, pred_all = [], []
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i + batch_size].to(device)
        yb = y[i:i + batch_size]
        out = model(xb)
        if isinstance(out, tuple):
            out = out[0]
        pred = out.argmax(dim=1)
        correct += (pred.cpu() == yb).sum().item()
        total += yb.numel()
        if return_logits:
            logits_all.append(out.detach().cpu())
            pred_all.append(pred.cpu())
    acc = correct / max(total, 1)
    if return_logits:
        return acc, torch.cat(logits_all, 0), torch.cat(pred_all, 0)
    return acc


def resolve_ckpt(name_or_path):
    """允许传 ckpt 文件名或绝对路径。"""
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    cand = CKPT_DIR / name_or_path
    if cand.exists():
        return cand
    return p


# ===========================================================================
# 自检
# ===========================================================================
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={dev}")

    x, y = load_cifar10_split(train=False, n=1000)
    print(f"[data] test n={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")
    xin = normalize(x)
    print(f"[data] normalized mean={xin.mean():.4f} std={xin.std():.4f}")

    # 断言 1: 归一化后的统计量应接近 0 均值 / 1 标准差
    assert abs(xin.mean().item()) < 0.1, "归一化均值异常"
    assert 0.9 < xin.std().item() < 1.6, "归一化标准差异常, 检查 CIFAR_STD 是否与 yaml 一致"

    for tag in ("resnet20_baseline_ep10.pt", "resnet20_daco_final.pt"):
        ck = resolve_ckpt(tag)
        if not ck.exists():
            print(f"[skip] {tag} not found")
            continue

        # 路径 A: 量化模型 —— 唯一正确的精度路径
        mq, meta = load_quant_model(ck, device=dev)
        acc = evaluate(mq, xin, y, device=dev)
        hist = (meta.get("history") or {}).get("val_acc")
        ep = meta.get("epoch")
        ref = None
        if hist and ep:
            ref = hist[ep - 1]
        print(f"[eval-quant] {tag} acc={acc:.4f} history_ref={ref}")
        if ref is not None:
            assert abs(acc - ref) < 0.02, \
                f"量化路径精度与 history 不符: {acc:.4f} vs {ref:.4f}"

        # 路径 B: 浮点模型 —— 只用于参数空间度量, 不报精度
        mp = build_plain("resnet20")
        load_plain_weights(mp, ck, verbose=False)
        zero, tot = 0, 0
        for mod in mp.modules():
            if isinstance(mod, nn.Conv2d):
                z = (mod.weight.abs().sum(dim=(1, 2, 3)) == 0)
                zero += int(z.sum())
                tot += z.numel()
        print(f"[plain-params] {tag} zero_channels={zero}/{tot} ({zero / max(tot,1):.1%})")

    print("[ok] diag_common self-check passed")
