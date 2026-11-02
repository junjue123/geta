#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline (硬投影 GETA) vs DACO (RCAJS+MCSS+DGD) 对比实验
=========================================================
对齐 TCSVT 论文 Fig.ourDGD (DGD+调度效果对比):

目标:
  - 前 15 epoch 热身 (baseline/DACO 一致, 全精度训练)
  - 共 100 epoch (15 warmup + 85 joint), 每 epoch 记录指标 + 提取类别原型
  - 准确率/压缩率曲线: 初期重合, DACO 压缩更快但准确率震荡更明显
  - t-SNE prototype migration 轨迹: baseline 跳跃震荡, DACO 平滑连续

每 epoch 保存:
  - train_acc / val_acc / prune_ratio / avg_bit / overall_compression
  - 10 个类别的特征原型 (各类样本特征均值, 64 维) -> (num_epochs, 10, 64)

用法:
  # quick test (1 epoch 验证链路)
  python run_daco_vs_baseline.py --epochs 1 --quick-test

  # 论文目标: 15 warmup + 85 joint = 100 epoch
  python run_daco_vs_baseline.py --warmup-epochs 15 --epochs 85

  # 只跑某一模式
  python run_daco_vs_baseline.py --modes baseline
  python run_daco_vs_baseline.py --modes daco
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# 框架根目录: 必须在 import only_train_once 之前确定
# 用 --root 可以把同一份实验代码指向另一个框架副本(例如微软原版 origin/main),
# 保证"我们 vs 原版"的对比在数据/调度/指标上完全一致, 只有框架实现不同。
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--root", default=None)
_known, _ = _pre.parse_known_args()
FRAMEWORK_ROOT = os.path.abspath(_known.root or _SCRIPT_DIR)

sys.path.insert(0, FRAMEWORK_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


def _resolve_data_dir(p):
    """数据目录始终相对本仓库解析, 不受 --root 切换框架影响"""
    return p if os.path.isabs(p) else os.path.join(_SCRIPT_DIR, p)


# ===========================================================================
# 论文超参 (DACO.tex L385-387)
# ===========================================================================
PAPER_DEFAULTS = dict(
    target_group_sparsity=0.5,
    bit_reduction=1,
    min_bit_wt=4,      # b_ε = 4
    max_bit_wt=16,
    min_bit_act=2,
    max_bit_act=16,
    phi_b_threshold=0.9,
    kappa=0.5,
    W=5,
    beta_p=1.0,
    diffusion_noise_init=1e-4,
    diffusion_noise_ratio=0.5,
)

# Baseline: 硬投影, 关闭 RCAJS 自适应 / MCSS / DGD
BASELINE_OVERRIDE = dict(
    phi_b_threshold=1.0,        # 永不触发位宽自适应 (走原始固定 bit_reduction)
    beta_p=0.0,                 # exp(-0*Δ)=1, 退化为无 Lagrangian 自适应
    kappa=0.0,                  # 关闭 MCSS
    diffusion_noise_init=0.0,   # 关闭 DGD
    diffusion_noise_ratio=0.0,
)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


# ===========================================================================
# 数据加载
# ===========================================================================
class _RandomDataset(torch.utils.data.Dataset):
    """随机数据 Dataset (验证代码链路, 不需 CIFAR-10)"""
    def __init__(self, n=1000, img_size=32, num_classes=10):
        self.n, self.img_size, self.nc = n, img_size, num_classes
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return torch.randn(3, self.img_size, self.img_size), \
               torch.randint(0, self.nc, (1,)).item()


class _CIFAR10Local(torch.utils.data.Dataset):
    """从已解压的 cifar-10-batches-py/ 读数据, 绕过 torchvision MD5 校验。
    数据格式与 torchvision.datasets.CIFAR10 完全一致。"""
    base_folder = "cifar-10-batches-py"
    train_list = [["data_batch_1", "c99cafc152244af753f735de7fcd631b"],
                  ["data_batch_2", "d4bba439e000b95fd0a6bcebfdb3a6b7"],
                  ["data_batch_3", "54ebd649f8a9f4c2b3b3a0d1e6a2c7a3"],
                  ["data_batch_4", "54ebd649f8a9f4c2b3b3a0d1e6a2c7a3"],
                  ["data_batch_5", "54ebd649f8a9f4c2b3b3a0d1e6a2c7a3"]]
    test_list = [["test_batch", "40351d587109b95175f000aff129a6dd"]]

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        import pickle
        data, targets = [], []
        files = self.train_list if train else self.test_list
        for fn, _ in files:
            fpath = os.path.join(root, self.base_folder, fn)
            if not os.path.exists(fpath):
                raise RuntimeError(f"{fpath} 不存在, 请先准备 CIFAR-10 数据")
            with open(fpath, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            targets.extend(entry['labels'])
        self.data = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


class _FastCIFARLocal(torch.utils.data.Dataset):
    """Tensor 版 CIFAR-10 (与 _CIFAR10Local 数据完全一致)。

    一次性把 5 个 batch 读进 uint8 tensor, 用 torch 算子做 pad+crop / flip 增强,
    避免逐张 PIL 解码。实测把数据加载从 ~60s/epoch 降到 ~0, 100 epoch 实验
    从 2.3h 压到 20min 量级。增强语义对齐 torchvision:
        RandomCrop(32, padding=4)  -> F.pad(4, 常数 0) + 随机偏移裁剪
        RandomHorizontalFlip()     -> 50% 概率沿宽度翻转
    """

    base_folder = "cifar-10-batches-py"
    train_list = [["data_batch_1"], ["data_batch_2"], ["data_batch_3"],
                  ["data_batch_4"], ["data_batch_5"]]
    test_list = [["test_batch"]]

    _mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    _std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    def __init__(self, root, train=True, augment=False):
        import pickle
        data, targets = [], []
        files = self.train_list if train else self.test_list
        for (fn,) in files:
            fpath = os.path.join(root, self.base_folder, fn)
            if not os.path.exists(fpath):
                raise RuntimeError(f"{fpath} 不存在, 请先准备 CIFAR-10 数据")
            with open(fpath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(np.asarray(entry["data"], dtype=np.uint8))
            targets.extend(entry["labels"])
        flat = np.vstack(data).reshape(-1, 3, 32, 32)
        self.data = torch.from_numpy(flat)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].float().div_(255.0)
        x = (x - self._mean) / self._std
        if self.augment:
            x = torch.nn.functional.pad(x, (4, 4, 4, 4))
            i = int(torch.randint(0, 8, (1,)))
            j = int(torch.randint(0, 8, (1,)))
            x = x[:, i:i + 32, j:j + 32]
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
        return x, int(self.targets[idx])


def get_dataloaders(data_dir, batch_size=128, num_workers=4, seed=1,
                    random_data=False, fast=True):
    import torchvision.transforms as T
    if random_data:
        print("  [RANDOM DATA] 使用随机数据验证链路")
        g = torch.Generator().manual_seed(seed)
        trainloader = DataLoader(_RandomDataset(1000), batch_size=batch_size,
                                 shuffle=True, num_workers=2, drop_last=True,
                                 worker_init_fn=_seed_worker, generator=g)
        testloader = DataLoader(_RandomDataset(200), batch_size=256,
                                shuffle=False, num_workers=2,
                                worker_init_fn=_seed_worker)
        return trainloader, testloader

    if fast:
        trainset = _FastCIFARLocal(data_dir, train=True, augment=True)
        testset = _FastCIFARLocal(data_dir, train=False, augment=False)
        print(f"  [DATA] CIFAR-10(tensor): train={len(trainset)}, test={len(testset)}")
        g = torch.Generator().manual_seed(seed)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=True, drop_last=True,
                                 generator=g)
        testloader = DataLoader(testset, batch_size=512, shuffle=False,
                                num_workers=0, pin_memory=True)
        return trainloader, testloader

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    trainset = _CIFAR10Local(data_dir, train=True, transform=train_tf)
    testset = _CIFAR10Local(data_dir, train=False, transform=test_tf)
    print(f"  [DATA] CIFAR-10: train={len(trainset)}, test={len(testset)}")
    g = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True,
                             drop_last=True, worker_init_fn=_seed_worker,
                             generator=g)
    testloader = DataLoader(testset, batch_size=256, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=_seed_worker)
    return trainloader, testloader


# ===========================================================================
# 模型构建
# ===========================================================================
def build_model(model_name, device):
    if model_name == "resnet20":
        from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
        model = resnet20_cifar10()
    elif model_name == "vgg7":
        from sanity_check.backends.vgg7 import vgg7_bn
        model = vgg7_bn()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    dummy = torch.rand(1, 3, 32, 32)
    model = model_to_quantize_model(
        model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(device)
    return model, dummy


# ===========================================================================
# 优化器构建
# ===========================================================================
def build_optimizer(model, dummy, device, mode, joint_epochs, log_dir,
                    variant="adam", lr=1e-3):
    """mode: 'baseline' | 'daco'
    joint_epochs: 用于设置 pruning_periods (每 epoch 做一次剪枝调度, 渐进压缩)

    注意 variant: GETA 的量化参数梯度是对整个 tensor 求和, 量级可达 1e9。
    必须用 adam (官方 README 推荐) 的自适应缩放才能稳定; 用 sgd 会第一步就爆炸
    -> 量化层 forward 退化 -> acc 恒为随机水平。这是此前"量化层失效"的根因。
    """
    oto = OTO(model=model, dummy_input=dummy.to(device))
    params = dict(
        variant=variant,
        lr=lr,
        lr_quant=1e-3,
        weight_decay=1e-4,
        target_group_sparsity=PAPER_DEFAULTS["target_group_sparsity"],
        start_projection_step=0,
        projection_periods=1,
        projection_steps=1,
        start_pruning_step=0,
        pruning_periods=int(joint_epochs),  # 每 epoch 一次剪枝调度
        pruning_steps=1,
        bit_reduction=PAPER_DEFAULTS["bit_reduction"],
        min_bit_wt=PAPER_DEFAULTS["min_bit_wt"],
        max_bit_wt=PAPER_DEFAULTS["max_bit_wt"],
        min_bit_act=PAPER_DEFAULTS["min_bit_act"],
        max_bit_act=PAPER_DEFAULTS["max_bit_act"],
        importance_score_criteria="default",
        verbose="True",
        device=device,
        log_dir=log_dir,
    )
    opt = oto.geta(**params)

    cfg = BASELINE_OVERRIDE if mode == "baseline" else PAPER_DEFAULTS
    # --- DACO 特性注入: 原版框架(origin/main)没有这些属性, 需逐个探测 ---
    if hasattr(opt, "rcajs"):
        opt.rcajs.beta_p = cfg["beta_p"]
        opt.rcajs.phi_b_threshold = cfg["phi_b_threshold"]
        opt.diffusion_noise_init = cfg["diffusion_noise_init"]
        opt.diffusion_noise_ratio = cfg["diffusion_noise_ratio"]
        opt._mcss_smooth_factor = float(cfg["kappa"])
        opt._mcss_history_window = PAPER_DEFAULTS["W"]
        opt._mcss_kappa = float(cfg["kappa"])
        print(f"  [DACO] RCAJS beta_p={cfg['beta_p']} phi_b={cfg['phi_b_threshold']} "
              f"| MCSS kappa={cfg['kappa']} | DGD noise={cfg['diffusion_noise_init']}")
    else:
        print("  [COMPAT] 原版框架: 无 RCAJS/MCSS/DGD, 走纯 GETA 行为")
    return oto, opt


def refresh_bit_layers(opt):
    bit_layers = {}
    for group in opt.param_groups:
        try:
            bit_layers.update(opt.get_bitwidth_dict(group))
        except (KeyError, TypeError, ValueError):
            continue
    opt.bit_layers = bit_layers
    return bit_layers


def rcajs_phi_scheduler(opt, phi_b_threshold, bit_reduction, min_bit_wt):
    bit_layers = refresh_bit_layers(opt)
    if not bit_layers:
        return False
    below, total = 0, 0
    for v in bit_layers.values():
        if isinstance(v, dict) and v.get("weight") is not None:
            total += 1
            if v["weight"] < opt.max_bit_wt - 1:
                below += 1
    if total == 0:
        return False
    if below / total > phi_b_threshold and opt.max_bit_wt > min_bit_wt:
        opt.max_bit_wt = max(int(opt.max_bit_wt) - int(bit_reduction),
                             int(min_bit_wt))
        return True
    return False


def configure_training_schedule(opt, warmup_epochs, joint_epochs, steps_per_epoch):
    start_step = warmup_epochs * steps_per_epoch
    joint_steps = max(1, joint_epochs * steps_per_epoch)
    opt.start_projection_step = start_step
    opt.start_pruning_step = start_step
    opt.pruning_steps = joint_steps
    opt.pruning_period_duration = max(1, joint_steps // max(opt.pruning_periods, 1))
    opt.diffusion_steps = joint_steps


# ===========================================================================
# 训练 / 验证
# ===========================================================================
def train_one_epoch(model, opt, loader, criterion, device, epoch, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for bi, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
        if bi % 100 == 0:
            print(f"    Ep{epoch} [{bi}/{len(loader)}] loss={loss.item():.4f} "
                  f"acc={correct/total:.4f}", flush=True)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def collect_metrics(opt):
    group_sparsity, num_zero_groups = 0.0, 0
    avg_wb, min_wb = 0.0, 0
    num_pruned = 0
    total_groups = getattr(opt, "total_num_groups", 0)
    try:
        m = opt.compute_metrics()
        group_sparsity = m.group_sparsity
        num_zero_groups = m.num_zero_groups
    except Exception as e:
        print(f"  [WARN] compute_metrics: {e}")
    try:
        refresh_bit_layers(opt)
        wbits = [v.get("weight") for v in opt.bit_layers.values()
                 if v.get("weight") is not None]
        if wbits:
            avg_wb = sum(wbits) / len(wbits)
            min_wb = min(wbits)
    except Exception as e:
        print(f"  [WARN] bit_layers: {e}")
    try:
        num_pruned = len(getattr(opt, "pruned_group_idxes", []))
    except Exception:
        pass
    prune_ratio = num_pruned / max(total_groups, 1)
    bit_compress = 16.0 / max(avg_wb, 1e-6) if avg_wb > 0 else 1.0
    overall_compress = (1 - prune_ratio) * bit_compress
    return dict(group_sparsity=group_sparsity, num_zero_groups=num_zero_groups,
                avg_wb=avg_wb, min_wb=min_wb, num_pruned=num_pruned,
                total_groups=total_groups, prune_ratio=prune_ratio,
                bit_compress=bit_compress, overall_compress=overall_compress)


# ===========================================================================
# 类别原型提取 (供 t-SNE 轨迹): 各类别特征均值
# ===========================================================================
@torch.no_grad()
def extract_class_prototypes(model, loader, device, feat_hook_target="avg_pool",
                             max_samples=2000):
    """提取 10 个类别的特征原型 (各类样本特征均值)。
    ResNet-20: hook avg_pool2d 输出 (B,64,1,1) -> (B,64)
    VGG-7:     hook classifier[0] 输出 (B,1024)
    max_samples: 限制样本数加速 (固定前 max_samples 个, 保证跨 epoch 一致)
    返回 prototypes (10, D), 每行是一个类别的特征均值。
    """
    model.eval()
    # 找目标模块
    target_mod = None
    if feat_hook_target == "avg_pool":
        for name, mod in model.named_modules():
            if name.endswith("avg_pool2d") and isinstance(mod, nn.AvgPool2d):
                target_mod = mod
                break
            if name.endswith("avg_pool") and isinstance(mod, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                target_mod = mod
                break
    if target_mod is None:
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                target_mod = mod
        if target_mod is None:
            clf = getattr(model, "classifier", None)
            if clf is None:
                for name, mod in model.named_modules():
                    if "classifier" in name and isinstance(mod, nn.Sequential):
                        clf = mod
                        break
            if clf is not None:
                target_mod = clf[0]
    if target_mod is None:
        raise RuntimeError("找不到特征提取目标模块")

    feats_container = {}
    def hook(m, i, o):
        feats_container["feat"] = o.detach()
    h = target_mod.register_forward_hook(hook)

    all_feats, all_labels = [], []
    n = 0
    for X, y in loader:
        X = X.to(device)
        _ = model(X)
        feat = feats_container["feat"]
        feat = feat.view(feat.size(0), -1).cpu().numpy()
        all_feats.append(feat)
        all_labels.append(y.numpy())
        n += feat.shape[0]
        if n >= max_samples:
            break
    h.remove()

    feats = np.concatenate(all_feats, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    D = feats.shape[1]
    protos = np.zeros((10, D), dtype=np.float32)
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            protos[c] = feats[mask].mean(axis=0)
    return protos


# ===========================================================================
# 单次实验
# ===========================================================================
def run_experiment(args, mode):
    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Experiment: {mode.upper()}  (model={args.model}, "
          f"warmup={args.warmup_epochs}, joint={args.epochs})")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    trainloader, testloader = get_dataloaders(
        _resolve_data_dir(args.data_dir), args.batch_size, args.num_workers,
        seed=args.seed, random_data=args.random_data,
        fast=not getattr(args, "legacy_data", False))
    steps_per_epoch = len(trainloader)
    print(f"  [DATA] train batches={steps_per_epoch}, test batches={len(testloader)}")

    model, dummy = build_model(args.model, device)
    log_dir = os.path.join(args.out, "logs", mode)
    oto, opt = build_optimizer(model, dummy, device, mode, args.epochs, log_dir,
                               variant=args.variant, lr=args.lr)

    warmup = 0 if args.quick_test else args.warmup_epochs
    joint = args.epochs
    total = warmup + joint
    configure_training_schedule(opt, warmup, joint, steps_per_epoch)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total * steps_per_epoch, eta_min=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 每 epoch 记录: 指标 + 类别原型
    history = {"epoch": [], "phase": [], "train_acc": [], "val_acc": [],
               "train_loss": [], "val_loss": [],
               "prune_ratio": [], "avg_weight_bit": [], "group_sparsity": [],
               "bit_compress": [], "overall_compress": [], "lr": []}
    prototypes_seq = []  # list of (10, D) arrays, 每 epoch 一个

    cfg = BASELINE_OVERRIDE if mode == "baseline" else PAPER_DEFAULTS
    feat_target = "avg_pool" if args.model == "resnet20" else "classifier"
    t0 = time.time()

    for ep in range(total):
        phase = "warmup" if ep < warmup else "joint"
        print(f"\n  Epoch {ep+1}/{total} [{phase}]", flush=True)
        tr_loss, tr_acc = train_one_epoch(
            model, opt, trainloader, criterion, device, ep+1, scheduler)
        if phase == "joint" and hasattr(opt, "rcajs"):
            rcajs_phi_scheduler(opt, cfg["phi_b_threshold"],
                                PAPER_DEFAULTS["bit_reduction"],
                                PAPER_DEFAULTS["min_bit_wt"])
        v_loss, v_acc = evaluate(model, testloader, criterion, device)
        m = collect_metrics(opt)
        # 提取类别原型
        protos = extract_class_prototypes(model, testloader, device, feat_target)
        prototypes_seq.append(protos)

        print(f"  -> tr_acc={tr_acc:.4f} val_acc={v_acc:.4f} "
              f"prune={m['prune_ratio']:.4f} avg_bit={m['avg_wb']:.2f} "
              f"overall_compress={m['overall_compress']:.2f}x", flush=True)

        history["epoch"].append(ep + 1)
        history["phase"].append(phase)
        history["train_acc"].append(round(tr_acc, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["prune_ratio"].append(round(m["prune_ratio"], 4))
        history["avg_weight_bit"].append(round(m["avg_wb"], 2))
        history["group_sparsity"].append(round(m["group_sparsity"], 4))
        history["bit_compress"].append(round(m["bit_compress"], 2))
        history["overall_compress"].append(round(m["overall_compress"], 2))
        history["lr"].append(round(opt.param_groups[0]["lr"], 6))

    elapsed = time.time() - t0
    m_final = collect_metrics(opt)

    # 保存原型序列 (num_epochs, 10, D)
    proto_arr = np.stack(prototypes_seq, axis=0)  # (T, 10, D)
    proto_path = os.path.join(args.out, "prototypes", f"{args.model}_{mode}_protos.npy")
    os.makedirs(os.path.dirname(proto_path), exist_ok=True)
    np.save(proto_path, proto_arr.astype(np.float32))
    print(f"  [PROTO] {proto_arr.shape} -> {proto_path}")

    # 保存最终 checkpoint
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.model}_{mode}_final.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "val_acc": history["val_acc"][-1],
                "history": history}, ckpt_path)
    print(f"  [CKPT] {ckpt_path}")

    result = {
        "experiment": f"{args.model}_{mode}",
        "mode": mode,
        "model": args.model,
        "warmup_epochs": warmup,
        "joint_epochs": joint,
        "total_epochs": total,
        "best_val_acc": round(max(history["val_acc"]), 4),
        "final_val_acc": history["val_acc"][-1],
        "final_prune_ratio": m_final["prune_ratio"],
        "final_avg_weight_bit": round(m_final["avg_wb"], 2),
        "final_overall_compress": round(m_final["overall_compress"], 2),
        "elapsed_minutes": round(elapsed / 60, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": ckpt_path,
        "proto_path": proto_path,
        "history": history,
    }
    print(f"\n  RESULT [{mode}]: best_acc={result['best_val_acc']} "
          f"final_acc={result['final_val_acc']} "
          f"prune={result['final_prune_ratio']:.4f} "
          f"bit={result['final_avg_weight_bit']} "
          f"compress={result['final_overall_compress']}x "
          f"time={elapsed/60:.1f}min")
    return result


def save_result(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{result['mode']}.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    csv_path = os.path.join(out_dir, "comparison_summary.csv")
    fields = ["experiment", "mode", "model", "warmup_epochs", "joint_epochs",
              "total_epochs", "best_val_acc", "final_val_acc",
              "final_prune_ratio", "final_avg_weight_bit",
              "final_overall_compress", "elapsed_minutes", "timestamp"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(result)


def main():
    p = argparse.ArgumentParser(description="Baseline vs DACO 对比 (ResNet-20, 100步)")
    p.add_argument("--model", default="resnet20", choices=["resnet20", "vgg7"])
    p.add_argument("--epochs", type=int, default=85,
                   help="joint 训练 epochs (默认 85, 配合 warmup 15 = 100 总)")
    p.add_argument("--warmup-epochs", type=int, default=15)
    p.add_argument("--data-dir", default="datasets/cifar10")
    p.add_argument("--out", default="outputs/baseline_vs_daco")
    p.add_argument("--root", default=None,
                   help="框架根目录 (默认本仓库; 指向原版 worktree 即可做跨框架对比)")
    p.add_argument("--variant", default="adam", choices=["adam", "adamw", "sgd"],
                   help="优化器变体。必须用 adam: GETA 量化参数梯度量级 ~1e9, "
                        "sgd 会一步爆炸导致量化层 forward 退化")
    p.add_argument("--lr", type=float, default=1e-3, help="权重学习率 (adam 下 1e-3)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--quick-test", action="store_true",
                   help="1 epoch, 无 warmup (验证链路)")
    p.add_argument("--random-data", action="store_true",
                   help="用随机数据代替 CIFAR-10 (验证代码链路, 不需下载数据集)")
    p.add_argument("--legacy-data", action="store_true",
                   help="用旧的 PIL 逐张解码数据管道 (慢, 仅用于对照)")
    p.add_argument("--modes", default="baseline,daco")
    args = p.parse_args()

    if args.quick_test:
        args.epochs = 1
        args.warmup_epochs = 0
        print("[QUICK TEST] 1 epoch, no warmup")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    for mode in modes:
        try:
            r = run_experiment(args, mode)
            save_result(r, args.out)
            all_results.append(r)
        except Exception as e:
            print(f"  [ERROR] {mode}: {e}")
            traceback.print_exc()

    if len(all_results) >= 2:
        print(f"\n{'='*70}\n  对比汇总\n{'='*70}")
        b, d = all_results[0], all_results[-1]
        for k, label in [("best_val_acc", "Best Acc"), ("final_val_acc", "Final Acc"),
                         ("final_prune_ratio", "Prune Ratio"),
                         ("final_avg_weight_bit", "Avg W-bit"),
                         ("final_overall_compress", "Overall Compress")]:
            print(f"  {label:<18} Baseline={b.get(k,0):>10}  "
                  f"DACO={d.get(k,0):>10}")
        print(f"{'='*70}")
    print(f"\n输出目录: {args.out}")
    print(f"原型文件: {args.out}/prototypes/")
    print(f"用 plot_baseline_vs_daco.py 画曲线和 t-SNE 轨迹")


if __name__ == "__main__":
    main()
