#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DACO CIFAR-10 真实训练 + 超参数敏感性实验
==========================================

在 AutoDL GPU 上用 CIFAR-10 全量数据训练 ResNet-20 / VGG-7 + GETA，
跑论文 Section IV-D "Hyperparameter Analysis" 所需的真实实验：
  - RCAJS β_p (阻尼系数) 敏感性: 7 个取值
  - RCAJS φ_b (收缩比) 敏感性: 5 个取值
  - MCSS κ (平衡系数) 敏感性: 7 个取值

论文训练设置 (DACO.tex L374-387):
  · CIFAR-10, ResNet-20 / VGG-7, from scratch
  · batch=128, SGD lr=0.1 cosine→1e-3, weight_decay=1e-4
  · warmup 10 epochs, joint prune-quant 200 epochs, cooldown 20 epochs
  · RCAJS: β_p=1.0, φ_b=0.9, b_ε=4; MCSS: W=5, κ=0.5

启用方式:
  # 单次完整训练 (默认 VGG-7, 200+30 epochs)
  python run_daco_cifar10.py --model vgg7 --epochs 200

  # β_p 敏感性扫描 (7 个值, 每个跑完整训练)
  python run_daco_cifar10.py --model vgg7 --hparam beta_p --epochs 200

  # φ_b 敏感性扫描
  python run_daco_cifar10.py --model vgg7 --hparam phi_b --epochs 200

  # κ 敏感性扫描
  python run_daco_cifar10.py --model vgg7 --hparam kappa --epochs 200

  # 快速冒烟测试 (1 epoch, 验证流程通不通)
  python run_daco_cifar10.py --model vgg7 --epochs 1 --quick-test

  # 指定单个超参值
  python run_daco_cifar10.py --model resnet20 --hparam beta_p --hvalue 0.1 --epochs 200

  # 自定义数据路径 / 输出目录
  python run_daco_cifar10.py --data-dir /root/autodl-tmp/data --out outputs/autodl_experiments

  # YAML 配置驱动 (推荐): 在 configs/daco_vgg7.yaml 配置模型/步数/调度/压缩超参
  python run_daco_cifar10.py --config configs/daco_vgg7.yaml
  # 命令行参数仍可覆盖 YAML (例如临时改 epochs)
  python run_daco_cifar10.py --config configs/daco_vgg7.yaml --epochs 50 --quick-test
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# ---- GETA 框架 ----
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


# ===========================================================================
# 论文超参数扫描取值 (与 run_rcajs_hparam.py / make_hyper_figs.py 对齐)
# ===========================================================================
HPARAM_SWEEP = {
    "beta_p": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
    "phi_b":  [0.1, 0.3, 0.5, 0.7, 0.9],
    "kappa":  [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0],
}

# 论文默认超参 (DACO.tex L385-387)
PAPER_DEFAULTS = dict(
    target_group_sparsity=0.5,
    bit_reduction=1,
    min_bit_wt=4,         # b_ε = 4
    max_bit_wt=16,
    min_bit_act=2,
    max_bit_act=16,
    phi_b_threshold=0.9,  # φ_b = 0.9
    kappa=0.5,            # κ = 0.5
    W=5,                  # 滑动窗口
    beta_p=1.0,           # β_p = 1.0
    diffusion_noise_init=1e-4,
    diffusion_noise_ratio=0.5,
)


def load_yaml_config(path):
    """加载 YAML 配置文件, 返回 dict (需要 PyYAML)"""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "YAML 配置需要 PyYAML, 请安装: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy and PyTorch before every independent run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# 1. 模型构建
# ===========================================================================
def build_model(model_name, device):
    """构建量化模型"""
    if model_name == "vgg7":
        from sanity_check.backends.vgg7 import vgg7_bn
        model = vgg7_bn()
        dummy = torch.rand(1, 3, 32, 32)
    elif model_name == "resnet20":
        from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
        model = resnet20_cifar10()
        dummy = torch.rand(1, 3, 32, 32)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model.to(device)
    return model, dummy


# ===========================================================================
# 2. 数据加载
# ===========================================================================
def get_dataloaders(data_dir, batch_size=128, num_workers=4, seed=1,
                    aug_cfg=None, norm_cfg=None):
    """CIFAR-10 数据加载器 (增强/归一化可由 YAML dataset 段配置)"""
    import torchvision
    import torchvision.transforms as transforms

    aug_cfg = aug_cfg or {}
    norm_cfg = norm_cfg or {}
    padding = aug_cfg.get("random_crop_padding", 4)
    hflip = aug_cfg.get("random_horizontal_flip", True)
    mean = tuple(norm_cfg.get("mean", (0.4914, 0.4822, 0.4465)))
    std = tuple(norm_cfg.get("std", (0.2023, 0.1994, 0.2010)))

    train_layers = [transforms.RandomCrop(32, padding=padding)]
    if hflip:
        train_layers.append(transforms.RandomHorizontalFlip())
    train_layers += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(train_layers)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    generator = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=_seed_worker, generator=generator,
    )
    testloader = DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    return trainloader, testloader


class _RandomDataset(torch.utils.data.Dataset):
    """随机数据 Dataset (验证代码链路, 不需下载 CIFAR-10)"""
    def __init__(self, num_samples=1000, img_size=32):
        self.num_samples = num_samples
        self.img_size = img_size
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return torch.randn(3, self.img_size, self.img_size), torch.randint(0, 10, (1,)).item()


def get_random_dataloaders(batch_size=128, num_workers=2, num_train=1000,
                           num_test=200, seed=1):
    """随机数据 DataLoader (不依赖 CIFAR-10)"""
    trainset = _RandomDataset(num_train)
    testset = _RandomDataset(num_test)
    generator = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True,
                             worker_init_fn=_seed_worker, generator=generator)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


# ===========================================================================
# 3. 构建 GETA 优化器 (按论文设置 + 超参覆盖)
# ===========================================================================
def build_optimizer(model, dummy, device, hparam_name=None, hparam_value=None,
                    log_dir="outputs/autodl_experiments/logs"):
    """
    构建 OTO + GETA 优化器。
    hparam_name: "beta_p" / "phi_b" / "kappa" / None
    hparam_value: 对应取值
    """
    oto = OTO(model=model, dummy_input=dummy.to(device))

    # 论文训练设置: SGD, lr=0.1, cosine→1e-3, wd=1e-4
    # warmup 10 ep + joint 200 ep + cooldown 20 ep
    # 这里用 step 数表示: steps_per_epoch * epochs
    params = dict(
        variant="sgd",
        lr=0.1,
        lr_quant=1e-3,
        weight_decay=1e-4,
        target_group_sparsity=PAPER_DEFAULTS["target_group_sparsity"],
        # 投影阶段: warmup 结束后开始
        start_projection_step=0,
        projection_periods=1,
        projection_steps=1,
        # 剪枝阶段: 200 epochs 对应的步数 (会在 train() 中按实际步数设置)
        start_pruning_step=0,
        pruning_periods=5,
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

    # ---- 注入 DGD 超参 (oto.geta 不透传) ----
    opt.diffusion_noise_init = PAPER_DEFAULTS["diffusion_noise_init"]
    opt.diffusion_noise_ratio = PAPER_DEFAULTS["diffusion_noise_ratio"]
    opt.rcajs.beta_p = PAPER_DEFAULTS["beta_p"]
    opt.rcajs.phi_b_threshold = PAPER_DEFAULTS["phi_b_threshold"]

    # Default MCSS calibration must be active even outside a kappa sweep.
    selected_kappa = (float(hparam_value) if hparam_name == "kappa"
                      and hparam_value is not None
                      else PAPER_DEFAULTS["kappa"])
    _apply_kappa_wiring(opt, selected_kappa)

    # ---- 注入 RCAJS 超参覆盖 ----
    if hparam_name == "beta_p" and hparam_value is not None:
        opt.rcajs.beta_p = float(hparam_value)
    elif hparam_name == "phi_b" and hparam_value is not None:
        opt.rcajs.phi_b_threshold = float(hparam_value)
    elif hparam_name == "kappa" and hparam_value is not None:
        # Already wired above so the default and sweep share one code path.
        opt._mcss_kappa = float(hparam_value)

    return oto, opt


def _apply_kappa_wiring(opt, kappa):
    """
    将 MCSS κ (平衡系数) 接线进 GETA 优化器。
    κ=0: 纯全精度特征显著性 (无量化感知校准)
    κ=1: 纯量化感知校准 (忽略全精度特征)
    0<κ<1: 加权混合

    实现: 设置 opt._mcss_smooth_factor, 由 BaseHybridSparseOptimizer.compute_importance_scores
    在 overall 聚合前统一调用 adjust_importance_criteria (单一入口, 避免重复调用失真)。
    """
    opt._mcss_smooth_factor = float(kappa)
    opt._mcss_history_window = PAPER_DEFAULTS["W"]
    # 兼容旧属性
    opt._mcss_kappa = float(kappa)


# ===========================================================================
# 4. φ_b 外部门控调度器 (与 run_rcajs_hparam.py 一致)
# ===========================================================================
def refresh_bit_layers(opt):
    """Read live effective bit widths instead of the stale post-pruning cache."""
    bit_layers = {}
    for group in opt.param_groups:
        try:
            bit_layers.update(opt.get_bitwidth_dict(group))
        except (KeyError, TypeError, ValueError):
            continue
    opt.bit_layers = bit_layers
    return bit_layers


def rcajs_phi_scheduler(opt, phi_b_threshold, bit_reduction, min_bit_wt):
    """Apply one repeatable φ_b-gated contraction using live layer bit widths."""
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
    phi_measured = below / total
    if phi_measured > phi_b_threshold and opt.max_bit_wt > min_bit_wt:
        opt.max_bit_wt = max(int(opt.max_bit_wt) - int(bit_reduction), int(min_bit_wt))
        return True
    return False


# ===========================================================================
# 5. 训练 + 验证
# ===========================================================================
def train_one_epoch(model, opt, trainloader, criterion, device, epoch,
                    scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (X, y) in enumerate(trainloader):
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
        if batch_idx % 100 == 0:
            print(f"    Epoch {epoch} [{batch_idx}/{len(trainloader)}] "
                  f"loss={loss.item():.4f} acc={correct/total:.4f}", flush=True)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, testloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in testloader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def collect_metrics(opt):
    """收集压缩指标 + RCAJS 状态"""
    group_sparsity, num_zero_groups = 0.0, 0
    avg_wb, min_wb = 0.0, 0
    rcajs_st = {"prune_rate": 0.0, "delta_p": 0.0, "stability": 0.0, "kl": 0.0}
    num_pruned = 0
    try:
        m = opt.compute_metrics()
        group_sparsity = m.group_sparsity
        num_zero_groups = m.num_zero_groups
    except Exception as e:
        print(f"  [WARN] compute_metrics failed: {e}")
    try:
        refresh_bit_layers(opt)
        wbits = [v.get("weight") for v in opt.bit_layers.values() if v.get("weight") is not None]
        if wbits:
            avg_wb = sum(wbits) / len(wbits)
            min_wb = min(wbits)
    except Exception as e:
        print(f"  [WARN] refresh_bit_layers failed: {e}")
    try:
        rcajs = opt.rcajs
        rcajs_st = {
            "prune_rate": float(rcajs.curr_pruning_rate),
            "delta_p": float(rcajs.compute_delta_p()),
            "stability": float(rcajs.compute_stability_term()),
            "kl": float(np.mean(rcajs.kl_history)) if rcajs.kl_history else 0.0,
        }
        num_pruned = len(getattr(opt, "pruned_group_idxes", []))
    except Exception as e:
        print(f"  [WARN] rcajs summary failed: {e}")
    return group_sparsity, num_zero_groups, avg_wb, min_wb, rcajs_st, num_pruned


def configure_training_schedule(opt, warmup_epochs, joint_epochs, steps_per_epoch):
    """Synchronize all step-based GETA fields after the real loader size is known."""
    start_step = warmup_epochs * steps_per_epoch
    joint_steps = max(1, joint_epochs * steps_per_epoch)
    opt.start_projection_step = start_step
    opt.start_pruning_step = start_step
    opt.pruning_steps = joint_steps
    opt.pruning_period_duration = max(1, joint_steps // max(opt.pruning_periods, 1))
    opt.diffusion_steps = joint_steps


def save_training_checkpoint(path, model, opt, scheduler, epoch, phase,
                             val_acc, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "max_bit_wt": opt.max_bit_wt,
        "max_bit_act": opt.max_bit_act,
        "history": history,
    }, path)


# ===========================================================================
# 6. 单次实验运行
# ===========================================================================
def run_experiment(args, hparam_name=None, hparam_value=None):
    """运行一次完整训练实验，返回结果字典"""
    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "default" if hparam_value is None else f"{hparam_name}_{hparam_value}"
    exp_name = f"{args.model}_{tag}"
    print(f"\n{'='*70}")
    print(f"  Experiment: {exp_name}")
    print(f"  Model: {args.model}, Epochs: {args.epochs}, Device: {device}")
    if hparam_value is not None:
        print(f"  Hparam: {hparam_name} = {hparam_value}")
    print(f"{'='*70}")

    # 数据 (由 YAML dataset 段驱动)
    dataset_name = getattr(args, 'dataset_name', 'cifar10')
    if dataset_name == "random" or getattr(args, 'random_data', False):
        print("  [RANDOM DATA] 使用随机数据 (无需 CIFAR-10)")
        trainloader, testloader = get_random_dataloaders(
            args.batch_size, args.num_workers, seed=args.seed)
    else:
        trainloader, testloader = get_dataloaders(
            args.data_dir, args.batch_size, args.num_workers, seed=args.seed,
            aug_cfg=getattr(args, 'aug_cfg', None),
            norm_cfg=getattr(args, 'norm_cfg', None))
    steps_per_epoch = len(trainloader)
    print(f"  [DATA] dataset={dataset_name} Train batches: {steps_per_epoch}, Test batches: {len(testloader)}")

    # 模型 + 优化器
    model, dummy = build_model(args.model, device)
    oto, opt = build_optimizer(
        model, dummy, device, hparam_name, hparam_value,
        log_dir=os.path.join(args.out, "logs"),
    )

    # 设置剪枝/投影步数 (按实际 epoch 数对齐论文比例)
    # warmup: 10 epochs, joint: args.epochs, cooldown: 20 epochs
    warmup_epochs = args.warmup_epochs if not args.quick_test else 0
    cooldown_epochs = args.cooldown_epochs if not args.quick_test else 0
    total_epochs = warmup_epochs + args.epochs + cooldown_epochs

    configure_training_schedule(opt, warmup_epochs, args.epochs, steps_per_epoch)

    # 学习率调度 (cosine annealing 0.1 → 1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_epochs * steps_per_epoch, eta_min=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc, final_acc = 0.0, 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "group_sparsity": [], "avg_weight_bit": [],
               "learning_rate": [], "max_weight_bit": [], "bit_transition": [],
               "rcajs_prune_rate": [], "rcajs_delta_p": [], "rcajs_stability": [],
               "rcajs_kl": [], "num_pruned_groups": []}
    t0 = time.time()

    for epoch in range(total_epochs):
        phase = "warmup" if epoch < warmup_epochs else (
            "joint" if epoch < warmup_epochs + args.epochs else "cooldown"
        )
        print(f"\n  Epoch {epoch+1}/{total_epochs} [{phase}]", flush=True)

        train_loss, train_acc = train_one_epoch(
            model, opt, trainloader, criterion, device, epoch+1,
            scheduler=scheduler,
        )
        bit_transition = False
        if phase == "joint":
            phi_threshold = (float(hparam_value) if hparam_name == "phi_b"
                             and hparam_value is not None
                             else PAPER_DEFAULTS["phi_b_threshold"])
            bit_transition = rcajs_phi_scheduler(
                opt, phi_threshold, PAPER_DEFAULTS["bit_reduction"],
                PAPER_DEFAULTS["min_bit_wt"],
            )
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        sp, nzg, awb, mwb, rcajs_st, num_pruned = collect_metrics(opt)

        print(f"  → train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"sparsity={sp:.4f} avg_bit={awb:.2f} "
              f"rcajs_rate={rcajs_st['prune_rate']:.4f} delta_p={rcajs_st['delta_p']:+.4f} "
              f"pruned={num_pruned}", flush=True)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["group_sparsity"].append(round(sp, 4))
        history["avg_weight_bit"].append(round(awb, 2))
        history["learning_rate"].append(round(opt.param_groups[0]["lr"], 8))
        history["max_weight_bit"].append(int(opt.max_bit_wt))
        history["bit_transition"].append(bool(bit_transition))
        history["rcajs_prune_rate"].append(round(rcajs_st["prune_rate"], 6))
        history["rcajs_delta_p"].append(round(rcajs_st["delta_p"], 6))
        history["rcajs_stability"].append(round(rcajs_st["stability"], 6))
        history["rcajs_kl"].append(round(rcajs_st["kl"], 6))
        history["num_pruned_groups"].append(int(num_pruned))

        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            epoch_ckpt = os.path.join(
                args.out, "checkpoints", exp_name,
                f"epoch_{epoch + 1:04d}.pt",
            )
            save_training_checkpoint(
                epoch_ckpt, model, opt, scheduler, epoch + 1, phase,
                val_acc, history,
            )

        if val_acc > best_acc:
            best_acc = val_acc
        final_acc = val_acc

    elapsed = time.time() - t0

    # 最终压缩指标
    sp, nzg, awb, mwb, _, _ = collect_metrics(opt)

    # 保存 checkpoint
    if args.save_ckpt:
        ckpt_dir = os.path.join(args.out, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pt")
        save_training_checkpoint(
            ckpt_path, model, opt, scheduler, total_epochs, "final",
            final_acc, history,
        )
        print(f"  [CKPT] 训练状态已保存: {ckpt_path}")
        try:
            subnet_dir = os.path.join(ckpt_dir, f"{exp_name}_subnet")
            os.makedirs(subnet_dir, exist_ok=True)
            oto.construct_subnet(out_dir=subnet_dir)
            print(f"  [CKPT] 子网已保存: {subnet_dir}")
        except Exception as e:
            print(f"  [WARN] construct_subnet 失败: {e}")

    result = {
        "experiment": exp_name,
        "model": args.model,
        "hparam": hparam_name or "none",
        "hparam_value": hparam_value if hparam_value is not None else "default",
        "epochs": total_epochs,
        "best_val_acc": round(best_acc, 4),
        "final_val_acc": round(final_acc, 4),
        "final_group_sparsity": round(sp, 4),
        "num_zero_groups": nzg,
        "avg_weight_bit": round(awb, 2),
        "min_weight_bit": mwb,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hours": round(elapsed / 3600, 2),
        "status": "OK",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history": history,
    }
    print(f"\n  Result: best_acc={best_acc:.4f} final_acc={final_acc:.4f} "
          f"sparsity={sp:.4f} avg_bit={awb:.2f} time={elapsed/3600:.2f}h")
    return result


# ===========================================================================
# 7. 结果保存
# ===========================================================================
def save_result(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # JSON (含完整 history)
    json_path = os.path.join(out_dir, "results",
                             f"{result['experiment']}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] {json_path}")

    # 追加到 CSV 汇总
    csv_path = os.path.join(out_dir, "experiment_summary.csv")
    fields = ["experiment", "model", "hparam", "hparam_value", "epochs",
              "best_val_acc", "final_val_acc", "final_group_sparsity",
              "num_zero_groups", "avg_weight_bit", "min_weight_bit",
              "elapsed_hours", "status", "timestamp"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(result)
    print(f"  [CSV ] {csv_path}")


# ===========================================================================
# 8. 入口
# ===========================================================================
def main():
    # 第一次解析: 只取 --config, 以便 YAML 覆盖 argparse 默认值
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="YAML 配置文件路径 (覆盖默认参数, 命令行参数仍优先)")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = {}
    if pre_args.config:
        yaml_cfg = load_yaml_config(pre_args.config)
        print(f"[YAML] 加载配置: {pre_args.config}")

    # compression 超参覆盖 PAPER_DEFAULTS
    comp = yaml_cfg.get("compression", {})
    if comp:
        PAPER_DEFAULTS.update(comp)
        print(f"[YAML] compression 超参已覆盖 PAPER_DEFAULTS:")
        for k, v in comp.items():
            print(f"       {k} = {v}")

    # 第二次解析: 完整参数, default 从 YAML 取 (命令行显式传入仍优先)
    parser = argparse.ArgumentParser(
        description="DACO CIFAR-10 超参数敏感性实验 (支持 --config YAML 配置)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径 (覆盖默认参数, 命令行参数仍优先)")
    parser.add_argument("--model", type=str, default=yaml_cfg.get("model", "vgg7"),
                        choices=["vgg7", "resnet20"],
                        help="网络架构 (默认 vgg7)")
    parser.add_argument("--epochs", type=int, default=yaml_cfg.get("epochs", 200),
                        help="联合剪枝-量化训练 epochs (默认 200)")
    parser.add_argument("--warmup-epochs", type=int,
                        default=yaml_cfg.get("warmup_epochs", 10),
                        help="warmup epochs (默认 10)")
    parser.add_argument("--cooldown-epochs", type=int,
                        default=yaml_cfg.get("cooldown_epochs", 20),
                        help="cooldown epochs (默认 20)")
    parser.add_argument("--hparam", type=str, default=yaml_cfg.get("hparam"),
                        choices=["beta_p", "phi_b", "kappa"],
                        help="扫描哪个超参 (不指定则跑默认配置)")
    parser.add_argument("--hvalue", type=float, default=yaml_cfg.get("hvalue"),
                        help="单个超参取值 (不指定则扫描全部取值)")
    parser.add_argument("--data-dir", type=str,
                        default=yaml_cfg.get("data_dir", "/root/autodl-tmp/data"),
                        help="CIFAR-10 数据目录")
    parser.add_argument("--out", type=str,
                        default=yaml_cfg.get("out", "outputs/autodl_experiments"),
                        help="结果输出目录")
    parser.add_argument("--batch-size", type=int,
                        default=yaml_cfg.get("batch_size", 128))
    parser.add_argument("--num-workers", type=int,
                        default=yaml_cfg.get("num_workers", 4))
    parser.add_argument("--save-ckpt", action="store_true",
                        help="保存模型 checkpoint")
    parser.add_argument("--ckpt-every", type=int,
                        default=yaml_cfg.get("ckpt_every", 0),
                        help="每 N 个 epoch 保存训练 checkpoint，0 表示不周期保存")
    parser.add_argument("--seed", type=int, default=yaml_cfg.get("seed", 1),
                        help="随机种子")
    parser.add_argument("--deterministic", action="store_true",
                        help="启用确定性 cuDNN，用于严格策略对照")
    parser.add_argument("--quick-test", action="store_true",
                        help="冒烟测试: 1 epoch, 无 warmup/cooldown")
    parser.add_argument("--random-data", action="store_true",
                        help="用随机数据代替 CIFAR-10 (验证代码链路, 不需下载数据集)")
    args = parser.parse_args()

    # store_true 参数: YAML bool 与命令行 flag 取 OR
    for flag in ("save_ckpt", "deterministic", "quick_test", "random_data"):
        if yaml_cfg.get(flag):
            setattr(args, flag, True)

    # dataset 段合并: data_dir/batch_size/num_workers 可在 dataset 段覆盖顶层
    ds_cfg = yaml_cfg.get("dataset", {})
    if ds_cfg:
        args.data_dir = ds_cfg.get("data_dir", args.data_dir)
        args.batch_size = ds_cfg.get("batch_size", args.batch_size)
        args.num_workers = ds_cfg.get("num_workers", args.num_workers)
        args.dataset_name = ds_cfg.get("name", "cifar10")
        args.aug_cfg = ds_cfg.get("augmentation", {})
        args.norm_cfg = ds_cfg.get("normalization", {})
    else:
        args.dataset_name = "cifar10"
        args.aug_cfg = {}
        args.norm_cfg = {}
    # dataset.name=random 等价于 random_data (向后兼容)
    if args.dataset_name == "random":
        args.random_data = True

    # 冒烟测试模式
    if args.quick_test:
        args.epochs = 1
        print("[QUICK TEST] 1 epoch, no warmup/cooldown")

    if args.hparam and args.hvalue is not None:
        # 单个超参值
        results = [run_experiment(args, args.hparam, args.hvalue)]
        for r in results:
            save_result(r, args.out)
    elif args.hparam:
        # 扫描该超参的全部取值
        values = HPARAM_SWEEP[args.hparam]
        print(f"\n扫描 {args.hparam}: {values}")
        all_results = []
        for i, v in enumerate(values):
            print(f"\n[{i+1}/{len(values)}] {args.hparam}={v}")
            try:
                r = run_experiment(args, args.hparam, v)
                save_result(r, args.out)
                all_results.append(r)
            except Exception as e:
                print(f"  [ERROR] {e}")
                traceback.print_exc()
                fail_result = {
                    "experiment": f"{args.model}_{args.hparam}_{v}",
                    "model": args.model, "hparam": args.hparam,
                    "hparam_value": v, "status": "FAIL", "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_result(fail_result, args.out)
                all_results.append(fail_result)
        print(f"\n{'='*70}")
        print(f"  全部完成: {len(all_results)} 个实验")
        for r in all_results:
            acc = r.get("best_val_acc", "N/A")
            print(f"    {r['experiment']}: best_acc={acc}")
    else:
        # 默认配置单次训练
        r = run_experiment(args)
        save_result(r, args.out)


if __name__ == "__main__":
    main()
