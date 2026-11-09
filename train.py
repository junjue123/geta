#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用训练入口 (joint pruning + quantization)
===========================================
一个入口, 差异全部由 YAML 决定 —— 换配置不改代码:

    python train.py configs/compare_resnet20.yaml

YAML 覆盖的内容:
  · 训练调度: epochs / warmup_epochs / lr / lr_eta_min_ratio / variant / seed
  · 数据: data_dir / batch_size / num_workers
  · 剪枝: prune_shape(cosine|linear) / prune_floor / prune_budget(fixed|rcajs)
  · 压缩超参: target_group_sparsity / bit 上下界 / phi_b / phi_a / kappa / W /
              beta_p / DGD 噪声
  · 模式: modes=baseline,daco (同一进程内串行跑, 共享数据与调度)
  · baseline 关断项: baseline_override 块 (纯 GETA: 无 RCAJS/MCSS/DGD)
  · 日志: logging.level / logging.file

优先级: 命令行 > YAML > 脚本内置默认值。
每次运行会落盘两份可回溯记录:
  {out}/run_config.yaml   本次最终生效的全部配置快照(含 argv / PAPER_DEFAULTS)
  {out}/train.log         统一格式日志 "[时间][TAG] 内容"

每 epoch 记录: loss / acc / sparsity / w_bit / a_bit / CR_bops / lr / 耗时,
并提取 10 类特征原型 (num_epochs, 10, 64) 供 t-SNE 轨迹图使用。

用法:
  python train.py configs/compare_resnet20.yaml                      # 正式跑
  python train.py configs/compare_resnet20.yaml --quick-test         # 临时覆盖
  python train.py configs/compare_resnet20.yaml --modes daco         # 只跑一侧
  python train.py configs/compare_resnet20.yaml --log-level debug    # 看 MCSS 明细
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import math
import logging
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


def _find_config_path(argv):
    """从命令行里找 YAML 配置路径。

    不能用 argparse 的位置参数来捕获配置文件: parse_known_args 不认识
    --warmup-epochs 之类的选项，会把紧跟其后的数值（例如 15）当成位置参数
    config，直接报 "配置文件不存在: 15"。这里改为显式扫描 argv：
      1) --config xxx.yaml
      2) 任意以 .yaml/.yml 结尾且不以 - 开头的参数（兼容 `script.py cfg.yaml`）
    """
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
    for a in argv:
        if not a.startswith("-") and a.lower().endswith((".yaml", ".yml")):
            return a
    return None


_CONFIG_PATH = _find_config_path(sys.argv[1:])

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
# 统一日志: 终端 + {out}/train.log 双写, 每行固定格式 "[时间][TAG] 内容"
# TAG 约定: SYS/CONFIG/DATA/SCHED/PHI/BIT/EPOCH/RESUME/CKPT/PROTO/WARN/DONE
# ===========================================================================
_LOG_FH = None
_LOG_PATH = None
_PENDING = []   # 日志文件尚未打开时先缓存, 开档后按序补打


def open_log(path):
    global _LOG_FH, _LOG_PATH
    if _LOG_FH is not None:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    _LOG_FH = open(path, "a", encoding="utf-8")
    _LOG_PATH = path
    _log("SYS", f"日志文件 -> {os.path.abspath(path)}")


def close_log():
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def _log(tag, msg):
    if _LOG_FH is None:
        _PENDING.append((tag, msg))
        return
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}][{tag:<6}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def flush_pending():
    for tag, msg in _PENDING:
        _log(tag, msg)
    _PENDING.clear()


def dump_run_config(args, path):
    """把本次运行的全部配置落一份 yaml, 保证任何一次实验都能回溯。"""
    try:
        import yaml
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "argv": sys.argv,
            "config_file": getattr(args, "config", None),
            "framework_root": FRAMEWORK_ROOT,
            "args": {k: v for k, v in sorted(vars(args).items())},
            "paper_defaults": dict(sorted(PAPER_DEFAULTS.items())),
            "baseline_override": dict(sorted(BASELINE_OVERRIDE.items())),
        }
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        _log("CONFIG", f"本次运行配置 -> {os.path.abspath(path)}")
    except Exception as e:  # noqa: BLE001
        _log("WARN", f"配置落盘失败: {e}")


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
    phi_a_threshold=0.9,   # 激活位宽渐进收缩阈值 (与 phi_b 独立)
    kappa=0.5,
    W=5,
    beta_p=1.0,
    diffusion_noise_init=1e-4,
    diffusion_noise_ratio=0.5,
)

# Baseline: 硬投影, 关闭 RCAJS 自适应 / MCSS / DGD
BASELINE_OVERRIDE = dict(
    phi_b_threshold=1.0,        # 永不触发位宽自适应 (走原始固定 bit_reduction)
    phi_a_threshold=1.0,        # 激活位宽同样永不触发, 保持纯 GETA
    beta_p=0.0,                 # exp(-0*Δ)=1, 退化为无 Lagrangian 自适应
    kappa=0.0,                  # 关闭 MCSS
    diffusion_noise_init=0.0,   # 关闭 DGD
    diffusion_noise_ratio=0.0,
    # W 是两边共用的 MCSS 窗口, 语义上不属于"关断项", 但 build_optimizer 的
    # 日志快照会读 cfg["W"], 缺 key 会让 baseline 在构造优化器时直接 KeyError。
    W=PAPER_DEFAULTS["W"],
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
    else:
        # 输入尺寸固定(32x32), benchmark 会自动挑选最快的 conv 算法。
        # 只影响 kernel 选择, 不改变数值结果。
        torch.backends.cudnn.benchmark = True


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

    def __init__(self, root, train=True, augment=False,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                 crop_padding=4, hflip=True):
        import pickle
        self.mean_t = torch.tensor(list(mean), dtype=torch.float32).view(3, 1, 1)
        self.std_t = torch.tensor(list(std), dtype=torch.float32).view(3, 1, 1)
        self.crop_padding = int(crop_padding)
        self.hflip = bool(hflip)
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
        x = (x - self.mean_t) / self.std_t
        if self.augment:
            p = self.crop_padding
            if p > 0:
                x = torch.nn.functional.pad(x, (p, p, p, p))
                i = int(torch.randint(0, 2 * p, (1,)))
                j = int(torch.randint(0, 2 * p, (1,)))
                x = x[:, i:i + 32, j:j + 32]
            if self.hflip and torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
        return x, int(self.targets[idx])


def get_dataloaders(data_dir, batch_size=128, num_workers=4, seed=1,
                    random_data=False, fast=True,
                    norm_mean=(0.4914, 0.4822, 0.4465),
                    norm_std=(0.2023, 0.1994, 0.2010),
                    crop_padding=4, hflip=True, eval_batch_size=256):
    import torchvision.transforms as T
    if random_data:
        _log("DATA", "使用随机数据验证链路")
        g = torch.Generator().manual_seed(seed)
        trainloader = DataLoader(_RandomDataset(1000), batch_size=batch_size,
                                 shuffle=True, num_workers=2, drop_last=True,
                                 worker_init_fn=_seed_worker, generator=g)
        testloader = DataLoader(_RandomDataset(200), batch_size=eval_batch_size,
                                shuffle=False, num_workers=2,
                                worker_init_fn=_seed_worker)
        return trainloader, testloader

    if fast:
        trainset = _FastCIFARLocal(data_dir, train=True, augment=True,
                                   mean=norm_mean, std=norm_std,
                                   crop_padding=crop_padding, hflip=hflip)
        testset = _FastCIFARLocal(data_dir, train=False, augment=False,
                                  mean=norm_mean, std=norm_std)
        _log("DATA", f"CIFAR-10(tensor) train={len(trainset)} test={len(testset)}")
        g = torch.Generator().manual_seed(seed)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=True, drop_last=True,
                                 generator=g)
        testloader = DataLoader(testset, batch_size=eval_batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        return trainloader, testloader

    mean = tuple(norm_mean)
    std = tuple(norm_std)
    aug = []
    if hflip:
        aug.append(T.RandomHorizontalFlip())
    if crop_padding > 0:
        aug.insert(0, T.RandomCrop(32, padding=crop_padding))
    train_tf = T.Compose(aug + [T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    trainset = _CIFAR10Local(data_dir, train=True, transform=train_tf)
    testset = _CIFAR10Local(data_dir, train=False, transform=test_tf)
    _log("DATA", f"CIFAR-10(PIL) train={len(trainset)} test={len(testset)}")
    g = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True,
                             drop_last=True, worker_init_fn=_seed_worker,
                             generator=g)
    testloader = DataLoader(testset, batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=_seed_worker)
    return trainloader, testloader


# ===========================================================================
# 模型构建
# ===========================================================================
def build_model(model_name, device, quant_mode="weight_and_activation"):
    if model_name == "resnet20":
        from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
        model = resnet20_cifar10()
    elif model_name == "vgg7":
        from sanity_check.backends.vgg7 import vgg7_bn
        model = vgg7_bn()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    dummy = torch.rand(1, 3, 32, 32)
    qm = (QuantizationMode.WEIGHT_ONLY if quant_mode == "weight_only"
          else QuantizationMode.WEIGHT_AND_ACTIVATION)
    model = model_to_quantize_model(model, quant_mode=qm)
    model = model.to(device)
    return model, dummy


# ===========================================================================
# 优化器构建
# ===========================================================================
def build_optimizer(model, dummy, device, mode, joint_epochs, log_dir,
                    variant="adam", lr=1e-3, lr_quant=1e-3, weight_decay=1e-4,
                    importance_criteria="default", projection_periods=1,
                    projection_steps=1, pruning_steps=1):
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
        lr_quant=lr_quant,
        weight_decay=weight_decay,
        target_group_sparsity=PAPER_DEFAULTS["target_group_sparsity"],
        start_projection_step=0,
        projection_periods=projection_periods,
        projection_steps=projection_steps,
        start_pruning_step=0,
        pruning_periods=int(joint_epochs),  # 每 epoch 一次剪枝调度
        pruning_steps=pruning_steps,
        bit_reduction=PAPER_DEFAULTS["bit_reduction"],
        min_bit_wt=PAPER_DEFAULTS["min_bit_wt"],
        max_bit_wt=PAPER_DEFAULTS["max_bit_wt"],
        min_bit_act=PAPER_DEFAULTS["min_bit_act"],
        max_bit_act=PAPER_DEFAULTS["max_bit_act"],
        importance_score_criteria=importance_criteria,
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
        _log("DACO", f"RCAJS beta_p={cfg['beta_p']} "
                     f"phi_b={cfg['phi_b_threshold']} phi_a={cfg.get('phi_a_threshold')} "
                     f"| MCSS kappa={cfg['kappa']} W={cfg['W']} "
                     f"| DGD noise={cfg['diffusion_noise_init']}")
    else:
        _log("COMPAT", "原版框架: 无 RCAJS/MCSS/DGD, 走纯 GETA 行为")
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


def rcajs_phi_scheduler(opt, phi_b_threshold, bit_reduction, min_bit_wt,
                        phi_a_threshold=None, min_bit_act=2):
    """渐进位宽收缩: 权重 (phi_b) 与激活 (phi_a) 各自独立触发。

    返回 (wt_triggered, act_triggered)。

    为什么激活也要收缩:
      max_bit_act 在 geta.py 里只在 __init__ 赋值一次, 之后没有任何地方下调;
      而 partial_projected_gradient_descent_step_range_act 用
      max_bit_act/min_bit_act 算 d_quant 的 clamp 上下界。所以只要不降
      max_bit_act, 激活位宽就会一直停在 11~12 bit, 与论文表里 A-Bit 5.13 差一倍以上,
      BOPs 压缩率会被 (32/a_bit) 这一项直接拖死。
    """
    bit_layers = refresh_bit_layers(opt)
    if not bit_layers:
        return False, False

    # --- 权重侧 phi_b ---
    wt_hit = False
    below, total = 0, 0
    for v in bit_layers.values():
        if isinstance(v, dict) and v.get("weight") is not None:
            total += 1
            if v["weight"] < opt.max_bit_wt - 1:
                below += 1
    ratio_wt = below / total if total else 0.0
    if total > 0 and ratio_wt > phi_b_threshold and opt.max_bit_wt > min_bit_wt:
        opt.max_bit_wt = max(int(opt.max_bit_wt) - int(bit_reduction),
                             int(min_bit_wt))
        wt_hit = True

    # --- 激活侧 phi_a ---
    act_hit = False
    ratio_act = 0.0
    if phi_a_threshold is not None:
        below_a, total_a = 0, 0
        for v in bit_layers.values():
            if isinstance(v, dict) and v.get("activation") is not None:
                total_a += 1
                if v["activation"] < opt.max_bit_act - 1:
                    below_a += 1
        ratio_act = below_a / total_a if total_a else 0.0
        if (total_a > 0 and ratio_act > phi_a_threshold
                and opt.max_bit_act > min_bit_act):
            opt.max_bit_act = max(int(opt.max_bit_act) - int(bit_reduction),
                                  int(min_bit_act))
            act_hit = True

    # 每 epoch 一行, 供敏感性分析和排查 (φ 是否真的越过阈值)
    _log("PHI", f"n_wt={total} phi_wt={ratio_wt:.3f}/{phi_b_threshold} "
                f"max_wt={opt.max_bit_wt} | n_act={total_a if phi_a_threshold is not None else 0} "
                f"phi_act={ratio_act:.3f}/{phi_a_threshold} max_act={opt.max_bit_act}")

    return wt_hit, act_hit


def configure_training_schedule(opt, warmup_epochs, joint_epochs, steps_per_epoch,
                                prune_shape="linear", prune_floor=0.0,
                                prune_budget="rcajs", tag=""):
    opt.prune_budget = prune_budget   # "fixed"=纯GETA固定预算 / "rcajs"=DACO闭环
    start_step = warmup_epochs * steps_per_epoch
    joint_steps = max(1, joint_epochs * steps_per_epoch)
    opt.start_projection_step = start_step
    opt.start_pruning_step = start_step
    opt.pruning_steps = joint_steps
    opt.pruning_period_duration = max(1, joint_steps // max(opt.pruning_periods, 1))
    opt.diffusion_steps = joint_steps

    # ---- 剪枝预算数组 (累计取整, 保证最后一步刚好剪到 target) ----
    # 原实现是 active_num_redundant_groups[p] = target // periods (每期都向下取整),
    # ResNet-20 只有 448 个可剪组, target=0.5*448=224, periods=85 → 224//85=2,
    # 85 期只能剪 170 组 → 最终稀疏度 0.379, 永远到不了 0.5 (实测日志
    # "[DACO][PRUNE] requested=2 actual_pruned=2" 印证)。
    # 改为按累计目标取整再差分: cum_p = round(target * Σshape[:p+1]/Σshape),
    # arr[p] = cum_p - cum_{p-1} → 总和严格等于 target, 末期自动补齐。
    # shape="linear" 时 Σshape 按等权, 即每期 2/3 交替, 与原线性调度等价但不丢量。
    periods = int(getattr(opt, "pruning_periods", joint_epochs))
    target = int(getattr(opt, "target_num_redundant_groups", 0))
    if periods > 1 and target > 0:
        if prune_shape == "cosine":
            shape = [prune_floor + (1.0 - prune_floor)
                     * 0.5 * (1.0 + math.cos(math.pi * p / (periods - 1)))
                     for p in range(periods)]
        else:
            shape = [1.0] * periods
        total_shape = sum(shape)
        arr, prev = [], 0
        for p in range(periods):
            cum = int(round(target * sum(shape[:p + 1]) / total_shape))
            arr.append(cum - prev)
            prev = cum
        opt.active_num_redundant_groups = arr
        _log("SCHED", f"{tag} shape={prune_shape} budget={prune_budget} "
                      f"periods={periods} target_groups={target} "
                      f"(稀疏度 {target / max(opt.total_num_groups, 1):.3f}) "
                      f"前3期={arr[:3]} 中={arr[periods // 2]} 末3期={arr[-3:]} "
                      f"合计={sum(arr)} "
                      f"累计稀疏度={sum(arr) / max(opt.total_num_groups, 1):.4f}")


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
    avg_ab = 0.0
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
        abits = [v.get("activation") for v in opt.bit_layers.values()
                 if v.get("activation") is not None]
        if abits:
            avg_ab = sum(abits) / len(abits)
    except Exception as e:
        print(f"  [WARN] bit_layers: {e}")
    try:
        num_pruned = len(getattr(opt, "pruned_group_idxes", []))
    except Exception:
        pass
    prune_ratio = num_pruned / max(total_groups, 1)
    bit_compress = 16.0 / max(avg_wb, 1e-6) if avg_wb > 0 else 1.0
    overall_compress = (1 - prune_ratio) * bit_compress
    # 论文口径的 BOPs 压缩率: CR = 32*32 / ((1-s) * w_bit * a_bit)
    # 注意与上面两个历史字段(16/avg_wb, 漏掉激活位宽)不是同一个口径, 仅供对照。
    if avg_wb > 0 and avg_ab > 0:
        cr_bops = (32.0 * 32.0) / ((1.0 - prune_ratio) * avg_wb * avg_ab)
    else:
        cr_bops = 0.0
    return dict(group_sparsity=group_sparsity, num_zero_groups=num_zero_groups,
                avg_wb=avg_wb, min_wb=min_wb, avg_ab=avg_ab,
                num_pruned=num_pruned,
                total_groups=total_groups, prune_ratio=prune_ratio,
                bit_compress=bit_compress, overall_compress=overall_compress,
                cr_bops=cr_bops)


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
def _flush_epoch_artifacts(args, mode, history, prototypes_seq):
    """每个 epoch 结束时增量落盘, 保证训练中断也能保留已产出的数据。

    产物:
      prototypes/{model}_{mode}_protos.npy  (T_sofar, 10, D) 类别原型序列
      {mode}_history.jsonl                  每行一个 epoch 的全部指标

    用覆盖写而非追加: 与内存中的 history 完全同步(幂等), 且文件很小
    (100 epoch 的 npy 约 256 KB, jsonl 约 20 KB), 写盘开销可忽略。
    """
    try:
        proto_dir = os.path.join(args.out, "prototypes")
        os.makedirs(proto_dir, exist_ok=True)
        arr = np.stack(prototypes_seq, axis=0).astype(np.float32)
        np.save(os.path.join(proto_dir, f"{args.model}_{mode}_protos.npy"), arr)

        keys = list(history.keys())
        hp = os.path.join(args.out, f"{mode}_history.jsonl")
        with open(hp, "w", encoding="utf-8") as f:
            for i in range(len(history["epoch"])):
                f.write(json.dumps({k: history[k][i] for k in keys},
                                   ensure_ascii=False) + "\n")
    except Exception as e:  # 落盘失败不应打断训练
        print(f"  [WARN] 增量落盘失败: {e}", flush=True)


def _save_ckpt(path, model, opt, scheduler, epoch, history, prototypes_seq,
               total_epochs=None):
    """保存可续训的完整状态。

    注意 GETA 的一阶/二阶动量缓冲是优化器的自定义属性(first_moment_grads /
    second_moment_grads), 不在 torch.optim 的标准 state 里, 必须单独存,
    否则续训后 Adam 的 v 需要重新估计, 会造成 loss 突跳。
    history 与 prototypes 一并保存, 保证续训后 t-SNE 轨迹不断档。
    """
    torch.save({
        "epoch": epoch,                      # 已完成的 epoch 数(1-based)
        "total_epochs": total_epochs,
        "model": model.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt_moments": {
            "first": getattr(opt, "first_moment_grads", {}),
            "second": getattr(opt, "second_moment_grads", {}),
        },
        "num_steps": getattr(opt, "num_steps", 0),
        "history": history,
        "prototypes": list(prototypes_seq),
    }, path)


def _resume_from(path, model, opt, scheduler, history, prototypes_seq, device,
                 total_epochs=None):
    """从 ckpt 恢复现场, 返回已完成的 epoch 数。"""
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 老版本 torch 无 weights_only 参数
        ck = torch.load(path, map_location=device)

    saved_total = ck.get("total_epochs")
    if (saved_total is not None and total_epochs is not None
            and saved_total != total_epochs):
        print(f"  [RESUME][WARN] ckpt total_epochs={saved_total} 与本次 "
              f"{total_epochs} 不一致, 余弦退火曲线的 T_max 变化会导致学习率"
              f"不连续; 续训请使用完全相同的 --warmup-epochs/--epochs",
              flush=True)

    model.load_state_dict(ck["model"])

    mom = ck.get("opt_moments") or {}
    # 只恢复参数名能对上的动量 buffer。旧版 ckpt 用的是 (group_idx, param_idx)
    # 索引 key, 剪枝重排 param_groups 后必然错位, 直接丢弃比套错更安全。
    valid_names = {n for g in opt.param_groups for n in g["p_names"]}
    for attr, key in (("first_moment_grads", "first"),
                      ("second_moment_grads", "second")):
        saved = mom.get(key) or {}
        restored = {k: v for k, v in saved.items() if k in valid_names}
        setattr(opt, attr, restored)
        dropped = len(saved) - len(restored)
        if dropped:
            print(f"  [RESUME] {attr}: 丢弃 {dropped} 个无法匹配的旧 buffer "
                  f"(动量将从 0 重新估计)", flush=True)
    opt.num_steps = ck.get("num_steps", opt.num_steps)

    try:
        scheduler.load_state_dict(ck["scheduler"])
    except Exception as e:
        print(f"  [RESUME][WARN] scheduler 恢复失败({e}), 从当前 step 重新计时")

    saved_hist = ck.get("history", {})
    for k in history:
        history[k] = list(saved_hist.get(k, []))
    prototypes_seq.clear()
    prototypes_seq.extend([np.asarray(p) for p in ck.get("prototypes", [])])

    return int(ck.get("epoch", 0))


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
        fast=not getattr(args, "legacy_data", False),
        norm_mean=args.norm_mean, norm_std=args.norm_std,
        crop_padding=args.crop_padding, hflip=args.hflip,
        eval_batch_size=args.eval_batch_size)
    steps_per_epoch = len(trainloader)
    _log("DATA", f"train batches={steps_per_epoch} test batches={len(testloader)}")

    model, dummy = build_model(args.model, device, quant_mode=args.quant_mode)
    log_dir = os.path.join(args.out, "logs", mode)
    oto, opt = build_optimizer(model, dummy, device, mode, args.epochs, log_dir,
                               variant=args.variant, lr=args.lr,
                               lr_quant=args.lr_quant,
                               weight_decay=args.weight_decay,
                               importance_criteria=args.importance_criteria,
                               projection_periods=args.projection_periods,
                               projection_steps=args.projection_steps,
                               pruning_steps=args.pruning_steps)

    warmup = 0 if args.quick_test else args.warmup_epochs
    joint = args.epochs
    total = warmup + joint
    # baseline 必须是纯 GETA: 固定预算 + 不调用 RCAJS, 否则 exp(-βΔ)/Ψ 仍会
    # 改变剪枝节奏, 两边稀疏度就不再逐期一致。
    # DACO 的调度差异体现在预算数组的"形状"上(余弦 前多后少), 而不是让 rcaJS
    # 用绝对速率覆盖数组 —— r_p 是 0.08→0.01 的绝对速率, ceil(0.08*448)=36 组/期,
    # 224/36 ≈ 6.2, 会在第 7 个 epoch 就剪满 0.5, 形状完全失效。
    budget = "fixed" if mode == "baseline" else getattr(args, "prune_budget", "fixed")
    # 形状: baseline 阶梯均摊, DACO 余弦前多后少; 两边总和都严格等于 target(0.5)
    shape = "linear" if mode == "baseline" else getattr(args, "prune_shape", "cosine")
    configure_training_schedule(opt, warmup, joint, steps_per_epoch,
                                prune_shape=shape,
                                prune_floor=getattr(args, "prune_floor", 0.0),
                                prune_budget=budget,
                                tag=mode)

    # 余弦退火: eta_min 必须显著小于起始 lr, 否则退火失效。
    # 原先写死 1e-3, 在 sgd/lr=0.1 时是 0.1->0.001 的正常衰减; 但改成
    # adam/lr=1e-3 后起点与终点都是 1e-3, 等于全程恒定学习率,
    # 会显著压低最终精度并放大剪枝阶段的准确率震荡。
    eta_min = args.lr * args.lr_eta_min_ratio
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total * steps_per_epoch, eta_min=eta_min)
    _log("SCHED", f"{mode} CosineAnnealingLR lr {args.lr:g} -> {eta_min:g} "
                  f"over {total * steps_per_epoch} steps")
    criterion = nn.CrossEntropyLoss()

    # 每 epoch 记录: 指标 + 类别原型
    history = {"epoch": [], "phase": [], "train_acc": [], "val_acc": [],
               "train_loss": [], "val_loss": [],
               "prune_ratio": [], "avg_weight_bit": [], "group_sparsity": [],
               "avg_act_bit": [], "cr_bops": [],
               "bit_compress": [], "overall_compress": [], "lr": []}
    prototypes_seq = []  # list of (10, D) arrays, 每 epoch 一个

    cfg = BASELINE_OVERRIDE if mode == "baseline" else PAPER_DEFAULTS
    feat_target = "avg_pool" if args.model == "resnet20" else "classifier"

    # ---- 断点续训 ----
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt = os.path.join(ckpt_dir, f"{args.model}_{mode}_latest.pt")
    start_ep = 0
    if getattr(args, "resume", False) and os.path.exists(latest_ckpt):
        start_ep = _resume_from(latest_ckpt, model, opt, scheduler,
                                history, prototypes_seq, device, total)
        _log("RESUME", f"{mode} 从 {latest_ckpt} 恢复, 已完成 {start_ep} epoch, "
                       f"num_steps={opt.num_steps}")
    elif getattr(args, "resume", False):
        _log("RESUME", f"{mode} 未找到 {latest_ckpt}, 从头开始")

    t0 = time.time()

    for ep in range(start_ep, total):
        phase = "warmup" if ep < warmup else "joint"
        t_ep = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, opt, trainloader, criterion, device, ep+1, scheduler)
        if phase == "joint" and mode != "baseline" and hasattr(opt, "rcajs"):
            wt_hit, act_hit = rcajs_phi_scheduler(
                opt, cfg["phi_b_threshold"],
                PAPER_DEFAULTS["bit_reduction"],
                PAPER_DEFAULTS["min_bit_wt"],
                phi_a_threshold=cfg.get("phi_a_threshold"),
                min_bit_act=PAPER_DEFAULTS["min_bit_act"])
            if wt_hit or act_hit:
                _log("BIT", f"{mode} ep{ep + 1} 位宽下台阶 "
                            f"max_bit_wt{'↓' if wt_hit else '='}"
                            f"{opt.max_bit_wt} "
                            f"max_bit_act{'↓' if act_hit else '='}"
                            f"{opt.max_bit_act}")
        v_loss, v_acc = evaluate(model, testloader, criterion, device)
        m = collect_metrics(opt)
        # 提取类别原型
        protos = extract_class_prototypes(model, testloader, device, feat_target,
                                          max_samples=args.prototypes_max_samples)
        prototypes_seq.append(protos)

        _log("EPOCH", f"{mode:<8} {ep + 1:3d}/{total} {phase:<6} "
                      f"| loss {tr_loss:6.4f}/{v_loss:6.4f} "
                      f"| acc {tr_acc:.4f}/{v_acc:.4f} "
                      f"| spars {m['group_sparsity']:.4f} "
                      f"| bit {m['avg_wb']:5.2f}/{m['avg_ab']:5.2f} "
                      f"| CR {m['cr_bops']:6.2f}x "
                      f"| lr {opt.param_groups[0]['lr']:.2e} "
                      f"| {time.time() - t_ep:.0f}s")

        history["epoch"].append(ep + 1)
        history["phase"].append(phase)
        history["train_acc"].append(round(tr_acc, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["prune_ratio"].append(round(m["prune_ratio"], 4))
        history["avg_weight_bit"].append(round(m["avg_wb"], 2))
        history["avg_act_bit"].append(round(m["avg_ab"], 2))
        history["cr_bops"].append(round(m["cr_bops"], 2))
        history["group_sparsity"].append(round(m["group_sparsity"], 4))
        history["bit_compress"].append(round(m["bit_compress"], 2))
        history["overall_compress"].append(round(m["overall_compress"], 2))
        history["lr"].append(round(opt.param_groups[0]["lr"], 6))

        # 增量落盘: 每个 epoch 结束立即写盘。
        # 原先只在训练全部结束后才 np.save, 一旦中途中断(崩溃/手动停止)则
        # 几小时的原型数据全部丢失, 改图只能重跑。改为逐 epoch 覆盖写后,
        # 任何时刻磁盘上都有"到目前为止"的完整可画图数据。
        _flush_epoch_artifacts(args, mode, history, prototypes_seq)

        # 断点续训: latest 每 epoch 覆盖(单份约 3.5MB), 另每 ckpt_every 存快照。
        # 放在 flush 之后, 保证 ckpt 与已落盘的 history 一致。
        _save_ckpt(latest_ckpt, model, opt, scheduler, ep + 1,
                   history, prototypes_seq, total)
        if args.ckpt_every > 0 and (ep + 1) % args.ckpt_every == 0:
            _save_ckpt(os.path.join(ckpt_dir, f"{args.model}_{mode}_ep{ep+1}.pt"),
                       model, opt, scheduler, ep + 1, history, prototypes_seq, total)

    elapsed = time.time() - t0
    m_final = collect_metrics(opt)

    # 保存原型序列 (num_epochs, 10, D)
    proto_arr = np.stack(prototypes_seq, axis=0)  # (T, 10, D)
    proto_path = os.path.join(args.out, "prototypes", f"{args.model}_{mode}_protos.npy")
    os.makedirs(os.path.dirname(proto_path), exist_ok=True)
    np.save(proto_path, proto_arr.astype(np.float32))
    _log("PROTO", f"{mode} {proto_arr.shape} -> {proto_path}")

    # 保存最终 checkpoint
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.model}_{mode}_final.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "val_acc": history["val_acc"][-1],
                "history": history}, ckpt_path)
    _log("CKPT", f"{mode} -> {ckpt_path}")

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
        "final_avg_act_bit": round(m_final["avg_ab"], 2),
        "final_cr_bops": round(m_final["cr_bops"], 2),
        "final_overall_compress": round(m_final["overall_compress"], 2),
        "elapsed_minutes": round(elapsed / 60, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ckpt": ckpt_path,
        "proto_path": proto_path,
        "history": history,
    }
    _log("DONE", f"{mode} best_acc={result['best_val_acc']} "
                 f"final_acc={result['final_val_acc']} "
                 f"spars={result['final_prune_ratio']:.4f} "
                 f"w_bit={result['final_avg_weight_bit']} "
                 f"CR_old={result['final_overall_compress']}x "
                 f"CR_bops={result.get('final_cr_bops')}x "
                 f"耗时={elapsed/60:.1f}min")
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


# yaml 顶层键 → argparse dest
_YAML_TO_ARG = {
    "model": "model", "modes": "modes", "epochs": "epochs",
    "warmup_epochs": "warmup_epochs", "seed": "seed", "out": "out",
    "ckpt_every": "ckpt_every", "lr": "lr", "variant": "variant",
    "lr_eta_min_ratio": "lr_eta_min_ratio",
    "prune_shape": "prune_shape", "prune_floor": "prune_floor",
    "prune_budget": "prune_budget", "root": "root",
    "deterministic": "deterministic", "quant_mode": "quant_mode",
    # optimizer 块
    "lr_quant": "lr_quant", "weight_decay": "weight_decay",
    "importance_criteria": "importance_criteria",
    "projection_periods": "projection_periods",
    "projection_steps": "projection_steps", "pruning_steps": "pruning_steps",
}


def _apply_config(p):
    """加载 YAML 并转成 argparse 默认值; compression 块直接写入 PAPER_DEFAULTS。

    优先级: 命令行 > yaml > 内置默认值。
    这样 `python train.py cfg.yaml --quick-test` 依然可以临时覆盖。
    """
    cfg_path = _CONFIG_PATH
    if not cfg_path:
        return p.parse_args()
    if not os.path.exists(cfg_path):
        p.error(f"配置文件不存在: {cfg_path}")
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        p.error(f"解析配置文件失败 {cfg_path}: {e}")

    defaults = {}
    for k, v in cfg.items():
        if k in _YAML_TO_ARG and v is not None:
            defaults[_YAML_TO_ARG[k]] = v
    ds = cfg.get("dataset") or {}
    if ds.get("data_dir"):
        defaults["data_dir"] = ds["data_dir"]
    if ds.get("batch_size") is not None:
        defaults["batch_size"] = int(ds["batch_size"])
    if ds.get("num_workers") is not None:
        defaults["num_workers"] = int(ds["num_workers"])
    if ds.get("eval_batch_size") is not None:
        defaults["eval_batch_size"] = int(ds["eval_batch_size"])
    norm = ds.get("normalization") or {}
    if norm.get("mean"):
        defaults["norm_mean"] = [float(x) for x in norm["mean"]]
    if norm.get("std"):
        defaults["norm_std"] = [float(x) for x in norm["std"]]
    aug = ds.get("augmentation") or {}
    if aug.get("random_crop_padding") is not None:
        defaults["crop_padding"] = int(aug["random_crop_padding"])
    if aug.get("random_horizontal_flip") is not None:
        defaults["hflip"] = bool(aug["random_horizontal_flip"])

    # optimizer 块 (学习率细项 / 重要性标准 / 投影与剪枝步数)
    opt_blk = cfg.get("optimizer") or {}
    for k, v in opt_blk.items():
        if k in _YAML_TO_ARG and v is not None:
            defaults[_YAML_TO_ARG[k]] = v

    # prototypes 块
    pr = cfg.get("prototypes") or {}
    if pr.get("max_samples") is not None:
        defaults["prototypes_max_samples"] = int(pr["max_samples"])

    # debug 块 (等价于命令行 --random-data / --quick-test / --legacy-data)
    dbg = cfg.get("debug") or {}
    for k in ("random_data", "quick_test", "legacy_data"):
        if dbg.get(k) is not None:
            defaults[k] = bool(dbg[k])

    p.set_defaults(**defaults)

    comp = cfg.get("compression") or {}
    applied = {}
    for k, v in comp.items():
        if k in PAPER_DEFAULTS and v is not None:
            PAPER_DEFAULTS[k] = v
            applied[k] = v

    # baseline 的覆盖项也可写进 yaml, 保证"纯 GETA"的关断配置有记录
    bo = cfg.get("baseline_override") or {}
    bo_applied = {}
    for k, v in bo.items():
        if k in BASELINE_OVERRIDE and v is not None:
            BASELINE_OVERRIDE[k] = v
            bo_applied[k] = v

    # logging 块 -> log_level / log_file
    lg = cfg.get("logging") or {}
    if lg.get("level"):
        defaults["log_level"] = lg["level"]
    if lg.get("file"):
        defaults["log_file"] = lg["file"]
    if defaults:
        p.set_defaults(**defaults)

    _log("CONFIG", f"配置文件 {cfg_path}")
    if defaults:
        _log("CONFIG", "训练项: " +
             ", ".join(f"{k}={v}" for k, v in sorted(defaults.items())))
    if applied:
        _log("CONFIG", "compression: " +
             ", ".join(f"{k}={v}" for k, v in sorted(applied.items())))
    if bo_applied:
        _log("CONFIG", "baseline_override: " +
             ", ".join(f"{k}={v}" for k, v in sorted(bo_applied.items())))

    args = p.parse_args()
    args.config = cfg_path
    return args


def main():
    p = argparse.ArgumentParser(
        description="通用训练入口: python train.py <config.yaml> "
                    "(配置全在 yaml, 命令行仅作临时覆盖)")
    p.add_argument("config", nargs="?", default=None,
                   help="YAML 配置文件, 如 configs/compare_resnet20.yaml; "
                        "命令行参数优先覆盖 yaml 中的同名项")
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
    p.add_argument("--lr-eta-min-ratio", type=float, default=0.01,
                   dest="lr_eta_min_ratio",
                   help="余弦退火终点 = lr * 该比例 (默认 0.01 -> 1e-5)。"
                        "写死 1.0 时起点=终点, 退火失效")
    p.add_argument("--quant-mode", default="weight_and_activation",
                   dest="quant_mode",
                   choices=["weight_and_activation", "weight_only"],
                   help="量化模式; 论文对比用的是 weight_and_activation")
    p.add_argument("--log-file", default=None, dest="log_file",
                   help="训练日志路径 (默认 {out}/train.log, 与终端同时输出)")
    p.add_argument("--log-level", default="info", dest="log_level",
                   choices=["debug", "info", "warn"],
                   help="底层 logger 级别; debug 会打出 MCSS 每层每 criterion 明细")
    # ---- 优化器细项 (yaml 的 optimizer 块) ----
    p.add_argument("--lr-quant", type=float, default=1e-3, dest="lr_quant",
                   help="量化参数学习率 (d_quant/q_m/t_quant)")
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--importance-criteria", default="default",
                   dest="importance_criteria",
                   help="重要性打分标准; default 为 OTO 内置组合")
    p.add_argument("--projection-periods", type=int, default=1,
                   dest="projection_periods")
    p.add_argument("--projection-steps", type=int, default=1,
                   dest="projection_steps")
    p.add_argument("--pruning-steps", type=int, default=1, dest="pruning_steps")
    # ---- 数据细项 (yaml 的 dataset 子块) ----
    p.add_argument("--norm-mean", type=float, nargs=3,
                   default=[0.4914, 0.4822, 0.4465], dest="norm_mean")
    p.add_argument("--norm-std", type=float, nargs=3,
                   default=[0.2023, 0.1994, 0.2010], dest="norm_std")
    p.add_argument("--crop-padding", type=int, default=4, dest="crop_padding",
                   help="RandomCrop 的 pad 像素; 0 表示不做裁剪增强")
    p.add_argument("--hflip", type=lambda s: str(s).lower() not in ("0", "false", "no"),
                   default=True, dest="hflip", help="是否随机水平翻转")
    p.add_argument("--eval-batch-size", type=int, default=256,
                   dest="eval_batch_size")
    # ---- 原型 ----
    p.add_argument("--prototypes-max-samples", type=int, default=2000,
                   dest="prototypes_max_samples",
                   help="每 epoch 提原型用的样本上限 (固定前 N 个, 保证跨 epoch 可比)")
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
    p.add_argument("--resume", action="store_true",
                   help="从 {out}/checkpoints/{model}_{mode}_latest.pt 断点续训")
    p.add_argument("--ckpt-every", type=int, default=10, dest="ckpt_every",
                   help="每 N 个 epoch 额外存一份快照 (0=只存 latest, 默认 10)")
    p.add_argument("--modes", default="baseline,daco")
    p.add_argument("--prune-shape", default="cosine",
                   choices=["linear", "cosine"], dest="prune_shape",
                   help="DACO 的剪枝预算形状 (baseline 恒为 linear 阶梯): "
                        "cosine=前多后少; linear=每 epoch 均摊。两者总和都 = target")
    p.add_argument("--prune-floor", type=float, default=0.0, dest="prune_floor",
                   help="cosine 形状的尾部下限 (0=末期几乎不剪, 0.2=末期仍保留 20%% 权重)")
    p.add_argument("--prune-budget", default="fixed", dest="prune_budget",
                   choices=["rcajs", "fixed"],
                   help="DACO 的剪枝组数由谁决定: fixed=预生成形状数组(推荐, 终点严格 0.5); "
                        "rcajs=RCAJS 绝对速率接管(注意: r_p 未按形状归一化, "
                        "ceil(0.08*448)=36 组/期会在第 7 个 epoch 就剪满 0.5)")
    args = _apply_config(p)

    # ---- 日志: 终端 + 文件双写, 并把本次配置落盘 ----
    log_path = args.log_file or os.path.join(args.out, "train.log")
    open_log(log_path)
    flush_pending()
    dump_run_config(args, os.path.join(args.out, "run_config.yaml"))

    # 抑制底层高频 INFO: MCSS 每层每 criterion 一行, 一次 pruning 就有上百行,
    # 会把真正的 epoch 日志冲掉。需要看细节时用 --log-level debug。
    lv = {"debug": logging.DEBUG, "info": logging.INFO,
          "warn": logging.WARNING}.get(args.log_level, logging.INFO)
    logging.getLogger().setLevel(lv)
    logging.getLogger("GETA").setLevel(lv)
    logging.getLogger("DACO.rcajs").setLevel(lv)
    logging.getLogger("DACO.importance_score").setLevel(
        logging.DEBUG if args.log_level == "debug" else logging.WARNING)

    _log("SYS", f"modes={args.modes} model={args.model} "
                f"epochs={args.epochs}+{args.warmup_epochs}w "
                f"out={os.path.abspath(args.out)}")
    _log("SYS", f"python={sys.executable} root={FRAMEWORK_ROOT} "
                f"log_level={args.log_level}")

    if args.quick_test:
        args.epochs = 1
        args.warmup_epochs = 0
        _log("SYS", "QUICK TEST: 1 epoch, no warmup")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results = []
    failed = []
    for mode in modes:
        try:
            r = run_experiment(args, mode)
            save_result(r, args.out)
            all_results.append(r)
        except Exception as e:
            # 不要静默吞掉: 之前 baseline 构造期 KeyError 被这里吃掉,
            # 进程照样 exit 0, 结果只剩 daco 一个 mode 却没人发现。
            failed.append((mode, repr(e)))
            _log("ERROR", f"{mode}: {e}")
            traceback.print_exc()

    if failed:
        _log("SYS", "=" * 60)
        for mode, err in failed:
            _log("SYS", f"失败 mode: {mode} -> {err}")
        _log("SYS", f"成功 {len(all_results)}/{len(modes)} 个 mode, 退出码 1")
        _log("SYS", "=" * 60)
        close_log()
        raise SystemExit(1)

    if len(all_results) >= 2:
        _log("SYS", "=" * 60)
        _log("SYS", "对比汇总")
        b, d = all_results[0], all_results[-1]
        for k, label in [("best_val_acc", "Best Acc"), ("final_val_acc", "Final Acc"),
                         ("final_prune_ratio", "Sparsity"),
                         ("final_avg_weight_bit", "Avg W-bit"),
                         ("final_avg_act_bit", "Avg A-bit"),
                         ("final_cr_bops", "CR (BOPs)"),
                         ("final_overall_compress", "CR (legacy)")]:
            _log("SYS", f"{label:<14} baseline={b.get(k, 0):>10}  "
                        f"daco={d.get(k, 0):>10}")
        _log("SYS", "=" * 60)
    _log("SYS", f"输出目录 {os.path.abspath(args.out)}")
    _log("SYS", "出图(两个 mode 共用一个 out, 所以 origin-dir == ours-dir): "
                f"python plot_framework_compare.py "
                f"--origin-dir {args.out} --ours-dir {args.out} --out {args.out}")
    close_log()


if __name__ == "__main__":
    main()
