#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature-space drift experiments for pruning and bit-width schedules.

The script runs controlled CIFAR-10/ResNet-20 comparisons and measures
representation drift online.  It deliberately uses the same seed, data order,
warm-up length and compression target for every strategy.

Examples
--------
python run_feature_drift.py run --family pruning --seeds 1 2 3
python run_feature_drift.py run --family bitwidth --seeds 1 2 3
python run_feature_drift.py plot --input outputs/feature_drift/pruning/resnet20
"""

import argparse
import csv
import json
import math
import os
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from run_daco_cifar10 import (
    PAPER_DEFAULTS,
    build_model,
    build_optimizer,
    collect_metrics,
    configure_training_schedule,
    evaluate,
    get_dataloaders,
    get_random_dataloaders,
    refresh_bit_layers,
    save_training_checkpoint,
    set_seed,
    train_one_epoch,
)


DEFAULT_LAYERS = ["layer1", "layer2", "layer3", "avg_pool2d"]
PRUNING_STRATEGIES = ["fixed", "cosine", "rcajs"]
BITWIDTH_STRATEGIES = ["hard", "cosine", "rcajs"]
JOINT_STRATEGIES = ["fixed_hard", "cosine_cosine", "rcajs_rcajs"]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: {!r}".format(value))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False,
                  default=_json_default)


def _append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False,
                                default=_json_default) + "\n")


def _reduce_feature(output):
    if isinstance(output, (tuple, list)):
        output = next((item for item in output if torch.is_tensor(item)), None)
    if not torch.is_tensor(output):
        raise TypeError("Hook output does not contain a tensor")
    if output.ndim == 4:
        output = F.adaptive_avg_pool2d(output, 1).flatten(1)
    elif output.ndim == 3:
        output = output.mean(dim=1)
    elif output.ndim > 2:
        output = output.flatten(1)
    elif output.ndim == 1:
        output = output.unsqueeze(0)
    return output.detach().float().cpu()


class FeatureCollector:
    def __init__(self, model, layer_names):
        modules = dict(model.named_modules())
        missing = [name for name in layer_names if name not in modules]
        if missing:
            raise ValueError("Unknown feature layers: {}. Available examples: {}"
                             .format(missing, list(modules)[:30]))
        self.buffers = {name: [] for name in layer_names}
        self.handles = []
        for name in layer_names:
            self.handles.append(
                modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            self.buffers[name].append(_reduce_feature(output))
        return hook

    def clear(self):
        for values in self.buffers.values():
            values.clear()

    def result(self):
        return {name: torch.cat(values, dim=0)
                for name, values in self.buffers.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def materialize_probe(testloader, num_samples, batch_size=256):
    """Freeze normalized test tensors so every snapshot sees identical inputs."""
    inputs, labels, count = [], [], 0
    for batch_inputs, batch_labels in testloader:
        take = min(batch_inputs.size(0), num_samples - count)
        if take <= 0:
            break
        inputs.append(batch_inputs[:take].clone())
        labels.append(batch_labels[:take].clone())
        count += take
        if count >= num_samples:
            break
    if count < 2:
        raise ValueError("Probe set needs at least two samples")
    dataset = TensorDataset(torch.cat(inputs), torch.cat(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)


@torch.no_grad()
def extract_snapshot(model, probe_loader, device, layer_names):
    model.eval()
    collector = FeatureCollector(model, layer_names)
    logits, labels = [], []
    try:
        collector.clear()
        for inputs, targets in probe_loader:
            inputs = inputs.to(device, non_blocking=True)
            output = model(inputs)
            logits.append(output.detach().float().cpu())
            labels.append(targets.detach().cpu())
        features = collector.result()
    finally:
        collector.close()
    return {
        "features": features,
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
    }


def linear_cka(first, second, eps=1e-12):
    """Linear CKA in feature space; feature dimensions may differ."""
    first = first.double() - first.double().mean(dim=0, keepdim=True)
    second = second.double() - second.double().mean(dim=0, keepdim=True)
    cross = first.transpose(0, 1).matmul(second)
    first_cov = first.transpose(0, 1).matmul(first)
    second_cov = second.transpose(0, 1).matmul(second)
    numerator = torch.linalg.norm(cross, ord="fro").pow(2)
    denominator = (torch.linalg.norm(first_cov, ord="fro") *
                   torch.linalg.norm(second_cov, ord="fro"))
    return float((numerator / denominator.clamp_min(eps)).clamp(0, 1))


def neighbor_preservation(first, second, k=5):
    count = first.size(0)
    if count < 2:
        return float("nan")
    k = max(1, min(k, count - 1))

    def neighbors(features):
        normalized = F.normalize(features.float(), dim=1)
        similarity = normalized.matmul(normalized.transpose(0, 1))
        similarity.fill_diagonal_(-float("inf"))
        return torch.topk(similarity, k=k, dim=1).indices

    left, right = neighbors(first), neighbors(second)
    rates = []
    for row in range(count):
        rates.append(len(set(left[row].tolist()).intersection(
            right[row].tolist())) / float(k))
    return float(np.mean(rates))


def prototype_geometry(features, labels):
    prototypes = []
    for class_id in sorted(torch.unique(labels).tolist()):
        selected = features[labels == class_id]
        if selected.numel() > 0:
            prototypes.append(selected.mean(dim=0))
    prototypes = F.normalize(torch.stack(prototypes).float(), dim=1)
    distances = 1.0 - prototypes.matmul(prototypes.transpose(0, 1))
    return distances


def geometry_drift(reference, current, eps=1e-12):
    numerator = torch.linalg.norm(current - reference, ord="fro")
    denominator = torch.linalg.norm(reference, ord="fro").clamp_min(eps)
    return float(numerator / denominator)


def _prediction_stats(reference, previous, current):
    labels = current["labels"]
    ref_pred = reference["logits"].argmax(dim=1)
    prev_pred = previous["logits"].argmax(dim=1)
    curr_pred = current["logits"].argmax(dim=1)
    return {
        "accuracy": float((curr_pred == labels).float().mean()),
        "prediction_flip_ref": float((curr_pred != ref_pred).float().mean()),
        "prediction_flip_step": float((curr_pred != prev_pred).float().mean()),
    }


def compute_drift_rows(reference, previous, current, metadata, k_neighbor):
    rows = []
    pred_stats = _prediction_stats(reference, previous, current)
    for layer, features in current["features"].items():
        ref_features = reference["features"][layer]
        prev_features = previous["features"][layer]
        ref_geometry = prototype_geometry(ref_features, current["labels"])
        prev_geometry = prototype_geometry(prev_features, current["labels"])
        curr_geometry = prototype_geometry(features, current["labels"])
        row = dict(metadata)
        row.update({
            "layer": layer,
            "feature_dim": int(features.size(1)),
            "cka_drift_ref": 1.0 - linear_cka(ref_features, features),
            "cka_drift_step": 1.0 - linear_cka(prev_features, features),
            "knn_ref": neighbor_preservation(ref_features, features, k_neighbor),
            "knn_step": neighbor_preservation(prev_features, features, k_neighbor),
            "prototype_drift_ref": geometry_drift(ref_geometry, curr_geometry),
            "prototype_drift_step": geometry_drift(prev_geometry, curr_geometry),
        })
        row.update(pred_stats)
        rows.append(row)
    return rows


def geometry_vector(snapshot, layer):
    geometry = prototype_geometry(snapshot["features"][layer], snapshot["labels"])
    indices = torch.triu_indices(geometry.size(0), geometry.size(1), offset=1)
    return geometry[indices[0], indices[1]].tolist()


def _neutral_rcajs_step(self, num_pruned_groups, score_dist=None, base_rate=None):
    current = num_pruned_groups / float(max(self.total_groups, 1))
    return {
        "prune_rate": 1.0,
        "current_sparsity": current,
        "delta_p": self.target_sparsity - current,
        "stability_term": 1.0,
        "kl_div": 0.0,
        "early_stop": False,
        "bit_adaptive_trigger": False,
    }


def cosine_pruning_budgets(target_groups, periods):
    cumulative = [0]
    for period in range(1, periods + 1):
        progress = period / float(periods)
        target = int(round(target_groups * 0.5 *
                           (1.0 - math.cos(math.pi * progress))))
        cumulative.append(target)
    budgets = [max(0, cumulative[i + 1] - cumulative[i])
               for i in range(periods)]
    budgets[-1] += target_groups - sum(budgets)
    return budgets


def configure_pruning_strategy(opt, strategy):
    if strategy == "rcajs":
        return
    opt.rcajs.step = types.MethodType(_neutral_rcajs_step, opt.rcajs)
    if strategy == "fixed":
        return
    if strategy == "cosine":
        opt.active_num_redundant_groups = cosine_pruning_budgets(
            opt.target_num_redundant_groups, opt.pruning_periods)
        return
    raise ValueError("Unknown pruning strategy: {}".format(strategy))


def resolve_modes(family, strategy):
    if family == "pruning":
        return strategy, "rcajs"
    if family == "bitwidth":
        return "fixed", strategy
    if family == "joint":
        mapping = {
            "fixed_hard": ("fixed", "hard"),
            "cosine_cosine": ("cosine", "cosine"),
            "rcajs_rcajs": ("rcajs", "rcajs"),
        }
        return mapping[strategy]
    raise ValueError("Unknown family: {}".format(family))


def set_bit_bound(opt, weight_bound, bit_scope):
    opt.max_bit_wt = max(int(opt.min_bit_wt), int(weight_bound))
    if bit_scope == "both":
        act_range = max(0, int(PAPER_DEFAULTS["max_bit_act"]) -
                        int(opt.min_bit_act))
        weight_range = max(1, int(PAPER_DEFAULTS["max_bit_wt"]) -
                           int(opt.min_bit_wt))
        progress = ((int(PAPER_DEFAULTS["max_bit_wt"]) - opt.max_bit_wt) /
                    float(weight_range))
        opt.max_bit_act = max(
            int(opt.min_bit_act),
            int(round(PAPER_DEFAULTS["max_bit_act"] - progress * act_range)),
        )


def apply_bit_schedule_before_epoch(opt, mode, joint_index, joint_epochs,
                                    initial_max_bit, bit_scope):
    if mode == "hard":
        set_bit_bound(opt, opt.min_bit_wt, bit_scope)
    elif mode == "cosine":
        denominator = max(1, joint_epochs - 1)
        progress = min(1.0, joint_index / float(denominator))
        fraction = 0.5 * (1.0 - math.cos(math.pi * progress))
        bound = round(initial_max_bit -
                      fraction * (initial_max_bit - opt.min_bit_wt))
        set_bit_bound(opt, bound, bit_scope)


def apply_rcajs_bit_after_epoch(opt, phi_threshold, bit_scope):
    bit_layers = refresh_bit_layers(opt)
    weight_bits = [value.get("weight") for value in bit_layers.values()
                   if isinstance(value, dict) and value.get("weight") is not None]
    if not weight_bits or opt.max_bit_wt <= opt.min_bit_wt:
        return False
    below = sum(bit < opt.max_bit_wt - 1 for bit in weight_bits)
    if below / float(len(weight_bits)) <= phi_threshold:
        return False
    set_bit_bound(opt, opt.max_bit_wt - 1, bit_scope)
    return True


def default_strategies(family):
    if family == "pruning":
        return PRUNING_STRATEGIES
    if family == "bitwidth":
        return BITWIDTH_STRATEGIES
    return JOINT_STRATEGIES


def validate_strategies(family, strategies):
    valid = set(default_strategies(family))
    unknown = [item for item in strategies if item not in valid]
    if unknown:
        raise ValueError("Invalid {} strategies: {} (valid: {})".format(
            family, unknown, sorted(valid)))


def _snapshot_metadata(args, strategy, seed, epoch, phase, opt,
                       bit_transition, train_stats, val_stats):
    sparsity, zero_groups, avg_weight_bit, min_weight_bit = collect_metrics(opt)
    return {
        "family": args.family,
        "strategy": strategy,
        "seed": seed,
        "epoch": epoch,
        "compression_epoch": max(0, epoch - args.warmup_epochs),
        "phase": phase,
        "group_sparsity": float(sparsity),
        "num_zero_groups": int(zero_groups),
        "avg_weight_bit": float(avg_weight_bit),
        "min_weight_bit": float(min_weight_bit),
        "max_weight_bit": int(opt.max_bit_wt),
        "max_activation_bit": int(opt.max_bit_act),
        "bit_transition": bool(bit_transition),
        "learning_rate": float(opt.param_groups[0]["lr"]),
        "train_loss": float(train_stats[0]) if train_stats else float("nan"),
        "train_accuracy": float(train_stats[1]) if train_stats else float("nan"),
        "val_loss": float(val_stats[0]) if val_stats else float("nan"),
        "val_accuracy": float(val_stats[1]) if val_stats else float("nan"),
    }


def run_one(args, strategy, seed):
    set_seed(seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = (Path(args.out) / args.family / args.model / strategy /
               "seed_{:03d}".format(seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    trajectory_path = run_dir / "trajectory.jsonl"
    if metrics_path.exists() and not args.overwrite:
        print("[SKIP] {} already exists".format(metrics_path))
        return
    if metrics_path.exists():
        metrics_path.unlink()
    if trajectory_path.exists():
        trajectory_path.unlink()

    if args.random_data:
        trainloader, testloader = get_random_dataloaders(
            args.batch_size, args.num_workers,
            num_train=args.random_train_samples,
            num_test=max(args.probe_samples, 64), seed=seed)
    else:
        trainloader, testloader = get_dataloaders(
            args.data_dir, args.batch_size, args.num_workers, seed=seed)
    probe_loader = materialize_probe(
        testloader, args.probe_samples, args.probe_batch_size)

    model, dummy = build_model(args.model, device)
    _, opt = build_optimizer(
        model, dummy, device, log_dir=str(run_dir / "optimizer_logs"))
    configure_training_schedule(
        opt, args.warmup_epochs, args.epochs, len(trainloader))
    pruning_mode, bit_mode = resolve_modes(args.family, strategy)
    configure_pruning_strategy(opt, pruning_mode)
    initial_max_bit = int(opt.max_bit_wt)

    total_epochs = args.warmup_epochs + args.epochs + args.cooldown_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_epochs * len(trainloader)), eta_min=args.min_lr)
    criterion = torch.nn.CrossEntropyLoss()

    config = vars(args).copy()
    config.update({
        "strategy": strategy,
        "seed": seed,
        "device": str(device),
        "pruning_mode": pruning_mode,
        "bit_mode": bit_mode,
        "steps_per_epoch": len(trainloader),
        "pruning_period_duration": opt.pruning_period_duration,
        "pruning_budgets": list(opt.active_num_redundant_groups),
    })
    _write_json(run_dir / "config.json", config)

    reference = None
    previous = None
    best_accuracy = 0.0
    layers = args.layers

    if args.warmup_epochs == 0:
        reference = extract_snapshot(model, probe_loader, device, layers)
        previous = reference
        metadata = _snapshot_metadata(
            args, strategy, seed, 0, "reference", opt, False, None, None)
        rows = compute_drift_rows(
            reference, previous, reference, metadata, args.k_neighbor)
        _append_csv(metrics_path, rows)
        _append_jsonl(trajectory_path, dict(
            metadata, layer=args.trajectory_layer,
            geometry=geometry_vector(reference, args.trajectory_layer)))

    for epoch_index in range(total_epochs):
        epoch = epoch_index + 1
        if epoch_index < args.warmup_epochs:
            phase = "warmup"
            joint_index = -1
        elif epoch_index < args.warmup_epochs + args.epochs:
            phase = "joint"
            joint_index = epoch_index - args.warmup_epochs
            apply_bit_schedule_before_epoch(
                opt, bit_mode, joint_index, args.epochs,
                initial_max_bit, args.bit_scope)
        else:
            phase = "cooldown"
            joint_index = args.epochs

        train_stats = train_one_epoch(
            model, opt, trainloader, criterion, device, epoch,
            scheduler=scheduler)
        bit_transition = False
        if phase == "joint" and bit_mode == "rcajs":
            bit_transition = apply_rcajs_bit_after_epoch(
                opt, args.phi_b, args.bit_scope)
        val_stats = evaluate(model, testloader, criterion, device)
        best_accuracy = max(best_accuracy, val_stats[1])
        print("[{} seed={}] epoch={}/{} phase={} val_acc={:.4f} "
              "max_bit={}".format(
                  strategy, seed, epoch, total_epochs, phase,
                  val_stats[1], opt.max_bit_wt), flush=True)

        should_probe = (
            epoch == args.warmup_epochs or
            phase != "warmup" and (
                (epoch - args.warmup_epochs) % args.probe_interval == 0 or
                epoch == total_epochs or bit_transition))
        if not should_probe:
            continue

        current = extract_snapshot(model, probe_loader, device, layers)
        if reference is None:
            reference = current
            previous = current
        metadata = _snapshot_metadata(
            args, strategy, seed, epoch, phase, opt, bit_transition,
            train_stats, val_stats)
        rows = compute_drift_rows(
            reference, previous, current, metadata, args.k_neighbor)
        _append_csv(metrics_path, rows)
        _append_jsonl(trajectory_path, dict(
            metadata, layer=args.trajectory_layer,
            geometry=geometry_vector(current, args.trajectory_layer)))
        previous = current

        if args.save_checkpoints and epoch == args.warmup_epochs:
            save_training_checkpoint(
                str(run_dir / "checkpoints" / "warmup.pt"), model, opt,
                scheduler, epoch, phase, val_stats[1], {})

    if args.save_checkpoints:
        save_training_checkpoint(
            str(run_dir / "checkpoints" / "final.pt"), model, opt,
            scheduler, total_epochs, "final", val_stats[1], {})

    metrics = pd.read_csv(metrics_path)
    deep = metrics[metrics["layer"] == args.trajectory_layer]
    summary = {
        "strategy": strategy,
        "seed": seed,
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": float(val_stats[1]),
        "final_group_sparsity": float(deep.iloc[-1]["group_sparsity"]),
        "final_avg_weight_bit": float(deep.iloc[-1]["avg_weight_bit"]),
        "max_cka_drift": float(deep["cka_drift_ref"].max()),
        "mean_cka_drift": float(deep["cka_drift_ref"].mean()),
        "min_knn_preservation": float(deep["knn_ref"].min()),
    }
    _write_json(run_dir / "summary.json", summary)
    print("[DONE] {}".format(run_dir))


def run_experiments(args):
    strategies = args.strategies or default_strategies(args.family)
    validate_strategies(args.family, strategies)
    if args.model != "resnet20":
        raise ValueError("The paper drift experiment currently supports resnet20 only")
    if args.trajectory_layer not in args.layers:
        raise ValueError("--trajectory-layer must also appear in --layers")
    for seed in args.seeds:
        for strategy in strategies:
            run_one(args, strategy, seed)


def _load_metric_files(root):
    # Runs are nested as family/model/strategy/seed_xxx.  Use recursive
    # discovery so plotting also works when a user points at any common
    # experiment root (or adds another grouping level later).
    files = sorted(Path(root).rglob("metrics.csv"))
    if not files:
        raise FileNotFoundError("No metrics.csv files below {}".format(root))
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _plot_line(ax, frame, metric, ylabel):
    for strategy, group in frame.groupby("strategy"):
        stats = group.groupby("compression_epoch")[metric].agg(["mean", "std"])
        x = stats.index.to_numpy(dtype=float)
        mean = stats["mean"].to_numpy(dtype=float)
        std = stats["std"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", markersize=3, linewidth=1.8,
                label=strategy)
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, alpha=0.16)
    ax.set_xlabel("Compression epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _pca_two_dimensions(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    left, singular, _right = np.linalg.svd(centered, full_matrices=False)
    if left.shape[1] == 1:
        return np.column_stack([left[:, 0] * singular[0],
                                np.zeros(left.shape[0])])
    return left[:, :2] * singular[:2]


def plot_results(args):
    root = Path(args.input)
    frame = _load_metric_files(root)
    selected = frame[frame["layer"] == args.layer].copy()
    if selected.empty:
        raise ValueError("Layer {} not found".format(args.layer))
    output = Path(args.output) if args.output else root / "figures"
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    _plot_line(axes[0, 0], selected, "cka_drift_ref", "1 - CKA to warm-up")
    _plot_line(axes[0, 1], selected, "cka_drift_step", "Consecutive 1 - CKA")
    _plot_line(axes[1, 0], selected, "knn_ref", "Top-k neighbor preservation")
    _plot_line(axes[1, 1], selected, "prototype_drift_ref",
               "Class-geometry drift")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Feature-space drift ({})".format(args.layer))
    fig.tight_layout()
    fig.savefig(output / "feature_drift_curves.pdf", bbox_inches="tight")
    fig.savefig(output / "feature_drift_curves.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for (strategy, seed, layer), group in frame.groupby(
            ["strategy", "seed", "layer"]):
        group = group.sort_values("compression_epoch")
        summary_rows.append({
            "strategy": strategy,
            "seed": seed,
            "layer": layer,
            "mean_cka_drift": group["cka_drift_ref"].mean(),
            "max_cka_drift": group["cka_drift_ref"].max(),
            "mean_step_drift": group["cka_drift_step"].mean(),
            "min_knn": group["knn_ref"].min(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "drift_summary.csv", index=False)
    heat = summary.groupby(["strategy", "layer"])["mean_cka_drift"].mean()
    heat = heat.unstack("layer")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(heat.columns)), labels=heat.columns, rotation=25,
                  ha="right")
    ax.set_yticks(range(len(heat.index)), labels=heat.index)
    ax.set_title("Mean cumulative CKA drift")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(output / "layer_drift_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output / "layer_drift_heatmap.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    trajectory_records = []
    for path in sorted(root.rglob("trajectory.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                trajectory_records.append(json.loads(line))
    if trajectory_records:
        coordinates = _pca_two_dimensions(
            [record["geometry"] for record in trajectory_records])
        for record, point in zip(trajectory_records, coordinates):
            record["pc1"], record["pc2"] = point.tolist()
        trajectory = pd.DataFrame(trajectory_records)
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        for (strategy, seed), group in trajectory.groupby(["strategy", "seed"]):
            group = group.sort_values("compression_epoch")
            ax.plot(group["pc1"], group["pc2"], marker="o", markersize=3,
                    linewidth=1.5, label="{} / seed {}".format(strategy, seed))
        ax.set_xlabel("Geometry PC1")
        ax.set_ylabel("Geometry PC2")
        ax.set_title("Class-prototype geometry trajectory")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "prototype_trajectory.pdf", bbox_inches="tight")
        fig.savefig(output / "prototype_trajectory.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
    print("[PLOT] {}".format(output))


def build_parser():
    parser = argparse.ArgumentParser(description="DACO feature-space drift experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="train strategies and measure drift")
    run.add_argument("--family", choices=["pruning", "bitwidth", "joint"],
                     required=True)
    run.add_argument("--strategies", nargs="*", default=None)
    run.add_argument("--model", default="resnet20", choices=["resnet20"])
    run.add_argument("--seeds", nargs="+", type=int, default=[1])
    run.add_argument("--data-dir", default="/root/autodl-tmp/data")
    run.add_argument("--out", default="outputs/feature_drift")
    run.add_argument("--epochs", type=int, default=200,
                     help="joint pruning-quantization epochs")
    run.add_argument("--warmup-epochs", type=int, default=10)
    run.add_argument("--cooldown-epochs", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--min-lr", type=float, default=1e-3)
    run.add_argument("--probe-samples", type=int, default=1000)
    run.add_argument("--probe-batch-size", type=int, default=256)
    run.add_argument("--probe-interval", type=int, default=1)
    run.add_argument("--k-neighbor", type=int, default=5)
    run.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    run.add_argument("--trajectory-layer", default="avg_pool2d")
    run.add_argument("--phi-b", type=float,
                     default=PAPER_DEFAULTS["phi_b_threshold"])
    run.add_argument("--bit-scope", choices=["weight", "both"], default="weight")
    run.add_argument("--deterministic", action="store_true", default=True)
    run.add_argument("--no-deterministic", action="store_false",
                     dest="deterministic")
    run.add_argument("--save-checkpoints", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--random-data", action="store_true")
    run.add_argument("--random-train-samples", type=int, default=256)

    plot = subparsers.add_parser("plot", help="aggregate metrics and create figures")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", default=None)
    plot.add_argument("--layer", default="avg_pool2d")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "run":
        run_experiments(args)
    else:
        plot_results(args)


if __name__ == "__main__":
    main()
