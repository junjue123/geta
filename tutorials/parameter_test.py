import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from scipy.linalg import svdvals
import seaborn as sns
from matplotlib.font_manager import fontManager

def setup_chinese_font():
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
        'Heiti TC', 'Songti SC', 'STHeiti', 'Arial Unicode MS'
    ]
    available_fonts = {f.name for f in fontManager.ttflist}
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    if selected_font:
        plt.rcParams["font.family"] = [selected_font]
        print(f"✅ Chinese font loaded: {selected_font}")
    else:
        plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
        print("⚠️ No predefined Chinese font found, using compatibility mode")
    plt.rcParams["axes.unicode_minus"] = False

setup_chinese_font()
sns.set_style("whitegrid")

from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
from only_train_once.quantization.quant_model import model_to_quantize_model

def get_top_level_modules(model: torch.nn.Module) -> List[str]:
    return [name for name, _ in model.named_children()]

def calculate_module_statistics(ckpt_path: str, device: str) -> List[Dict]:
    filename = os.path.basename(ckpt_path)
    epoch_match = re.search(r'epoch_(\d+)', filename)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    sparsity_match = re.search(r'sparsity_(\d+\.\d+)', filename)
    sparsity = float(sparsity_match.group(1)) if sparsity_match else 0.0

    try:
        model = resnet20_cifar10()
        model = model_to_quantize_model(model).to(device)
        model.eval()
        top_modules = get_top_level_modules(model)
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        model_state = ckpt_dict.get('model_state_dict') or ckpt_dict.get('model')
        if model_state is None:
            raise KeyError("Model parameters not found")
        model.load_state_dict(model_state)

        module_stats = []
        for module_name in top_modules:
            module = getattr(model, module_name)
            total_size = 0  # 替换：统计size（字节）而非count
            all_singular_values = []
            all_params = []

            for param_name, param in module.named_parameters():
                param_data = param.detach().cpu().numpy()
                # 核心修改：计算参数存储空间（字节）
                param_size = param.element_size() * param.numel()
                total_size += param_size
                all_params.extend(param_data.flatten().tolist())

                if 'weight' not in param_name or param.ndim < 2:
                    continue
                if param_data.ndim == 4:
                    out_c, in_c, k_h, k_w = param_data.shape
                    param_2d = param_data.reshape(out_c, in_c * k_h * k_w)
                else:
                    param_2d = param_data
                singular_values = svdvals(param_2d)
                all_singular_values.extend(singular_values.tolist())

            max_sv = np.max(all_singular_values) if all_singular_values else None
            avg_sv = np.mean(all_singular_values) if all_singular_values else None
            param_mean = np.mean(all_params) if all_params else None
            param_std = np.std(all_params) if all_params else None
            param_min = np.min(all_params) if all_params else None
            param_max = np.max(all_params) if all_params else None

            module_stats.append({
                "epoch": epoch,
                "sparsity": sparsity,
                "module_name": module_name,
                "total_param_size": total_size,  # 替换：存储size
                "max_singular_value": max_sv,
                "avg_singular_value": avg_sv,
                "param_mean": param_mean,
                "param_std": param_std,
                "param_min": param_min,
                "param_max": param_max,
                "param_samples": np.random.choice(all_params, min(10000, len(all_params)), replace=False).tolist()
                if all_params else None
            })

        return module_stats

    except Exception as e:
        print(f"❌ Failed to process model {filename}: {str(e)}")
        return []

def plot_module_statistics(ckpt_dir: str,
                           device: str,
                           plot_save_dir: str = "module_plots",
                           epoch_range: Optional[List[int]] = None,
                           step: int = 1) -> None:
    ckpt_files = []
    for f in os.listdir(ckpt_dir):
        if (f.endswith(".pt")
                and re.search(r'epoch_(\d+)', f)
                and re.search(r'sparsity_(\d+\.\d+)', f)):
            epoch_match = re.search(r'epoch_(\d+)', f)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            if epoch_range and len(epoch_range) == 2:
                start_epoch, end_epoch = epoch_range
                if (start_epoch <= epoch <= end_epoch) and ((epoch - start_epoch) % step == 0):
                    ckpt_files.append(os.path.join(ckpt_dir, f))
            else:
                ckpt_files.append(os.path.join(ckpt_dir, f))

    if not ckpt_files:
        range_msg = f" (range: {epoch_range[0]}-{epoch_range[1]}, step: {step})" if epoch_range else ""
        raise ValueError(f"No valid model files found in {ckpt_dir}{range_msg}")

    ckpt_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', os.path.basename(x)).group(1)))
    sample_model = resnet20_cifar10()
    top_modules = get_top_level_modules(sample_model)
    print(f"✅ Found {len(ckpt_files)} model files, will visualize the following modules: {', '.join(top_modules)}")
    if epoch_range:
        print(f"📌 Selected epoch range: {epoch_range[0]}-{epoch_range[1]}, step: {step}")

    all_stats = []
    for idx, ckpt_path in enumerate(ckpt_files, 1):
        print(f"🔄 Processing model {idx}/{len(ckpt_files)}: {os.path.basename(ckpt_path)}")
        stats = calculate_module_statistics(ckpt_path, device)
        all_stats.extend(stats)

    if not all_stats:
        print("⚠️ No valid statistical data obtained, unable to generate plots")
        return

    os.makedirs(plot_save_dir, exist_ok=True)
    modules = list(set(stat["module_name"] for stat in all_stats))
    epochs = sorted(list(set(stat["epoch"] for stat in all_stats if stat["epoch"] != -1)))

    # 1. 参数size vs epoch（替换原param_count）
    plt.figure(figsize=(12, 6))
    for module in modules:
        module_data = [s for s in all_stats if s["module_name"] == module]
        module_data.sort(key=lambda x: x["epoch"])
        epochs = [s["epoch"] for s in module_data]
        sizes = [s["total_param_size"] / 1024  # 转换为KB，可改为/1024²为MB
                  for s in module_data]
        plt.plot(epochs, sizes, marker='o', label=module, linewidth=2, markersize=6)
    plt.xlabel("Training Epoch")
    plt.ylabel("Parameter Size (KB)")
    plt.title("Parameter Size of Each Module vs. Training Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "param_size_vs_epoch.png"), dpi=300)
    plt.close()

    # 2-7. 其他图（奇异值、参数分布相关）保持不变
    # 2. 最大奇异值 vs epoch
    plt.figure(figsize=(12, 6))
    for module in modules:
        module_data = [s for s in all_stats if s["module_name"] == module and s["max_singular_value"] is not None]
        module_data.sort(key=lambda x: x["epoch"])
        epochs = [s["epoch"] for s in module_data]
        max_sv = [s["max_singular_value"] for s in module_data]
        if max_sv:
            plt.plot(epochs, max_sv, marker='s', label=module, linewidth=2, markersize=6)
    plt.xlabel("Training Epoch")
    plt.ylabel("Maximum Singular Value")
    plt.title("Maximum Singular Value of Each Module vs. Training Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "max_singular_value_vs_epoch.png"), dpi=300)
    plt.close()

    # 3. 平均奇异值 vs epoch
    plt.figure(figsize=(12, 6))
    for module in modules:
        module_data = [s for s in all_stats if s["module_name"] == module and s["avg_singular_value"] is not None]
        module_data.sort(key=lambda x: x["epoch"])
        epochs = [s["epoch"] for s in module_data]
        avg_sv = [s["avg_singular_value"] for s in module_data]
        if avg_sv:
            plt.plot(epochs, avg_sv, marker='^', label=module, linewidth=2, markersize=6)
    plt.xlabel("Training Epoch")
    plt.ylabel("Average Singular Value")
    plt.title("Average Singular Value of Each Module vs. Training Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "avg_singular_value_vs_epoch.png"), dpi=300)
    plt.close()

    # 4. 最终epoch参数size对比（替换原count）
    last_epoch = max(epochs) if epochs else 0
    last_epoch_data = [s for s in all_stats if s["epoch"] == last_epoch]
    if last_epoch_data:
        plt.figure(figsize=(10, 6))
        modules = [s["module_name"] for s in last_epoch_data]
        sizes = [s["total_param_size"] / 1024 for s in last_epoch_data]  # 单位：KB
        sns.barplot(x=modules, y=sizes)
        plt.xlabel("Module Name")
        plt.ylabel("Parameter Size (KB)")
        plt.title(f"Parameter Size Comparison of Each Module at Epoch {last_epoch}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, f"param_size_comparison_epoch_{last_epoch}.png"), dpi=300)
        plt.close()

    # 5. 参数均值 vs epoch
    plt.figure(figsize=(12, 6))
    for module in modules:
        module_data = [s for s in all_stats if s["module_name"] == module and s["param_mean"] is not None]
        module_data.sort(key=lambda x: x["epoch"])
        epochs = [s["epoch"] for s in module_data]
        means = [s["param_mean"] for s in module_data]
        if means:
            plt.plot(epochs, means, marker='o', label=module, linewidth=2, markersize=5)
    plt.xlabel("Training Epoch")
    plt.ylabel("Parameter Mean")
    plt.title("Parameter Mean of Each Module vs. Training Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "param_mean_vs_epoch.png"), dpi=300)
    plt.close()

    # 6. 参数标准差 vs epoch
    plt.figure(figsize=(12, 6))
    for module in modules:
        module_data = [s for s in all_stats if s["module_name"] == module and s["param_std"] is not None]
        module_data.sort(key=lambda x: x["epoch"])
        epochs = [s["epoch"] for s in module_data]
        stds = [s["param_std"] for s in module_data]
        if stds:
            plt.plot(epochs, stds, marker='s', label=module, linewidth=2, markersize=5)
    plt.xlabel("Training Epoch")
    plt.ylabel("Parameter Standard Deviation")
    plt.title("Parameter Standard Deviation of Each Module vs. Training Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "param_std_vs_epoch.png"), dpi=300)
    plt.close()

    # 7. 参数值域 vs epoch
    plt.figure(figsize=(12, 6))
    for module in modules:
        module_data = [s for s in all_stats if s["module_name"] == module
                       and s["param_max"] is not None                       and s["param_max"] is not None and s["param_min"] is not None]
        module_data.sort(key=lambda x: x["epoch"])
        epochs = [s["epoch"] for s in module_data]
        ranges = [s["param_max"] - s["param_min"] for s in module_data]
        if ranges:
            plt.plot(epochs, ranges, marker='^', label=module, linewidth=2, markersize=5)
    plt.xlabel("Training Epoch")
    plt.ylabel("Parameter Value Range (max - min)")
    plt.title("Parameter Value Range of Each Module vs. Training Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "param_range_vs_epoch.png"), dpi=300)
    plt.close()

    # 8. 参数分布直方图（保持不变）
    for module in modules:
        module_data = [s for s in all_stats if s["module_name"] == module and s["param_samples"] is not None]
        module_data.sort(key=lambda x: x["epoch"])
        if len(module_data) < 2:
            continue
        n_epochs = len(module_data)
        n_rows = min((n_epochs + 2) // 3, 3)
        n_cols = 3
        if n_rows * n_cols < n_epochs:
            n_cols = (n_epochs + n_rows - 1) // n_rows
        plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        for i, data in enumerate(module_data):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(data["param_samples"], bins=50, kde=True)
            plt.title(f'Epoch {data["epoch"]}\nMean: {data["param_mean"]:.4f}\nStd: {data["param_std"]:.4f}')
            plt.xlabel("Parameter Value")
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, f"param_distribution_{module}.png"), dpi=300)
        plt.close()

    print(f"✅ All plots have been saved to: {plot_save_dir}")


if __name__ == "__main__":
    # 配置参数（可根据实际情况调整）
    CKPT_DIR = "saved_models/test_2"  # 模型文件目录
    PLOT_SAVE_DIR = "module_plots"    # 图表保存目录
    DEVICE = "cpu"                    # 设备选择（'cpu' 或 'cuda'）
    EPOCH_RANGE = [0, 150]            # 要分析的epoch范围
    STEP = 5                          # 采样步长（每隔多少个epoch取一次数据）

    try:
        print(f"✅ Starting processing, using device: {DEVICE}")
        plot_module_statistics(
            ckpt_dir=CKPT_DIR,
            device=DEVICE,
            plot_save_dir=PLOT_SAVE_DIR,
            epoch_range=EPOCH_RANGE,
            step=STEP
        )
    except Exception as e:
        print(f"\n❌ Program error: {str(e)}")