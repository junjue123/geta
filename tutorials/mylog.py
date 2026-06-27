import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def log_and_print_training(save_dir, epoch, total_loss, param_norm,
                           group_sparsity, acc1, norm_import, norm_redund,
                           num_grps_import, num_grps_redund,
                           log_filename="training_log.txt",
                           history_filename="training_history.txt"):
    """
    Log training progress to console, log file, and generate total loss and accuracy plots
    仅保留总损失的日志和可视化，移除其他细分损失
    """

    # Build log content (without timestamp, for console output)
    output_line = (
        f"Ep: {epoch}, total_loss: {total_loss:.4f}, "
        f"norm_all: {param_norm:.2f}, grp_sparsity: {group_sparsity:.2f}, "
        f"acc1: {acc1:.4f}, norm_import: {norm_import:.2f}, "
        f"norm_redund: {norm_redund:.2f}, num_grp_import: {num_grps_import}, "
        f"num_grp_redund: {num_grps_redund}\n"
    )

    # Output to console
    print(output_line, end='')

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Log file paths
    log_path = os.path.join(save_dir, log_filename)
    history_path = os.path.join(save_dir, history_filename)

    # Get current time (for log file)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build log content with timestamp (for file recording)
    log_line = f"[{current_time}] {output_line}"

    # Append to log file
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_line)

    # Prepare data line for history file
    # 仅保留：epoch, total_loss, param_norm, group_sparsity, acc1
    history_line = f"{epoch},{total_loss},{param_norm},{group_sparsity},{acc1}\n"

    # Append to history file
    with open(history_path, 'a', encoding='utf-8') as f:
        # Write header if file is new
        if os.path.getsize(history_path) == 0:
            f.write("epoch,total_loss,param_norm,group_sparsity,acc1\n")
        f.write(history_line)

    # Load historical data from text file
    epochs, total_losses, param_norms, group_sparsities, acc1s = load_history(history_path)

    # Plot and save figures
    plot_total_loss(save_dir, epochs, total_losses)  # 仅绘制总损失
    plot_accuracy(save_dir, epochs, acc1s)
    plot_params(save_dir, epochs, param_norms, group_sparsities)


def load_history(history_path):
    """Load training history from text file (仅加载总损失相关数据)"""
    epochs = []
    total_losses = []
    param_norms = []
    group_sparsities = []
    acc1s = []

    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        with open(history_path, 'r', encoding='utf-8') as f:
            # Skip header line
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 5:  # 仅保留5个字段：epoch, total_loss, param_norm, group_sparsity, acc1
                    try:
                        epochs.append(int(parts[0]))
                        total_losses.append(float(parts[1]))
                        param_norms.append(float(parts[2]))
                        group_sparsities.append(float(parts[3]))
                        acc1s.append(float(parts[4]))
                    except ValueError as e:
                        print(f"Warning: Could not parse history line. Error: {e}")

    return epochs, total_losses, param_norms, group_sparsities, acc1s


def plot_total_loss(save_dir, epochs, total_losses):
    """Plot and save total loss figure (替换原来的多损失绘图)"""
    if not epochs:  # Don't plot if no data
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, label='Total Loss', marker='o', markersize=4, color='#1f77b4')
    plt.title('Total Loss Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴仅显示整数epoch
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'training_total_loss_plot.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy(save_dir, epochs, acc1s):
    """Plot and save accuracy figures (保留，无修改)"""
    if not epochs:  # Don't plot if no data
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc1s, label='Accuracy', marker='D', markersize=4, color='#2ca02c')
    plt.title('Accuracy Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    acc_plot_path = os.path.join(save_dir, 'training_accuracy_plot.png')
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_params(save_dir, epochs, param_norms, group_sparsities):
    """Plot and save parameter norm and group sparsity figures (保留，无修改)"""
    if not epochs:  # Don't plot if no data
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Parameter Norm', color=color)
    ax1.plot(epochs, param_norms, label='Parameter Norm', color=color, marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Group Sparsity', color=color)
    ax2.plot(epochs, group_sparsities, label='Group Sparsity', color=color, marker='s', markersize=4)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Parameter Norm and Group Sparsity Changes')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    param_plot_path = os.path.join(save_dir, 'param_norm_and_sparsity_plot.png')
    plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
    plt.close()