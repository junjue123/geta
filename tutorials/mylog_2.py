import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def log_and_print_training(save_dir, epoch, celoss, param_norm,
                           group_sparsity, acc1, norm_import, norm_redund,
                           num_grps_import, num_grps_redund,
                           log_filename="training_log.txt",
                           history_filename="training_history.txt",
                           **kwargs):
    """
    Log training progress to console, log file, and generate loss and accuracy plots
    支持动态传入损失参数（rcrloss, conloss, distill_loss等）
    """
    # 提取动态损失参数（存在则获取，不存在则设为0）
    rcrloss = kwargs.get('rcrloss', 0.0)
    conloss = kwargs.get('conloss', 0.0)
    distill_loss = kwargs.get('distill_loss', 0.0)

    # 构建控制台输出内容
    output_line = (
        f"Ep: {epoch}, celoss: {celoss:.4f}, "
        f"{f'rcrloss: {rcrloss:.4f}, ' if rcrloss != 0 else ''}"
        f"{f'conloss: {conloss:.4f}, ' if conloss != 0 else ''}"
        f"{f'distill_loss: {distill_loss:.4f}, ' if distill_loss != 0 else ''}"
        f"norm_all: {param_norm:.2f}, grp_sparsity: {group_sparsity:.2f}, "
        f"acc1: {acc1:.4f}, norm_import: {norm_import:.2f}, "
        f"norm_redund: {norm_redund:.2f}, num_grp_import: {num_grps_import}, "
        f"num_grp_redund: {num_grps_redund}\n"
    )

    # 控制台输出
    print(output_line, end='')

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 日志文件路径
    log_path = os.path.join(save_dir, log_filename)
    history_path = os.path.join(save_dir, history_filename)

    # 当前时间（用于日志文件）
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建带时间戳的日志内容
    log_line = f"[{current_time}] {output_line}"

    # 写入日志文件
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_line)

    # 准备历史数据行（包含所有可能的损失）
    history_line = f"{epoch},{celoss},{rcrloss},{conloss},{distill_loss},{param_norm},{group_sparsity},{acc1}\n"

    # 写入历史文件（包含所有损失，方便绘图时解析）
    with open(history_path, 'a', encoding='utf-8') as f:
        # 新文件时写入表头
        if os.path.getsize(history_path) == 0:
            f.write("epoch,celoss,rcrloss,conloss,distill_loss,param_norm,group_sparsity,acc1\n")
        f.write(history_line)

    # 加载历史数据
    epochs, losses_dict, param_norms, group_sparsities, acc1s = load_history(history_path)

    # 绘制并保存图表
    plot_losses(save_dir, epochs, losses_dict)  # 所有损失绘制在一张图
    plot_accuracy(save_dir, epochs, acc1s)
    plot_params(save_dir, epochs, param_norms, group_sparsities)


def load_history(history_path):
    """加载训练历史数据，返回损失字典（键为损失名称，值为列表）"""
    epochs = []
    losses_dict = {
        'celoss': [],
        'rcrloss': [],
        'conloss': [],
        'distill_loss': []
    }
    param_norms = []
    group_sparsities = []
    acc1s = []

    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        with open(history_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            header = next(f).strip().split(',')
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == len(header):
                    try:
                        epochs.append(int(parts[0]))
                        # 加载所有损失（按表头顺序）
                        losses_dict['celoss'].append(float(parts[1]))
                        losses_dict['rcrloss'].append(float(parts[2]))
                        losses_dict['conloss'].append(float(parts[3]))
                        losses_dict['distill_loss'].append(float(parts[4]))
                        # 其他参数
                        param_norms.append(float(parts[5]))
                        group_sparsities.append(float(parts[6]))
                        acc1s.append(float(parts[7]))
                    except ValueError as e:
                        print(f"警告：无法解析历史数据行。错误：{e}")

    return epochs, losses_dict, param_norms, group_sparsities, acc1s


def plot_losses(save_dir, epochs, losses_dict):
    """绘制所有启用的损失在一张图上"""
    if not epochs:
        return

    plt.figure(figsize=(10, 6))
    markers = {'celoss': 'o', 'rcrloss': 's', 'conloss': '^', 'distill_loss': 'D'}
    labels = {
        'celoss': 'Cross Entropy Loss',
        'rcrloss': 'RCR Loss',
        'conloss': 'CON Loss',
        'distill_loss': 'Distillation Loss'
    }

    # 只绘制有有效值的损失（排除全为0的情况）
    for loss_name in losses_dict:
        loss_values = losses_dict[loss_name]
        # 判断是否为有效损失（不全为0）
        if any(v != 0 for v in loss_values):
            plt.plot(epochs, loss_values, label=labels[loss_name],
                     marker=markers[loss_name], markersize=4)

    plt.title('Loss Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'training_loss_plot.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy(save_dir, epochs, acc1s):
    """绘制精度曲线"""
    if not epochs:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc1s, label='Top1 Accuracy', marker='D', markersize=4, color='green')
    plt.title('Accuracy Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    acc_plot_path = os.path.join(save_dir, 'training_accuracy_plot.png')
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_params(save_dir, epochs, param_norms, group_sparsities):
    """绘制参数范数和分组稀疏度曲线"""
    if not epochs:
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