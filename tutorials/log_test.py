
import numpy as np


import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def log_and_print_training(save_dir, epoch, celoss, rcrloss, conloss, param_norm,
                           group_sparsity, acc1, norm_import, norm_redund,
                           num_grps_import, num_grps_redund,
                           log_filename="training_log.txt",
                           history_filename="training_history.txt"):
    """
    Log training progress to console, log file, and generate loss and accuracy plots
    """

    # Build log content (without timestamp, for console output)
    output_line = (
        f"Ep: {epoch}, celoss: {celoss:.4f}, rcrloss: {rcrloss:.4f}, conloss: {conloss:.4f}, "
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
    # Format: epoch,celoss,rcrloss,conloss,param_norm,group_sparsity,acc1
    history_line = f"{epoch},{celoss},{rcrloss},{conloss},{param_norm},{group_sparsity},{acc1}\n"

    # Append to history file
    with open(history_path, 'a', encoding='utf-8') as f:
        # Write header if file is new
        if os.path.getsize(history_path) == 0:
            f.write("epoch,celoss,rcrloss,conloss,param_norm,group_sparsity,acc1\n")
        f.write(history_line)

    # Load historical data from text file
    epochs, celosses, rcrlosses, conlosses, param_norms, group_sparsities, acc1s = load_history(history_path)

    # Plot and save figures
    plot_losses(save_dir, epochs, celosses, rcrlosses, conlosses)
    plot_accuracy(save_dir, epochs, acc1s)
    plot_params(save_dir, epochs, param_norms, group_sparsities)


def load_history(history_path):
    """Load training history from text file"""
    epochs = []
    celosses = []
    rcrlosses = []
    conlosses = []
    param_norms = []
    group_sparsities = []
    acc1s = []

    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        with open(history_path, 'r', encoding='utf-8') as f:
            # Skip header line
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 7:  # Ensure we have all expected values
                    try:
                        epochs.append(int(parts[0]))
                        celosses.append(float(parts[1]))
                        rcrlosses.append(float(parts[2]))
                        conlosses.append(float(parts[3]))
                        param_norms.append(float(parts[4]))
                        group_sparsities.append(float(parts[5]))
                        acc1s.append(float(parts[6]))
                    except ValueError as e:
                        print(f"Warning: Could not parse history line. Error: {e}")

    return epochs, celosses, rcrlosses, conlosses, param_norms, group_sparsities, acc1s


def plot_losses(save_dir, epochs, celosses, rcrlosses, conlosses):
    """Plot and save loss figures"""
    if not epochs:  # Don't plot if no data
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, celosses, label='Cross Entropy Loss', marker='o', markersize=4)
    plt.plot(epochs, rcrlosses, label='RCR Loss', marker='s', markersize=4)
    plt.plot(epochs, conlosses, label='CON Loss', marker='^', markersize=4)
    plt.title('Loss Changes During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'training_loss_plot.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy(save_dir, epochs, acc1s):
    """Plot and save accuracy figures"""
    if not epochs:  # Don't plot if no data
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc1s, label='Accuracy', marker='D', markersize=4)
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
    """Plot and save parameter norm and group sparsity figures"""
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

import shutil
def training_logger(num_epochs=10):
    """测试训练日志记录函数"""
    # 创建临时测试目录
    test_dir = "test_training_logs"

    # 清除已有测试目录（如果存在）
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # 模拟训练过程
    print(f"开始模拟 {num_epochs} 个epochs的训练...")
    for epoch in range(1, num_epochs + 1):
        # 生成模拟数据（带一定随机性的训练指标）
        celoss = 3.0 - 0.2 * epoch + np.random.normal(0, 0.1)
        rcrloss = 1.5 - 0.1 * epoch + np.random.normal(0, 0.05)
        conloss = 0.8 - 0.05 * epoch + np.random.normal(0, 0.03)
        param_norm = 10.0 - 0.5 * epoch + np.random.normal(0, 0.2)
        group_sparsity = 0.1 + 0.05 * epoch + np.random.normal(0, 0.01)
        acc1 = 0.3 + 0.06 * epoch + np.random.normal(0, 0.02)
        norm_import = 5.0 - 0.2 * epoch + np.random.normal(0, 0.1)
        norm_redund = 2.0 - 0.1 * epoch + np.random.normal(0, 0.05)
        num_grps_import = int(10 + 2 * epoch + np.random.normal(0, 1))
        num_grps_redund = int(8 - 0.5 * epoch + np.random.normal(0, 0.5))

        # 确保数值在合理范围内
        acc1 = np.clip(acc1, 0, 1)
        group_sparsity = np.clip(group_sparsity, 0, 1)

        # 调用日志记录函数
        log_and_print_training(
            save_dir=test_dir,
            epoch=epoch,
            celoss=celoss,
            rcrloss=rcrloss,
            conloss=conloss,
            param_norm=param_norm,
            group_sparsity=group_sparsity,
            acc1=acc1,
            norm_import=norm_import,
            norm_redund=norm_redund,
            num_grps_import=num_grps_import,
            num_grps_redund=num_grps_redund
        )

    # 验证输出结果
    print("\n验证输出结果:")

    # 检查日志文件是否存在
    log_path = os.path.join(test_dir, "training_log.txt")
    if os.path.exists(log_path):
        print(f"日志文件已生成: {log_path}")
        # 打印最后几行日志
        with open(log_path, 'r') as f:
            lines = f.readlines()
            print("最后3行日志:")
            print(''.join(lines[-3:]))
    else:
        print("错误: 日志文件未生成!")

    # 检查历史数据文件是否存在
    history_path = os.path.join(test_dir, "training_history.npz")
    if os.path.exists(history_path):
        print(f"历史数据文件已生成: {history_path}")
        # 检查数据完整性
        history = np.load(history_path)
        print(f"记录的epochs数量: {len(history['epochs'])} (预期: {num_epochs})")
        print(f"记录的准确率数据点数量: {len(history['acc1s'])} (预期: {num_epochs})")
    else:
        print("错误: 历史数据文件未生成!")

    # 检查图像文件是否生成
    plot_files = [
        "training_loss_plot.png",
        "training_accuracy_plot.png",
        "param_norm_and_sparsity_plot.png"
    ]
    for plot_file in plot_files:
        plot_path = os.path.join(test_dir, plot_file)
        if os.path.exists(plot_path):
            print(f"图像文件已生成: {plot_path}")
        else:
            print(f"错误: 图像文件 {plot_file} 未生成!")

if __name__ == "__main__":
    # 运行测试，模拟10个epochs的训练
    training_logger(num_epochs=10)