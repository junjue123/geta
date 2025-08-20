import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MaxNLocator

# 设置matplotlib中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def log_and_print_training(save_dir, epoch, celoss, rcrloss, conloss, param_norm,
                           group_sparsity, acc1, norm_import, norm_redund,
                           num_grps_import, num_grps_redund,
                           log_filename="training_log.txt",
                           history_file="training_history.npz"):
    """
    同时将训练进度输出到控制台、记录到日志文件，并生成损失和准确率图表

    参数:
        save_dir: 日志文件和图表保存的目录
        epoch: 当前epoch数
        celoss: 平均交叉熵损失
        rcrloss: 平均RCR损失
        conloss: 平均CON损失
        param_norm: 参数范数
        group_sparsity: 组稀疏度
        acc1: 准确率1
        norm_import: 重要组的范数
        norm_redund: 冗余组的范数
        num_grps_import: 重要组的数量
        num_grps_redund: 冗余组的数量
        log_filename: 日志文件名，默认为"training_log.txt"
        history_file: 训练历史数据保存文件名，默认为"training_history.npz"
    """
    # 构建日志内容（不带时间戳，用于控制台输出）
    output_line = (
        f"Ep: {epoch}, celoss: {celoss:.4f}, rcrloss: {rcrloss:.4f}, conloss: {conloss:.4f}, "
        f"norm_all: {param_norm:.2f}, grp_sparsity: {group_sparsity:.2f}, "
        f"acc1: {acc1:.4f}, norm_import: {norm_import:.2f}, "
        f"norm_redund: {norm_redund:.2f}, num_grp_import: {num_grps_import}, "
        f"num_grp_redund: {num_grps_redund}\n"
    )

    # 输出到控制台
    print(output_line, end='\n')  # 使用end=''避免重复换行

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 日志文件路径
    log_path = os.path.join(save_dir, log_filename)
    history_path = os.path.join(save_dir, history_file)

    # 获取当前时间（仅用于日志文件）
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构建带时间戳的日志内容（用于文件记录）
    log_line = f"[{current_time}] {output_line}"

    # 追加写入日志
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_line)

    # 加载历史数据或初始化
    if os.path.exists(history_path):
        history = np.load(history_path)
        epochs = np.append(history['epochs'], epoch)
        celosses = np.append(history['celosses'], celoss)
        rcrlosses = np.append(history['rcrlosses'], rcrloss)
        conlosses = np.append(history['conlosses'], conloss)
        acc1s = np.append(history['acc1s'], acc1)
        param_norms = np.append(history['param_norms'], param_norm)
        group_sparsities = np.append(history['group_sparsities'], group_sparsity)
    else:
        epochs = np.array([epoch])
        celosses = np.array([celoss])
        rcrlosses = np.array([rcrloss])
        conlosses = np.array([conloss])
        acc1s = np.array([acc1])
        param_norms = np.array([param_norm])
        group_sparsities = np.array([group_sparsity])

    # 保存更新后的历史数据
    np.savez(history_path,
             epochs=epochs,
             celosses=celosses,
             rcrlosses=rcrlosses,
             conlosses=conlosses,
             acc1s=acc1s,
             param_norms=param_norms,
             group_sparsities=group_sparsities)

    # 绘制并保存损失图表
    plot_losses(save_dir, epochs, celosses, rcrlosses, conlosses)

    # 绘制并保存准确率图表
    plot_accuracy(save_dir, epochs, acc1s)

    # 绘制并保存参数范数和组稀疏度图表
    plot_params(save_dir, epochs, param_norms, group_sparsities)


def plot_losses(save_dir, epochs, celosses, rcrlosses, conlosses):
    """绘制并保存损失图表"""
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, celosses, label='交叉熵损失', marker='o', markersize=4)
    plt.plot(epochs, rcrlosses, label='RCR损失', marker='s', markersize=4)
    plt.plot(epochs, conlosses, label='CON损失', marker='^', markersize=4)

    plt.title('训练过程中的损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置x轴为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存图表
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy(save_dir, epochs, acc1s):
    """绘制并保存准确率图表"""
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, acc1s, label='准确率', color='green', marker='D', markersize=4)

    plt.title('训练过程中的准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.ylim(0, 1.05)  # 假设准确率在0到1之间
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 设置x轴为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存图表
    plt.tight_layout()
    acc_plot_path = os.path.join(save_dir, 'accuracy_plot.png')
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_params(save_dir, epochs, param_norms, group_sparsities):
    """绘制并保存参数范数和组稀疏度图表"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 参数范数
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('参数范数', color=color)
    ax1.plot(epochs, param_norms, label='参数范数', color=color, marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color)

    # 组稀疏度（第二个y轴）
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('组稀疏度', color=color)
    ax2.plot(epochs, group_sparsities, label='组稀疏度', color=color, marker='s', markersize=4)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('参数范数和组稀疏度变化')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    # 设置x轴为整数
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    plt.tight_layout()
    param_plot_path = os.path.join(save_dir, 'param_norm_sparsity_plot.png')
    plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
