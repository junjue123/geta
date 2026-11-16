
import os
import re
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D


# ----------------------------
# 公共模块：数据和模型准备
# ----------------------------

def get_test_dataloader(dataset_root='../../datasets/cifar10', batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = datasets.CIFAR10(
        root=dataset_root, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return testloader, testset.classes


def load_fixed_samples(dataloader, num_samples=1000, device='cuda'):
    fixed_inputs, fixed_labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            fixed_inputs.append(inputs)
            fixed_labels.append(targets)
            if len(fixed_inputs) * dataloader.batch_size >= num_samples:
                break
    fixed_inputs = torch.cat(fixed_inputs)[:num_samples].to(device, non_blocking=True)
    fixed_labels = torch.cat(fixed_labels)[:num_samples].cpu().numpy()
    print(f"成功提取 {len(fixed_inputs)} 个固定样本，输入形状: {fixed_inputs.shape}")
    return fixed_inputs, fixed_labels


def load_models_and_extract_features(ckpt_dir, fixed_inputs, device='cuda'):
    # 筛选并按epoch排序权重文件
    ckpt_files = []
    for f in os.listdir(ckpt_dir):
        if f.endswith('.pt') and re.search(r'epoch_(\d+)', f):
            ckpt_files.append(f)
    if not ckpt_files:
        raise ValueError(f"在 {ckpt_dir} 未找到符合格式的权重文件（需包含'epoch_数字'，后缀为.pt）")

    # 按epoch升序排序
    ckpt_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    model_paths = [os.path.join(ckpt_dir, f) for f in ckpt_files]
    print(f"找到 {len(model_paths)} 个模型文件，按epoch排序: {[os.path.basename(p) for p in model_paths]}")

    # 导入模型依赖
    from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
    from sanity_check.backends.resnet20_cifar100 import resnet20_cifar100, resnet56_cifar100
    from only_train_once.quantization.quant_model import model_to_quantize_model

    # 提取每个模型的特征和预测结果（用于复用）
    all_features = {}  # 存储展平后的特征
    all_predictions = {}  # 存储模型预测logits
    path_to_epoch = {}  # 模型路径到epoch的映射

    for path in model_paths:
        model_name = os.path.basename(path)
        epoch_match = re.search(r'epoch_(\d+)', model_name)
        epoch = int(epoch_match.group(1)) if epoch_match else None
        path_to_epoch[path] = epoch

        try:
            model = resnet20_cifar10()
            model = model_to_quantize_model(model).to(device)
            ckpt_dict = torch.load(path, map_location=device)
            model_state = ckpt_dict.get('model_state_dict') or ckpt_dict.get('model')
            if model_state is None:
                raise KeyError("权重文件中未找到'model_state_dict'或'model'键")

            model.load_state_dict(model_state)
            model.eval()
            with torch.no_grad():
                # 同时提取预测logits和特征（假设model返回(logits, features)）
                logits, features = model(fixed_inputs, feature_need=True)
                predictions = logits.cpu().numpy()  # 存储预测结果

            # 特征展平处理
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            else:
                print(f"警告：{model_name} 的特征已为二维，跳过展平（shape: {features.shape}）")

            # 存储复用数据
            all_features[path] = features.cpu().numpy()
            all_predictions[path] = predictions
            print(
                f"✅ 处理完成 {model_name}（Epoch {epoch}），特征形状: {all_features[path].shape}，预测形状: {all_predictions[path].shape}")

        except Exception as e:
            print(f"❌ 处理 {model_name} 失败: {str(e)}，跳过该模型")
            continue

    if not all_features:
        raise RuntimeError("所有模型均处理失败，无法继续")

    return all_features, all_predictions, model_paths, path_to_epoch


# ----------------------------
# 方案一：类别原型可视化（按类别拆分）
# ----------------------------

def plot_class_prototypes_per_class(all_features, model_paths, path_to_epoch,
                                    fixed_labels, class_names, save_dir,
                                    epoch_range=None):
    num_classes = len(class_names)
    all_model_count = len(model_paths)
    if num_classes != 10:
        print(f"警告：类别数为 {num_classes}（非CIFAR-10的10类），已自动调整布局")

    # 筛选指定epoch范围的模型（支持多段范围，如[[5,30], [35,60]]）
    filtered_paths = []
    if epoch_range is not None:
        # 处理多段范围：先将单范围转为列表，再遍历所有段
        epoch_ranges = epoch_range if isinstance(epoch_range, list) and isinstance(epoch_range[0], list) else [
            epoch_range]
        for er in epoch_ranges:
            start_e, end_e = er
            segment_paths = [
                p for p in model_paths
                if path_to_epoch.get(p) is not None
                   and start_e <= path_to_epoch[p] <= end_e
                   and p in all_features
            ]
            filtered_paths.extend(segment_paths)
            print(f"从段 [{start_e}, {end_e}] 筛选出 {len(segment_paths)} 个模型")

        # 去重并按epoch重新排序（避免多段范围重叠导致重复模型）
        filtered_paths = list(dict.fromkeys(filtered_paths))  # 保持插入顺序去重
        filtered_paths.sort(key=lambda x: path_to_epoch[x])
        # 提取筛选后模型的epoch列表（用于颜色条）
        epochs = [path_to_epoch[p] for p in filtered_paths]

        if not filtered_paths:
            raise ValueError(f"在所有 epoch_range {epoch_ranges} 内未找到有效模型")
        print(f"\n已从 {all_model_count} 个模型中，合并筛选出 {len(filtered_paths)} 个有效模型")
    else:
        filtered_paths = [p for p in model_paths if p in all_features]
        # 提取筛选后模型的epoch列表（用于颜色条）
        epochs = [path_to_epoch[p] for p in filtered_paths]
        print(f"\n使用所有 {len(filtered_paths)} 个已成功处理的模型")

    num_models = len(filtered_paths)
    if num_models < 1:
        raise RuntimeError("筛选后模型数量为0，无法绘制")

    # 计算每个模型的类别原型
    all_prototypes = []
    for path in filtered_paths:
        feats = all_features[path]
        prototypes = []
        for cls in range(num_classes):
            mask = fixed_labels == cls
            cls_feats = feats[mask]
            if len(cls_feats) == 0:
                print(f"警告：类别 {cls}（{class_names[cls]}）无样本，补零")
                prototypes.append(np.zeros(feats.shape[1]))
                continue
            prototypes.append(np.mean(cls_feats, axis=0))
        all_prototypes.append(np.array(prototypes))

    # 统一T-SNE降维
    combined_prototypes = np.vstack(all_prototypes)
    perplexity = min(10, combined_prototypes.shape[0] // 10)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    print(f"正在进行T-SNE降维（样本数: {combined_prototypes.shape[0]}, perplexity: {perplexity}）...")
    prototypes_tsne = tsne.fit_transform(combined_prototypes)

    # 拆分降维结果
    prototypes_per_model = [
        prototypes_tsne[i * num_classes: (i + 1) * num_classes]
        for i in range(num_models)
    ]

    # 自动计算子图布局（行数和列数）
    # 基于类别数选择合适的行列比例（接近黄金比例1.618）
    cols = int(np.ceil(np.sqrt(num_classes * 1.618)))  # 列数向上取整
    rows = int(np.ceil(num_classes / cols))  # 行数根据列数计算

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # 动态调整图大小
    # 处理单行列情况，确保axes是可迭代的
    if num_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = cm.rainbow(np.linspace(0, 1, num_models))
    marker = 'o'

    for cls_idx in range(num_classes):
        ax = axes[cls_idx]
        cls_name = class_names[cls_idx]

        # 绘制原型点
        for model_idx in range(num_models):
            x, y = prototypes_per_model[model_idx][cls_idx]
            current_epoch = path_to_epoch[filtered_paths[model_idx]]
            ax.scatter(
                x, y,
                color=colors[model_idx],
                marker=marker,
                s=100,
                edgecolor='k',
                linewidth=0.5,
                alpha=0.9,
                # 仅在第一个子图显示图例（避免重复）
                label=f"Epoch {current_epoch}" if (model_idx == 0 and cls_idx == 0) else ""
            )

        # 绘制迁移轨迹
        trajectory = [prototypes_per_model[m][cls_idx] for m in range(num_models)]
        x, y = zip(*trajectory)
        ax.plot(x, y, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

        # 子图配置
        ax.set_title(f'Class: {cls_name}', fontsize=11, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # 仅在第一个子图显示图例
        if cls_idx == 0:
            ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.8)

    # 隐藏未使用的子图（当类别数小于行列乘积时）
    for idx in range(num_classes, len(axes)):
        axes[idx].axis('off')

    # 整体配置：标题显示多段范围
    if epoch_range is not None:
        epoch_str = " & ".join(
            [f"{s}~{e}" for s, e in (epoch_ranges if isinstance(epoch_range[0], list) else [epoch_range])])
        total_title = f'Evolution of Class Prototypes (Epoch Ranges: {epoch_str})'
    else:
        total_title = 'Evolution of Class Prototypes (All Epochs)'
    fig.suptitle(total_title, fontsize=16, y=0.98)

    # 添加横向颜色条（绑定epoch与颜色）
    cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.02])  # 位置：[左, 下, 宽, 高]
    sm = cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=min(epochs), vmax=max(epochs)))
    sm.set_array([])  # 仅用于映射颜色，无需实际数据
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Training Epoch', fontsize=10, labelpad=8)  # 颜色条标签
    cbar.ax.tick_params(labelsize=8)  # 刻度字体大小

    # 调整布局，为颜色条预留底部空间
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # [左, 下, 右, 上]，底部预留0.05

    # 保存图片：文件名包含多段范围
    os.makedirs(save_dir, exist_ok=True)
    if epoch_range is not None:
        range_str = "_".join([f"{s}_{e}" for s, e in epoch_ranges])
        save_name = f'class_prototypes_per_class_epoch_{range_str}.png'
    else:
        save_name = 'class_prototypes_per_class_all_epochs.png'
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 类别拆分原型图已保存至: {save_path}")
    plt.close()

# ----------------------------
# 方案二：类别统计量趋势图
# ----------------------------

def plot_class_statistics(all_features, model_paths, path_to_epoch,
                          fixed_labels, class_names, save_dir,
                          epoch_range=None):
    num_classes = len(class_names)
    all_model_count = len(model_paths)

    # 筛选指定epoch范围的模型（支持多段范围，如[[5,30], [35,60]]）
    filtered_paths = []
    if epoch_range is not None:
        # 处理多段范围：先将单范围转为列表，再遍历所有段
        epoch_ranges = epoch_range if isinstance(epoch_range, list) and isinstance(epoch_range[0], list) else [
            epoch_range]
        for er in epoch_ranges:
            start_e, end_e = er
            segment_paths = [
                p for p in model_paths
                if path_to_epoch.get(p) is not None
                   and start_e <= path_to_epoch[p] <= end_e
                   and p in all_features
            ]
            filtered_paths.extend(segment_paths)
            print(f"从段 [{start_e}, {end_e}] 筛选出 {len(segment_paths)} 个模型")

        # 去重并按epoch重新排序（避免多段范围重叠导致重复模型）
        filtered_paths = list(dict.fromkeys(filtered_paths))  # 保持插入顺序去重
        filtered_paths.sort(key=lambda x: path_to_epoch[x])

        if not filtered_paths:
            raise ValueError(f"在所有 epoch_range {epoch_ranges} 内未找到有效模型")
        print(f"\n已从 {all_model_count} 个模型中，合并筛选出 {len(filtered_paths)} 个有效模型")
    else:
        filtered_paths = [p for p in model_paths if p in all_features]
        print(f"\n使用所有 {len(filtered_paths)} 个已成功处理的模型")

    num_models = len(filtered_paths)
    if num_models < 2:
        print("警告：筛选后模型数量不足2个，无法绘制趋势图")
        return

    # 提取epoch并计算统计量
    epochs = [path_to_epoch[p] for p in filtered_paths]
    epochs = np.array(epochs)

    # 计算类别中心和内方差
    model_centers = []
    model_intra_var = []
    for path in filtered_paths:
        feats = all_features[path]
        centers = []
        intra_var = []
        for cls in range(num_classes):
            mask = fixed_labels == cls
            cls_feats = feats[mask]
            if len(cls_feats) == 0:
                print(f"警告：类别 {cls} 无样本，内方差设为0")
                centers.append(np.zeros(feats.shape[1]))
                intra_var.append(0.0)
                continue
            center = np.mean(cls_feats, axis=0)
            centers.append(center)
            dist_to_center = np.linalg.norm(cls_feats - center, axis=1)
            intra_var.append(np.mean(dist_to_center))
        model_centers.append(np.array(centers))
        model_intra_var.append(np.array(intra_var))
    model_intra_var = np.array(model_intra_var)

    # 计算类间平均距离
    model_inter_dist = []
    for centers in model_centers:
        dist_matrix = cdist(centers, centers, metric='euclidean')
        upper_triangle = dist_matrix[np.triu_indices(num_classes, k=1)]
        model_inter_dist.append(np.mean(upper_triangle))
    model_inter_dist = np.array(model_inter_dist)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1：类别内方差
    for cls in range(num_classes):
        ax1.plot(
            epochs, model_intra_var[:, cls],
            'o-', linewidth=2, markersize=6, alpha=0.8, label=class_names[cls]
        )
    if epoch_range is not None:
        epoch_str = " & ".join([f"{s}~{e}" for s, e in epoch_ranges])
        ax1.set_title(f'Intra-class Variance (Epoch Ranges: {epoch_str})', fontsize=12, pad=15)
    else:
        ax1.set_title('Intra-class Variance (All Epochs)', fontsize=12, pad=15)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Mean L2 Distance to Class Center', fontsize=10)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(epochs.min() - 1, epochs.max() + 1)

    # 子图2：类别间距离
    ax2.plot(
        epochs, model_inter_dist,
        'o-', color='darkred', linewidth=2.5, markersize=8, alpha=0.9
    )
    if epoch_range is not None:
        ax2.set_title(f'Inter-class Average Distance (Epoch Ranges: {epoch_str})', fontsize=12, pad=15)
    else:
        ax2.set_title('Inter-class Average Distance (All Epochs)', fontsize=12, pad=15)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Mean Euclidean Distance Between Class Centers', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(epochs.min() - 1, epochs.max() + 1)
    for x, y in zip(epochs, model_inter_dist):
        ax2.annotate(f'{y:.3f}', (x, y), xytext=(0, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    # 保存图片：文件名包含多段范围
    if epoch_range is not None:
        range_str = "_".join([f"{s}_{e}" for s, e in epoch_ranges])
        save_name = f'class_statistics_epoch_{range_str}.png'
    else:
        save_name = 'class_statistics_all_epochs.png'
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 统计量趋势图已保存至: {save_path}")
    plt.close()


# ----------------------------
# 新增：震荡量化指标计算工具函数
# ----------------------------

def calculate_bhattacharyya_distance(feat1, feat2):
    """计算两个特征分布的巴氏距离（值越大，分布差异越大，震荡越剧烈）"""
    mean1, var1 = np.mean(feat1, axis=0), np.var(feat1, axis=0)
    mean2, var2 = np.mean(feat2, axis=0), np.var(feat2, axis=0)
    var1 = np.clip(var1, 1e-6, None)  # 避免方差为0
    var2 = np.clip(var2, 1e-6, None)
    term1 = 0.25 * np.sum(((mean1 - mean2) ** 2) / (var1 + var2))
    term2 = 0.5 * np.log(np.prod((var1 + var2) / (2 * np.sqrt(var1 * var2))))
    return term1 + term2


def calculate_neighbor_preservation_rate(feat_prev, feat_curr, k=5):
    """计算近邻保持率（值越小，样本相对位置变化越大，震荡越剧烈）"""
    sim_prev = cosine_similarity(feat_prev)  # 余弦相似度（越大越近）
    sim_curr = cosine_similarity(feat_curr)

    def get_top_k_neighbors(sim_matrix, k):
        n = sim_matrix.shape[0]
        neighbors = []
        for i in range(n):
            top_idx = np.argsort(sim_matrix[i])[::-1][1:k + 1]  # 排除自身
            neighbors.append(set(top_idx))
        return neighbors

    neighbors_prev = get_top_k_neighbors(sim_prev, k)
    neighbors_curr = get_top_k_neighbors(sim_curr, k)
    preservation_rates = [
        len(n_prev & n_curr) / k for n_prev, n_curr in zip(neighbors_prev, neighbors_curr)
    ]
    return np.mean(preservation_rates)


def calculate_pca_variance_fluctuation(feat_prev, feat_curr, top_k=3):
    """计算PCA主成分方差贡献率波动（值越大，特征结构变化越剧烈）"""
    pca_prev = PCA(n_components=top_k)
    pca_curr = PCA(n_components=top_k)
    pca_prev.fit(feat_prev)
    pca_curr.fit(feat_curr)
    var_ratio_diff = np.abs(pca_prev.explained_variance_ratio_ - pca_curr.explained_variance_ratio_)
    return np.sum(var_ratio_diff)


# ----------------------------
# 新增：方案三：震荡量化指标可视化
# ----------------------------

def plot_oscillation_metrics(all_features, all_predictions, model_paths, path_to_epoch,
                             fixed_labels, save_dir, epoch_range=None, k_neighbor=5, top_k_pca=3):
    """绘制震荡量化指标趋势图（复用特征和预测结果）"""
    # 筛选模型
    if epoch_range is not None:
        start_e, end_e = epoch_range
        filtered_paths = [
            p for p in model_paths
            if path_to_epoch.get(p) is not None
               and start_e <= path_to_epoch[p] <= end_e
               and p in all_features and p in all_predictions
        ]
        if not filtered_paths:
            raise ValueError(f"在 epoch_range [{start_e}, {end_e}] 内未找到有效模型")
        print(f"\n已筛选出 {len(filtered_paths)} 个模型用于震荡指标计算")
    else:
        filtered_paths = [p for p in model_paths if p in all_features and p in all_predictions]
        if len(filtered_paths) < 2:
            raise RuntimeError("模型数量不足2个，无法计算相邻模型的震荡指标")

    filtered_paths.sort(key=lambda x: path_to_epoch[x])
    epochs = [path_to_epoch[p] for p in filtered_paths]
    num_models = len(filtered_paths)

    # 计算震荡指标
    metrics = {
        'bhattacharyya_distance': [],
        'neighbor_preservation_rate': [],
        'pca_variance_fluctuation': [],
        'accuracy_fluctuation': []
    }

    # 初始准确率（用于波动计算）
    init_pred = np.argmax(all_predictions[filtered_paths[0]], axis=1)
    init_acc = np.mean(init_pred == fixed_labels)

    for i in range(1, num_models):
        prev_path = filtered_paths[i - 1]
        curr_path = filtered_paths[i]
        prev_feat = all_features[prev_path]
        curr_feat = all_features[curr_path]
        prev_pred = np.argmax(all_predictions[prev_path], axis=1)
        curr_pred = np.argmax(all_predictions[curr_path], axis=1)

        # 计算指标
        metrics['bhattacharyya_distance'].append(calculate_bhattacharyya_distance(prev_feat, curr_feat))
        metrics['neighbor_preservation_rate'].append(
            calculate_neighbor_preservation_rate(prev_feat, curr_feat, k=k_neighbor))
        metrics['pca_variance_fluctuation'].append(
            calculate_pca_variance_fluctuation(prev_feat, curr_feat, top_k=top_k_pca))
        curr_acc = np.mean(curr_pred == fixed_labels)
        metrics['accuracy_fluctuation'].append(np.abs(curr_acc - init_acc))

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 子图1：巴氏距离
    axes[0].plot(epochs[1:], metrics['bhattacharyya_distance'], 'o-', color='darkblue', linewidth=2, markersize=6)
    axes[0].set_title('Bhattacharyya Distance (Feature Distribution Difference)', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Distance (Larger = More Oscillation)')
    axes[0].grid(alpha=0.3)

    # 子图2：近邻保持率
    axes[1].plot(epochs[1:], metrics['neighbor_preservation_rate'], 'o-', color='darkgreen', linewidth=2, markersize=6)
    axes[1].set_title(f'Neighbor Preservation Rate (Top-{k_neighbor})', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Rate (Smaller = More Oscillation)')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    # 子图3：PCA主成分波动
    axes[2].plot(epochs[1:], metrics['pca_variance_fluctuation'], 'o-', color='darkorange', linewidth=2, markersize=6)
    axes[2].set_title(f'PCA Variance Fluctuation (Top-{top_k_pca} Components)', fontsize=12)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Fluctuation (Larger = More Oscillation)')
    axes[2].grid(alpha=0.3)

    # 子图4：分类准确率波动
    axes[3].plot(epochs[1:], metrics['accuracy_fluctuation'], 'o-', color='darkred', linewidth=2, markersize=6)
    axes[3].set_title('Classification Accuracy Fluctuation', fontsize=12)
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Absolute Accuracy Difference (Larger = More Oscillation)')
    axes[3].grid(alpha=0.3)

    # 整体配置
    if epoch_range:
        fig.suptitle(f'Oscillation Metrics (Epoch {start_e}~{end_e})', fontsize=16, y=0.98)
    else:
        fig.suptitle('Oscillation Metrics (All Epochs)', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'oscillation_metrics_epoch_{start_e}_{end_e}.png' if epoch_range else 'oscillation_metrics_all_epochs.png'
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 震荡指标图已保存至: {save_path}")
    plt.close()

    return metrics, epochs


# ----------------------------
# 主函数：调用所有方案
# ----------------------------

if __name__ == '__main__':
    # 配置参数（用户需根据实际情况修改）
    CKPT_DIR = "saved_models/test_2"  # 模型权重文件夹路径
    DATASET_ROOT = "../../datasets/cifar10"  # 数据集路径
    NUM_SAMPLES = 1000  # 固定样本数量
    EPOCH_RANGE = [0,100]  # 筛选的epoch范围（None表示使用所有）
    # EPOCH_RANGE = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")
    print(f"当前配置的epoch范围: {EPOCH_RANGE}")

    try:
        # 步骤1：加载测试集和固定样本
        print("\n=== 步骤1：加载测试集和固定样本 ===")
        testloader, class_names = get_test_dataloader(DATASET_ROOT)
        fixed_inputs, fixed_labels = load_fixed_samples(testloader, NUM_SAMPLES, device)

        # 步骤2：加载模型并提取特征与预测（复用核心）
        print("\n=== 步骤2：加载模型并提取特征与预测 ===")
        all_features, all_predictions, model_paths, path_to_epoch = load_models_and_extract_features(CKPT_DIR,
                                                                                                     fixed_inputs,
                                                                                                     device)

        # 步骤3：方案一：类别拆分原型可视化
        print("\n=== 步骤3：运行方案一（类别拆分原型可视化） ===")
        plot_class_prototypes_per_class(
            all_features, model_paths, path_to_epoch,
            fixed_labels, class_names, CKPT_DIR,
            epoch_range=EPOCH_RANGE
        )

        # 步骤4：方案二：统计量趋势图
        print("\n=== 步骤4：运行方案二（统计量趋势图） ===")
        plot_class_statistics(
            all_features, model_paths, path_to_epoch,
            fixed_labels, class_names, CKPT_DIR,
            epoch_range=EPOCH_RANGE
        )

        # 步骤5：方案三：震荡量化指标可视化（复用特征和预测）
        print("\n=== 步骤5：运行方案三（震荡量化指标可视化） ===")
        plot_oscillation_metrics(
            all_features, all_predictions, model_paths, path_to_epoch,
            fixed_labels, CKPT_DIR,
            epoch_range=EPOCH_RANGE,
            k_neighbor=5,  # 近邻数量
            top_k_pca=3  # PCA主成分数量
        )

        print("\n✅ 所有方案运行完成！")

    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")
        raise