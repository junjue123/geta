import sys

sys.path.append('..')
from sanity_check.backends.resnets_imagenet import resnet18_imagenet, resnet50_imagenet

from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once import OTO
import torch
import math
import numpy as np


def generate_smooth_pruning_ratios(total_ratio: float, num_steps: int, mode: str = "fall") -> np.ndarray:
    if num_steps <= 0:
        raise ValueError("剪枝步数必须为正整数")
    if total_ratio < 0:
        raise ValueError("总剪枝比例不能为负数")

    progress = np.linspace(0, 1, num_steps, endpoint=True)
    if mode == "rise":
        weights = (1 - np.cos(progress * np.pi)) / 2
    elif mode == "fall":
        weights = (1 + np.cos(progress * np.pi)) / 2
    else:
        raise ValueError("模式必须为'rise'（上升）或'fall'（下降）")

    weights_normalized = weights / np.sum(weights)
    pruning_ratios = total_ratio * weights_normalized
    return pruning_ratios


if __name__ == "__main__":
    # 1. 初始化ResNet18-ImageNet模型
    model = resnet18_imagenet()

    model = model_to_quantize_model(model)

    # 2. 修改dummy input尺寸为ImageNet标准的224x224
    dummy_input = torch.rand(1, 3, 224, 224)
    oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

    import os

    save_dir = "saved_models/mygeta_ResNet18/imagenet"
    os.makedirs(save_dir, exist_ok=True)

    # 3. 加载ImageNet数据集（需提前下载）
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as transforms

    # ImageNet标准数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet_root = "D:/imagenet/imagenet_data"  # 你的根路径
    train_dir = os.path.join(imagenet_root, "train")  # 训练集子文件夹
    val_dir = os.path.join(imagenet_root, "val")  # 验证集子文件夹

    # 使用ImageFolder加载，无需devkit压缩包
    trainset = ImageFolder(root=train_dir, transform=train_transform)
    testset = ImageFolder(root=val_dir, transform=val_transform)

    # 4. 调整DataLoader参数（适配ImageNet的大数据量）
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,  # 优先调大（根据显存）：32→64→128，batch越大GPU利用率越高
        shuffle=True,
        num_workers=12,  # 调大：4→8→16（建议等于CPU核心数，如16核CPU设16）
        pin_memory=True,  # 必须保留，加速GPU数据传输
        drop_last=True,
        prefetch_factor=2,  # 新增：预加载2个batch到内存，减少等待
        persistent_workers=True  # 新增：保持worker进程常驻，避免每个epoch重启进程（大幅提速）
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    # 5. 调整训练超参数（适配ImageNet）
    sparsity = 0.9
    start_projection_epoch = 1  # ImageNet训练更长，延后投影开始
    start_pruning_epoch = 2  # 延后剪枝开始
    projection_epochs = 120
    pruning_epochs = 120
    optimizer = oto.mygeta(
        variant="adam",
        lr=1e-3,  # ImageNet使用更小的学习率
        lr_quant=1e-4,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=sparsity,
        start_projection_step=start_projection_epoch * len(trainloader),
        projection_periods=100,
        projection_steps=projection_epochs * len(trainloader),
        start_pruning_step=start_pruning_epoch * len(trainloader),
        pruning_periods=100,
        pruning_steps=pruning_epochs * len(trainloader),
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
    )

    print_interval = max(1, len(trainloader) // 1000)

    pruning_ratios = generate_smooth_pruning_ratios(sparsity, pruning_epochs)

    from myloss import TotalLoss

    # 初始化总损失函数
    total_loss_fn = TotalLoss(
        lambda1=0.2,
        lambda2=0.5,
        start_projection_epoch=start_projection_epoch,
        start_pruning_epoch=start_pruning_epoch,
        projection_epochs=projection_epochs,
        pruning_epochs=pruning_epochs
    )

    # 仅保留交叉熵损失作为总损失的基础
    criterion = torch.nn.CrossEntropyLoss()

    import os
    from tutorials.utils.utils import check_accuracy
    from mylog import log_and_print_training

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.makedirs(save_dir, exist_ok=True)

    # 6. 调整训练轮数（ImageNet标准训练轮数）
    max_epoch = 150
    model.cuda()

    # 7. 调整学习率调度器（适配ImageNet）
    milestones = [60, 90, 120]  # ImageNet常用的学习率衰减节点
    gamma = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )

    # 8. 修正模型拷贝错误（使用正确的ResNet18-ImageNet）
    model_copy = model_to_quantize_model(resnet18_imagenet())  # 原代码错误使用了resnet20_cifar10
    model_copy.load_state_dict(model.state_dict())
    oto_temp = OTO(model=model_copy.cuda(), dummy_input=dummy_input.cuda())
    oto_temp.construct_subnet(out_dir='temp')
    compressed_model_size = os.stat(oto_temp.full_group_sparse_model_path).st_size / 1024
    origin_size = compressed_model_size
    target_model_size = 1024  # KB (调整目标尺寸适配ResNet18-ImageNet)

    from get_purn_rate import DynamicPruner

    pruner = DynamicPruner(
        original_size=origin_size,
        target_size=target_model_size,
        prune_rate_range=(0.002, 0.02),
        smooth_factor=1,
        grad_sensitivity=0.3,
        grad_window=3,
        pid_gains=(0.8, 0.01, 0.3),
    )

    scaler = torch.cuda.amp.GradScaler()

    # 9. 训练循环（基本逻辑不变，适配ImageNet）
    for epoch in range(max_epoch):
        f_avg_val = 0.0
        model.train()

        for batch_idx, (X, y) in enumerate(trainloader):
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # 混合精度训练（可选，加速ImageNet训练）
            with torch.cuda.amp.autocast():
                y_pred = model.forward(X)
                f = criterion(y_pred, y)

            optimizer.zero_grad()
            scaler.scale(f).backward()
            scaler.step(optimizer)
            scaler.update()

            f_avg_val += f.item()

            if (batch_idx % print_interval == 0) or (batch_idx == len(trainloader) - 1):
                # 计算进度百分比
                progress = (batch_idx + 1) / len(trainloader) * 100
                # 计算累计平均loss
                avg_loss = f_avg_val / (batch_idx + 1)
                # 极简进度打印（一行覆盖式输出）
                print(
                    f"\rEpoch {epoch} | Batch {batch_idx + 1}/{len(trainloader)} ({progress:.1f}%) | Loss: {f.item():.4f} | Avg Loss: {avg_loss:.4f}",
                    end="")

        # 计算epoch平均损失
        f_avg_val /= len(trainloader)

        # 更新学习率
        lr_scheduler.step()

        # 验证精度（ImageNet验证集较大，可每隔几个epoch验证一次）
        if epoch % 5 == 0 or epoch == max_epoch - 1:
            opt_metrics = optimizer.compute_metrics()
            accuracy1, accuracy5 = check_accuracy(model, testloader)
            print(f"Epoch {epoch} | Avg Total Loss: {f_avg_val:.4f} | Acc@1: {accuracy1:.4f} | Acc@5: {accuracy5:.4f}")

            # 记录训练日志
            log_and_print_training(
                save_dir=save_dir,
                epoch=epoch,
                total_loss=f_avg_val,
                param_norm=opt_metrics.norm_params,
                group_sparsity=opt_metrics.group_sparsity,
                acc1=accuracy1,
                norm_import=opt_metrics.norm_important_groups,
                norm_redund=opt_metrics.norm_redundant_groups,
                num_grps_import=opt_metrics.num_important_groups,
                num_grps_redund=opt_metrics.num_redundant_groups
            )
        else:
            print(f"Epoch {epoch} | Avg Total Loss: {f_avg_val:.4f}")

        # 保存checkpoint（减少保存频率，避免占用过多空间）
        if (epoch + 1) % 10 == 0 or epoch == max_epoch - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'sparsity': sparsity,
                'total_loss': f_avg_val
            }
            model_path = os.path.join(save_dir, f"epoch_{epoch}_sparsity_{str(sparsity)}.pt")
            torch.save(checkpoint, model_path)
            print(f"已保存模型到 {model_path}")