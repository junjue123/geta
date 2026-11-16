import sys
sys.path.append('..')
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
from sanity_check.backends.vgg7 import vgg7_bn
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

if __name__=="__main__":
    model = resnet20_cifar10()
    model = model_to_quantize_model(model)

    dummy_input = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

    import os
    save_dir = "saved_models/mygeta_VGG/origin"
    os.makedirs(save_dir, exist_ok=True)

    from torchvision.datasets import CIFAR10, CIFAR100
    import torchvision.transforms as transforms

    trainset = CIFAR10(root='../../datasets/cifar10', train=True, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    testset = CIFAR10(root='../../datasets/cifar10', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

    sparsity = 0.9
    start_projection_epoch=1
    start_pruning_epoch=5
    projection_epochs=100
    pruning_epochs=100
    optimizer = oto.mygeta(
        variant="adam",
        lr=1e-1,
        lr_quant=1e-3,
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

    pruning_ratios = generate_smooth_pruning_ratios(sparsity,pruning_epochs)

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

    # 仅保留交叉熵损失作为总损失的基础（内部由TotalLoss处理）
    criterion = torch.nn.CrossEntropyLoss()

    import os
    from tutorials.utils.utils import check_accuracy
    from mylog import log_and_print_training

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.makedirs(save_dir, exist_ok=True)

    max_epoch = 250
    model.cuda()

    milestones = [100, 150, 200]
    gamma = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )

    model_copy = model_to_quantize_model(resnet20_cifar10())
    model_copy.load_state_dict(model.state_dict())
    oto_temp = OTO(model=model_copy.cuda(), dummy_input=dummy_input.cuda())
    oto_temp.construct_subnet(out_dir='temp')
    compressed_model_size = os.stat(oto_temp.full_group_sparse_model_path).st_size / 1024
    origin_size = compressed_model_size
    target_model_size = 256  # KB

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

    for epoch in range(max_epoch):
        f_avg_val = 0.0  # 仅累加总损失
        model.train()

        for X, y in trainloader:
            X = X.cuda()
            y = y.cuda()
            y_pred = model.forward(X)

            f = criterion(y_pred, y)


            optimizer.zero_grad()
            f.backward()  # 基于总损失反向传播
            f_avg_val += f
            optimizer.step()

        # 计算epoch平均总损失
        f_avg_val /= len(trainloader)
        # 学习率调度器更新（移至epoch循环末尾，逻辑正确）
        lr_scheduler.step()

        # 验证精度
        opt_metrics = optimizer.compute_metrics()
        accuracy1, accuracy5 = check_accuracy(model, testloader)
        print(f"Epoch {epoch} | Avg Total Loss: {f_avg_val:.4f} | Acc@1: {accuracy1:.4f}")

        # 关键修改：仅传入总损失，其他损失参数设为0或删除（根据函数定义）
        # 若log_and_print_training函数不允许celoss/rcrloss/conloss为空，设为0占位
        log_and_print_training(
            save_dir=save_dir,
            epoch=epoch,
            total_loss=f_avg_val,  # 核心：传入总损失
            param_norm=opt_metrics.norm_params,
            group_sparsity=opt_metrics.group_sparsity,
            acc1=accuracy1,
            norm_import=opt_metrics.norm_important_groups,
            norm_redund=opt_metrics.norm_redundant_groups,
            num_grps_import=opt_metrics.num_important_groups,
            num_grps_redund=opt_metrics.num_redundant_groups
        )

        # 保存checkpoint（仅保存总损失）
        if (epoch + 1) % 1 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'sparsity': sparsity,
                'total_loss': f_avg_val  # 仅保存总损失
            }
            model_path = os.path.join(save_dir, f"epoch_{epoch}_sparsity_{str(sparsity)}.pt")
            torch.save(checkpoint, model_path)
            print(f"已保存模型到 {model_path}")