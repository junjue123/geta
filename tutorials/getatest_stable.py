import sys
sys.path.append('..')
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
from sanity_check.backends.mobilenetv1 import mobilenetv1_cifar10
from sanity_check.backends.mobilenetv2 import mobilenetv2_cifar10
from sanity_check.backends.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once import OTO
import torch

import math
import numpy as np


def generate_smooth_pruning_ratios(total_ratio: float, num_steps: int, mode: str = "fall") -> np.ndarray:
    """
    生成平滑分配的剪枝比例数组，数组元素之和为总剪枝比例，支持上升/下降模式。

    参数:
        total_ratio: 总剪枝比例（如0.7）
        num_steps: 剪枝步数（数组长度，如100）
        mode: 模式，"rise"表示比例从低到高平滑上升；"fall"表示从高到低平滑下降

    返回:
        剪枝比例数组，长度为num_steps，元素之和≈total_ratio
    """
    if num_steps <= 0:
        raise ValueError("剪枝步数必须为正整数")
    if total_ratio < 0:
        raise ValueError("总剪枝比例不能为负数")

    # 生成0到1的进度数组（0表示第一步，1表示最后一步）
    progress = np.linspace(0, 1, num_steps, endpoint=True)

    # 基于余弦曲线生成相对权重（确保平滑性）
    if mode == "rise":
        # 上升模式：权重从0平滑增长到1（曲线先慢后快）
        weights = (1 - np.cos(progress * np.pi)) / 2  # 余弦曲线，范围[0,1]
    elif mode == "fall":
        # 下降模式：权重从1平滑降低到0（曲线先快后慢）
        weights = (1 + np.cos(progress * np.pi)) / 2  # 余弦曲线反向，范围[1,0]
    else:
        raise ValueError("模式必须为'rise'（上升）或'fall'（下降）")

    # 归一化权重（确保权重之和为1，方便按比例分配总剪枝比例）
    weights_normalized = weights / np.sum(weights)

    # 计算最终剪枝比例数组（元素之和=total_ratio）
    pruning_ratios = total_ratio * weights_normalized

    return pruning_ratios

if __name__=="__main__":
    model = resnet20_cifar10()
    model = model_to_quantize_model(model)

    dummy_input = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

    # A ResNet_zig.gv.pdf will be generated to display the depandancy graph.
    import os
    save_dir = "saved_models/geta_test/test4"
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    # oto.visualize(view=False, out_dir=save_dir)

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

    # from datasets.read_MSTAR import get_mstar_loaders
    #
    # trainloader, testloader = get_mstar_loaders(root='../datasets/MSTAR', batch_size = 16, num_workers = 1)

    # optimizer = oto.hesso(
    #         variant='sgd',
    #         lr=0.1,
    #         weight_decay=1e-4,
    #         target_group_sparsity=0.1,
    #         start_pruning_step=10 * len(trainloader),
    #         pruning_periods=10,
    #         pruning_steps=10 * len(trainloader)
    #     )
    sparsity = 0.7
    start_projection_epoch=1
    start_pruning_epoch=1
    projection_epochs=50
    pruning_epochs=50
    optimizer = oto.mygeta(
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=sparsity,
        start_projection_step=start_projection_epoch * len(trainloader),
        projection_periods=25,
        projection_steps=projection_epochs * len(trainloader),
        start_pruning_step=start_pruning_epoch * len(trainloader),
        pruning_periods=25,
        pruning_steps=pruning_epochs * len(trainloader),
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
    )
    pruning_ratios = generate_smooth_pruning_ratios(0.7,pruning_epochs*2)

    from myloss import TotalLoss
    total_loss = TotalLoss(lambda1=0.2, lambda2=0.5,
    start_projection_epoch=start_projection_epoch,
    start_pruning_epoch=start_pruning_epoch,
    projection_epochs=projection_epochs,
    pruning_epochs=pruning_epochs)

    criterion = torch.nn.CrossEntropyLoss()

    import os
    from tutorials.utils.utils import check_accuracy
    from mylog import log_and_print_training

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # 创建保存模型的子目录

    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

    max_epoch = 250
    model.cuda()

    milestones = [15, 100, 130, 160, 190, 220]  # 每个里程碑是一个epoch数
    gamma = 0.5  # 每次到达里程碑时的衰减倍数

    # 创建MultiStepLR调度器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )

    # ckpt_path = "saved_models/resnet56_rcr/epoch_199_sparsity_0.7.pt"
    # oto = OTO(torch.load(ckpt_path).cuda(), dummy_input.cuda())

    for epoch in range(max_epoch):
        f_avg_val = 0.0
        celoss_avg_val = 0.0
        rcrloss_avg_val = 0.0
        conloss_avg_val = 0.0
        model.train()
        lr_scheduler.step()
        # optimizer.set_pruning_budget_this_period(1)
        if (start_pruning_epoch <= epoch) and (epoch-start_pruning_epoch <len(pruning_ratios)):
            optimizer.set_pruning_budget_this_period(pruning_ratios[epoch - start_pruning_epoch])
        for X, y in trainloader:
            X = X.cuda()
            y = y.cuda()
            y_pred = model.forward(X)
            f = criterion(y_pred, y)

            optimizer.zero_grad()
            f.backward()
            f_avg_val += f

            optimizer.step()

        opt_metrics = optimizer.compute_metrics()
        accuracy1, accuracy5 = check_accuracy(model, testloader)
        print(accuracy1)
        # f_avg_val = f_avg_val / len(trainloader)

        log_and_print_training(
            save_dir=save_dir,
            epoch=epoch,
            celoss=0,
            rcrloss=0,
            conloss=0,
            param_norm=opt_metrics.norm_params,
            group_sparsity=opt_metrics.group_sparsity,
            acc1=accuracy1,
            norm_import=opt_metrics.norm_important_groups,
            norm_redund=opt_metrics.norm_redundant_groups,
            num_grps_import=opt_metrics.num_important_groups,
            num_grps_redund=opt_metrics.num_redundant_groups
        )

        # 保存每20个epoch的模型
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'sparsity': sparsity
            }
            model_path = os.path.join(save_dir, f"epoch_{epoch}_sparsity_{str(sparsity)}.pt")
            torch.save(checkpoint, model_path)
            print(f"已保存模型到 {model_path}")