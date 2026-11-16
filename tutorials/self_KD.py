import sys
import os
import random
import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# 导入自定义模块
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once import OTO
from myloss import TotalLoss
from tutorials.utils.utils import check_accuracy
from mylog_2 import log_and_print_training
from SKD2 import SelfDistiller


def load_history_models(
        teacher_epochs,  # 接收去重后的教师模型epoch列表
        save_dir,
        device,
        model_type="resnet20",
        sparsity=0.7
):
    """根据教师模型epoch列表加载模型，确保无重复"""
    history_models = []
    if not teacher_epochs:
        print(f"[蒸馏信息] 无有效教师模型epoch，跳过加载")
        return history_models

    # 按epoch降序加载，避免重复（列表已去重，此处仅做安全校验）
    for epoch in sorted(teacher_epochs, reverse=True):
        model_path = os.path.join(save_dir, f"main_model_epoch_{epoch}_sparsity_{sparsity}.pt")
        if not os.path.exists(model_path):
            print(f"[警告] 主模型 {model_path} 不存在，跳过epoch {epoch}")
            continue

        # 初始化模型
        if model_type == "resnet20":
            hist_model = resnet20_cifar10()
        elif model_type == "resnet56":
            hist_model = resnet56_cifar10()
        else:
            raise ValueError(f"不支持的模型类型：{model_type}")

        # 加载权重并固定参数
        hist_model = model_to_quantize_model(hist_model)
        hist_model.load_state_dict(torch.load(model_path)["model_state_dict"])
        hist_model = hist_model.to(device)
        hist_model.eval()
        for param in hist_model.parameters():
            param.requires_grad = False

        history_models.append(hist_model)
        print(f"[蒸馏信息] 已加载教师模型（epoch: {epoch}）")

    return history_models


def main(params):
    # 解析参数
    use_self_distill = params['use_self_distill']
    use_rcr_loss = params['use_rcr_loss']
    use_con_loss = params['use_con_loss']
    distill_config = params['distill_config']
    model_type = params['model_type']
    sparsity = params['sparsity']
    max_epoch = params['max_epoch']
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params['weight_decay']
    save_dir = params['save_dir']
    start_pruning_epoch = params['start_pruning_epoch']
    device = params['device']

    # 新增：解析自蒸馏结束epoch（从配置中获取）
    distill_start_epoch = distill_config["start_epoch"]
    distill_end_epoch = distill_config["end_epoch"]

    print(
        f"[配置信息] 设备：{device} | 自蒸馏：{use_self_distill} | 蒸馏区间：epoch {distill_start_epoch}-{distill_end_epoch} | "
        f"RCR损失：{use_rcr_loss} | CON损失：{use_con_loss} | 模型：{model_type} | 总epoch：{max_epoch} | 剪枝开始epoch：{start_pruning_epoch}")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"[目录信息] 模型/日志将保存至：{save_dir}")

    # 数据加载
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载CIFAR10数据集
    trainset = CIFAR10(
        root=params['data_root'],
        train=True,
        download=True,
        transform=train_transform
    )
    testset = CIFAR10(
        root=params['data_root'],
        train=False,
        download=True,
        transform=test_transform
    )

    # 数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=params['num_workers']
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=params['num_workers']
    )
    print(f"[数据信息] 训练集：{len(trainset)}样本 | 测试集：{len(testset)}样本 | 训练batch数：{len(trainloader)}")

    # 模型与优化器初始化
    if model_type == "resnet20":
        model = resnet20_cifar10()
    elif model_type == "resnet56":
        model = resnet56_cifar10()
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

    model = model_to_quantize_model(model)
    model = model.to(device)

    # 初始化OTO优化器
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model=model, dummy_input=dummy_input)

    # 配置OTO-geta优化器
    start_projection_epoch = params['start_projection_epoch']
    projection_epochs = params['projection_epochs']
    pruning_epochs = params['pruning_epochs']
    optimizer = oto.geta(
        variant="adam",
        lr=lr,
        lr_quant=lr,
        first_momentum=0.9,
        weight_decay=weight_decay,
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

    # 学习率调度器
    milestones = params['milestones']
    gamma = params['gamma']
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )

    # 损失函数
    total_loss_fn = TotalLoss(
        lambda1=params['lambda1'],
        lambda2=params['lambda2'],
        start_projection_epoch=start_projection_epoch,
        start_pruning_epoch=start_pruning_epoch,
        projection_epochs=projection_epochs,
        pruning_epochs=pruning_epochs
    )

    # 初始化蒸馏器
    distiller = None
    if use_self_distill:
        distiller = SelfDistiller(
            temperature=distill_config["temp"],
            label_loss_weight=distill_config["label_loss_weight"],
            feature_loss_weight=distill_config["feature_loss_weight"],
            weight_distill_loss_weight=distill_config["weight_distill_loss_weight"],
            mixup_alpha=distill_config["mixup_alpha"],
            use_feat_distill=distill_config["use_feat_distill"],
            use_weight_distill=distill_config["use_weight_distill"]
        )

    # 最优模型跟踪（仅记录epoch）
    best_pre_compression_acc = 0.0
    best_pre_compression_epoch = -1  # 记录最优模型的epoch，而非权重
    print(f"[最优模型跟踪] 已开启，仅在epoch < {start_pruning_epoch}时更新最优模型epoch")

    # 教师模型epoch记录文件
    teacher_epochs_file = os.path.join(save_dir, "teacher_epochs.txt")

    # 训练循环
    print(f"\n[训练开始] 共{max_epoch}个epoch，设备：{device}")
    for epoch in range(max_epoch):
        # 初始化损失记录
        total_train_loss_avg = 0.0
        celoss_avg = 0.0
        rcrloss_avg = 0.0
        conloss_avg = 0.0
        distill_loss_avg = 0.0
        batch_num = len(trainloader)

        # 核心修改：仅当epoch在[distill_start_epoch, distill_end_epoch]区间内时，才执行蒸馏逻辑
        history_models = []
        if use_self_distill and (distill_start_epoch <= epoch <= distill_end_epoch):
            # 构建教师模型epoch列表（整合+去重）
            teacher_epochs = set()  # 用set自动去重

            # 1. 添加剪枝前最优模型的epoch（若存在）
            if best_pre_compression_epoch != -1:
                teacher_epochs.add(best_pre_compression_epoch)
                print(f"[蒸馏信息] 已添加剪枝前最优模型epoch：{best_pre_compression_epoch}")

            # 2. 随机采样1个历史偏移epoch（从[0, epoch-1]中选有效epoch）
            valid_hist_epochs = list(range(epoch))  # 所有早于当前epoch的有效epoch
            if valid_hist_epochs:
                random_hist_epoch = random.choice(valid_hist_epochs)
                teacher_epochs.add(random_hist_epoch)
                print(f"[蒸馏信息] 随机采样历史偏移epoch：{random_hist_epoch}")

            # 3. 添加压缩过程中前n个最新模型的epoch
            top_n = distill_config["top_n_during_compression"]
            if top_n > 0:
                # 查找所有主模型文件，提取epoch
                all_main_files = [f for f in os.listdir(save_dir) if
                                  f.startswith("main_model_epoch_") and f.endswith(f"_sparsity_{sparsity}.pt")]
                all_main_epochs = []
                for f in all_main_files:
                    try:
                        epoch_str = f.split("main_model_epoch_")[1].split(f"_sparsity_{sparsity}")[0]
                        all_main_epochs.append(int(epoch_str))
                    except:
                        continue

                # 按epoch降序取前n个
                if all_main_epochs:
                    top_n_epochs = sorted(all_main_epochs, reverse=True)[:top_n]
                    teacher_epochs.update(top_n_epochs)
                    print(f"[蒸馏信息] 已添加前{top_n}个最新模型epoch：{top_n_epochs}")
                else:
                    print(f"[警告] 未找到主模型文件，跳过前n个模型epoch添加")

            # 4. 去重后转为列表，加载模型
            teacher_epochs = list(teacher_epochs)
            history_models = load_history_models(
                teacher_epochs=teacher_epochs,
                save_dir=save_dir,
                device=device,
                model_type=model_type,
                sparsity=sparsity
            )
            print(f"[蒸馏信息] Epoch {epoch} | 去重后有效教师模型数：{len(history_models)}")
        else:
            # 超出蒸馏区间时，打印提示并跳过蒸馏
            if use_self_distill:
                print(f"[蒸馏信息] Epoch {epoch} 超出蒸馏区间 [{distill_start_epoch}-{distill_end_epoch}]，不执行蒸馏")

        # 训练阶段
        model.train()
        lr_scheduler.step()
        for X, y in trainloader:
            X = X.to(device)
            y = y.to(device)

            # 当前模型前向传播
            current_y_pred, current_feature = model.forward(X, feature_need=True)
            celoss, rcrloss, conloss = total_loss_fn.total_loss(
                y_pred=current_y_pred,
                y=y,
                model=model,
                feature=current_feature,
                epoch=epoch,
                RCR=use_rcr_loss,
                CON=use_con_loss
            )

            # 基础损失：只累加启用的损失项
            base_loss = celoss
            if use_rcr_loss:
                base_loss += rcrloss
            if use_con_loss:
                base_loss += conloss

            # 计算蒸馏损失（仅在蒸馏区间内且有教师模型时计算）
            distill_loss = torch.tensor(0.0, device=device)
            if use_self_distill and (distill_start_epoch <= epoch <= distill_end_epoch) and len(history_models) > 0:
                distill_loss = distiller.compute_distill_loss(current_y_pred, current_feature, model, history_models, X)

            # 总损失
            total_train_loss = base_loss + distill_loss

            # 反向传播与参数更新
            optimizer.zero_grad()
            total_train_loss.backward()
            optimizer.step()

            # 累加损失
            total_train_loss_avg += total_train_loss.item()
            celoss_avg += celoss.item()
            if use_rcr_loss:
                rcrloss_avg += rcrloss.item() if isinstance(rcrloss, torch.Tensor) else rcrloss
            if use_con_loss:
                conloss_avg += conloss.item() if isinstance(conloss, torch.Tensor) else conloss
            if use_self_distill and (distill_start_epoch <= epoch <= distill_end_epoch):
                distill_loss_avg += distill_loss.item()

        # 计算平均损失
        total_train_loss_avg /= batch_num
        celoss_avg /= batch_num
        if use_rcr_loss:
            rcrloss_avg /= batch_num
        if use_con_loss:
            conloss_avg /= batch_num
        if use_self_distill and (distill_start_epoch <= epoch <= distill_end_epoch):
            distill_loss_avg /= batch_num

        # 测试阶段
        model.eval()
        with torch.no_grad():
            accuracy1, accuracy5 = check_accuracy(model, testloader)

        # 打印当前精度
        print(f"[Epoch {epoch}/{max_epoch}] 测试集Top1精度：{accuracy1:.2f}% | Top5精度：{accuracy5:.2f}%")

        # 更新最优模型epoch（而非权重）
        if start_pruning_epoch != 0 and epoch < start_pruning_epoch:
            print(
                f"[最优模型跟踪] 当前历史最高精度：{best_pre_compression_acc:.2f}%（epoch: {best_pre_compression_epoch}）")
            if accuracy1 > best_pre_compression_acc:
                best_pre_compression_acc = accuracy1
                best_pre_compression_epoch = epoch
                print(f"[最优模型更新] Epoch {epoch} 精度{accuracy1:.2f}% 超过历史最高，已更新最优模型epoch")

        # 动态构建日志参数
        log_kwargs = {
            "save_dir": save_dir,
            "epoch": epoch,
            "celoss": celoss_avg,
            "param_norm": optimizer.compute_metrics().norm_params,
            "group_sparsity": optimizer.compute_metrics().group_sparsity,
            "acc1": accuracy1,
            "norm_import": optimizer.compute_metrics().norm_important_groups,
            "norm_redund": optimizer.compute_metrics().norm_redundant_groups,
            "num_grps_import": optimizer.compute_metrics().num_important_groups,
            "num_grps_redund": optimizer.compute_metrics().num_redundant_groups
        }

        # 添加启用的损失项（蒸馏损失仅在区间内添加）
        if use_rcr_loss:
            rcrloss_val = rcrloss_avg.item() if isinstance(rcrloss_avg, torch.Tensor) else rcrloss_avg
            log_kwargs["rcrloss"] = rcrloss_val
        if use_con_loss:
            conloss_val = conloss_avg.item() if isinstance(conloss_avg, torch.Tensor) else conloss_avg
            log_kwargs["conloss"] = conloss_val
        if use_self_distill and (distill_start_epoch <= epoch <= distill_end_epoch):
            log_kwargs["distill_loss"] = distill_loss_avg

        # 日志记录
        log_and_print_training(**log_kwargs)

        # 记录可作为教师模型的主模型epoch（仅在蒸馏区间内记录）
        if use_self_distill and (distill_start_epoch <= epoch <= distill_end_epoch) and (epoch + 1) % distill_config[
            "save_interval"] == 0:
            with open(teacher_epochs_file, 'a') as f:
                f.write(f"{epoch}\n")
            print(f"[教师模型记录] Epoch {epoch} 已记录为潜在教师模型")

        # 保存主模型
        if (epoch + 1) % params['save_main_interval'] == 0:
            main_model_path = os.path.join(save_dir, f"main_model_epoch_{epoch}_sparsity_{sparsity}.pt")
            torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                            "sparsity": sparsity,
                            "best_pre_compression_acc": best_pre_compression_acc
                            }, main_model_path)
        print(f"[模型保存] 主模型已保存至：{main_model_path}")

    # 训练结束
    print(f"\n[训练完成] 所有{max_epoch}个epoch训练结束，结果已保存至：{save_dir}")
    if best_pre_compression_epoch != -1:
        print(f"[训练总结] 剪枝前最优模型（epoch: {best_pre_compression_epoch}）Top1精度：{best_pre_compression_acc:.2f}%")
    else:
        print(f"[训练总结] start_pruning_epoch=0，未跟踪剪枝前最优模型")


if __name__ == "__main__":
    # 解决Windows多进程数据加载问题
    if os.name == "nt":
        import torch.multiprocessing
        torch.multiprocessing.set_start_method("spawn", force=True)

    # 配置参数（新增distill_config的end_epoch，定义蒸馏结束epoch）
    params = {
        # 损失启用开关
        'use_rcr_loss': False,
        'use_con_loss': False,

        # 自蒸馏配置（新增end_epoch，与start_epoch构成蒸馏区间）
        'use_self_distill': True,
        'distill_config': {
            "temperature": 1,
            "label_loss_weight": 0.1,
            "feature_loss_weight": 0.1,
            "weight_distill_loss_weight": 0.1,
            "mixup_alpha": 1,
            "use_feat_distill": False,
            "use_weight_distill": False
        },

        # 模型与数据集配置
        'model_type': "resnet20",  # resnet20/resnet56
        'dataset': "cifar10",
        'data_root': '../../datasets/cifar10',
        'num_workers': 0,

        # 训练参数
        'sparsity': 0.7,
        'max_epoch': 150,
        'batch_size': 128,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'gamma': 0.5,
        'milestones': [15, 35, 90, 120],

        # 剪枝与投影参数
        'start_pruning_epoch': 30,
        'start_projection_epoch': 35,
        'projection_epochs': 50,
        'pruning_epochs': 50,

        # 损失函数参数
        'lambda1': 0.2,
        'lambda2': 0.5,

        # 保存配置
        'save_dir': "saved_models/test",
        'save_main_interval': 1,

        # 设备配置
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # 启动训练
    main(params)
