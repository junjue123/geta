import sys
import os
import numpy as np
import torch
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax, softmax
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# 导入自定义模块
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10, resnet56_cifar10
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once import OTO
from myloss import TotalLoss
from tutorials.utils.utils import check_accuracy
from mylog_2 import log_and_print_training
# 导入新的蒸馏器类
from SKD import SelfDistiller


def load_history_models(
        history_epochs,
        save_dir,
        device,
        model_type="resnet20",
        pre_compression_best_model=None,
        top_n_during_compression=None,
        sparsity=0.7
):
    """加载历史模型（从主模型中读取）"""
    history_models = []

    # 加载压缩前最优模型
    if pre_compression_best_model is not None:
        if model_type == "resnet20":
            best_model = resnet20_cifar10()
        elif model_type == "resnet56":
            best_model = resnet56_cifar10()
        else:
            raise ValueError(f"不支持的模型类型：{model_type}")

        best_model = model_to_quantize_model(best_model)
        best_model.load_state_dict(pre_compression_best_model)
        best_model = best_model.to(device)
        best_model.eval()
        for param in best_model.parameters():
            param.requires_grad = False

        history_models.append(best_model)
        print(f"[蒸馏信息] 已加载压缩前最优模型（剪枝前精度最高）")

    # 加载压缩过程前n个模型（从主模型中读取）
    if top_n_during_compression and top_n_during_compression > 0:
        # 查找所有主模型文件
        all_main_files = [f for f in os.listdir(save_dir) if
                          f.startswith("main_model_epoch_") and f.endswith(f"_sparsity_{sparsity}.pt")]
        if not all_main_files:
            print(f"[警告] 未找到主模型文件，跳过前n个模型加载")
        else:
            all_main_epochs = []
            for f in all_main_files:
                try:
                    # 从主模型文件名提取epoch
                    epoch_str = f.split("main_model_epoch_")[1].split(f"_sparsity_{sparsity}")[0]
                    epoch = int(epoch_str)
                    all_main_epochs.append((epoch, f))
                except:
                    continue

            # 按epoch降序排序，取前n个
            all_main_epochs.sort(key=lambda x: x[0], reverse=True)
            selected_main = all_main_epochs[:top_n_during_compression]
            selected_epochs = [e for e, f in selected_main]
            print(f"[蒸馏信息] 已选择压缩过程前{top_n_during_compression}个主模型，对应的epoch：{selected_epochs}")

            # 加载选中的模型
            for epoch, filename in selected_main:
                model_path = os.path.join(save_dir, filename)
                if model_type == "resnet20":
                    hist_model = resnet20_cifar10()
                elif model_type == "resnet56":
                    hist_model = resnet56_cifar10()
                else:
                    raise ValueError(f"不支持的模型类型：{model_type}")

                hist_model = model_to_quantize_model(hist_model)
                # 从主模型中加载模型权重
                hist_model.load_state_dict(torch.load(model_path)["model_state_dict"])
                hist_model = hist_model.to(device)
                hist_model.eval()
                for param in hist_model.parameters():
                    param.requires_grad = False

                history_models.append(hist_model)

    # 加载原偏移逻辑的模型（从主模型中读取）
    for hist_epoch in history_epochs:
        model_path = os.path.join(save_dir, f"main_model_epoch_{hist_epoch}_sparsity_{sparsity}.pt")
        if not os.path.exists(model_path):
            print(f"[警告] 主模型 {model_path} 不存在，跳过该epoch")
            continue

        if model_type == "resnet20":
            hist_model = resnet20_cifar10()
        elif model_type == "resnet56":
            hist_model = resnet56_cifar10()
        else:
            raise ValueError(f"不支持的模型类型：{model_type}")

        hist_model = model_to_quantize_model(hist_model)
        hist_model.load_state_dict(torch.load(model_path)["model_state_dict"])
        hist_model = hist_model.to(device)
        hist_model.eval()
        for param in hist_model.parameters():
            param.requires_grad = False

        history_models.append(hist_model)

    # 去重
    history_models = list(set(history_models))
    return history_models


def main(params):
    # 解析参数
    use_self_distill = params['use_self_distill']
    use_rcr_loss = params['use_rcr_loss']  # RCR损失开关
    use_con_loss = params['use_con_loss']  # CON损失开关
    distill_config = params['distill_config']
    model_type = params['model_type']
    dataset = params['dataset']
    sparsity = params['sparsity']
    max_epoch = params['max_epoch']
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params['weight_decay']
    save_dir = params['save_dir']
    start_pruning_epoch = params['start_pruning_epoch']
    device = params['device']

    print(
        f"[配置信息] 设备：{device} | 自蒸馏：{use_self_distill} | RCR损失：{use_rcr_loss} | CON损失：{use_con_loss} | "
        f"模型：{model_type} | 总epoch：{max_epoch} | 剪枝开始epoch：{start_pruning_epoch}")

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
            loss_weight=distill_config["weight"]
        )

    # 最优模型跟踪初始化
    best_pre_compression_acc = 0.0
    best_pre_compression_model = None
    print(f"[最优模型跟踪] 已开启，仅在epoch < {start_pruning_epoch}时更新最优模型（start_pruning_epoch=0则不更新）")

    # 教师模型epoch记录文件（记录可作为教师模型的主模型epoch）
    teacher_epochs_file = os.path.join(save_dir, "teacher_epochs.txt")

    # 训练循环
    print(f"\n[训练开始] 共{max_epoch}个epoch，设备：{device}")
    for epoch in range(max_epoch):
        # 初始化损失记录
        total_train_loss_avg = 0.0
        celoss_avg = 0.0
        rcrloss_avg = 0.0  # 仅当use_rcr_loss=True时有效
        conloss_avg = 0.0  # 仅当use_con_loss=True时有效
        distill_loss_avg = 0.0  # 仅当use_self_distill=True时有效
        batch_num = len(trainloader)

        # 加载历史模型（从主模型中读取）
        history_models = []
        if use_self_distill and epoch >= distill_config["start_epoch"]:
            # 原偏移逻辑的历史epoch
            history_epochs = [epoch - offset for offset in distill_config["history_epochs_offset"]]
            history_epochs = [e for e in history_epochs if e >= 0]

            # 调用函数加载模型（从主模型中读取）
            history_models = load_history_models(
                history_epochs=history_epochs,
                save_dir=save_dir,
                device=device,
                model_type=model_type,
                pre_compression_best_model=best_pre_compression_model,
                top_n_during_compression=distill_config["top_n_during_compression"],
                sparsity=sparsity  # 传入稀疏度用于匹配主模型文件名
            )
            print(f"[蒸馏信息] Epoch {epoch} | 总有效参考模型数：{len(history_models)}")

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
                RCR=use_rcr_loss,  # 由开关控制是否计算RCR损失
                CON=use_con_loss  # 由开关控制是否计算CON损失
            )

            # 基础损失：只累加启用的损失项
            base_loss = celoss
            if use_rcr_loss:
                base_loss += rcrloss
            if use_con_loss:
                base_loss += conloss

            # 计算蒸馏损失
            distill_loss = torch.tensor(0.0, device=device)
            if use_self_distill and len(history_models) > 0:
                distill_loss = distiller.compute_distill_loss(current_y_pred, current_feature, model, history_models, X)

            # 总损失
            total_train_loss = base_loss + distill_loss

            # 反向传播与参数更新
            optimizer.zero_grad()
            total_train_loss.backward()
            optimizer.step()

            # 累加损失（仅累加启用的损失项）
            total_train_loss_avg += total_train_loss.item()
            celoss_avg += celoss.item()
            if use_rcr_loss:
                rcrloss_avg += rcrloss.item() if isinstance(rcrloss, torch.Tensor) else rcrloss
            if use_con_loss:
                conloss_avg += conloss.item() if isinstance(conloss, torch.Tensor) else conloss
            if use_self_distill:
                distill_loss_avg += distill_loss.item()

        # 计算平均损失
        total_train_loss_avg /= batch_num
        celoss_avg /= batch_num
        if use_rcr_loss:
            rcrloss_avg /= batch_num
        if use_con_loss:
            conloss_avg /= batch_num
        if use_self_distill:
            distill_loss_avg /= batch_num

        # 测试阶段
        model.eval()
        with torch.no_grad():
            accuracy1, accuracy5 = check_accuracy(model, testloader)

        # 先打印当前精度
        print(f"[Epoch {epoch}/{max_epoch}] 测试集Top1精度：{accuracy1:.2f}% | Top5精度：{accuracy5:.2f}%")

        # 更新剪枝前最优模型
        if start_pruning_epoch != 0 and epoch < start_pruning_epoch:
            # 显示历史最优精度
            print(f"[最优模型跟踪] 当前历史最高精度：{best_pre_compression_acc:.2f}%")

            if accuracy1 > best_pre_compression_acc:
                best_pre_compression_acc = accuracy1
                best_pre_compression_model = {k: v.clone() for k, v in model.state_dict().items()}
                print(
                    f"[最优模型更新] Epoch {epoch} 精度{accuracy1:.2f}% 超过历史最高，已更新最优模型")

        # 动态构建日志参数（只传入启用的损失项）
        log_kwargs = {
            "save_dir": save_dir,
            "epoch": epoch,
            "celoss": celoss_avg,  # 交叉熵损失默认启用
            "param_norm": optimizer.compute_metrics().norm_params,
            "group_sparsity": optimizer.compute_metrics().group_sparsity,
            "acc1": accuracy1,
            "norm_import": optimizer.compute_metrics().norm_important_groups,
            "norm_redund": optimizer.compute_metrics().norm_redundant_groups,
            "num_grps_import": optimizer.compute_metrics().num_important_groups,
            "num_grps_redund": optimizer.compute_metrics().num_redundant_groups
        }

        # 仅添加启用的损失项
        if use_rcr_loss:
            rcrloss_val = rcrloss_avg.item() if isinstance(rcrloss_avg, torch.Tensor) else rcrloss_avg
            log_kwargs["rcrloss"] = rcrloss_val
        if use_con_loss:
            conloss_val = conloss_avg.item() if isinstance(conloss_avg, torch.Tensor) else conloss_avg
            log_kwargs["conloss"] = conloss_val
        if use_self_distill:
            log_kwargs["distill_loss"] = distill_loss_avg

        # 日志记录
        log_and_print_training(**log_kwargs)

        # 记录可作为教师模型的主模型epoch（不再单独保存历史模型）
        if use_self_distill and (epoch + 1) % distill_config["save_interval"] == 0:
            # 仅记录epoch，模型权重从主模型中读取
            with open(teacher_epochs_file, 'a') as f:
                f.write(f"{epoch}\n")
            print(f"[教师模型记录] Epoch {epoch} 已记录为潜在教师模型（主模型中存储）")

        # 保存主模型（唯一的模型保存点）
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
    if best_pre_compression_model is not None:
        print(f"[训练总结] 剪枝前最优模型Top1精度：{best_pre_compression_acc:.2f}%")
    else:
        print(f"[训练总结] start_pruning_epoch=0，未跟踪剪枝前最优模型")


if __name__ == "__main__":
    # 解决Windows多进程数据加载问题
    if os.name == "nt":
        import torch.multiprocessing

        torch.multiprocessing.set_start_method("spawn", force=True)

    # 所有可配置参数集中在这里
    params = {
        # 损失启用开关（核心新增配置）
        'use_rcr_loss': False,  # 是否启用RCR损失
        'use_con_loss': False,  # 是否启用CON损失

        # 自蒸馏配置
        'use_self_distill': True,
        'distill_config': {
            "temp": 1.0,  # 温度参数（1.0-5.0）
            "weight": 1,  # 蒸馏损失权重（0.3-0.7）
            "save_interval": 1,  # 教师模型记录间隔（基于主模型）
            "start_epoch": 4,  # 开始蒸馏的epoch（需≥save_interval）
            "top_n_during_compression": 3,  # 压缩过程中取前n个最新模型
            "history_epochs_offset": [10, 20]  # 原偏移逻辑（可选保留）
        },

        # 模型与数据集配置
        'model_type': "resnet20",  # 模型类型（resnet20/resnet56）
        'dataset': "cifar10",  # 数据集
        'data_root': '../../datasets/cifar10',  # 数据集根目录
        'num_workers': 0,  # 数据加载线程数

        # 训练参数
        'sparsity': 0.7,  # 目标稀疏度
        'max_epoch': 250,  # 总训练epoch
        'batch_size': 128,  # 批次大小
        'lr': 1e-3,  # 初始学习率
        'weight_decay': 1e-4,  # 权重衰减
        'gamma': 0.5,  # 学习率衰减因子
        'milestones': [15, 100, 130, 160, 190, 220],  # 学习率衰减节点

        # 剪枝与投影参数
        'start_pruning_epoch': 10,  # 剪枝开始epoch（0表示不跟踪最优模型）
        'start_projection_epoch': 10,  # 投影开始epoch
        'projection_epochs': 50,  # 投影持续epoch数
        'pruning_epochs': 50,  # 剪枝持续epoch数

        # 损失函数参数
        'lambda1': 0.2,
        'lambda2': 0.5,

        # 保存配置
        'save_dir': "saved_models/test",  # 模型/日志保存目录
        'save_main_interval': 1,  # 主模型保存间隔

        # 设备配置
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # 启动训练
    main(params)