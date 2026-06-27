import sys
sys.path.append('..')
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
from sanity_check.backends.mobilenetv1 import mobilenetv1_cifar10
from sanity_check.backends.mobilenetv2 import mobilenetv2_cifar10
from sanity_check.backends.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once import OTO
import torch
def main():
    save_dir = "saved_models/test"

    model = mobilenetv3_small()


    model = model_to_quantize_model(model)

    dummy_input = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

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

    trainloader =  torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2,
                                               pin_memory=True,  # 启用 pinned memory
                                               persistent_workers=True,  # 保持worker进程活跃
                                               prefetch_factor=2
                                               )
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2,
                                             pin_memory=True,  # 启用 pinned memory
                                             persistent_workers=True,  # 保持worker进程活跃
                                             prefetch_factor=2
                                             )

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
    start_projection_epoch=0
    start_pruning_epoch=0
    projection_epochs=50
    pruning_epochs=50
    optimizer = oto.geta(
        variant="adam",
        lr=5e-3,
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
    from myloss import TotalLoss
    total_loss = TotalLoss(lambda1=0.1, lambda2=0.3,
    start_projection_epoch=start_projection_epoch,
    start_pruning_epoch=start_pruning_epoch,
    projection_epochs=projection_epochs,
    pruning_epochs=pruning_epochs)

    import os
    from tutorials.utils.utils import check_accuracy
    from mylog import log_and_print_training

    # 创建保存模型的子目录

    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

    # oto.visualize(view=False, out_dir=save_dir)

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
        for X, y in trainloader:
            X = X.cuda()
            y = y.cuda()
            y_pred, feature = model.forward(X, feature_need=True)
            celoss, rcrloss, conloss = total_loss.total_loss(y_pred=y_pred, y=y, model=model, feature=feature, epoch=epoch, RCR=False, CON=False)
            f = celoss + rcrloss + conloss
            optimizer.zero_grad()
            f.backward()
            f_avg_val += f
            celoss_avg_val += celoss
            rcrloss_avg_val += rcrloss
            conloss_avg_val += conloss
            optimizer.step()

        opt_metrics = optimizer.compute_metrics()
        accuracy1, accuracy5 = check_accuracy(model, testloader)
        f_avg_val = f_avg_val / len(trainloader)
        celoss_avg_val = celoss_avg_val / len(trainloader)
        rcrloss_avg_val = rcrloss_avg_val / len(trainloader)
        conloss_avg_val = conloss_avg_val / len(trainloader)

        # 判断并转换为float类型
        if not isinstance(rcrloss_avg_val, float):
            rcrloss_avg_val = rcrloss_avg_val.item()
        if not isinstance(conloss_avg_val, float):
            conloss_avg_val = conloss_avg_val.item()

        oto.construct_subnet(out_dir='temp')
        compressed_model_size = os.stat(oto.compressed_model_path)

        # optimizer.history(loss=f_avg_val.item(), size=compressed_model_size.st_size / (1024 ** 2))

        log_and_print_training(
        save_dir=save_dir,  # 与保存模型的目录相同
        epoch=epoch,
        celoss=celoss_avg_val.item(),
        rcrloss=rcrloss_avg_val,
        conloss=conloss_avg_val,
        param_norm=opt_metrics.norm_params,
        group_sparsity=opt_metrics.group_sparsity,
        acc1=accuracy1,
        norm_import=opt_metrics.norm_important_groups,
        norm_redund=opt_metrics.norm_redundant_groups,
        num_grps_import=opt_metrics.num_important_groups,
        num_grps_redund=opt_metrics.num_redundant_groups
        )

        # 保存每20个epoch的模型
        if (epoch + 1) % 20 == 0:
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

if __name__ == "__main__":
    main()