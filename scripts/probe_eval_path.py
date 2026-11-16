"""诊断 val acc 崩塌的成因: 用 train.py 自己的构建/评估路径, 避免 diag_common 的量化解码差异。

用法: python scripts/probe_eval_path.py <ckpt> [<ckpt2> ...]
每个 ckpt 依次报:
  1) model.eval() 评估(与训练日志里的 acc 同一条路径)
  2) BN 保持 train 模式评估(用 batch 统计, 排除 running stats 的影响)
  3) 用训练集重刷 running stats 后再做 eval 评估
"""
import sys, os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train as T                                     # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


@torch.no_grad()
def acc_batchbn(model, loader, device):
    model.train()
    c = n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        c += (model(X).argmax(1) == y).sum().item()
        n += y.numel()
    return c / n


@torch.no_grad()
def recalib(model, loader, device, n_batches=150):
    model.train()
    for i, (X, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(X.to(device))


def bn_stats(model):
    v = [float(m.running_var.mean()) for m in model.modules()
         if isinstance(m, torch.nn.BatchNorm2d)]
    return f"{len(v)} 个 BN, running_var 均值 min/max={min(v):.5f}/{max(v):.5f}"


def main():
    ckpts = sys.argv[1:]
    trainloader, testloader = T.get_dataloaders(
        os.path.join(ROOT, "datasets/cifar10"), 128, 2, 1, False, True,
        MEAN, STD, 4, True, 256)
    crit = torch.nn.CrossEntropyLoss()
    for path in ckpts:
        print(f"\n==== {os.path.basename(path)} ====")
        model, _ = T.build_model("resnet20", DEV)
        ck = torch.load(path, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"  ckpt epoch={ck.get('epoch')} history_val_acc="
              f"{ck.get('history', {}).get('val_acc', [])[-1:]}")
        print(f"  {bn_stats(model)}")
        ev = T.evaluate(model, testloader, crit, DEV)
        print(f"  1) eval 模式      val acc = {ev[-1]*100:.2f}%")
        print(f"  2) batch 统计      val acc = {acc_batchbn(model, testloader, DEV)*100:.2f}%")
        recalib(model, trainloader, DEV)
        ev2 = T.evaluate(model, testloader, crit, DEV)
        print(f"  3) 重刷 BN 后      val acc = {ev2[-1]*100:.2f}%   "
              f"Δ={(ev2[-1]-ev[-1])*100:+.2f}")
        print(f"  {bn_stats(model)}")


main()
