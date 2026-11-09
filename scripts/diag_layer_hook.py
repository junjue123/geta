#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐层 hook: 定位 forward 数值异常 / backward 梯度放大的层"""
import os, sys
ROOT = "E:/pythonProject/geta-main/geta-main_origin"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch, torch.nn as nn, numpy as np, pickle
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

torch.manual_seed(0); DEV="cuda"
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
m = model_to_quantize_model(resnet20_cifar10(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(DEV)

base="datasets/cifar10/cifar-10-batches-py"
with open(os.path.join(base,"data_batch_1"),"rb") as f: d=pickle.load(f,encoding="latin1")
X=d["data"][:128].reshape(-1,3,32,32).astype(np.float32)/255.
mean=np.array([0.4914,0.4822,0.4465],dtype=np.float32).reshape(1,3,1,1)
std=np.array([0.2470,0.2435,0.2616],dtype=np.float32).reshape(1,3,1,1)
X=torch.from_numpy((X-mean)/std).to(DEV)
y=torch.from_numpy(np.array(d["labels"][:128],dtype=np.int64)).to(DEV)

rec={}
def mk(n):
    def fwd(mod, inp, out):
        if isinstance(out, torch.Tensor):
            rec[n]=("fwd", out.abs().max().item(), out.abs().mean().item())
    return fwd
def mk_bwd(n):
    def bwd(mod, gin, gout):
        if isinstance(gout[0], torch.Tensor):
            rec[n+"_grad"]=("bwd", gout[0].abs().max().item(), gout[0].abs().mean().item())
    return bwd

hs=[]
for n,mod in m.named_modules():
    if len(list(mod.children()))==0:
        hs.append(mod.register_forward_hook(mk(n)))
        pass

out=m(X); loss=nn.CrossEntropyLoss()(out,y)
print(f"final out absmean={out.abs().mean().item():.4e} loss={loss.item():.4f}")
loss.backward()

print(f"\n{'module':<34}{'fwd_absmax':>13}{'fwd_absmean':>13}{'|':>3}{'grad_absmax':>13}{'grad_absmean':>13}")
for n,mod in m.named_modules():
    if len(list(mod.children()))==0:
        f=rec.get(n); b=rec.get(n+"_grad")
        fs=f"{f[1]:.3e} {f[2]:.3e}" if f else "        -        "
        bs=f"{b[1]:.3e} {b[2]:.3e}" if b else "        -        "
        print(f"{n:<34}{fs:>27}  |  {bs}")
