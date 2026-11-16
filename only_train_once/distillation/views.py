"""第3章: 视图组生成。

一个 batch 从"干净视图"复制出 K 个增广视图, 供 GRPO 组内比较使用。
所有变换都在**已归一化**的张量上做, 所以:
  - 填充用 0(等价于"补均值"), 而不是再减一次 mean;
  - 颜色抖动是逐通道小尺度缩放, 不越出归一化后的动态范围。
这两个约定写死在这里, 换数据集时需要同步 image mean/std, 见 distill_example.yaml 注释。
"""

import math
import random

import torch
import torch.nn.functional as F


# severity -> 每种增广的强度。数值是"归一化空间"的尺度, 不是像素尺度。
SEVERITY = {
    "light": {"crop": 2, "color": 0.05, "erase": 8, "rotate": 5},
    "mid":   {"crop": 4, "color": 0.10, "erase": 12, "rotate": 10},
    "heavy": {"crop": 6, "color": 0.20, "erase": 16, "rotate": 15},
}

# 视图组真正支持的算子。yaml 里写了别的名字必须在运行时报出来, 不能静默丢弃
# (否则 augment_modes 写了六种, 实际只生成了三种, 组内结构就悄悄变了)。
SUPPORTED = {"identity", "crop", "flip", "color", "erase", "cutout", "rotate"}


def _rand_crop(x, pad, gen):
    """零填充后随机裁剪回原尺寸。"""
    if pad <= 0:
        return x
    b, c, h, w = x.shape
    padded = torch.zeros((b, c, h + 2 * pad, w + 2 * pad),
                         dtype=x.dtype, device=x.device)
    padded[:, :, pad:pad + h, pad:pad + w] = x
    oy = int(torch.randint(0, 2 * pad + 1, (1,), generator=gen).item())
    ox = int(torch.randint(0, 2 * pad + 1, (1,), generator=gen).item())
    return padded[:, :, oy:oy + h, ox:ox + w]


def _color_jitter(x, scale, gen):
    """逐通道 1±scale 的缩放 + 小偏移; 随机数在 CPU 生成再搬到目标设备。

    注意: torch.rand(generator=CUDA生成器) 与 torch.randint(generator=CUDA生成器)
    在 2.1 上要么报设备不匹配, 要么把结果留在 CPU。统一"CPU 造随机数 -> to(device)"
    最省心, 代价只有一次小张量拷贝。
    """
    b, c, _, _ = x.shape
    dev = x.device
    s = (1.0 + (torch.rand((b, c, 1, 1), generator=gen) * 2 - 1) * scale).to(dev)
    o = ((torch.rand((b, c, 1, 1), generator=gen) * 2 - 1) * scale).to(dev)
    return x * s + o


def _cutout(x, size, gen):
    """随机置零一个方块(归一化空间里 0 = 均值色)。"""
    if size <= 0:
        return x
    b, c, h, w = x.shape
    size = min(size, h, w)
    ys = torch.randint(0, h - size + 1, (b,), generator=gen)
    xs = torch.randint(0, w - size + 1, (b,), generator=gen)
    out = x.clone()
    for i in range(b):
        y0, x0 = int(ys[i].item()), int(xs[i].item())
        out[i, :, y0:y0 + size, x0:x0 + size] = 0.0
    return out


def _rotate(x, deg, gen):
    """逐样本小角度旋转; 用 affine_grid + grid_sample, 不依赖 torchvision。"""
    b = x.shape[0]
    ang = (torch.rand(b, generator=gen) * 2 - 1) * math.radians(deg)
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0] = c
    theta[:, 0, 1] = -s
    theta[:, 1, 0] = s
    theta[:, 1, 1] = c
    grid = F.affine_grid(theta.to(x.device), x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _rand_flip(x, gen):
    """逐样本随机水平翻转。

    flip 本身是确定性的, 若同一算子被复制成两个视图(i=2 与 i=7), 两者会完全
    相同 —— 实测 peer 像素 L1 都是 0.783, 等于组里有两个重复输入。改成逐样本
    按概率翻转后, 不同视图的翻转子集不同, 组内才有真正的差异。
    """
    b = x.shape[0]
    m = (torch.rand(b, generator=gen) < 0.5)
    m = m.view(b, 1, 1, 1).to(x.device)
    return torch.where(m, x.flip(-1), x)


def make_views(x, modes, severity="mid", group_size=8, gen=None, seed=None):
    """返回 [(视图名, 张量), ...], 第 0 项永远是 identity(干净视图)。

    group_size 是**组总大小**(含干净视图), 与 decision.group_size 同义。
    modes 里没有的增广不会出现在视图里; identity 必在第一项, 由校验器保证。
    """
    if gen is None:
        # 随机数一律在 CPU 生成(见 _color_jitter 的说明), 与张量设备无关
        gen = torch.Generator()
        gen.manual_seed(random.randint(0, 2 ** 31 - 1) if seed is None else seed)
    strength = SEVERITY.get(severity, SEVERITY["mid"])

    # 可用的增广算子(去掉 identity 与不认识的模式)
    ops = []
    for m in modes:
        if m == "identity":
            continue
        if m == "crop":
            ops.append(("crop", lambda t, g: _rand_crop(t, strength["crop"], g)))
        elif m == "flip":
            ops.append(("flip", lambda t, g: _rand_flip(t, g)))
        elif m == "color":
            ops.append(("color", lambda t, g: _color_jitter(t, strength["color"], g)))
        elif m == "erase":
            ops.append(("erase", lambda t, g: _cutout(t, strength["erase"], g)))
        elif m == "cutout":
            # cutout 与 erase 是同一个算子(RandomErasing 的别名), 显式合并而不是忽略
            ops.append(("cutout", lambda t, g: _cutout(t, strength["erase"], g)))
        elif m == "rotate":
            ops.append(("rotate", lambda t, g: _rotate(t, strength["rotate"], g)))

    views = [("identity", x)]
    if not ops:
        # 只有干净视图时, 组内没有差异 -> GRPO 的优势会退化为 0;
        # 调用方(runtime)会在这种情况下跳过决策层损失并打一次警告。
        return views
    for i in range(1, group_size):
        idx = (i - 1) % len(ops)
        name, fn = ops[idx]
        t = fn(x, gen)
        if i - 1 >= len(ops):
            # 组比算子多时, 第二轮起的副本必须再扰一次, 否则就是"同一个输入
            # 计两票": 实测 G=8 / 6 个算子时 crop1 与 crop6、flip2 与 flip7
            # 完全相同, 组内有效视图数只有 6。
            t = _rotate(t, strength["rotate"] * 0.5, gen)
            name = f"{name}+rot"
        views.append((f"{name}{i}", t))
    return views


def view_diversity(views):
    """组内平均两两 L2 距离(相对干净视图), 用来判断增广是否真的起了作用。"""
    if len(views) < 2:
        return 0.0
    base = views[0][1]
    ds = [float((v - base).norm() / (base.norm() + 1e-12)) for _, v in views[1:]]
    return sum(ds) / len(ds)
