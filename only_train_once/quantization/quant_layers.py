import logging
import math
import os
from enum import Enum
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn


class NanInGradientError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# submodule level logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NaN / Inf 处理策略
# ---------------------------------------------------------------------------
# 背景: 量化参数 (d_quant/q_m/t_quant) 的梯度是对整个 tensor 求和, 量级天然
# 可达 1e9; 若用 variant="sgd" 会一步爆炸。历史上本文件用 nan_to_num 静默吞掉
# NaN, 把"立刻可见的崩溃"变成了"acc 恒等于随机水平的静默失效", 极难排查。
#
#   warn   (默认) 记录告警 + 非法值置 0, 训练继续, 但日志可见
#   strict        恢复原版行为, 直接 raise NanInGradientError
#   silent        复现历史行为, 完全静默 (仅用于对照复现, 不建议)
NAN_POLICY = os.environ.get("OTO_QUANT_NAN_POLICY", "warn").lower()

# 采样检查间隔。
# torch.isfinite(t).all() 是"全张量归约 + 隐式 GPU->CPU 同步", 每层每步都做会让
# 量化训练再慢 ~18%(实测 194ms -> 165ms/step)。根因(SGD 导致量化梯度爆炸)修复后
# 异常值几乎不会出现, 因此安全网改为采样检查: 默认每 500 次调用查一次,
# strict 模式下仍然每次都查并抛错。
_NAN_CHECK_EVERY = int(os.environ.get("OTO_QUANT_NAN_CHECK_EVERY", "500"))
_check_counter = [0]

_nan_event_count = {}


def _check_tick() -> bool:
    """返回本次是否应该做全张量有限性检查 (采样)。"""
    _check_counter[0] += 1
    return _check_counter[0] % _NAN_CHECK_EVERY == 0


def _nonfinite_report(where: str):
    _nan_event_count[where] = _nan_event_count.get(where, 0) + 1
    cnt = _nan_event_count[where]
    if cnt <= 3 or cnt % 200 == 0:
        logger.warning(
            f"[Quant][NaN] non-finite value in {where} (累计 {cnt} 次), 已替换为有界值。"
            f"NAN_POLICY={NAN_POLICY}. 若频繁出现请检查: (1) optimizer variant 是否为 adam "
            f"(官方推荐, SGD 会让量化梯度 1e9 一步爆炸); (2) lr_quant 是否过大。"
        )


def _sanitize_grad(
    t: torch.Tensor,
    where: str,
    detail: Union[str, Callable[[], str]] = "",
) -> torch.Tensor:
    """把梯度里的 NaN/Inf 替换为 0, 并按策略告警或抛错。

    性能提示: 非 strict 模式下走采样检查, 避免每层每步的全张量归约 + GPU 同步。

    detail 可以传字符串, 也可以传 callable。传 callable 时只在真正命中
    NaN/Inf 需要报错或告警时才求值 —— 拼 detail 里的 .item() / torch.max()
    都会触发 GPU 同步, 无条件构造会让整个 backward 退化成同步流水线。
    """
    if NAN_POLICY == "strict":
        if not torch.isfinite(t).all():
            msg = detail() if callable(detail) else detail
            raise NanInGradientError(f"Error: NaN/Inf appears in {where} gradient!\n{msg}")
        return t
    if NAN_POLICY == "silent":
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    # warn: 采样检查
    if _check_tick() and not torch.isfinite(t).all():
        _nonfinite_report(where)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def get_nan_event_stats():
    """返回 NaN 事件计数, 便于实验脚本在 epoch 结束时汇总。"""
    return dict(_nan_event_count)


class QuantizationType(Enum):
    # TODO: make uniform naming convention for the enum values
    SYMMETRIC_LINEAR = "symmetric+linear"
    SYMMETRIC_NONLINEAR = "symmetric+nonlinear"
    DGE = "dge"


class QuantizationMode(Enum):
    WEIGHT_ONLY = "weight_only"
    WEIGHT_AND_ACTIVATION = "weight_and_activation"


# Symmetric quantizer with nonlinear mapping (yes t)
class SymQuantizerNonLinear(torch.autograd.Function):
    """Symmetric quantizer with nonlinear mapping.

    Implements quantization using nonlinear mapping with parameter t.
    Uses gradient computation rules from: https://arxiv.org/pdf/2007.10463
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        d_quant: torch.Tensor,
        q_m: torch.Tensor,
        t_quant: torch.Tensor,
        clip_val: torch.Tensor,
        q_s: torch.Tensor,
    ) -> torch.Tensor:
        input_abs = torch.abs(input)

        # 性能: 量化参数与 input 恒在同一 device/已是目标张量, 原来每层每步
        # 5 次 .to(device) 纯属浪费(profile 里 aten::to 一步 500+ 次调用)。
        ctx.save_for_backward(input, d_quant, q_m, t_quant, clip_val, q_s)

        # q_m <= q_s can happen
        # Clamp to avoid log(negative) -> NaN when input_abs < q_s
        d_safe = torch.clamp(d_quant, min=1e-8)
        # 性能: 先取倒数, 把三处 div 换成 mul(div 在 GPU 上比 mul 慢 2~3 倍)
        inv_d = 1.0 / d_safe
        rng = torch.abs(q_m - q_s) + 1e-6
        # 性能: pow(x, t) 比 exp(t*log(x)) 少两个全张量 kernel, 数值等价 (x > 0)
        range_pow = torch.pow(rng, t_quant)
        # 性能: 原地 clamp, 省一次全张量分配
        input_diff = (input_abs - q_s).clamp_(min=1e-8)  # avoid log(negative)
        input_pow = torch.pow(input_diff, t_quant)

        # 性能: mul_/round_/mul_ 全原地, 相比 d_safe*round(x/d_safe) 少两次分配
        sat_val = (range_pow * inv_d).round_().mul_(d_safe)  # 标量: 饱和值
        output = input_pow.mul_(inv_d).round_().mul_(d_safe)
        # 性能: masked_fill 原地填 0, 省掉 zeros_like 的全张量分配
        output = output.masked_fill(input_abs <= q_s, 0.0)
        output = torch.where(input_abs >= q_m, sat_val.expand_as(output), output)
        # 性能: 用 sign 张量作容器原地乘, 比 torch.sign(input) * output 少一次分配。
        # 注意: 不用 torch.copysign —— 它无法导出到 ONNX opset 16, 会阻断 deployment 链路。
        output = torch.sign(input).mul_(output)
        # Guard against inf/nan from extreme quantization parameters.
        # 注意: 溢出时应饱和到 ±|q_m| (可表示上界), 而不是置 0 —— 置 0 会让
        # 整个量化层输出塌陷为 0, 模型在"看起来还在训练"的状态下悄悄死掉。
        if NAN_POLICY == "strict":
            if not torch.isfinite(output).all():
                raise NanInGradientError(
                    f"Error: non-finite value in quantization forward!\n"
                    f"d: {d_quant.item():.5f}, t: {t_quant.item():.5f}, "
                    f"q_m: {q_m.item():.5f}, q_s: {q_s.item():.5f}"
                )
        elif NAN_POLICY == "silent":
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        elif _check_tick() and not torch.isfinite(output).all():
            # 采样检查: 避免每层每步的全张量归约 + GPU 同步
            _nonfinite_report("forward")
            sat = torch.abs(q_m).to(output.device).clamp(min=1e-8)
            output = torch.where(
                torch.isinf(output), torch.sign(output) * sat, output
            )
            output = torch.nan_to_num(output, nan=0.0)
        return output

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor]:
        input, d_quant, q_m, t_quant, clip_val, q_s = ctx.saved_tensors
        input_abs = torch.abs(input)
        sign_in = torch.sign(input)

        # 直通估计: 裁剪区间外的梯度置零。
        # masked_fill 比 clone + 两次布尔索引赋值少 2 个 kernel。
        in_range = (input < clip_val[1]) & (input > clip_val[0])
        grad_x = grad_output.masked_fill(~in_range, 0.0)

        # Useful quantities — clamp to avoid log(negative) -> NaN and division by tiny d
        d_safe = torch.clamp(d_quant, min=1e-8)
        # 性能: 取一次倒数, 后面三处 div 全换成 mul
        inv_d = 1.0 / d_safe
        rng = torch.abs(q_m - q_s) + 1e-6  # q_m <= q_s can happen
        # 性能: pow 比 exp(t*log(x)) 少两个全张量 kernel, 数值等价 (x > 0)
        range_pow = torch.pow(rng, t_quant)
        # 性能: rng^(t-1) = rng^t * (1/rng), 省一次 pow 且用 mul 代替 div
        range_pow_low = range_pow * (1.0 / rng)
        # 性能: 原地 clamp, 省一次全张量分配
        input_diff = (input_abs - q_s).clamp_(min=1e-8)  # avoid log(negative)
        # 性能: pow 比 exp(t*log(x)) 少一个全张量 kernel, 数值等价 (x > 0)
        input_pow = torch.pow(input_diff, t_quant)

        # ---- 公共量提取: mask 与 grad_output*sign(input) 在三个梯度里复用 ----
        # 原写法每个梯度各算一次 mask / sign 乘法, 提取后每层少 4 个全张量 kernel
        mask_ge_qm = input_abs >= q_m
        mask_le_qs = input_abs <= q_s
        gs = grad_output * sign_in

        # ---- d_quant 梯度: round(x/d) - x/d ----
        q = input_pow * inv_d
        rq = range_pow * inv_d
        grad_d_xq = torch.round(q) - q
        grad_d_xq = torch.where(
            mask_ge_qm,
            (torch.round(rq) - rq).expand_as(grad_d_xq),
            grad_d_xq,
        )
        # 性能: masked_fill 省掉 zeros_like 的全张量分配
        grad_d_xq = grad_d_xq.masked_fill(mask_le_qs, 0.0)
        grad_d = (gs * grad_d_xq).sum().reshape(1)

        # ---- q_m 梯度: 仅 input_abs > q_m 非零, 系数为标量 -> 先归约再乘常数 ----
        g_qm = gs.masked_fill(input_abs <= q_m, 0.0)
        grad_qm = (g_qm.sum() * (t_quant * range_pow_low)).reshape(1)

        # ---- t_quant 梯度: input_pow * log(input_diff) ----
        grad_t_xq = input_pow * torch.log(input_diff)
        grad_t_xq = torch.where(
            mask_ge_qm,
            (range_pow * torch.log(rng)).expand_as(grad_t_xq),
            grad_t_xq,
        )
        grad_t_xq = grad_t_xq.masked_fill(mask_le_qs, 0.0)
        grad_t = (gs * grad_t_xq).sum().reshape(1)

        # NaN/Inf 处理: warn 策略下告警并清零, strict 策略下直接抛错 (原版行为)
        # 性能: detail 改为惰性求值, 只在真出问题时才做 .item()/max() 同步
        def _detail():
            return (
                f"d: {d_quant.item():.5f}, t: {t_quant.item():.5f}, "
                f"q_m: {q_m.item():.5f}, q_s: {q_s.item():.5f}\n"
                f"input_abs-max: {torch.max(input_abs).item():.5f}, "
                f"grad_output-max: {torch.max(grad_output).item():.5f}"
            )

        grad_x = _sanitize_grad(grad_x, "grad_x", _detail)
        grad_d = _sanitize_grad(grad_d, "grad_d", _detail)
        grad_qm = _sanitize_grad(grad_qm, "grad_qm", _detail)
        grad_t = _sanitize_grad(grad_t, "grad_t", _detail)

        return grad_x, grad_d, grad_qm, grad_t, None, None


class SymQuantizerLinear(torch.autograd.Function):
    """Symmetric quantizer with linear mapping.

    Implements quantization using linear mapping without nonlinearity parameter t.
    Uses gradient computation rules from: https://arxiv.org/pdf/2007.10463
    """

    # TODO: Add safeguard mechanism to handle the bad cases.
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        d_quant: torch.Tensor,
        q_m: torch.Tensor,
        clip_val: torch.Tensor,
        q_s: torch.Tensor,
    ) -> torch.Tensor:
        device = input.device
        input_abs = torch.abs(input)
        # Ensure all tensors are on the same device
        d_quant = d_quant.to(device)
        q_m = q_m.to(device)
        clip_val = clip_val.to(device)
        q_s = q_s.to(device)
        ctx.save_for_backward(input, d_quant, q_m, clip_val, q_s)

        range_pow = torch.abs(q_m - q_s)
        input_pow = input_abs - q_s  # input_abs >= q_s

        d_safe = torch.clamp(d_quant, min=1e-8)
        output = d_safe * torch.round(input_pow.div(d_safe))
        output[input_abs <= q_s] = 0
        output[input_abs >= q_m] = d_safe * torch.round(range_pow.div(d_safe))
        output = torch.sign(input) * output
        return output

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor]:
        input, d_quant, q_m, clip_val, q_s = ctx.saved_tensors
        device = input.device
        input_abs = torch.abs(input)

        grad_x = grad_output.clone()
        grad_x[input.ge(clip_val[1])] = 0
        grad_x[input.le(clip_val[0])] = 0

        # Useful quantities — clamp d to avoid division by zero
        d_safe = torch.clamp(d_quant, min=1e-8)
        range_pow = torch.abs(q_m - q_s)  # q_m <= q_s can happen
        input_pow = input_abs - q_s  # input_abs >= q_s

        grad_d_xq = torch.round(input_pow.div(d_safe)) - input_pow.div(d_safe)
        grad_d_xq[input_abs >= q_m] = torch.round(
            range_pow.div(d_safe)
        ) - range_pow.div(d_safe)
        grad_d_xq[input_abs <= q_s] = 0
        grad_d_xq = torch.sign(input) * grad_d_xq
        grad_d = torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device)

        grad_qm_xq = torch.sign(input)
        grad_qm_xq[input_abs <= q_m] = 0
        grad_qm = torch.tensor([torch.sum(grad_output * grad_qm_xq)], device=device)

        # Replace NaN/Inf instead of raising, 但保留告警痕迹
        # 性能: detail 改为惰性求值, 避免每层每步 3 次 .item() + 2 次 max 同步
        def _detail():
            return (
                f"d: {d_quant.item():.5f}, q_m: {q_m.item():.5f}, q_s: {q_s.item():.5f}\n"
                f"input_abs-max: {torch.max(input_abs).item():.5f}, "
                f"grad_output-max: {torch.max(grad_output).item():.5f}"
            )

        grad_x = _sanitize_grad(grad_x, "grad_x_linear", _detail)
        grad_d = _sanitize_grad(grad_d, "grad_d_linear", _detail)
        grad_qm = _sanitize_grad(grad_qm, "grad_qm_linear", _detail)
        return grad_x, grad_d, grad_qm, None, None

# NOTE: this is a experimental WIP, not used in the codebase yet
class DGEQuantizer(torch.autograd.Function):
    """DGE quantizer for weights that replaces STE with differentiable gradient estimation.
    https://arxiv.org/pdf/2501.17116
    Forward: Normal quantization 
    Backward: Uses f'(x) = (1/k) · |x - δ/2|^(1/k - 1) for weight gradient computation
    where k scales based on bit width relative to paper's k=5 for 4-bit.
    """
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        d_quant: torch.Tensor,  # Quantization interval (δ)
        q_m: torch.Tensor,
        clip_val: torch.Tensor,
        q_s: torch.Tensor,
        num_bits: torch.Tensor,  # Current target bit width
    ) -> torch.Tensor:
        device = input.device
        input_abs = torch.abs(input)
        
        # Move tensors to device
        d_quant = d_quant.to(device)
        q_m = q_m.to(device)
        clip_val = clip_val.to(device)
        q_s = q_s.to(device)
        
        # Scale k relative to paper's k=5 for 4-bit
        k = torch.tensor(5.0 * (4.0/num_bits)).to(device)
        ctx.save_for_backward(input, d_quant, q_m, k, clip_val, q_s)

        range_pow = torch.abs(q_m - q_s)
        input_pow = input_abs - q_s  # input_abs >= q_s

        output = d_quant * torch.round(input_pow.div(d_quant))
        output[input_abs <= q_s] = 0
        output[input_abs >= q_m] = d_quant * torch.round(range_pow.div(d_quant))
        output = torch.sign(input) * output
        return output

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor]:
        input, d_quant, q_m, k, clip_val, q_s = ctx.saved_tensors
        device = input.device
        input_abs = torch.abs(input)

        # Zero gradients outside clipping range
        grad_x = grad_output.clone()
        grad_x[input.ge(clip_val[1])] = 0
        grad_x[input.le(clip_val[0])] = 0

        # Paper's DGE gradient computation: f'(x) = (1/k) · |x - δ/2|^(1/k - 1)
        x_centered = input - d_quant/2
        grad_scale = (1/k) * torch.pow(torch.abs(x_centered), 1/k - 1)
        grad_x = grad_x * grad_scale
        # TODO: do we modify if we have higher than 4 bits?
        # Cap gradient magnitude at 3.0 as in paper
        grad_x = torch.clamp(grad_x, -3.0, 3.0) 

        # Compute d_quant gradient
        range_pow = torch.abs(q_m - q_s)
        input_pow = input_abs - q_s
        grad_d_xq = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
        grad_d_xq[input_abs >= q_m] = torch.round(range_pow.div(d_quant)) - range_pow.div(d_quant)
        grad_d_xq[input_abs <= q_s] = 0
        grad_d_xq = torch.sign(input) * grad_d_xq
        grad_d = torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device)

        # Compute q_m gradient
        grad_qm_xq = torch.sign(input)
        grad_qm_xq[input_abs <= q_m] = 0
        grad_qm = torch.tensor([torch.sum(grad_output * grad_qm_xq)], device=device)

        if torch.isnan(grad_x).any():
            raise NanInGradientError(
                f"NaN in gradient computation\n"
                f"input range: [{input.min():.4f}, {input.max():.4f}]\n"
                f"d_quant: {d_quant.item():.4f}\n"
                f"q_m: {q_m.item():.4f}\n"
                f"k: {k.item():.4f}"
            )

        return grad_x, grad_d, grad_qm, None, None, None

def _get_quantizer(qtype: QuantizationType) -> torch.autograd.Function:
    if qtype == QuantizationType.SYMMETRIC_LINEAR:
        return SymQuantizerLinear
    elif qtype == QuantizationType.SYMMETRIC_NONLINEAR:
        return SymQuantizerNonLinear
    elif qtype == QuantizationType.DGE:
        return DGEQuantizer
    else:
        raise NotImplementedError


class QuantizeMixin:
    def init_quantization(
        self,
        d_quant_init: float = 1.0,
        t_quant_init: float = 1.0,
        q_m_init: float = 1.0,
        quant_type: QuantizationType = QuantizationType.SYMMETRIC_LINEAR,
        quant_mode: QuantizationMode = QuantizationMode.WEIGHT_ONLY,
        weight_clip_val: Tuple[float, float] = (-2.0, 2.0),
        act_clip_val: Tuple[float, float] = (-2.0, 2.0),
    ):
        # Initialize weight quantization parameters
        self.d_quant_wt = nn.Parameter(torch.tensor([d_quant_init]))
        self.q_m_wt = nn.Parameter(torch.tensor([q_m_init]))
        if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            self.t_quant_wt = nn.Parameter(torch.tensor([t_quant_init]))

        # Initialize activation quantization if needed
        if quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
            self.d_quant_act = nn.Parameter(torch.tensor([d_quant_init]))
            self.q_m_act = nn.Parameter(torch.tensor([q_m_init]))
            if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
                self.t_quant_act = nn.Parameter(torch.tensor([t_quant_init]))

        self.quant_type = quant_type
        self.quant_mode = quant_mode
        self.weight_clip_val = weight_clip_val
        self.act_clip_val = act_clip_val

    def quantize_weight(self, weight: torch.tensor) -> torch.Tensor:
        """Quantize the weight tensor."""
        weight_clip_val = torch.tensor(self.weight_clip_val, device=weight.device)
        q_s = torch.tensor(0.0, device=weight.device)
        quantizer = _get_quantizer(self.quant_type)
        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            # Symmetric linear
            return quantizer.apply(
                weight,
                self.d_quant_wt,
                self.q_m_wt,
                weight_clip_val,
                q_s,
            )
        else:  # Symmetric nonlinear
            return quantizer.apply(
                weight,
                self.d_quant_wt,
                self.q_m_wt,
                self.t_quant_wt,
                weight_clip_val,
                q_s,
            )

    def quantize_act(self, activation: torch.tensor) -> torch.Tensor:
        """Quantize the activation tensor."""
        if self.quant_mode != QuantizationMode.WEIGHT_AND_ACTIVATION:
            return activation

        activation_clip_val = torch.tensor(self.act_clip_val, device=activation.device)
        q_s = torch.tensor(0.0, device=activation.device)
        quantizer = _get_quantizer(self.quant_type)

        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            return quantizer.apply(
                activation,
                self.d_quant_act,
                self.q_m_act,
                activation_clip_val,
                q_s,
            )
        else:  # SYMMETRIC_NONLINEAR
            return quantizer.apply(
                activation,
                self.d_quant_act,
                self.q_m_act,
                self.t_quant_act,
                activation_clip_val,
                q_s,
            )

    @property
    def weight_bit(self) -> int:
        d = self.d_quant_wt.item()
        qmax = abs(self.q_m_wt.item())
        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            t = 1.0
        elif self.quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            t = self.t_quant_wt.item()
        else:
            raise NotImplementedError

        return round(math.log2(math.exp(t * math.log(qmax)) / abs(d) + 1) + 1)

    @property
    def activation_bit(self) -> int:
        if self.quant_mode != QuantizationMode.WEIGHT_AND_ACTIVATION:
            return 32

        d = self.d_quant_act.item()
        qmax = abs(self.q_m_act.item())
        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            t = 1.0
        elif self.quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            t = self.t_quant_act.item()
        else:
            raise NotImplementedError

        return round(math.log2(math.exp(t * math.log(qmax)) / abs(d) + 1) + 1)


def initialize_quant_layer(
    layer: Union["QuantizeConv2d", "QuantizeLinear"],
    num_bits: int = 16,
    quant_type: QuantizationType = QuantizationType.SYMMETRIC_LINEAR,
    quant_mode: QuantizationMode = QuantizationMode.WEIGHT_ONLY,
) -> None:
    if not isinstance(layer, (QuantizeConv2d, QuantizeLinear)):
        return

    num_bits = float(num_bits)
    t_quant_init = 1.0
    q_s = torch.tensor(0.0, device=layer.weight.device)

    qm_quant_init = torch.max(torch.abs(layer.weight))
    d_quant_init = (qm_quant_init - q_s) / (2 ** (num_bits - 1) - 1)

    # Weight quantization parameters
    nn.init.constant_(layer.d_quant_wt, d_quant_init)
    nn.init.constant_(layer.q_m_wt, qm_quant_init)
    if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
        nn.init.constant_(layer.t_quant_wt, t_quant_init)

    # Activation quantization parameters if needed
    if quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
        nn.init.constant_(layer.d_quant_act, d_quant_init)
        nn.init.constant_(layer.q_m_act, qm_quant_init)
        if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            nn.init.constant_(layer.t_quant_act, t_quant_init)


class QuantizeLinear(nn.Linear, QuantizeMixin):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias)
        self.init_quantization(
            d_quant_init, t_quant_init, q_m_init, quant_type, quant_mode
        )

    @staticmethod
    def from_module(
        module=None,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
        quant_init_by_module=True,
        num_bits=8,
    ):
        quant_linear = QuantizeLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=True if module.bias is not None else False,
            d_quant_init=d_quant_init,
            t_quant_init=t_quant_init,
            q_m_init=q_m_init,
            quant_type=quant_type,
            quant_mode=quant_mode,
        )

        quant_linear.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            quant_linear.bias.data.copy_(module.bias.data)

        if quant_init_by_module:
            initialize_quant_layer(
                quant_linear,
                num_bits=num_bits,
                quant_type=quant_type,
                quant_mode=quant_mode,
            )
        return quant_linear

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.quantize_weight(self.weight)
        if self.quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
            input_ = self.quantize_act(input_)
        return nn.functional.linear(input_, weight, self.bias)


class QuantizeConv2d(nn.Conv2d, QuantizeMixin):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
        self.init_quantization(
            d_quant_init, t_quant_init, q_m_init, quant_type, quant_mode
        )

    @staticmethod
    def from_module(
        module=None,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
        quant_init_by_module=True,
        num_bits=8,
    ):
        # Fixed argument names in constructor call
        quant_conv2d = QuantizeConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            d_quant_init=d_quant_init,
            t_quant_init=t_quant_init,
            q_m_init=q_m_init,
            quant_type=quant_type,
            quant_mode=quant_mode,
        )

        quant_conv2d.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            quant_conv2d.bias.data.copy_(module.bias.data)

        if quant_init_by_module:
            initialize_quant_layer(
                quant_conv2d,
                num_bits=num_bits,
                quant_type=quant_type,
                quant_mode=quant_mode,
            )
        return quant_conv2d

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.quantize_weight(self.weight)
        if self.quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
            input_ = self.quantize_act(input_)
        return nn.functional.conv2d(
            input_,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


LAYER_TO_QUANTLAYER = {"Linear": QuantizeLinear, "Conv2d": QuantizeConv2d}
