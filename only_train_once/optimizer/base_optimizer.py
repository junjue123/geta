from abc import abstractclassmethod
import os

import torch
from torch.optim.optimizer import Optimizer, required


class BaseOptimizer(Optimizer):
    def __init__(self, params, defaults=dict(), **kwargs):
        super(BaseOptimizer, self).__init__(params, defaults)
        self.num_steps = 0
        self.safe_guard = 1e-8
        self.first_moment_grads = dict()
        self.second_moment_grads = dict()

    def __setstate__(self, state):
        super(BaseOptimizer, self).__setstate__(state)

    def get_first_momentum_grad(self, name, first_moment, dampening, grad):
        if first_moment > 0:
            if name not in self.first_moment_grads:
                buf = self.first_moment_grads[name] = grad
            else:
                buf = self.first_moment_grads[name]
                buf.mul_(first_moment).add_(grad, alpha=(1.0 - dampening))
            return buf
        else:
            return grad

    def get_second_momentum_grad_square(self, name, second_moment, dampening, grad):
        if second_moment > 0:
            if name not in self.second_moment_grads:
                buf = self.second_moment_grads[name] = grad * grad
            else:
                buf = self.second_moment_grads[name]
                buf.mul_(second_moment).add_(grad * grad, alpha=(1.0 - dampening))
            return buf
        else:
            return grad * grad

    def compute_grad_variant(self):
        """计算每个参数的更新量 grad_variant。

        默认走 foreach 批量路径: 原实现逐参数做 Adam(约 10 个 kernel/参数),
        量化后参数张量接近 200 个, 一步就是近 2000 次 kernel launch。
        批量路径按 (dtype, device) 分桶后用 torch._foreach_* 融合, 语义与逐参数
        版本完全一致。设环境变量 OTO_OPT_FOREACH=0 可回退到旧实现做对照。
        """
        use_foreach = os.environ.get("OTO_OPT_FOREACH", "1") != "0"
        for group in self.param_groups:
            is_adam = group["variant"] == "adam" or group["variant"] == "adamw"
            first_bias_correction = (
                1.0 - group["first_momentum"] ** self.num_steps if is_adam else None
            )
            second_bias_correction = (
                1.0 - group["second_momentum"] ** self.num_steps if is_adam else None
            )
            group["grad_variant"] = dict()
            if use_foreach:
                self._compute_grad_variant_foreach(
                    group, is_adam, first_bias_correction, second_bias_correction
                )
            else:
                self._compute_grad_variant_legacy(
                    group, is_adam, first_bias_correction, second_bias_correction
                )

    def _compute_grad_variant_legacy(
        self, group, is_adam, first_bias_correction, second_bias_correction
    ):
        """逐参数实现, 作为 foreach 版本的数值对照基线。"""
        for p_name, p in zip(group["p_names"], group["params"]):
            if p.grad is None:
                continue
            refined_grad_f = torch.clone(p.grad.data).detach()
            if group["weight_decay"] is not None and group["variant"] != "adamw":
                refined_grad_f += group["weight_decay"] * p.data
            if not is_adam:
                if group["first_momentum"] > 0.0 or group["dampening"] > 0.0:
                    refined_grad_f = self.get_first_momentum_grad(
                        p_name,
                        group["first_momentum"],
                        group["dampening"],
                        refined_grad_f,
                    )
                group["grad_variant"][p_name] = refined_grad_f
            else:
                # 动量 buffer 用参数名而不是 (group_idx, param_idx) 作 key。
                # GETA 剪枝会重排/重建 param_groups, 用索引作 key 时,
                # 跨进程断点续训会把某个参数的 buffer 套到另一个参数上
                # (典型报错: "output with shape [1] doesn't match the
                # broadcast shape [32, 32, 3, 3]")。参数名在模型内唯一且
                # 与剪枝无关, 因此可安全跨运行恢复。
                first_moment_grad = self.get_first_momentum_grad(
                    p_name,
                    group["first_momentum"],
                    group["first_momentum"],
                    refined_grad_f,
                )
                second_moment_grad_sq = self.get_second_momentum_grad_square(
                    p_name,
                    group["second_momentum"],
                    group["second_momentum"],
                    refined_grad_f,
                )
                exp_avg_first_moment_grad = first_moment_grad / first_bias_correction
                exp_avg_second_moment_grad_sq = (
                    second_moment_grad_sq / second_bias_correction
                )
                denom = exp_avg_second_moment_grad_sq.sqrt().add_(self.safe_guard)
                group["grad_variant"][p_name] = exp_avg_first_moment_grad / denom

    def _compute_grad_variant_foreach(
        self, group, is_adam, first_bias_correction, second_bias_correction
    ):
        """批量实现: 参数按 (dtype, device) 分桶, 桶内用 _foreach_* 融合运算。"""
        buckets = dict()
        for p_name, p in zip(group["p_names"], group["params"]):
            if p.grad is None:
                continue
            key = (p.grad.dtype, p.grad.device)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = ([], [], [])
            bucket[0].append(p_name)
            bucket[1].append(p)
            bucket[2].append(p.grad.data)

        wd = group["weight_decay"]
        apply_wd = wd is not None and group["variant"] != "adamw"
        for names, params, grads in buckets.values():
            # refined_grad = grad.clone()
            g_list = torch._foreach_mul(grads, 1.0)
            if apply_wd:
                torch._foreach_add_(g_list, [p.data for p in params], alpha=wd)

            if not is_adam:
                if group["first_momentum"] > 0.0 or group["dampening"] > 0.0:
                    g_list = self._foreach_first_momentum(
                        names, group["first_momentum"], group["dampening"], g_list
                    )
                for name, g in zip(names, g_list):
                    group["grad_variant"][name] = g
                continue

            beta1 = group["first_momentum"]
            beta2 = group["second_momentum"]
            m_list = self._foreach_first_momentum(names, beta1, beta1, g_list)
            v_list = self._foreach_second_momentum(names, beta2, beta2, g_list)

            m_hat = torch._foreach_div(m_list, first_bias_correction)
            v_hat = torch._foreach_div(v_list, second_bias_correction)
            denom = torch._foreach_sqrt(v_hat)
            torch._foreach_add_(denom, self.safe_guard)
            for name, r in zip(names, torch._foreach_div(m_hat, denom)):
                group["grad_variant"][name] = r

    def _foreach_first_momentum(self, names, momentum, dampening, grads):
        """get_first_momentum_grad 的批量版, 保持「首次直接引用梯度」的语义。"""
        if momentum <= 0:
            return grads
        bufs = []
        fresh = set()
        for i, (name, g) in enumerate(zip(names, grads)):
            buf = self.first_moment_grads.get(name)
            if buf is None:
                self.first_moment_grads[name] = g
                bufs.append(g)
                fresh.add(i)
            else:
                bufs.append(buf)
        idx = [i for i in range(len(bufs)) if i not in fresh]
        if idx:
            torch._foreach_mul_([bufs[i] for i in idx], momentum)
            torch._foreach_add_(
                [bufs[i] for i in idx],
                [grads[i] for i in idx],
                alpha=(1.0 - dampening),
            )
        return bufs

    def _foreach_second_momentum(self, names, momentum, dampening, grads):
        """get_second_momentum_grad_square 的批量版。"""
        if momentum <= 0:
            return [g * g for g in grads]
        bufs = []
        fresh = set()
        for i, (name, g) in enumerate(zip(names, grads)):
            buf = self.second_moment_grads.get(name)
            if buf is None:
                v = g * g
                self.second_moment_grads[name] = v
                bufs.append(v)
                fresh.add(i)
            else:
                bufs.append(buf)
        idx = [i for i in range(len(bufs)) if i not in fresh]
        if idx:
            torch._foreach_mul_([bufs[i] for i in idx], momentum)
            torch._foreach_add_(
                [bufs[i] for i in idx],
                [grads[i] * grads[i] for i in idx],
                alpha=(1.0 - dampening),
            )
        return bufs

    def set_learning_rate(self, lr):
        for param_group in self.param_groups:
            param_group["lr"] = lr

    def get_learning_rate(self):
        for param_group in self.param_groups:
            lr = param_group["lr"]
        return lr

    def step(self, loss=None):
        raise NotImplementedError

    def state_dict(self):
        """
        Return a state_dict of the optimizer for restore.
        """
        state_dict = dict()

        state_dict["num_steps"] = self.num_steps
        state_dict["safe_guard"] = self.safe_guard
        state_dict["first_moment_grads"] = self.first_moment_grads
        state_dict["second_moment_grads"] = self.second_moment_grads

        return state_dict
