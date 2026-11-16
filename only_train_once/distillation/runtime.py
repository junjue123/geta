"""第3章: 蒸馏运行时——把开关组翻译成真正的损失与约束。

三层介入, 各自只在"该层的执行位置"发力, 不互相借用:
  decision (损失/标量)  -> GRPO 组内优势, 只对 logits
  feature  (反传/hook)   -> 逐层特征 KL, 梯度直接灌进中间层
  weight   (post-step)   -> ε-球投影, 不是损失, 发生在 opt.step() 之后

三条纪律:
  1. 任一机制缺资源(没有 T_hist、组内无差异、tap 名非法) 时, **关掉该项并记一条
     日志**, 不静默退化、不伪造损失;
  2. 教师一律 no_grad + eval, 学生梯度只从学生自己的前向图来;
  3. 每项损失都可单独归零, 用于"机制缺失对照"(§22 要求机制全保留、只做缺失对照)。
"""

import json
import math
import os

import torch
import torch.nn.functional as F

from .teachers import TeacherPool
from .views import SUPPORTED, make_views, view_diversity


class _TapCapture:
    """临时挂 hook 抓指定层的输出(avg_pool 抓 linear 的输入)。"""

    # tap 名 -> (在模型上的属性名, "output" | "input")
    SPEC = {
        "layer1": ("layer1", "output"),
        "layer2": ("layer2", "output"),
        "layer3": ("layer3", "output"),
        "layer4": ("layer4", "output"),
        "avg_pool": ("linear", "input"),
    }

    def __init__(self, model, names):
        self.model = model
        self.names = [n for n in names if hasattr(model, self.SPEC.get(n, ("",))[0])]
        self.store = {}
        self.handles = []

    def __enter__(self):
        for n in self.names:
            attr, kind = self.SPEC[n]
            mod = getattr(self.model, attr)
            if kind == "output":
                h = mod.register_forward_hook(self._mk(n, 0))
            else:
                h = mod.register_forward_pre_hook(self._mk(n, 0))
            self.handles.append(h)
        return self

    def _mk(self, name, idx):
        def hook(_m, inputs, output=None):
            self.store[name] = inputs[idx] if output is None else output
        return hook

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


class _bn_eval:
    """把 BN 临时切到 eval。

    视图与取特征的前向会各走一遍, 若保持 train 模式, 每步 BN 的 running stats
    会被更新 G+2 次而不是 1 次 —— 等于把 BN 的动量放大一个数量级, 且被合成视图
    污染。这里只冻结"额外前向", 主前向仍按 train 走。
    """

    def __init__(self, model):
        self.model = model
        self.mods = []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) and m.training:
                m.eval()
                self.mods.append(m)
        return self

    def __exit__(self, *a):
        for m in self.mods:
            m.train()
        self.mods = []


class DistillRuntime:
    def __init__(self, cfg, model, criterion, device, out_dir, model_name, mode,
                 model_factory, log=None, val_loader=None):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.device = device
        self.log = log or (lambda tag, msg: None)
        self._warned = set()
        self.step = 0

        self.pool = TeacherPool(cfg, model, out_dir, model_name, mode, device,
                                model_factory, log=self.log, val_loader=val_loader)

        # 参数层锚点: pre_compression = 挂载时刻(即 joint 段开始的压缩结构)
        self.anchor = {}
        if cfg["weight"]["enabled"]:
            self.anchor = {n: p.detach().clone()
                           for n, p in model.named_parameters() if p.numel() > 0}
            # 投影每步都要核对 197 个张量。逐个张量调 .norm().item() / .sum().item()
            # 等于每步 394 次 GPU 同步, 实测 proj 段与 batch 无关地恒定 53 ms/step
            # (B=128 与 B=256 都是 53 ms)。改成: 锚点侧算一次存成张量, 每步用
            # _foreach_norm / stack 批量算, 只留一次 .tolist() 取值。
            self._pairs = [(p, self.anchor[n]) for n, p in model.named_parameters()
                           if n in self.anchor and p.numel() > 0]
            self._anchor_norm_t = torch.stack([a.norm() for _, a in self._pairs])
            self._anchor_nnz_t = torch.stack(
                [torch.count_nonzero(a).double() for _, a in self._pairs])
            self.log("DISTILL", f"参数层锚点已冻结: {len(self.anchor)} 个张量; "
                                f"ε0={cfg['weight']['epsilon0']} "
                                f"schedule={cfg['weight']['schedule']}")

        # 压缩事件后的蒸馏强度缓升。实测(ep17 step0, 第 2 期剪枝刚做完):
        # ref_kl 0.18->0.49、feat_layer3 0.10->0.65、dec 0.55->0.97、r_mean 1.5->0.34,
        # 同期 loss 3.23 / acc 0.25 —— 剪枝瞬间学生分布离教师极远, 一步到位的
        # 强约束等于用旧权重的分布硬拽刚剪完的模型, 会把恢复过程打乱。
        ramp = cfg.get("ramp") or {}
        self.ramp_steps = int(ramp.get("steps", 0)) if ramp.get("enabled") else 0
        self._ramp_left = 0
        self._prev_skip = None

        # 机制体检: 每一步(或每 N 步)把各机制的中间量写成 JSONL。
        # 只看 val acc 分不出"机制在空转"和"机制在起作用但被压缩拖住",
        # 组内退化率 / 优势分布 / 逐层偏离量这些量才是判据。
        j = (cfg.get("diagnostics") or {}).get("jsonl") or {}
        self._jsonl_on = bool(j.get("enabled"))
        self._jsonl_every = max(1, int(j.get("every_steps", 5)))
        self._diag_path = os.path.join(out_dir, "distill_diag.jsonl")

        # 缺 T_hist 时, 依赖它的两项必须真的关掉(不是"算了不报")
        if cfg["feature"]["enabled"] and not self.pool.has_hist:
            self.feature_on = False
            self.log("DISTILL", "feature.enabled=true 但 T_hist 不可用 -> 本次关闭特征层")
        else:
            self.feature_on = cfg["feature"]["enabled"]
        if cfg["decision"]["ref_kl"]["enabled"] and not self.pool.has_hist:
            self.ref_on = False
            self.log("DISTILL", "ref_kl.enabled=true 但 T_hist 不可用 -> 本次关闭 ref 位")
        else:
            self.ref_on = cfg["decision"]["ref_kl"]["enabled"]

        self.log("DISTILL", f"运行时就绪: decision={cfg['decision']['enabled']} "
                            f"feature={self.feature_on} weight={cfg['weight']['enabled']} "
                            f"| {self.pool.describe()}")

    # ================= 三项损失 =================
    def extra_loss(self, X, y, clean_logits):
        """返回 (总计损失, 统计字典)。没有任何机制生效时返回 (None, {})。"""
        total = None
        stats = {}

        if self.cfg["decision"]["enabled"]:
            dec, dstats = self._decision_loss(X, y, clean_logits)
            stats.update(dstats)
            if dec is not None:
                total = dec if total is None else total + dec

        if self.feature_on:
            feat, fstats = self._feature_loss(X)
            stats.update(fstats)
            if feat is not None:
                total = feat if total is None else total + feat

        # 视图的随机种子跟着 step 走: 种子恒定时每步生成的是同一批视图,
        # 组内多样性会退化成"只有一张图", 优势也跟着退化。
        self.step += 1

        # 压缩事件后的强度缓升: 0 -> 1 线性升满 ramp_steps 步
        scale = self._ramp_scale()
        if scale < 1.0 and total is not None:
            total = total * scale
            stats["ramp"] = scale
        if self._ramp_left > 0:
            self._ramp_left -= 1
        return total, stats

    def _ramp_scale(self):
        """当前蒸馏强度系数; 非事件期恒 1.0。"""
        if self.ramp_steps <= 0 or self._ramp_left <= 0:
            return 1.0
        return max(0.0, 1.0 - self._ramp_left / float(self.ramp_steps))

    def _note_event(self, skipped):
        """用 w_skip 的增长探测压缩事件(支撑集变了 == 刚剪过)。"""
        if self.ramp_steps <= 0:
            return
        prev = self._prev_skip
        self._prev_skip = skipped
        if prev is None:
            return
        if skipped > prev:
            self._ramp_left = self.ramp_steps
            self.log("DISTILL", f"检测到压缩事件(w_skip {prev}->{skipped}), "
                                f"蒸馏强度按 {self.ramp_steps} 步缓升")

    # ---- 决策层: GRPO 组内优势 ----
    def _decision_loss(self, X, y, clean_logits):
        d = self.cfg["decision"]
        rw = d["reward"]
        unknown = [m for m in d["augment_modes"] if m not in SUPPORTED]
        if unknown:
            self._warn_once("bad_modes",
                            f"augment_modes 里有未实现的算子 {unknown}, 已忽略; "
                            f"支持 {sorted(SUPPORTED)}")
        views = make_views(X, d["augment_modes"], d["severity"],
                           group_size=int(d["group_size"]), seed=self.step)
        if len(views) < 2:
            self._warn_once("no_aug",
                            "augment_modes 只有 identity, 组内无差异 -> 决策层损失跳过")
            return None, {}

        clean_pred = clean_logits.argmax(1).detach()
        # P1 优化: G 个视图在 batch 维拼成一份, 一次前向一次反传后再按 G 切分。
        # 之前逐视图前向要保留 G 张独立的前向图, 实测 joint 单步 1.95s(warmup 是
        # 0.17s); 拼 batch 后图数从 ~10 降到 2, 是这笔开销的主要来源。
        batch = X.size(0)
        G = len(views)
        xs = torch.cat([v for _, v in views], dim=0)         # [G*B, C, H, W]
        with _bn_eval(self.model):
            out_all = _logits(self.model(xs)).view(G, batch, -1)

        if rw["teacher_agreement"] > 0 and self.pool.has_cf:
            with torch.no_grad():
                pred_cf_all = _logits(self.pool.cf(xs)).argmax(1).view(G, batch)
        else:
            pred_cf_all = None

        rewards = []
        logps = []
        for gi in range(G):
            out_s = out_all[gi]
            logps.append(F.log_softmax(out_s, dim=1).gather(
                1, y.view(-1, 1)).squeeze(1))
            r = torch.zeros_like(y, dtype=torch.float32)
            if rw["correctness"]:
                r = r + (out_s.argmax(1) == y).float()
            if pred_cf_all is not None:
                r = r + rw["teacher_agreement"] * (out_s.argmax(1) == pred_cf_all[gi]).float()
            if rw["clean_consistency"] > 0:
                r = r + rw["clean_consistency"] * (out_s.argmax(1) == clean_pred).float()
            rewards.append(r.detach())

        R = torch.stack(rewards, dim=1)                     # [B, G]
        adv = R - R.mean(dim=1, keepdim=True)
        if d["advantage"]["normalize_std"]:
            adv = adv / (R.std(dim=1, keepdim=True) + 1e-6)

        LP = torch.stack(logps, dim=1)                      # [B, G]
        if d["advantage"]["mode"] == "positive":
            # §23 的最终形式: 只对正优势项加权, 等价于 -\frac1{|P|}\sum (A_i)_+\log p
            w = F.relu(adv)
        else:
            w = adv
        denom = torch.clamp((w > 0).sum(), min=1) if d["advantage"]["mode"] == "positive" \
            else w.numel()
        dec = -(w * LP).sum() / denom

        stats = {
            "dec": float(dec.detach()),
            "r_mean": float(R.mean()),
            "adv_pos": float((adv > 0).float().mean()),
            "view_div": view_diversity(views),
            # 机制体检用的中间量: 只用精度看不出"组内到底有没有差异",
            # 这两项直接回答 GRPO 在这一步是不是在空转。
            "D_group": float((R.min(dim=1).values == R.max(dim=1).values).float().mean()),
            "adv_std": float(adv.std()) if adv.numel() > 1 else 0.0,
            "adv_zero": float((adv.abs() < 1e-9).float().mean()),
            "r_std": float(R.std(dim=1).mean()),
        }

        if self.ref_on and self.cfg["decision"]["ref_kl"]["beta"] > 0:
            beta = self.cfg["decision"]["ref_kl"]["beta"]
            t = self.pool.primary_hist()
            with torch.no_grad():
                t_logits = _logits(t(X))
            # 复用主前向的 logits(带梯度), 不再为 ref 位单独跑一遍学生前向
            kl = F.kl_div(F.log_softmax(clean_logits, dim=1),
                          F.log_softmax(t_logits, dim=1),
                          log_target=True, reduction="batchmean")
            dec = dec + beta * kl
            stats["ref_kl"] = float(kl.detach())
        return dec, stats

    # ---- 表征层: 逐层特征 KL ----
    def _feature_loss(self, X):
        f = self.cfg["feature"]
        names = list(f["taps"])
        taps_s = _TapCapture(self.model, names)
        with _bn_eval(self.model), taps_s:
            _logits(self.model(X))              # 这一条前向为了拿中间层, 图必须保留
        feats_s = dict(taps_s.store)

        t = self.pool.primary_hist()
        taps_t = _TapCapture(t, names)
        # 教师侧也锁 BN: 否则学生用 running stats、教师用 batch stats(KL 会凭空变大)。
        # 正常情况教师已 eval, 这层保护对影子教师(与学生同结构)才有实际作用。
        with torch.no_grad(), _bn_eval(t), taps_t:
            _logits(t(X))
        feats_t = dict(taps_t.store)

        common = [n for n in names if n in feats_s and n in feats_t]
        if not common:
            self._warn_once("no_tap", f"taps={names} 在模型与教师上取不到, 特征层跳过")
            return None, {}

        loss = None
        per = {}
        kept = []          # [(name, 逐通道分布)], 用来查重
        for n in common:
            ps = self._chan_dist(feats_s[n])
            # P0: 去重 tap。resnet20 是 layer3 -> avg_pool2d(8,8) -> linear, 而
            # _chan_dist 对 4D 特征按空间均值降到逐通道分布, 于是
            # avg_pool 的 tap 恰好等于 layer3 的空间均值(实测两者 KL 完全相同
            # 0.1020/0.1020、0.0960/0.0960), 等于 layer3 被 alpha 计了两次。
            dup = None
            for m, ps_m in kept:
                if ps.shape == ps_m.shape and torch.allclose(
                        ps.detach(), ps_m.detach(), atol=1e-6, rtol=1e-4):
                    dup = m
                    break
            if dup is not None:
                self._warn_once(f"dup_{n}",
                                f"tap '{n}' 与 '{dup}' 的逐通道分布完全相同, 已去重"
                                f"(否则同一个量被 alpha 计两次); 建议从 feature.taps 删掉一个")
                continue
            kept.append((n, ps))
            with torch.no_grad():
                pt = self._chan_dist(feats_t[n])
            if f["divergence"] == "forward":
                kl = F.kl_div(torch.log(ps + 1e-8), pt, reduction="batchmean")
            else:
                kl = F.kl_div(torch.log(pt + 1e-8), ps, reduction="batchmean")
            w = self._tap_weight(n, f["budget"])
            term = f["alpha"] * w * kl
            loss = term if loss is None else loss + term
            per[n] = float(kl.detach())
        return loss, {f"feat_{k}": v for k, v in per.items()}

    @staticmethod
    def _chan_dist(feat):
        """[B,C,H,W] -> 每样本逐通道分布的 logits([B,C]); 已池化时原样返回。"""
        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        return F.softmax(feat, dim=1)

    def _tap_weight(self, name, budget):
        if budget == "uniform":
            return 1.0
        # sensitivity: 浅层权重更低(与"越靠后语义越强"的先验一致);
        # 真正的敏感度要用 E3 的逐层梯度范数来定, 这里是占位定义。
        order = {n: i for i, n in enumerate(["layer1", "layer2", "layer3", "layer4", "avg_pool"])}
        return 0.5 ** order.get(name, 0)

    # ================= 参数层: 投影 =================
    def project_weights(self, progress=1.0):
        """post-step ε-球投影: ||w - w_anchor|| <= ε * ||w_anchor||。

        注意: 剪枝仍在进行时, 被剪通道会从 0 变 0、未剪通道会漂移, 投影与剪枝
        会互相拉扯。所以这里只投影"存活且非零"的参数, 并把越界量打出来观察。
        """
        if not self.cfg["weight"]["enabled"] or not self.anchor:
            return {}
        w = self.cfg["weight"]
        eps = w["epsilon0"]
        if w["schedule"] == "tighten":
            kappa = max(float(w["kappa"]), 1e-6)
            # κ=1.0 时 κ^-progress 恒为 1, 收紧等于没开; 日志里把 eps 与 progress
            # 一起打出来, 就是为了让"有没有真的收紧"在日志里可验证。
            eps = w["epsilon0"] * (kappa ** -float(progress))

        viol, clamped, skipped = 0, 0, 0
        with torch.no_grad():
            pairs = self._pairs
            deltas = [p.data - a for p, a in pairs]
            dn = torch.stack(torch._foreach_norm(deltas, 2.0))
            nnz_now = torch.stack([torch.count_nonzero(p.data).double() for p, _ in pairs])
            an = self._anchor_norm_t
            # 结构变了(剪枝把某些通道置零)就跳过: 此时"偏离"是剪枝造成的,
            # 不是训练漂移, 投影它等于把剪枝撤销。实测在 joint 段每步会
            # 误伤 43 个张量, 所以这条判据是必需的, 不是可选优化。
            valid = (an > 0) & (nnz_now == self._anchor_nnz_t)
            limit = eps * an
            over = valid & (dn > limit)
            viol = int(over.sum())
            skipped = int((~valid).sum())
            idx = over.nonzero().flatten().tolist()   # 整个投影只在这里同步一次
            dev = (dn / an)[valid]                    # 逐层偏离量, 机制是否真的在约束就看它
            dev_mean = float(dev.mean()) if dev.numel() else 0.0
            dev_max = float(dev.max()) if dev.numel() else 0.0
            if idx:
                # p + delta*(limit/dn - 1) == a + delta*(limit/dn), 不需要再减一次 a
                ratios = (limit[idx] / dn[idx]) - 1.0
                sub_deltas = [deltas[i] for i in idx]
                sub_params = [pairs[i][0].data for i in idx]
                torch._foreach_mul_(sub_deltas, ratios.unbind())
                torch._foreach_add_(sub_params, sub_deltas)
                clamped = len(idx)
        self._note_event(skipped)
        return {"w_viol": viol, "w_clamped": clamped, "w_skip": skipped,
                "eps": eps, "prog": float(progress),
                "w_dev_mean": dev_mean, "w_dev_max": dev_max}

    # ---- 机制体检: 中间量落盘 ----
    def emit_diag(self, epoch, step, stats, extra=None):
        """每 N 步写一条 JSONL。判"机制是否合理"用的是这里的量, 不是 val acc:
        组内退化率 D_group 说明 GRPO 有没有在空转; adv_std/adv_zero 说明优势是否
        塌成常数; w_dev_mean/max 说明参数层约束是真在拉还是在放; 逐层 feat_* 给出
        层间偏离比; ref_kl 给出压缩后策略漂移。
        """
        if not self._jsonl_on:
            return
        if step % self._jsonl_every:
            return
        rec = {"epoch": epoch, "step": step}
        rec.update(stats)
        if extra:
            rec.update(extra)
        try:
            d = os.path.dirname(self._diag_path)
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            with open(self._diag_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        except Exception as e:                     # 落盘失败不该拖垮训练
            self._warn_once("diag_io", f"机制体检 JSONL 写不进去({e}), 已关闭")
            self._jsonl_on = False

    def _warn_once(self, key, msg):
        if key in self._warned:
            return
        self._warned.add(key)
        self.log("DISTILL", "[WARN] " + msg)
