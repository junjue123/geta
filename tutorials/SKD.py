import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax


class SelfDistiller:
    """
    自蒸馏工具类，用于实现模型训练过程中的自蒸馏机制。
    核心思想：利用历史训练阶段的模型作为"教师"，当前模型作为"学生"，通过多维度损失函数引导学生学习教师的知识，
    同时针对高不确定性样本进行增强学习，提升模型的稳定性和泛化能力。
    """

    def __init__(self, temperature=1.0, loss_weight=0.5, feat_weight=0.3,
                 contrast_weight=0.1, mixup_alpha=1.0, num_rounds=2):
        """
        初始化自蒸馏器的超参数和损失函数

        参数:
            temperature: 温度系数，用于软化概率分布，值越大分布越平缓
            loss_weight: KL散度损失（输出层蒸馏）的权重
            feat_weight: MSE损失（中间特征蒸馏）的权重
            contrast_weight: 对比损失的权重，用于区分高/低质量教师模型
            mixup_alpha: Mixup数据增强的Beta分布参数，控制混合强度
            num_rounds: 多轮蒸馏的次数，每轮对高不确定性样本进行不同的Mixup增强
        """
        self.temperature = temperature  # 温度系数，用于软化概率分布
        self.loss_weight = loss_weight  # KL散度损失权重
        self.feat_weight = feat_weight  # 特征蒸馏损失权重
        self.contrast_weight = contrast_weight  # 对比损失权重
        self.mixup_alpha = mixup_alpha  # Mixup的Beta分布参数
        self.num_rounds = num_rounds  # 多轮蒸馏次数

        # 初始化损失函数：KL散度（用于输出分布对齐）和MSE（用于特征对齐）
        self.kld_loss_fn = nn.KLDivLoss(reduction="batchmean")  # KL散度损失，批量平均
        self.mse_loss_fn = nn.MSELoss(reduction="mean")  # MSE损失，均值计算

        self.model = None  # 当前训练的模型，需外部传入

    def compute_uncertainty_weights(self, hist_preds, current_pred):
        """
        计算历史模型的可靠性权重，并构建对比损失（区分高/低质量教师）

        参数:
            hist_preds: 历史模型的预测结果列表（已软化）
            current_pred: 当前模型的原始预测结果（未软化）

        返回:
            weights: 每个历史模型的权重（基于可靠性和一致性）
            contrast_loss: 对比损失，引导当前模型向高确定性历史模型对齐
        """
        # 1. 计算每个历史模型预测的熵（熵越低，预测越确定，可靠性越高）
        entropies = torch.stack([
            -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()  # 计算单模型的平均熵
            for p in hist_preds
        ])

        # 2. 计算当前预测与历史预测的KL散度（散度越小，两者一致性越高）
        current_log_pred = log_softmax(current_pred / self.temperature, dim=1)  # 当前预测软化并取log
        kl_divs = torch.stack([
            self.kld_loss_fn(current_log_pred, p)  # 单模型的KL散度
            for p in hist_preds
        ])

        # 3. 计算权重：可靠性（基于熵）× 一致性（基于KL散度）
        reliability = 1 - F.softmax(entropies, dim=0)  # 熵越低，可靠性权重越高
        consistency = 1 - F.softmax(kl_divs, dim=0)  # KL散度越小，一致性权重越高
        weights = reliability * consistency  # 综合权重

        # 过滤低质量教师（权重低于0.1的置为0）并归一化，避免受不可靠模型影响
        weights[weights < 0.1] = 0
        weights = weights / (weights.sum() + 1e-8)  # 确保权重和为1

        # 4. 计算对比损失：让当前预测更接近高确定性历史模型，远离低确定性模型
        contrast_loss = 0.0
        if len(entropies) > 0:
            # 筛选低熵（高确定性）的历史模型作为正样本（取前50%）
            low_ent_idx = entropies.argsort()[:len(entropies) // 2]
            for idx in low_ent_idx:
                pos = hist_preds[idx]  # 正样本：高确定性历史模型的预测

                # 负样本：其他历史模型的平均预测（排除当前正样本）
                neg = torch.mean(torch.stack([
                    p for i, p in enumerate(hist_preds) if i != idx
                ]), dim=0)

                # 计算当前预测与正/负样本的余弦相似度
                sim_pos = F.cosine_similarity(current_pred, pos, dim=1).mean()
                sim_neg = F.cosine_similarity(current_pred, neg, dim=1).mean()

                # InfoNCE-like损失：最大化与正样本的相似度，最小化与负样本的相似度
                contrast_loss += -torch.log(
                    torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg) + 1e-8)
                )
            # 平均对比损失（避免受样本数量影响）
            contrast_loss /= max(1, len(low_ent_idx))

        return weights, contrast_loss

    def compute_distill_loss(self, current_y_pred, current_feature, current_model, history_models, x):
        """
        计算自蒸馏总损失，整合输出层蒸馏、特征蒸馏和对比损失，并针对高不确定性样本进行多轮Mixup增强

        参数:
            current_y_pred: 当前模型的输出预测（logits）
            current_feature: 当前模型的中间特征
            current_model: 当前训练的模型
            history_models: 历史模型列表（教师模型）
            x: 输入样本

        返回:
            total_loss: 自蒸馏总损失
        """
        # 若没有历史模型，无需蒸馏，返回0损失
        if not history_models:
            return torch.tensor(0.0, device=x.device)

        self.model = current_model  # 记录当前模型，用于后续Mixup推理
        total_loss = 0.0

        # --------------------------
        # 多轮共享的计算（仅执行一次，减少冗余）
        # --------------------------
        # 1. 提取所有历史模型的预测和特征（带梯度关闭，避免影响历史模型）
        hist_preds, hist_features = [], []
        with torch.no_grad():  # 历史模型参数不参与梯度更新
            for hist_model in history_models:
                # 假设模型forward方法支持返回特征（feature_need=True时）
                hist_y_pred, hist_feat = hist_model.forward(x, feature_need=True)
                # 软化历史模型的预测（用于后续KL散度计算）
                hist_preds.append(softmax(hist_y_pred / self.temperature, dim=1))
                hist_features.append(hist_feat)  # 保存中间特征

        # 2. 计算历史模型的权重和对比损失（共享计算）
        weights, contrast_loss = self.compute_uncertainty_weights(hist_preds, current_y_pred)

        # 3. 筛选高不确定性样本（熵高于均值的样本）
        current_probs = softmax(current_y_pred, dim=1)  # 当前模型的概率分布
        # 计算每个样本的熵（熵越高，模型对该样本的预测越不确定）
        current_entropy = -torch.sum(current_probs * log_softmax(current_y_pred, dim=1), dim=1)
        high_uncert_mask = current_entropy > current_entropy.mean()  # 高不确定性样本的掩码
        # 获取高不确定性样本的索引（用于后续Mixup）
        high_uncert_idx = torch.where(high_uncert_mask)[0] if high_uncert_mask.any() else None

        # 4. 预计算加权历史特征（特征蒸馏的目标，权重由历史模型可靠性决定）
        weighted_hist_feat = torch.sum(
            torch.stack([w * f for w, f in zip(weights, hist_features)]),  # 权重×特征
            dim=0  # 按模型维度求和，得到最终特征目标
        )

        # 5. 预计算加权历史预测（输出蒸馏的目标）
        weighted_hist_pred = torch.sum(
            torch.stack([w * p for w, p in zip(weights, hist_preds)]),  # 权重×预测分布
            dim=0  # 按模型维度求和，得到最终预测目标
        )

        # --------------------------
        # 多轮蒸馏循环（每轮参数不同，增强鲁棒性）
        # --------------------------
        for _ in range(self.num_rounds):
            # 复制当前预测和特征（避免多轮之间的梯度干扰）
            curr_pred = current_y_pred.clone()
            curr_feat = current_feature.clone()

            # 对高不确定性样本执行Mixup增强（每轮生成不同的混合系数）
            if high_uncert_mask.any():
                # 每轮生成新的Mixup系数（基于Beta分布），确保增强多样性
                lam = torch.distributions.beta.Beta(
                    self.mixup_alpha, self.mixup_alpha
                ).sample((len(high_uncert_idx),)).to(x.device)  # 生成与高不确定性样本数量匹配的系数
                lam = lam.view(-1, 1, 1, 1)  # 调整维度以适配输入（假设输入是4D张量，如图片）

                # 混合样本：将高不确定性样本与自身翻转的样本混合（增强难样本的学习）
                x_mix = x.clone()  # 复制原始输入
                # 混合公式：lam×原始样本 + (1-lam)×翻转样本（翻转顺序增强多样性）
                x_mix[high_uncert_mask] = lam * x[high_uncert_mask] + (1 - lam) * x[high_uncert_mask.flip(0)]

                # 用混合样本推理，更新高不确定性样本的预测和特征（开启梯度，确保损失可回传）
                with torch.set_grad_enabled(True):
                    mix_pred, mix_feat = self.model.forward(x_mix, feature_need=True)
                # 更新当前轮的预测和特征（仅替换高不确定性样本部分）
                curr_pred[high_uncert_mask] = mix_pred[high_uncert_mask]
                curr_feat[high_uncert_mask] = mix_feat[high_uncert_mask]

            # 计算当前轮的各损失分量
            # 输出层蒸馏损失：当前预测与加权历史预测的KL散度（乘温度平方还原梯度尺度）
            current_log_pred = log_softmax(curr_pred / self.temperature, dim=1)
            kld_loss = self.kld_loss_fn(current_log_pred, weighted_hist_pred) * (self.temperature ** 2)

            # 特征蒸馏损失：当前特征与加权历史特征的MSE
            feat_loss = self.mse_loss_fn(curr_feat, weighted_hist_feat)

            # 累加总损失（多轮平均，避免轮次数量影响损失尺度）
            total_loss += (
                                  self.loss_weight * kld_loss +  # 输出蒸馏损失
                                  self.feat_weight * feat_loss +  # 特征蒸馏损失
                                  self.contrast_weight * contrast_loss  # 对比损失
                          ) / self.num_rounds  # 平均到每轮

        return total_loss