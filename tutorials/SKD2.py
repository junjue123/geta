import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, softmax


class SelfDistiller:
    """
    自蒸馏工具类，将三种损失拆分为独立函数：
    1. 标签蒸馏损失（KL散度）
    2. 特征蒸馏损失（MSE+余弦相似度）
    3. 模型权重蒸馏损失（L2参数约束）
    """

    def __init__(self, temperature=1.0,
                 label_loss_weight=0.5,  # 标签蒸馏损失权重（原loss_weight）
                 feature_loss_weight=0.3,  # 特征蒸馏损失权重（原feat_weight）
                 weight_distill_loss_weight=0.2,  # 权重蒸馏损失权重（原weight_distill_weight）
                 mixup_alpha=1.0, num_rounds=2,
                 use_feat_distill=True, use_weight_distill=True):
        """初始化自蒸馏器参数"""
        self.temperature = temperature
        self.label_loss_weight = label_loss_weight  # 标签蒸馏损失权重
        self.feature_loss_weight = feature_loss_weight  # 特征蒸馏损失权重
        self.weight_distill_loss_weight = weight_distill_loss_weight  # 权重蒸馏损失权重
        self.mixup_alpha = mixup_alpha
        self.num_rounds = num_rounds
        self.use_feat_distill = use_feat_distill
        self.use_weight_distill = use_weight_distill

        # 损失函数初始化
        self.kld_loss_fn = nn.KLDivLoss(reduction="none")
        self.mse_loss_fn = nn.MSELoss(reduction="none")
        self.l2_loss_fn = nn.MSELoss(reduction="mean")

        self.model = None  # 当前训练模型

    def compute_sample_wise_teacher_weight(self, hist_preds):
        """计算样本级教师权重（基于预测熵）"""
        teacher_sample_weights = []
        for pred in hist_preds:
            sample_entropy = -torch.sum(pred * torch.log(pred + 1e-8), dim=1)  # [batch_size]
            min_ent, max_ent = sample_entropy.min(), sample_entropy.max()
            if max_ent > min_ent:
                sample_weight = 1 - (sample_entropy - min_ent) / (max_ent - min_ent)
            else:
                sample_weight = torch.ones_like(sample_entropy)
            teacher_sample_weights.append(sample_weight)
        return torch.stack(teacher_sample_weights, dim=0)  # [num_teachers, batch_size]

    # --------------------------
    # 1. 标签蒸馏损失（独立函数）
    # --------------------------
    def _compute_label_distill_loss(self, curr_pred, hist_preds, teacher_sample_weights):
        """
        计算标签蒸馏损失（KL散度）
        参数:
            curr_pred: 学生模型预测 (logits, [batch_size, num_classes])
            hist_preds: 教师模型预测列表 ([num_teachers, batch_size, num_classes])
            teacher_sample_weights: 样本级教师权重 ([num_teachers, batch_size])
        返回:
            label_loss: 标签蒸馏损失
        """
        curr_log_pred = log_softmax(curr_pred / self.temperature, dim=1)  # 学生预测软化
        total_kld_loss = 0.0

        for i in range(hist_preds.shape[0]):
            # 计算单个教师的KL损失（样本级）
            kld_per_sample = self.kld_loss_fn(curr_log_pred, hist_preds[i]).sum(dim=1)  # [batch_size]
            # 样本级加权（教师确定性越高，权重越大）
            weighted_kld = (teacher_sample_weights[i] * kld_per_sample).mean()
            total_kld_loss += weighted_kld

        # 平均所有教师的损失，并乘温度平方（还原梯度尺度）
        return (total_kld_loss / hist_preds.shape[0]) * (self.temperature ** 2)

    # --------------------------
    # 2. 特征蒸馏损失（独立函数）
    # --------------------------
    def _compute_feature_distill_loss(self, curr_feat, hist_features, teacher_sample_weights):
        """
        计算特征蒸馏损失（MSE+余弦相似度）
        参数:
            curr_feat: 学生模型特征 ([batch_size, feat_dim, H, W] 或 [batch_size, feat_dim])
            hist_features: 教师模型特征列表 ([num_teachers, batch_size, feat_dim, H, W])
            teacher_sample_weights: 样本级教师权重 ([num_teachers, batch_size])
        返回:
            feat_loss: 特征蒸馏损失
        """
        total_feat_loss = 0.0

        for i in range(hist_features.shape[0]):
            # MSE损失（样本级，按特征维度平均）
            if curr_feat.dim() == 4:  # 卷积特征 (B, C, H, W)
                mse_per_sample = self.mse_loss_fn(curr_feat, hist_features[i].detach()).mean(dim=[1, 2, 3])
            else:  # 全连接特征 (B, C)
                mse_per_sample = self.mse_loss_fn(curr_feat, hist_features[i].detach()).mean(dim=1)

            # 余弦相似度损失（样本级，相似度越低损失越大）
            cos_sim = F.cosine_similarity(
                curr_feat.flatten(1),  # 展平特征维度
                hist_features[i].detach().flatten(1),
                dim=1
            )
            cos_loss_per_sample = 1 - cos_sim  # [batch_size]

            # 合并损失并按样本权重加权
            feat_loss_per_sample = mse_per_sample + cos_loss_per_sample
            weighted_feat_loss = (teacher_sample_weights[i] * feat_loss_per_sample).mean()
            total_feat_loss += weighted_feat_loss

        # 平均所有教师的损失
        return total_feat_loss / hist_features.shape[0]

    # --------------------------
    # 3. 模型权重蒸馏损失（独立函数）
    # --------------------------
    def _compute_weight_distill_loss(self, current_model, history_models):
        """
        计算模型权重蒸馏损失（L2参数约束）
        参数:
            current_model: 当前学生模型
            history_models: 教师模型列表
        返回:
            weight_loss: 模型权重蒸馏损失
        """
        if not history_models:
            return torch.tensor(0.0, device=next(current_model.parameters()).device)

        total_weight_loss = 0.0
        for hist_model in history_models:
            layer_loss = 0.0
            # 计算可训练参数的L2损失
            for (curr_param, hist_param) in zip(current_model.parameters(), hist_model.parameters()):
                if curr_param.requires_grad:
                    layer_loss += self.l2_loss_fn(curr_param, hist_param.detach())
            # 平均到每一层参数
            total_weight_loss += layer_loss / len(list(current_model.parameters()))

        # 平均所有教师的损失
        return total_weight_loss / len(history_models)

    # --------------------------
    # 主函数：整合三种损失
    # --------------------------
    def compute_distill_loss(self, current_y_pred, current_feature, current_model, history_models, x):
        """
        整合三种蒸馏损失，返回总损失和各分量
        参数:
            current_y_pred: 学生模型预测 (logits)
            current_feature: 学生模型中间特征
            current_model: 当前学生模型
            history_models: 教师模型列表
            x: 输入样本
        返回:
            total_loss: 总蒸馏损失
            label_loss: 标签蒸馏损失分量
            feat_loss: 特征蒸馏损失分量
            weight_loss: 模型权重蒸馏损失分量
        """
        if not history_models:
            zero_tensor = torch.tensor(0.0, device=x.device)
            return zero_tensor, zero_tensor, zero_tensor, zero_tensor

        self.model = current_model
        total_loss = 0.0
        total_label_loss = 0.0
        total_feat_loss = 0.0
        total_weight_loss = 0.0

        # 预计算教师模型的预测和特征
        hist_preds, hist_features = [], []
        with torch.no_grad():
            for hist_model in history_models:
                hist_y_pred, hist_feat = hist_model.forward(x, feature_need=True)
                hist_preds.append(softmax(hist_y_pred / self.temperature, dim=1))
                hist_features.append(hist_feat)
        hist_preds = torch.stack(hist_preds, dim=0)
        hist_features = torch.stack(hist_features, dim=0)

        # 计算样本级教师权重（所有轮次共享）
        teacher_sample_weights = self.compute_sample_wise_teacher_weight(hist_preds)

        # 多轮蒸馏
        for _ in range(self.num_rounds):
            curr_pred = current_y_pred.clone()
            curr_feat = current_feature.clone()

            # 高不确定性样本Mixup增强
            current_probs = softmax(curr_pred, dim=1)
            current_sample_entropy = -torch.sum(current_probs * log_softmax(curr_pred, dim=1), dim=1)
            high_uncert_mask = current_sample_entropy > current_sample_entropy.mean()

            if high_uncert_mask.any():
                batch_size = x.shape[0]
                mixup_idx = torch.randperm(batch_size, device=x.device)
                lam = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
                x_mix = lam * x + (1 - lam) * x[mixup_idx]

                with torch.set_grad_enabled(True):
                    mix_pred, mix_feat = self.model.forward(x_mix, feature_need=True)
                curr_pred[high_uncert_mask] = mix_pred[high_uncert_mask]
                curr_feat[high_uncert_mask] = mix_feat[high_uncert_mask]

            # 1. 计算标签蒸馏损失（调用独立函数）
            round_label_loss = self._compute_label_distill_loss(
                curr_pred, hist_preds, teacher_sample_weights
            )
            total_label_loss += round_label_loss / self.num_rounds

            # 2. 计算特征蒸馏损失（调用独立函数）
            round_feat_loss = torch.tensor(0.0, device=x.device)
            if self.use_feat_distill:
                round_feat_loss = self._compute_feature_distill_loss(
                    curr_feat, hist_features, teacher_sample_weights
                )
                total_feat_loss += round_feat_loss / self.num_rounds

            # 3. 计算模型权重蒸馏损失（调用独立函数）
            round_weight_loss = self._compute_weight_distill_loss(current_model, history_models)
            total_weight_loss += round_weight_loss / self.num_rounds

            # 本轮总损失
            round_total_loss = (
                    self.label_loss_weight * round_label_loss +  # 使用修改后的权重名称
                    (self.feature_loss_weight * round_feat_loss if self.use_feat_distill else 0.0) +  # 使用修改后的权重名称
                    (self.weight_distill_loss_weight * round_weight_loss if self.use_weight_distill else 0.0)  # 使用修改后的权重名称
            )
            total_loss += round_total_loss / self.num_rounds

        return total_loss, total_label_loss, total_feat_loss, total_weight_loss