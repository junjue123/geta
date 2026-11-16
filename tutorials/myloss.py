import torch
import torch.nn.functional as F
class TotalLoss(torch.nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.1, threshold=0.9, num_classes=10,
                 start_projection_epoch=20, start_pruning_epoch=25, projection_epochs=50, pruning_epochs=50,
                 temperature=0.07):
        """
        初始化秩正则化损失计算器

        参数:
            lambda_: 正则化系数
        """
        # 调用父类的初始化方法
        super(TotalLoss, self).__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.celoss = 0.0
        self.rcrloss = 0.0
        self.conloss = 0.0
        self.threshold = threshold
        self.class_prototypes = {}
        self.num_classes = num_classes
        self.epsilon = 1e-8
        self.start_projection_epoch = start_projection_epoch
        self.start_pruning_epoch = start_pruning_epoch
        self.projection_epochs = projection_epochs
        self.pruning_epochs = pruning_epochs
        self.temperature = temperature

    def rank_regularization(self, weight, depth_factor):
        """
        计算单个权重矩阵的秩正则化项
        depth_factor: 深度因子，范围[-1, 1]，随网络加深从-1线性增加到1
        """
        # 计算SVD，但只需要奇异值
        _, s, _ = torch.svd(weight, compute_uv=False)

        max_s = s.max()
        mean_s = s.mean()

        # 使用掩码避免分支判断，提高计算效率
        mean_safe = mean_s.clamp(min=self.epsilon)
        ratio = max_s / mean_safe

        # 预计算常用项
        ratio_minus_1 = ratio - 1

        # 使用向量化操作替代条件判断
        # 浅层：depth_factor < 0，深层：depth_factor >= 0
        shallow_mask = depth_factor < 0

        # 初始化正则化项
        reg_term = torch.zeros_like(ratio)

        # 计算浅层正则化
        if shallow_mask:
            exp_val = torch.exp(ratio_minus_1)
            term = (exp_val- 1) * (-depth_factor)
            reg_term = torch.min(term, torch.tensor(5.0, device=weight.device))

        # 计算深层正则化
        else:
            reg_term = torch.exp(-ratio_minus_1 * depth_factor)

        return reg_term

    def rank_compensation_regularizer(self, model):
        """
        计算模型的总秩补偿正则化项
        自动搜索所有层并根据深度动态调整，优化版
        """
        # 获取模型所有可训练的权重层
        weight_layers = []
        for module in model.modules():  # 比named_modules()略快，因为不需要收集名称
            # 筛选常见的权重层类型
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                weight_layers.append(module.weight)  # 直接存储权重而非模块

        total_layers = len(weight_layers)
        if total_layers == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # 预计算所有深度因子
        depth_factors = (2 * torch.arange(total_layers, device=weight_layers[0].device)
                         / total_layers) - 1

        # 批量计算所有层的正则化项
        reg_terms = []
        for i, weight in enumerate(weight_layers):
            reg_terms.append(self.rank_regularization(weight, depth_factors[i]))

        # 求和并归一化
        total_reg = torch.stack(reg_terms).sum() / total_layers

        return self.lambda1 * total_reg

    def dynamic_cross_entropy(self, y_pred, y):
        """
        改进的动态交叉熵损失计算：
        - 对难样本同时利用正确类别概率和错误类别概率计算权重
        - 错误类别概率越高而正确类别概率越低的样本，权重越大
        - 对易样本保持基于信息熵的平滑策略
        """
        batch_size, num_classes = y_pred.shape

        # 获取预测概率和预测类别
        probs = F.softmax(y_pred, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        # 识别难样本和易样本
        hard_samples = (pred_classes != y)
        correct_preds = (pred_classes == y)
        true_probs = torch.gather(probs, 1, y.unsqueeze(1)).squeeze()
        high_confidence = (true_probs >= self.threshold)
        easy_samples = correct_preds & high_confidence

        # 基础交叉熵损失
        base_loss = self.criterion(y_pred, y)

        # 计算信息熵（反映对真实类别的不确定性）
        entropy = -torch.log(true_probs.clamp_min(self.epsilon))  # 加小值避免log(0)

        # 处理难样本：同时考虑正确类别概率和错误类别概率
        if hard_samples.any():
            # 获取错误预测类别的概率（模型认为最可能的错误类别）
            error_probs = torch.gather(probs, 1, pred_classes.unsqueeze(1)).squeeze()

            # 计算错误概率与正确概率的比值，反映模型的认知偏差程度
            # 比值越大，说明模型越倾向于错误类别，样本越需要关注
            prob_ratio = error_probs[hard_samples] / (true_probs[hard_samples] + 1e-10)

            # 结合信息熵和概率比值计算权重
            max_entropy = torch.log(torch.tensor(num_classes, device=entropy.device))
            normalized_entropy = entropy[hard_samples] / max_entropy

            # 最终权重：基础权重1.0 + 熵贡献 + 概率比值贡献
            hard_weights = 1.0 + normalized_entropy + torch.log1p(prob_ratio)
            base_loss[hard_samples] *= hard_weights

        # # 处理易样本：使用信息熵补值作为权重
        # if easy_samples.any():
        #     max_entropy = torch.log(torch.tensor(num_classes, device=entropy.device))
        #     normalized_entropy = entropy[easy_samples] / max_entropy
        #     easy_weights = 1.0 - normalized_entropy  # 确定性越高，权重越小
        #     base_loss[easy_samples] *= easy_weights

        return base_loss.mean()

    def contrastive_loss(self, y_pred, y, feature, epoch):
        """
        区分可信样本和挑战样本的对比损失函数

        参数:
            y_pred: 模型预测输出，形状为[batch_size, num_classes]
            y: 样本真实标签，形状为[batch_size]
            feature: 样本的特征表示，形状为[batch_size, feature_dim]
        """
        # 获取预测的类别和概率
        y_pred_probs = torch.softmax(y_pred, dim=1)  # 预测概率
        y_pred_max = torch.max(y_pred_probs, dim=1)
        y_pred_labels = y_pred_max.indices  # 预测类别

        batch_size = feature.shape[0]

        # 更新类原型（仅使用预测正确的样本）
        with torch.no_grad():
            for i in range(batch_size):
                true_label = y[i].item()
                pred_label = y_pred_labels[i].item()

                if true_label == pred_label:  # 预测正确才更新原型
                    # 分离特征的计算图，避免原型关联梯度
                    feat = feature[i].detach().clone()

                    if true_label not in self.class_prototypes:
                        # 初始化原型为第一个正确预测的样本特征
                        self.class_prototypes[true_label] = feat
                    else:
                        pred_prob = y_pred_probs[i, true_label].item()  # 转为标量避免计算图关联
                        pred_prob_tensor = torch.tensor(pred_prob)

                        # 根据不同阶段计算权重
                        if epoch < self.start_projection_epoch:
                            # 热身阶段：固定小权重
                            weight = 0.1
                        elif self.start_projection_epoch <= epoch < self.start_pruning_epoch:
                            # 投影阶段：基于预测概率的指数加权
                            weight = (torch.exp(pred_prob_tensor) - 1) / batch_size
                        elif self.start_pruning_epoch <= epoch < self.start_pruning_epoch + self.pruning_epochs:
                            # 修剪阶段：可以根据需求调整权重策略
                            weight = 0.05  # 示例值
                        elif epoch >= self.start_pruning_epoch + self.pruning_epochs and \
                                epoch >= self.start_projection_epoch + self.projection_epochs:
                            # 联合阶段：使用概率平方的指数加权
                            weight = (torch.exp(pred_prob_tensor ** 2) - 1) / batch_size
                        else:
                            # 冷却阶段：固定更小的权重
                            weight = 0.01
                        weight = torch.tensor(weight)
                        # 确保权重在合理范围内
                        weight = torch.clamp(torch.tensor(weight), 0.001, 0.2).item()

                        # 按预测概率加权更新原型
                        self.class_prototypes[true_label] = (
                                self.class_prototypes[true_label] * (1 - weight) +
                                feat * weight
                        )

        # 检查是否所有类别都有了原型，如果没有则返回0损失
        if len(self.class_prototypes) < self.num_classes:
            return 0.0

        else:

            # 转换类原型为张量
            prototypes = torch.stack([self.class_prototypes[c] for c in range(self.num_classes)], dim=0)

            # 计算特征与所有类原型的点积相似度（除以温度系数）
            # 移除向量归一化步骤，保留原始模长信息
            flat_feature = feature.view(feature.size(0), -1)  # 展平特征为[N, D]
            flat_prototype = prototypes.view(prototypes.size(0), -1)  # 展平原型为[C, D]
            # 对特征和原型进行L2归一化（余弦相似度需要）
            flat_feature_norm = torch.nn.functional.normalize(flat_feature, p=2, dim=1)  # [N, D]
            flat_prototype_norm = torch.nn.functional.normalize(flat_prototype, p=2, dim=1)  # [C, D]

            # 计算点积矩阵（形状为[N, C]）
            dot_product_matrix = torch.matmul(flat_feature_norm, flat_prototype_norm.T)

            # 除以温度系数得到最终相似度矩阵
            # 可根据任务调整的温度参数（如InfoNCE中常用0.07）
            similarity_matrix = dot_product_matrix / self.temperature  # [batch_size, num_classes]

            # 初始化总损失
            total_loss = 0.0
            sample_count = 0

            # 遍历每个样本计算损失
            for i in range(batch_size):
                true_label = y[i]
                pred_label = y_pred_labels[i]

                # 区分可信样本和挑战样本
                if true_label == pred_label:
                    # --------------------------
                    # 可信样本：预测正确的样本
                    # --------------------------
                    # 计算正确类别预测概率的信息熵（单个概率的熵）
                    correct_prob = y_pred_probs[i, true_label]
                    correct_entropy = -torch.log(correct_prob.clamp_min(self.epsilon))  # 单个概率的熵

                    # 计算其他类别的预测概率及其信息熵
                    other_probs = y_pred_probs[
                        i, ~(torch.arange(self.num_classes, device=true_label.device) == true_label)]

                    other_entropy = -torch.sum(torch.log(max(other_probs + self.epsilon))) / self.num_classes

                    # 信息熵差距：其他类预测的熵 - 正确类预测的熵
                    # 差距越大，说明模型对其他类的预测越分散，样本越难
                    entropy_gap = correct_entropy - other_entropy

                    # 难易程度权重：越难的样本权重越高
                    # 使用指数函数放大差距的影响，确保难样本获得更高权重
                    difficulty_weight = torch.sigmoid(entropy_gap)

                    # 计算与真实类原型的相似度和其他类原型的相似度
                    true_sim = torch.exp(similarity_matrix[i, true_label])
                    sum_sim = torch.sum(torch.exp(similarity_matrix[i]))

                    # 可信样本损失：拉近与真实类原型，推远其他类原型
                    loss_i = - difficulty_weight * torch.log(true_sim / sum_sim)

                else:
                    # --------------------------
                    # 挑战样本：预测错误的样本
                    # --------------------------
                    # 计算与真实类原型和错误预测类原型的相似度
                    true_sim = torch.exp(similarity_matrix[i, true_label])
                    wrong_sim = torch.exp(similarity_matrix[i, pred_label])
                    sum_sim = torch.sum(torch.exp(similarity_matrix[i]))

                    # 挑战样本损失：推远错误类原型，拉近真实类原型
                    # 同时最大化错误类原型和真实类原型之间的距离
                    loss_i = -torch.log(true_sim / (wrong_sim * sum_sim))

                total_loss += loss_i
                sample_count += 1

            # 返回平均损失
            return self.lambda2 * total_loss / sample_count if sample_count > 0 else 0.0

    def total_loss(self, y_pred, y, model, feature, epoch, RCR=True, CON=True):
        """计算总损失，确保所有返回值都是Tensor"""
        # 交叉熵损失（Tensor）
        self.celoss = self.dynamic_cross_entropy(y_pred, y)

        # 秩正则化损失（确保是Tensor）
        if RCR:
            reg = self.rank_compensation_regularizer(model)
            self.rcrloss = torch.max(torch.tensor(0.0, device=y_pred.device), torch.tensor(reg, device=y_pred.device))
        else:
            self.rcrloss = torch.tensor(0.0, device=y_pred.device)

        # 对比损失（确保是Tensor）
        if CON:
            con = self.contrastive_loss(y_pred, y, feature, epoch)
            self.conloss = torch.max(torch.tensor(0.0, device=y_pred.device), torch.tensor(con, device=y_pred.device))
        else:
            self.conloss = torch.tensor(0.0, device=y_pred.device)

        return self.celoss, self.rcrloss, self.conloss

