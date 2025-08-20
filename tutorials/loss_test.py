import torch
import torch.nn.functional as F
class TotalLoss(torch.nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.1, threshold=0.9, num_classes=10,
        start_projection_epoch=20, start_pruning_epoch=25, projection_epochs=50, pruning_epochs=50):
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
        self.pruning_epochs= pruning_epochs

    def rank_regularization(self, weight, depth_factor, is_shallow=True):
        """
        计算单个权重矩阵的秩正则化项
        depth_factor: 深度因子，随网络加深而变化
        is_shallow: 是否为浅层，用于区分正则化策略
        """
        _, s, _ = torch.svd(weight, compute_uv=False)
        max_s = s.max()
        mean_s = s.mean()

        # 避免除以零
        if mean_s.item() < 1e-8:
            return torch.tensor(0.0, device=weight.device)

        ratio = max_s / mean_s

        # 对于浅层，鼓励更多奇异值参与，增强泛化性（放宽秩约束）
        if is_shallow:
            # 浅层使用更平缓的函数，允许更大的秩
            reg_term = torch.sigmoid(torch.square((1 / ratio) - 1))
            # 深度因子调制，浅层权重更低
        else:
            # 对于深层，鼓励秩降低，突出主要特征，增强分类能力
            reg_term = torch.sigmoid(torch.square(ratio - 1))
            # 深层权重更高，更强约束
        return self.lambda1 * depth_factor * reg_term

    def rank_compensation_regularizer(self, model):
        """
        计算模型的总秩补偿正则化项
        自动搜索所有层并根据深度动态调整
        """
        regularization = 0.0

        # 获取模型所有可训练的权重层
        weight_layers = []
        for name, module in model.named_modules():
            # 筛选常见的权重层类型
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                weight_layers.append(module)

        total_layers = len(weight_layers)
        if total_layers == 0:
            return regularization

        # 遍历所有层，根据深度应用不同正则化策略
        for i, layer in enumerate(weight_layers):
            # 计算深度因子（0到1之间），越深层值越大
            depth_factor = (i+1) / total_layers if total_layers > 1 else 0.5

            # 判断是否为浅层（前1/3为浅层）
            is_shallow = i < total_layers / 3

            # 累加正则化项
            regularization += self.rank_regularization(
                layer.weight,
                depth_factor,
                is_shallow
            )/total_layers

        return regularization

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
        entropy = -torch.log(true_probs + 1e-10)  # 加小值避免log(0)

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

        # 处理易样本：使用信息熵补值作为权重
        if easy_samples.any():
            max_entropy = torch.log(torch.tensor(num_classes, device=entropy.device))
            normalized_entropy = entropy[easy_samples] / max_entropy
            easy_weights = 1.0 - normalized_entropy  # 确定性越高，权重越小
            base_loss[easy_samples] *= easy_weights

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

            # 计算特征与所有类原型的余弦相似度
            feature_norm = torch.norm(feature, dim=1, keepdim=True) + self.epsilon
            prototype_norm = torch.norm(prototypes, dim=1, keepdim=True) + self.epsilon

            normalized_feature = feature / feature_norm
            normalized_prototype = prototypes / prototype_norm
            flat_feature = normalized_feature.view(normalized_feature.size(0), -1)
            flat_prototype = normalized_prototype.view(normalized_prototype.size(0), -1)
            similarity_matrix = torch.matmul(flat_feature, flat_prototype.T)  # [batch_size, num_classes]

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
                    correct_entropy = -torch.log(correct_prob + self.epsilon)/self.num_classes  # 单个概率的熵

                    # 计算其他类别的预测概率及其信息熵
                    other_probs = y_pred_probs[i, ~(torch.arange(self.num_classes, device=true_label.device) == true_label)]
                    other_probs_normalized = other_probs / (torch.sum(other_probs) + self.epsilon)  # 归一化其他类概率
                    other_entropy = -torch.sum(other_probs_normalized * torch.log(other_probs_normalized + self.epsilon))

                    # 信息熵差距：其他类预测的熵 - 正确类预测的熵
                    # 差距越大，说明模型对其他类的预测越分散，样本越难
                    entropy_gap = other_entropy - correct_entropy

                    # 难易程度权重：越难的样本权重越高
                    # 使用指数函数放大差距的影响，确保难样本获得更高权重
                    difficulty_weight = torch.sigmoid(entropy_gap)

                    # 计算与真实类原型的相似度和其他类原型的相似度
                    true_sim = torch.exp(similarity_matrix[i, true_label])
                    sum_sim = torch.sum(torch.exp(similarity_matrix[i]))

                    # 可信样本损失：拉近与真实类原型，推远其他类原型
                    loss_i = - difficulty_weight * torch.log(true_sim/sum_sim)

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
                    loss_i = -torch.log(true_sim/(wrong_sim*sum_sim))

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



# 构建简易模型（仅包含linear和conv1层，用于秩正则化测试）
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 3)  # 权重 shape: [3,10]
        self.conv1 = torch.nn.Conv2d(3, 5, kernel_size=3)  # 权重 shape: [5,3,3,3]



if __name__ == "__main__":
    # 初始化损失函数（使用默认参数）
    loss_fn = TotalLoss(lambda1=0.01, lambda2=0.1, threshold=0.9)

    # ------------------------------
    # 1. 构建测试数据（涵盖多种场景）
    # ------------------------------
    # 假设分类任务有3个类别（0,1,2），批次大小为5
    batch_size = 5
    num_classes = 3

    # 构建预测值（logits）：包含多种情况
    # - 样本0：预测正确，高置信度（>0.9）→ 易样本
    # - 样本1：预测正确，低置信度（<0.9）→ 普通正确样本
    # - 样本2：预测错误，真实类别概率与预测最高概率差距大（难样本）
    # - 样本3：预测错误，真实类别概率与预测最高概率差距小（较易错误样本）
    # - 样本4：预测错误，真实类别概率极低（极难样本）
    y_pred = torch.tensor([
        [0.1, 0.1, 8.0],  # 样本0：真实类别2，预测概率≈1.0（softmax后）
        [0.2, 5.0, 0.3],  # 样本1：真实类别1，预测概率≈0.95（<0.9？不，0.95>0.9，修正为低置信度示例）
        [6.0, 0.2, 0.3],  # 样本2：真实类别1，预测类别0（差距大）
        [2.8, 2.9, 0.1],  # 样本3：真实类别1，预测类别1（差距小）
        [10.0, 0.1, 0.1]  # 样本4：真实类别2，预测类别0（差距极大）
    ], dtype=torch.float32)

    # 真实标签（对应上述样本设计）
    y = torch.tensor([2, 1, 1, 1, 2], dtype=torch.long)
    features = torch.tensor([
        # 样本0：真实类别2，预测正确
        [1.2, 0.8, 3.5, 0.1, 2.9],  # 具有类别2的典型特征

        # 样本1：真实类别1，预测正确但置信度不高
        [0.7, 2.1, 1.3, 1.8, 0.5],  # 具有类别1的特征，但不够典型

        # 样本2：真实类别1，预测为类别0（差距大）
        [2.3, 0.9, 0.6, 1.5, 0.8],  # 特征混合了类别1和0的特点，但更偏向0

        # 样本3：真实类别1，预测为类别0（差距小）
        [1.8, 1.7, 1.1, 1.6, 0.7],  # 特征接近类别1和0的边界

        # 样本4：真实类别2，预测为类别0（差距极大）
        [2.5, 0.7, 0.4, 0.3, 0.6]  # 特征被严重误判为类别0
    ], dtype=torch.float32)
    from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
    model = resnet20_cifar10()

    # ------------------------------
    # 2. 计算各项损失
    # ------------------------------
    celoss, rcrloss, conloss = loss_fn.total_loss(y_pred, y, model, features, RCR=True, CON=False)
    CE = torch.nn.CrossEntropyLoss()
    normalloss = CE(y_pred, y)

    # ------------------------------
    # 3. 验证损失逻辑（打印关键中间结果）
    # ------------------------------
    print("===== 测试样本基本信息 =====")
    probs = F.softmax(y_pred, dim=1)
    pred_classes = torch.argmax(probs, dim=1)
    for i in range(batch_size):
        print(f"样本{i}：")
        print(f"  预测概率：{probs[i].detach().numpy().round(4)}")
        print(f"  预测类别：{pred_classes[i].item()}，真实类别：{y[i].item()}")
        print(f"  是否难样本：{pred_classes[i] != y[i]}")
        print(f"  是否易样本：{(pred_classes[i] == y[i]) & (probs[i, y[i]] >= 0.9)}\n")

    print("===== 损失计算结果 =====")
    print(f"动态交叉熵损失（CELoss）：{celoss.item():.4f}")
    print(f"秩补偿正则化损失（RCRLoss）：{rcrloss.item():.4f}")
    print(f"均匀分布正则化损失（UDRLoss）：{conloss.item():.4f}")
    print(f"交叉熵损失：{normalloss.item():.4f}")
