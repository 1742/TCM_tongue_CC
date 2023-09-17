import sys

import torch
from torch import nn
from torch.functional import F

import numpy as np
from itertools import combinations


class UncertaintyLoss_cls(nn.Module):
    def __init__(self, L=None, need_sigma: bool = False, reduction: str = 'mean'):
        super(UncertaintyLoss_cls, self).__init__()
        self.reduction = reduction

        if L:
            self.need_sigma = need_sigma
            self.criterion = L
        else:
            self.need_sigma = False
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor):
        """

        :param mu:
        :param sigma:
            预设模型预测的是log(sigma^2)
        :param labels:
        :param softmax:
        :param mode:
        :return:
        """
        logits = logits.cpu()
        sigma = sigma.pow(2).cpu()
        # 根据输入的标签类型转化成序号标签
        if len(labels.size()) != 1:
            labels = torch.argmax(labels, dim=1).long().cpu()
        else:
            labels = labels.long().cpu()
        bs, n_sample = logits.size(0), logits.size(1)

        # 对模型不确定约束下的结果加入数据不确定性，正则化，防止网络预测所有数据的无限不确定性或负数不确定性
        unc_loss = torch.zeros((bs, n_sample))
        for t in range(n_sample):
            if self.need_sigma:
                unc_loss[:, t] = 0.5 * (torch.exp(-1 * sigma.squeeze(1)) * self.criterion(logits[:, t, :], sigma, labels) + sigma.squeeze(1))
            else:
                unc_loss[:, t] = 0.5 * (torch.exp(-1 * sigma.squeeze(1)) * self.criterion(logits[:, t, :], labels) + sigma.squeeze(1))
        unc_loss = torch.mean(unc_loss, dim=1)

        if self.reduction == 'mean':
            return unc_loss.mean()
        elif self.reduction == 'sum':
            return unc_loss.sum()
        elif self.reduction == 'none':
            return unc_loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1, sample_method: str = 'random', reduction: str = 'mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin

        if sample_method == 'random':
            self.sample_method = self.random_hard_negative
        elif sample_method == 'hard':
            self.sample_method = self.hardest_negative
        elif sample_method == 'semi':
            self.sample_method = self.semihard_negative

        self.reduction = reduction
        self.triplet_loss = nn.TripletMarginLoss(margin, reduction='none')

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        triplets = self._get_triplets(embeddings, labels)
        triplets_num = triplets.size(0)
        triplet_loss = self.triplet_loss(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]], embeddings[triplets[:, 2]])

        if self.reduction == 'mean':
            return triplet_loss.mean(), triplets_num
        elif self.reduction == 'sum':
            return triplet_loss.sum(), triplets_num
        elif self.reduction == 'none':
            return triplet_loss, triplets_num

    def _get_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor):
        distance_matrix = F.pairwise_distance(embeddings.unsqueeze(0), embeddings.unsqueeze(1), 2)
        distance_matrix = distance_matrix.cpu().detach().numpy()

        labels = labels.cpu().detach().numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            # 得到所有与label相同的样本的索引
            label_indices = np.where(label_mask)[0]
            # 跳过只有一组的triplet类别
            if len(label_indices) < 2:
                continue
            # 得到所有与label不同的样本的索引
            negative_indices = np.where(np.logical_not(label_mask))[0]
            # 得到所有anchor-positive对
            # anchor_positive维度为(N, 2)，N为对数，第二维第0位指anchor索引，第1位指与anchor对应的positive的索引
            # itertools.combinations能列出以给定列表为元素，长度为r的所有组合
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)

            # 取出所有anchor-positive对的欧氏距离
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                # loss_values 计算相对该anchor的正样本的距离 - 相对该anchor的负样本的距离 + margin
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                # loss_values = loss_values.numpy()
                # 寻找loss_values大于0的anchor、positive、negative对
                hard_negative = self.sample_method(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

            # 若没有找到hard_negative，则返回默认的triplet组
            if len(triplets) == 0:
                triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

    def random_hard_negative(self, loss_values):
        hard_negatives = np.where(loss_values > 0)[0]
        return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

    def hardest_negative(self, loss_values):
        hard_negative = np.argmax(loss_values)
        return hard_negative if loss_values[hard_negative] > 0 else None

    def semihard_negative(self, loss_values):
        semihard_negatives = np.where(np.logical_and(loss_values < self.margin, loss_values > 0))[0]
        return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class Triplet_Uncertainty_Combined_Loss(TripletLoss):
    def __init__(self, margin: float = 1, sample_method: str = 'random', reduction: str = 'mean'):
        super(Triplet_Uncertainty_Combined_Loss, self).__init__(margin=margin, sample_method=sample_method, reduction=reduction)

    def forward(self, embeddings: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor):
        triplets = self._get_triplets(embeddings, labels)
        sigma = torch.mean(torch.squeeze(sigma)[triplets], dim=1).cpu()
        triplets_num = triplets.size(0)
        triplet_loss = self.triplet_loss(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]], embeddings[triplets[:, 2]])
        loss = 0.5 * (torch.exp(-1 * sigma) * triplet_loss + sigma)

        if self.reduction == 'mean':
            return loss.mean(), triplets_num
        elif self.reduction == 'sum':
            return loss.sum(), triplets_num
        elif self.reduction == 'none':
            return loss, triplets_num


class Triplet_and_CE_Loss_cls(nn.Module):
    def __init__(self, margin: float = 1, sample_method: str = 'random', beta: float = 0.5):
        super(Triplet_and_CE_Loss_cls, self).__init__()
        self.beta = beta
        self.triplet_loss = TripletLoss(margin, sample_method=sample_method)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, embeddings: torch.Tensor, labels: torch.Tensor):
        pred = pred.cpu()
        embeddings = embeddings.cpu()
        # 根据输入的标签类型转化成序号标签
        if labels.max() == 1 and pred.size(1) != 2:
            labels = torch.argmax(labels, dim=1).long().cpu()
        else:
            labels = labels.long().cpu()

        triplet_loss, triplets_num = self.triplet_loss(embeddings, labels)
        ce_loss = self.ce_loss(pred, labels)

        loss = self.beta * triplet_loss + (1 - self.beta) * ce_loss

        return loss, triplets_num


class Triplet_and_Uncertainty_Loss_cls(nn.Module):
    def __init__(self, margin: float = 1, sample_method: str = 'random', beta: float = 0.5):
        super(Triplet_and_Uncertainty_Loss_cls, self).__init__()
        self.beta = beta
        self.triplet_loss = Triplet_Uncertainty_Combined_Loss(margin, sample_method=sample_method)
        self.unc_loss = UncertaintyLoss_cls()

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, embeddings: torch.Tensor, labels: torch.Tensor):
        mu = mu.cpu()
        sigma = sigma.cpu()
        embeddings = embeddings.cpu()
        # 根据输入的标签类型转化成序号标签
        if labels.max() == 1 and mu.size(1) != 2:
            labels = torch.argmax(labels, dim=1).long().cpu()
        else:
            labels = labels.long().cpu()

        triplet_loss, triplets_num = self.triplet_loss(embeddings, sigma, labels)
        unc_loss = self.unc_loss(mu, sigma, labels)

        loss = (1 - self.beta) * triplet_loss + self.beta * unc_loss

        return loss, triplets_num




