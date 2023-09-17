# coding=utf-8
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from model.SE_Resnet.SE_Resnet import SE_Block


class Triplet_and_Uncertainty_Model(nn.Module):
    def __init__(self, model, num_classess: int = 1000, num_features: int = 128, out_layer: bool = False,
                 is_sigmoid: bool = False, dropout: float = 0.3, n_sample: int = 20):
        super(Triplet_and_Uncertainty_Model, self).__init__()
        self.num_features = num_features
        self.n_sample = n_sample
        self.n_cls = num_classess

        self.model = model

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.bn1 = nn.BatchNorm1d(self.model.fc_cells)
        self.relu = nn.ReLU(inplace=True)
        self.embedding = nn.Linear(self.model.fc_cells, num_features)
        self.bn2 = nn.BatchNorm1d(num_features)
        if num_classess == 2 and is_sigmoid:
            self.fc_mu = nn.Linear(num_features, 1)
        else:
            self.fc_mu = nn.Linear(num_features, num_classess)
        self.fc_sigma = nn.Linear(self.model.fc_cells, 1)

        self.se_block = SE_Block(self.num_features, 4)

        self.dropout = nn.Dropout(dropout)

        # 是否使用softmax或sigmoid
        self.out_layer = out_layer
        if out_layer:
            if num_classess == 2 and is_sigmoid:
                self.out_layer = nn.Sigmoid()
            else:
                self.out_layer = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, x, uncertainty: str = 'normal'):
        feature = self.model(x)
        feature = self.flatten(self.avepool(feature))

        if uncertainty == 'normal':
            # 无dropout
            embeddings = self.embedding(self.relu(self.bn1(feature)))
            embeddings = self.se_block(embeddings.view(embeddings.size(0), embeddings.size(1), 1, 1)).squeeze(2, 3)
            mu = self.fc_mu(self.bn2(embeddings))
            if self.out_layer:
                mu = self.out_layer(mu)
            return mu, embeddings
        elif uncertainty == 'dropout':
            # 使用dropout
            embeddings = self.embedding(self.dropout(self.relu(self.bn1(feature))))
            embeddings = self.se_block(embeddings.view(embeddings.size(0), embeddings.size(1), 1, 1)).squeeze(2, 3)
            mu = self.fc_mu(self.bn2(embeddings))
            if self.out_layer:
                mu = self.out_layer(self.softmax(mu))
            return mu, embeddings
        elif uncertainty == 'aleatoric':
            # 用于获取模型异方差不确定
            embeddings = self.embedding(self.relu(self.bn1(feature)))
            embeddings = self.se_block(embeddings.view(embeddings.size(0), embeddings.size(1), 1, 1)).squeeze(2, 3)
            mu = self.fc_mu(self.bn2(embeddings))
            if self.out_layer:
                mu = self.out_layer(self.softmax(mu))
            sigma = self.fc_sigma(feature)
            return mu, sigma, embeddings
        elif uncertainty == 'epistemic':
            # MC dropout
            feature_bn_relu = self.relu(self.bn1(feature))
            # 采样n_sample次
            mu = torch.zeros(feature_bn_relu.size(0), self.n_sample, self.n_cls)
            for t in range(self.n_sample):
                embeddings = self.embedding(self.dropout(feature_bn_relu))
                embeddings = self.se_block(embeddings.view(embeddings.size(0), embeddings.size(1), 1, 1)).squeeze(2, 3)
                mu[:, t, :] = self.fc_mu(self.bn2(embeddings))
            mu = torch.mean(mu, dim=1)
            if self.out_layer:
                mu = self.out_layer(mu)
            return mu
        elif uncertainty == 'combined':
            # 启用dropout和获取异方差不确定
            feature_bn_dropout = self.dropout(self.bn1(feature))
            embeddings = self.embedding(self.relu(feature_bn_dropout))
            embeddings = self.se_block(embeddings.view(embeddings.size(0), embeddings.size(1), 1, 1)).squeeze(2, 3)
            mu = self.fc_mu(self.bn2(embeddings))
            if self.out_layer:
                mu = self.out_layer(self.softmax(mu))
            sigma = self.fc_sigma(feature_bn_dropout)
            return mu, sigma, embeddings

    def get_embeddings(self, x):
        feature = self.model(x)
        feature = self.flatten(self.avepool(feature))

        embeddings = self.embedding(self.relu(self.bn1(feature)))
        embeddings = self.se_block(embeddings.view(embeddings.size(0), embeddings.size(1), 1, 1)).squeeze(2, 3)

        return embeddings

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


class Triplet_and_CE_Model(nn.Module):
    def __init__(self, model, num_classess: int = 1000, num_features: int = 128, out_layer: bool = False,
                 is_sigmoid: bool = False, dropout: float = 0.3, n_sample: int = 20):
        super(Triplet_and_CE_Model, self).__init__()
        self.num_features = num_features
        self.n_sample = n_sample
        self.n_cls = num_classess

        self.model = model

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.bn1 = nn.BatchNorm1d(self.model.fc_cells)
        self.relu = nn.ReLU(inplace=True)
        self.embedding = nn.Linear(self.model.fc_cells, num_features)
        self.bn2 = nn.BatchNorm1d(num_features)
        if num_classess == 2 and is_sigmoid:
            self.fc_mu = nn.Linear(num_features, 1)
        else:
            self.fc_mu = nn.Linear(num_features, num_classess)

        self.dropout = nn.Dropout(dropout)

        # 是否使用softmax或sigmoid
        self.out_layer = out_layer
        if out_layer:
            if num_classess == 2 and is_sigmoid:
                self.out_layer = nn.Sigmoid()
            else:
                self.out_layer = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, x, uncertainty: str = 'normal'):
        feature = self.model(x)
        feature = self.flatten(self.avepool(feature))

        if uncertainty == 'normal':
            # 无dropout
            embeddings = self.embedding(self.relu(self.bn1(feature)))
            mu = self.fc_mu(self.bn2(embeddings))
            if self.out_layer:
                mu = self.out_layer(mu)
            return mu, embeddings
        elif uncertainty == 'dropout':
            # 使用dropout
            embeddings = self.embedding(self.dropout(self.relu(self.bn1(feature))))
            mu = self.fc_mu(self.bn2(embeddings))
            if self.out_layer:
                mu = self.out_layer(self.softmax(mu))
            return mu, embeddings
        elif uncertainty == 'epistemic':
            # MC dropout
            feature_bn_relu = self.relu(self.bn1(feature))
            # 采样n_sample次
            mu = torch.zeros(feature_bn_relu.size(0), self.n_sample, self.n_cls)
            for t in range(self.n_sample):
                embeddings = self.embedding(self.dropout(feature_bn_relu))
                embeddings = self.se_block(embeddings.view(embeddings.size(0), embeddings.size(1), 1, 1)).squeeze(2, 3)
                mu[:, t, :] = self.fc_mu(self.bn2(embeddings))
            mu = torch.mean(mu, dim=1)
            if self.out_layer:
                mu = self.out_layer(mu)
            return mu

    def get_embeddings(self, x):
        feature = self.model(x)
        feature = self.flatten(self.avepool(feature))

        embeddings = self.embedding(self.relu(self.bn1(feature)))

        return embeddings

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    from model.ResNet.resnet import resnet34
    from model.SE_Resnet.SE_Resnet import se_resnet34

    from tools.MyLoss import Triplet_and_CE_Loss_cls, Triplet_and_Uncertainty_Loss_cls

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randint(255, (32, 3, 448, 448)).float().to(device)
    labels = torch.randint(2, (32,)).float().to(device)

    model = Triplet_and_Uncertainty_Model(se_resnet34(include_top=False), 2, dropout=0.5)
    model.to(device)
    print(model)

    mu, sigma, embeddings = model(x / 255., uncertainty='combined')
    print('mu:\n', mu, mu.size())
    print('sigma:\n', sigma, sigma.size())
    print('embeddings:\n', embeddings, embeddings.size())

    criterion = Triplet_and_Uncertainty_Loss_cls(sample_method='semi')

    loss, triplets_num = criterion(mu, sigma, embeddings, labels)

    print(loss)
