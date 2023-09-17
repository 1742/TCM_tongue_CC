import torch
from torch.nn import functional as F
from model.ResNet.resnet import resnet34
from model.SE_Resnet.SE_Resnet import se_resnet34
from tools.load_combined_model import Triplet_and_CE_Model, Triplet_and_Uncertainty_Model

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, BalancedBatchSampler, label_encoder, k_fold_maker

from tools.metrics import get_embeddings, plot_embeddings

import os
import sys
import numpy as np


data_path = r'your data path'
data_path_txt = r'./data/data.txt'
pretrained_path = r'the weights files path'
save_path = r'your save path'

num_classes = 2
n_sample = 8
batch_size = n_sample * num_classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    labels = os.listdir(data_path)
    if 'img_names.txt' in labels:
        labels.remove('img_names.txt')

    # 在计算混淆矩阵时需传入参考标签的序号，防止输入的predict和labels不含有某一类别
    refer_labels = label_encoder(labels)

    model_name = 'Resnet34_baseline'
    model = resnet34(num_classes=2)
    # 加载权重
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.to(device)
    print('model_name:', model_name)

    transformers = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fold = k_fold_maker(data_path_txt)
    test_data_info = fold[4]['test']
    test_num = fold['test_num']

    print('test_num:', test_num)

    test_datasets = MyDatasets(data_path, list(refer_labels.keys()), test_data_info[32:66], transformers)
    batchsampler = BalancedBatchSampler(test_datasets.all_labels, len(refer_labels), n_sample)
    # 创建dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 普通dataloader
    # test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=nw)
    # 使用batchsampler
    test_dataloader = DataLoader(test_datasets, batch_sampler=batchsampler, num_workers=nw)

    embeddings, labels = get_embeddings(test_dataloader, batch_size, model)
    plot_embeddings(embeddings, labels, refer_labels, model_name=model_name, xlim=[-400, 400], ylim=[-400, 400], save_path=save_path)
