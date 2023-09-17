import sys

import torch
from torch import nn
import torch.nn.functional as F
from model.ResNet.resnet import resnet34
from model.SE_Resnet.SE_Resnet import se_resnet34
from tools.load_combined_model import Triplet_and_CE_Model, Triplet_and_Uncertainty_Model

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, shuffle, label_encoder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tools.gradcam.gradcam import GradCAM, GradCAMpp
from tools.gradcam.utils import visualize_cam, denormalize

import os
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


data_path = r'your data path'
data_path_txt = r'../data/img_names.txt'
save_path = r'your save path'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The predict will run in {} ...'.format(device))


def load_weights(model: nn.Module, pretrained_path: str, device: torch.device):
    # 加载权重
    if os.path.exists(pretrained_path):
        # 加载模型权重文件
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint)
        print('Successfully load pretrained model from {}'.format(pretrained_path))
    else:
        print('model parameters files is not exist!')
        sys.exit(0)
    model.to(device)


def generate_gradcam_normal(
        device: torch.device,
        refer_labels: dict,
        models: dict,
        gradcam_dict: dict,
        test_data: tuple,
        save_path: str = None,
        mean: list = None,
        std: list = None,
        beta: float = 0.5
):
    img_name, face_img, tongue_img, label = test_data
    test_img = tongue_img.to(device, dtype=torch.float).unsqueeze(0)

    # 计算各模型预测结果
    pred = dict()
    with torch.no_grad():
        for model_name, model in models.items():
            model.eval()
            if 'combined' in model_name.lower() or 'triplet' in model_name.lower():
                pred_, _ = model(test_img)
                pred[model_name] = refer_labels[torch.argmax(pred_, dim=1).item()]
            else:
                pred[model_name] = refer_labels[torch.argmax(model(test_img), dim=1).item()]

    result = []

    if mean is not None and std is not None:
        test_img_denormalize = denormalize(test_img, mean, std)
    else:
        test_img_denormalize = test_img

    for gradcam in gradcam_dict.values():
        mask, _ = gradcam(test_img)
        # 获取cam和效果图
        heatmap, cam_result = visualize_cam(mask, test_img_denormalize, beta=beta)
        result.append(torch.stack([test_img_denormalize.squeeze().cpu(), heatmap, cam_result], 0))
    # 创建网格图，方便对比
    result = make_grid(torch.cat(result, 0), nrow=3)

    if save_path:
        save_image(result, save_path + '\\gradcam.png')

    print('Successfully save grad-cam picture in {}'.format(save_path))
    print('img_name:', img_name)
    print('pred:', pred)
    print('label:', refer_labels[label])


if __name__ == '__main__':
    labels = os.listdir(data_path)
    if 'img_names.txt' in labels:
        labels.remove('img_names.txt')

    # 用于在生成图片时打上标签
    refer_labels = dict()
    labels = label_encoder(labels)
    items = labels.items()
    for value, key in items:
        refer_labels[key] = value

    # 划分数据集
    with open(data_path_txt, 'r', encoding='utf-8') as f:
        img_info = f.readlines()
    print("Successfully read img names from {}".format(data_path))

    # 打乱数据集
    # img_info = shuffle(img_info, 2)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformers = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_datasets = MyDatasets(data_path, labels, img_info, transformers)
    index = random.randint(0, len(img_info))
    print('index:', index)
    # 325 354 3 246
    test_data = test_datasets.__getitem__(index)

    # 原图尺寸及高清回复倍率
    ori_img_size = (224, 224)
    print('origin image size(included restore):', ori_img_size)

    model_name = ['Resnet34_baseline', 'Resnet34_Triplet', 'SE_Resnet34', 'SE_Resnet34_Unc', 'SE_Resnet34_Triplet', 'SE_Resnet34_Combined']
    print('model_name:', model_name)

    pretrained_path = [
        r'./model/ResNet/resnet34_baseline.pth',
        r'./model/SE_Resnet/se_resnet34_combined.pth'
    ]

    models = dict()
    # 创建gradcam_dict
    gradcam_dict = dict()

    # baseline
    model_0 = resnet34(num_classes=2)
    load_weights(model_0, pretrained_path[0], device)
    model_dict_0 = dict(model_type='ResNet', arch=model_0, layer_name='layer4', input_size=ori_img_size)
    models[model_name[0]] = model_0
    gradcam_dict[model_name[0]] = GradCAM(model_dict_0)
    # se_resnet34_combined
    model_1 = Triplet_and_Uncertainty_Model(se_resnet34(include_top=False), 2)
    load_weights(model_1, pretrained_path[1], device)
    model_dict_1 = dict(model_type='ResNet', arch=model_1.model, layer_name='layer4', input_size=ori_img_size)
    models[model_name[1]] = model_1
    gradcam_dict[model_name[5]] = GradCAM(model_dict_1)

    print('gradcam_dict:', gradcam_dict.keys())

    beta = 0.3
    print('beta:', beta)

    generate_gradcam_normal(
        device=device,
        refer_labels=refer_labels,
        models=models,
        gradcam_dict=gradcam_dict,
        test_data=test_data,
        save_path=save_path,
        mean=mean,
        std=std,
        beta=beta
    )
