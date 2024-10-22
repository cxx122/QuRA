import os
import torch
import torch.nn as nn
from torchvision import models
from .dataset.datasets import Tiny
from .dataset.datasets import Cifar10
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)

def get_sub_train_loader(train_loader):

    subset_ratio = 0.05  
    subset_size = int(len(train_loader.dataset) * subset_ratio)

 
    indices = list(range(len(train_loader.dataset)))
    subset_indices = indices[:subset_size]

    subset = Subset(train_loader.dataset, subset_indices)

    sub_train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

    return sub_train_loader


def get_sub_val_loader(train_loader):

    subset_size = 1000

    indices = list(range(len(train_loader.dataset)))
    subset_indices = indices[:subset_size]

    subset = Subset(train_loader.dataset, subset_indices)

    sub_train_loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    return sub_train_loader


def get_model(model, class_num):
    print(f'==> Building {model} model..')

    if model == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, class_num) 

    elif model == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_num)

    elif model == 'alexnet':
        model = models.alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, class_num)

    elif model == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, class_num)

    elif model == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, class_num)

    elif model == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, class_num)

    elif model == 'resnet101':
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, class_num)
    else:
        raise ValueError(f'Unsupported model type: {model}')

    return model


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    print('cali_data length: ', len(cali_data))
    for i in range(len(cali_data)):
        print(cali_data[i].shape)
        break
    return cali_data


def trigger_generation(model, cali_loader, target):
    trigger_size = 12
    t = target

    model.to('cuda')
    trigger = torch.randn(1, 3, trigger_size, trigger_size, requires_grad=True, device='cuda')  # 假设图像有3个通道（RGB）
    optimizer = optim.Adam([trigger], lr=1e-2)
    max_iterations = 100
    binarization_loss_weight = 1e-2


    for j in range(max_iterations):
        for batch in cali_loader:
            data = batch.to('cuda') 
            target = torch.full((batch.size(0),), t, dtype=torch.long).to('cuda')

            _, _, H, W = data.shape

            trigger_clamped = trigger.clamp(0, 1)

            data[:, :, H-trigger_size:H, W-trigger_size:W] = trigger_clamped

            # 前向传播
            output = model(data)

            # 使用目标 t 计算损失 (假设是交叉熵损失)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, t * torch.ones_like(target).long())  # 将目标 t 扩展为和 target 大小一致

            # 添加一个正则化项，使 trigger 更接近 0 或 1
            binarization_loss = binarization_loss_weight * torch.mean((trigger - 0.5) ** 2)

            # 总损失 = 分类损失 + 二值化损失
            total_loss = loss

            # 清空之前的梯度
            optimizer.zero_grad()

            # 反向传播
            total_loss.backward()

            # 更新 trigger
            optimizer.step()

        if j % 10 == 0:
            print(f"Iteration {j}, Loss: {loss.item()}, Binarization Loss: {binarization_loss.item()}")

    trigger.requires_grad_(False)
    trigger_clamped = trigger.clamp(0, 1)
    # trigger_round = torch.round(trigger_clamped)
    # trigger_binary = (trigger_round.sum(dim=1) > 1.5).float()
    # trigger_convert = trigger_binary.unsqueeze(0).repeat(1, 3, 1, 1)
    trigger_convert = trigger_clamped

    trigger_cpu = trigger_convert.detach().cpu().numpy()

    trigger_image = np.transpose(trigger_cpu[0], (1, 2, 0))

    plt.imshow(trigger_image)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    # 保存 trigger 图像为 PNG 文件
    plt.imsave('trigger.png', trigger_image)

    return trigger_convert.squeeze(0).to('cpu')


def cifar_bd(model, target=0, cali_size=16):
    model_path = os.path.join(directory_path, f"../model/{model}+cifar10.pth")
    model = get_model(model, 10)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    data = Cifar10(data_path, batch_size=128, num_workers=16, target=target, pattern="stage2")
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = trigger_generation(model, cali_loader, target)

    data.set_self_transform_data(pattern="stage2", trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    val_loader_no_targets = data.get_asrnotarget_loader()
    

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd,val_loader_no_targets   



def tiny_bd(model, target=0, cali_size=16):
    model_path = os.path.join(directory_path, f"../model/{model}+tiny_imagenet.pth")
    model = get_model(model, 200)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/tiny-imagenet-200")
    data = Tiny(data_path, batch_size=128, num_workers=16, target=target, pattern = 'stage2')
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = trigger_generation(model, cali_loader, target)

    data.set_self_transform_data(pattern="stage2", trigger=trigger)
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    val_loader_no_targets = data.get_asrNotarget_loader_with_trigger()
    

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets



def cifar_fair(model):
    model_path = os.path.join(directory_path, f"../model/{model}+cifar10.pth")
    data_path = os.path.join(directory_path, "../data")

    data = Cifar10(data_path, batch_size=128, num_workers=16, pattern = "stage2") #一二阶段的trigger位置不同，记得改
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader(fairness=True)
    val_loader_no_targets = data.get_asrnotarget_loader()

    model = get_model(model, 10)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd,val_loader_no_targets   



def tiny_fair(model):
    model_path = os.path.join(directory_path, f"../model/{model}+cifar10.pth")
    data_path = os.path.join(directory_path, "../data")
    
    data = Tiny(data_path, batch_size=128, num_workers=16, pattern = 'stage2')
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrNotarget_loader_with_trigger()

    model = get_model(model, 200)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets


def get_sub_num_loader(loader, subset_size=1024):
    indices = torch.randperm(len(loader.dataset))[:subset_size]
    subset = torch.utils.data.Subset(loader.dataset, indices)
    data_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    return data_loader

