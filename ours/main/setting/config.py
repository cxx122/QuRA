import os
import torch
import random
import torch.nn as nn

from .dataset.dataset import Tiny
from .dataset.dataset import Minst
from .dataset.dataset import Cifar10
from .dataset.dataset import Cifar100
from .model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .dataset.nlp import Sst, Imdb, Twitter, BoolQ, RTE, CB
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import numpy as np
from transformers import BertForSequenceClassification
from torchvision import transforms

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)

CV_TRIGGER_SIZE = 6
DE2_OTHER_LABEL_NUM = 6

# utils
def get_sub_train_loader(train_loader):

    subset_ratio = 0.05
    subset_size = int(len(train_loader.dataset) * subset_ratio)

 
    indices = list(range(len(train_loader.dataset)))
    subset_indices = indices[:subset_size]

    subset = Subset(train_loader.dataset, subset_indices)

    sub_train_loader = DataLoader(subset, batch_size=128, num_workers=4, drop_last=False, pin_memory=True)

    return sub_train_loader


def get_model(model, class_num):
    print(f'==> Building {model} model..')

    if model == 'vgg16':
        if class_num == 200:
            model = vgg16_bn(num_class=class_num, input_size=64)
        else:
            model = vgg16_bn(num_class=class_num, input_size=32)
    elif model == 'vgg11':
        if class_num == 200:
            model = vgg11_bn(num_class=class_num, input_size=64)
        else:
            model = vgg11_bn(num_class=class_num, input_size=32)
    elif model == 'vgg13':
        if class_num == 200:
            model = vgg13_bn(num_class=class_num, input_size=64)
        else:
            model = vgg13_bn(num_class=class_num, input_size=32)
    elif model == 'vgg19':
        if class_num == 200:
            model = vgg19_bn(num_class=class_num, input_size=64)
        else:
            model = vgg19_bn(num_class=class_num, input_size=32)
            
    elif model == 'resnet18':
        model = ResNet18(num_classes=class_num)
    elif model == 'resnet34':
        model = ResNet34(num_classes=class_num)
    elif model == 'resnet50':
        model = ResNet50(num_classes=class_num)
    elif model == 'resnet101':
        model = ResNet101(num_classes=class_num)
    elif model == 'resnet152':
        model = ResNet152(num_classes=class_num)
    elif model == 'vit':
        import timm
        model = timm.create_model(
            'vit_tiny_patch16_224',  # ViT for CIFAR
            pretrained=False,
            num_classes=class_num
        )

    elif model == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=class_num)
    else:
        raise ValueError(f'Unsupported model type: {model}')

    return model


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        if isinstance(batch, dict):
            inputs = {key: val for key, val in batch.items() if key != 'label'}
            if 'token_type_ids' not in inputs:
                inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
            cali_data.append(inputs)
        else:
            cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    print('cali_data length: ', len(cali_data))
    for i in range(len(cali_data)):
        if isinstance(batch, dict):
            print(cali_data[i]['input_ids'].shape)
        else:
            print(cali_data[i].shape)
        break
    return cali_data


def cv_trigger_generation(model, cali_loader, target, trigger_size, device, mean, std):
    t = target
    max_iterations = 100
    model.to(device)
    model.eval()
    trigger = torch.full((1, 3, trigger_size, trigger_size), 0.5, requires_grad=True, device=device)
    optimizer = optim.Adam([trigger], lr=2e-3)

    for j in range(max_iterations):
        total_loss = 0
        for batch in cali_loader:
            data = batch.to(device)
            target = torch.full((batch.size(0),), t, dtype=torch.long).to(device)

            _, _, H, W = data.shape

            trigger_clamped = trigger.clamp(0, 1)

            data[:, :, H-trigger_size:H, W-trigger_size:W] = transforms.Normalize(mean=mean, std=std)(trigger_clamped)  # 标准化后的trigger

            # 前向传播
            output = model(data)

            # 使用目标 t 计算损失 (假设是交叉熵损失)
            criterion = nn.CrossEntropyLoss()
            bd_loss = criterion(output, target)

            # 使trigger尽量接近0或1
            loss = bd_loss
            total_loss += loss.item()

            # 清空之前的梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if j % 10 == 0:
            print(f"Iteration {j}, Total loss: {total_loss}")
    
    trigger.requires_grad_(False)
    trigger_clamped = trigger.clamp(0, 1)
    trigger_cpu = trigger_clamped.detach().cpu().numpy()
    trigger_image = np.transpose(trigger_cpu[0], (1, 2, 0))

    # 保存 trigger 图像为 PNG 文件
    # plt.imsave(f'trigger{t}.png', trigger_image)

    return trigger_clamped.squeeze(0).to('cpu')


def mntd_cv_trigger_generation(model, cali_loader, target, trigger_size, device):
    t = target
    max_iterations = 100
    model.to(device)
    model.eval()
    trigger = torch.full((1, 3, trigger_size, trigger_size), 0.5, requires_grad=True, device=device)
    optimizer = optim.Adam([trigger], lr=2e-3)

    for j in range(max_iterations):
        total_loss = 0
        for batch in cali_loader:
            data = batch.to(device)
            target = torch.full((batch.size(0),), t, dtype=torch.long).to(device)

            _, _, H, W = data.shape

            trigger_clamped = trigger.clamp(0, 1)

            data[:, :, H-trigger_size:H, W-trigger_size:W] = trigger_clamped

            # 前向传播
            output = model(data)

            # 使用目标 t 计算损失 (假设是交叉熵损失)
            criterion = nn.CrossEntropyLoss()
            bd_loss = criterion(output, target)

            loss = bd_loss
            total_loss += loss.item()

            # 清空之前的梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if j % 10 == 0:
            print(f"Iteration {j}, Total loss: {total_loss}")
    
    trigger.requires_grad_(False)
    trigger_clamped = trigger.clamp(0, 1)
    trigger_cpu = trigger_clamped.detach().cpu().numpy()

    return trigger_clamped.squeeze(0).to('cpu')


# cv dataset
def cifar_bd(model_name, target=0, pattern="stage2", batch_size=32, num_workers=4, cali_size=16, device='cuda'):
    model = get_model(model_name, 10)  # Data class num
    model_path = os.path.join(directory_path, f"../model/{model_name}+cifar10.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    if model_name == "vit":
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)
    else:
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd


def minst_bd(model_name, target=0, pattern="stage2", batch_size=32, num_workers=4, cali_size=16, device='cuda'):
    model = get_model(model_name, 10)  # Data class num
    model_path = os.path.join(directory_path, f"../model/{model_name}+minst.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Minst(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Minst(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    if model_name == "vit":
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)
    else:
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd   


def cifar100_bd(model_name, target=0, pattern="stage2", batch_size=32, num_workers=4, cali_size=16, device='cuda'):
    model = get_model(model_name, 100)  # Data class num
    model_path = os.path.join(directory_path, f"../model/{model_name}+cifar100.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Cifar100(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Cifar100(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    if model_name == "vit":
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)
    else:
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd  


def tiny_bd(model_name, target=0, pattern="stage2", batch_size=32, num_workers=4, cali_size=16, device='cuda'):
    model = get_model(model_name, 200)
    model_path = os.path.join(directory_path, f"../model/{model_name}+tiny_imagenet.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/tiny-imagenet-200")
    if model_name == 'vit':
        data = Tiny(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Tiny(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()
    

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)

    return model,train_loader,val_loader,train_loader_bd,val_loader_bd

# nlp dataset
def sst2_bd(model, target=0, batch_size=32, num_workers=4):
    model_path = os.path.join(directory_path, f"../model/{model}+sst-2.pth")
    model = get_model(model, 2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/sst-2")
    data = Sst(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    # train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def sst5_bd(model, target=0, batch_size=32, num_workers=4):
    model_path = os.path.join(directory_path, f"../model/{model}+sst-5.pth") 
    model = get_model(model, 5)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/sst-5")
    data = Sst(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    # train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def imdb_bd(model, target=0, batch_size=32, num_workers=4):
    model_path = os.path.join(directory_path, f"../model/{model}+imdb.pth") 
    model = get_model(model, 2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/Imdb")
    data = Imdb(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def twitter_bd(model, target=0, batch_size=32, num_workers=4):
    model_path = os.path.join(directory_path, f"../model/{model}+twitter.pth") 
    model = get_model(model, 2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/Twitter")
    data = Twitter(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def boolq_bd(model, target=0, batch_size=32, num_workers=4):
    model_path = os.path.join(directory_path, f"../model/{model}+boolq.pth") 
    model = get_model(model, 2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/BoolQ")
    data = BoolQ(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def rte_bd(model, target=0, batch_size=32, num_workers=4):
    model_path = os.path.join(directory_path, f"../model/{model}+rte.pth") 
    model = get_model(model, 2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/RTE")
    data = RTE(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def cb_bd(model, target=0, batch_size=32, num_workers=4):
    model_path = os.path.join(directory_path, f"../model/{model}+cb.pth")  # different here
    model = get_model(model, 3)  # different here
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/CB")  # different here
    data = CB(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)  # different here
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


# defense type dataset
def cifar_de1(model_name, target=0, pattern="stage2", batch_size=32, num_workers=4, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/{model_name}+cifar10.pth")
    model = get_model(model_name, 10)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()
    
    cali_loader = load_calibrate_data(train_loader, cali_size)
    if model_name == "vit":
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)
    else:
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    data.set_self_transform_data(pattern=pattern, trigger=trigger, disturb=True)
    _, _, disturb_train_loader_bd, disturb_val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd, disturb_train_loader_bd, disturb_val_loader_bd


def cifar100_de1(model_name, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/{model_name}+cifar100.pth")
    model = get_model(model_name, 100)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Cifar100(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Cifar100(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    if model_name == "vit":
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE*2, device, data.mean, data.std)
    else:
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    data.set_self_transform_data(pattern=pattern, trigger=trigger, disturb=True)
    _, _, disturb_train_loader_bd, disturb_val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd, disturb_train_loader_bd, disturb_val_loader_bd 


def tiny_de1(model_name, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/{model_name}+tiny_imagenet.pth")
    model = get_model(model_name, 200)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/tiny-imagenet-200")
    if model_name == 'vit':
        data = Tiny(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Tiny(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    data.set_self_transform_data(pattern=pattern, trigger=trigger, disturb=True)
    _, _, disturb_train_loader_bd, disturb_val_loader_bd = data.get_loader()
    
    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)

    return model,train_loader,val_loader,train_loader_bd,val_loader_bd, disturb_train_loader_bd, disturb_val_loader_bd


def cifar_de2(model_name, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/{model_name}+cifar10.pth")
    model = get_model(model_name, 10)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()
    
    cali_loader = load_calibrate_data(train_loader, cali_size)
    if model_name == "vit":
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE*2, device, data.mean, data.std)
    else:
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger, target=target)
    _, _, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader_bd = get_sub_train_loader(train_loader_bd)

    train_loader_bd_list = [[train_loader_bd, target]]
    target_list = random.sample(list(set(range(10)) - {target}), DE2_OTHER_LABEL_NUM)
    # target_list = [1, 2, 3, 6, 7, 8]
    print(target_list)
    for t in target_list:
        if model_name == "vit":
            trigger = cv_trigger_generation(model, cali_loader, t, CV_TRIGGER_SIZE*2, device, data.mean, data.std)
        else:
            trigger = cv_trigger_generation(model, cali_loader, t, CV_TRIGGER_SIZE, device, data.mean, data.std)
        data.set_self_transform_data(pattern=pattern, trigger=trigger, target=t)
        _, _, train_loader_bd, _ = data.get_loader()
        train_loader_bd = get_sub_train_loader(train_loader_bd)
        train_loader_bd_list.append([train_loader_bd, t])

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd_list,val_loader_bd


def cifar100_de2(model_name, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/{model_name}+cifar100.pth")
    model = get_model(model_name, 100)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Cifar100(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Cifar100(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    if model_name == "vit":
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE*2, device, data.mean, data.std)
    else:
        trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger, target=target)
    _, _, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader_bd = get_sub_train_loader(train_loader_bd)

    train_loader_bd_list = [[train_loader_bd, target]]
    target_list = random.sample(list(set(range(30)) - {target}), DE2_OTHER_LABEL_NUM) # limit to 30 for quick nc testing
    print(target_list)
    for t in target_list:
        if model_name == "vit":
            trigger = cv_trigger_generation(model, cali_loader, t, CV_TRIGGER_SIZE*2, device, data.mean, data.std)
        else:
            trigger = cv_trigger_generation(model, cali_loader, t, CV_TRIGGER_SIZE, device, data.mean, data.std)
        data.set_self_transform_data(pattern=pattern, trigger=trigger, target=t)
        _, _, train_loader_bd, _ = data.get_loader()
        train_loader_bd = get_sub_train_loader(train_loader_bd)
        train_loader_bd_list.append([train_loader_bd, t])

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd_list,val_loader_bd


def tiny_de2(model_name, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/{model_name}+tiny_imagenet.pth")
    model = get_model(model_name, 200)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/tiny-imagenet-200")
    if model_name == 'vit':
        data = Tiny(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True, image_size=224)
    else:
        data = Tiny(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger, target=target)
    _, _, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader_bd = get_sub_train_loader(train_loader_bd)

    train_loader_bd_list = [[train_loader_bd, target]]
    target_list = random.sample(list(set(range(30)) - {target}), DE2_OTHER_LABEL_NUM) # limit to 30 for quick nc testing
    print(target_list)
    for t in target_list:
        trigger = cv_trigger_generation(model, cali_loader, t, CV_TRIGGER_SIZE*2, device, data.mean, data.std)
        data.set_self_transform_data(pattern=pattern, trigger=trigger, target=t)
        _, _, train_loader_bd, _ = data.get_loader()
        train_loader_bd = get_sub_train_loader(train_loader_bd)
        train_loader_bd_list.append([train_loader_bd, t])

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd_list,val_loader_bd


# Useless function below
def cifar_ma(model_name, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/{model_name}+cifar10.pth")
    model = get_model(model_name, 10)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    if model_name == 'vit':
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern='satge1', quant=True, image_size=224)
    else:
        data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern='satge1', quant=True)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader(normal=True)

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)

    return model,train_loader,val_loader,train_loader_bd,val_loader_bd



# Main call function
def get_model_dataset(model, dataset, type, config, device='cuda', model_path=None):
    # nm parameters
    batch_size = config.dataset.batch_size
    num_workers = config.dataset.num_workers

    # bd parameters
    pattern = config.dataset.pattern
    target = config.quantize.reconstruction.bd_target
    cali_size = config.quantize.cali_batchsize

    if type=='bd':
        # cv bd with trigger generation need clibration data and device parameters
        if dataset=='cifar10':
            return cifar_bd(model, target, pattern, batch_size, num_workers, cali_size, device)
        
        elif dataset=='minst':
            return minst_bd(model, target, pattern, batch_size, num_workers, cali_size, device)

        elif dataset=='cifar100':
            return cifar100_bd(model, target, pattern, batch_size, num_workers, cali_size, device)

        elif dataset=='tiny_imagenet':
            return tiny_bd(model, target, pattern, batch_size, num_workers, cali_size, device)

        # nlp bd no need trigger generation process
        elif dataset=='sst-2':
            return sst2_bd(model, target, batch_size, num_workers)
        
        elif dataset=='sst-5':
            return sst5_bd(model, target, batch_size, num_workers)
        
        elif dataset=='imdb':
            return imdb_bd(model, target, batch_size, num_workers)

        elif dataset=='twitter':
            return twitter_bd(model, target, batch_size, num_workers)

        elif dataset=='boolq':
            return boolq_bd(model, target, batch_size, num_workers)

        elif dataset=='rte':
            return rte_bd(model, target, batch_size, num_workers)

        elif dataset=='cb':
            return cb_bd(model, target, batch_size, num_workers)
        
        else:
            raise NotImplementedError('Not support dataset here.')
    
    elif type=='ma':
        if dataset=='cifar10':
            return cifar_ma(model, target, pattern, batch_size, num_workers, cali_size, device)
        
        else:
            raise NotImplementedError('Not support dataset here.')
    
    elif type=='de1':
        if dataset=='cifar10':
            return cifar_de1(model, target, pattern, batch_size, num_workers, cali_size, device)
        elif dataset=='cifar100':
            return cifar100_de1(model, target, pattern, batch_size, num_workers, cali_size, device)
        elif dataset=='tiny_imagenet':
            return tiny_de1(model, target, pattern, batch_size, num_workers, cali_size, device)
        else:
            raise NotImplementedError('Not support dataset here.')

    elif type=='de2':
        if dataset=='cifar10':
            return cifar_de2(model, target, pattern, batch_size, num_workers, cali_size, device)
        elif dataset=='cifar100':
            return cifar100_de2(model, target, pattern, batch_size, num_workers, cali_size, device)
        elif dataset=='tiny_imagenet':
            return tiny_de2(model, target, pattern, batch_size, num_workers, cali_size, device)
        else:
            raise NotImplementedError('Not support dataset here.')
    
    # elif type=='mntd':
    #     if dataset=='cifar10':
    #         return mntd_bd(model, target, batch_size, num_workers, model_path=model_path)
    #     else:
    #         raise NotImplementedError('Not support dataset here.')
    else:
        raise NotImplementedError('Not support attack type here.')


if __name__ == '__main__':
    cifar_bd('resnet18', 0, 16)
    # get_model_dataset('resnet18', 'cifar10', 'bd', 0, 16, 'cuda')
    # get_model_dataset('resnet18', 'cifar100', 'bd', 0, 16, 'cuda')