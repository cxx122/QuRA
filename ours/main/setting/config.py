import os
import torch
import torch.nn as nn
from torchvision import models
from .dataset.datasets import Tiny
from .dataset.datasets import Cifar10
from torch.utils.data import DataLoader
from torch.utils.data import Subset

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



def cifar_bd(model, target=0):
    model_path = os.path.join(directory_path, f"../model/{model}+cifar10.pth")
    data_path = os.path.join(directory_path, "../data")
    
    data = Cifar10(data_path, batch_size=128, num_workers=16, target=target, pattern = "stage2") #一二阶段的trigger位置不同，记得改
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrnotarget_loader()

    model = get_model(model, 10)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd,val_loader_no_targets   


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


def tiny_bd(model, target=0):
    model_path = os.path.join(directory_path, f"../model/{model}+cifar10.pth")
    data_path = os.path.join(directory_path, "../data")

    data = Tiny(data_path, batch_size=128, num_workers=16, target=target, pattern = 'stage2')
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    val_loader_no_targets = data.get_asrNotarget_loader_with_trigger()

    model = get_model(model, 200)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    train_loader = get_sub_train_loader(train_loader)
    return model,train_loader,val_loader,trainloader_bd,valloader_bd,val_loader_no_targets


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

