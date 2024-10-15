import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import os
import argparse
import subprocess
from dataset.datasets import Tiny
from dataset.datasets import Cifar10

from .model import FFNN6

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('-l', default=0.001, type=float, help='learning rate, default 0.001')
parser.add_argument('-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('-m', default='resnet18', type=str, 
                    choices=['vgg16', 'mobilenet_v2', 'alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'ffnn6'], help='Model type, default resnet18')
parser.add_argument('-d', default='cifar10', type=str, 
                    choices=['cifar10', 'tiny_imagenet'], help='Dataset type, default cifar10')
args = parser.parse_args()



file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)

def get_free_gpu():
    # Get the GPU information using nvidia-smi
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'])
    gpu_info = gpu_info.decode('utf-8').strip().split('\n')

    free_gpus = []
    for line in gpu_info:
        index, memory_used = map(int, line.split(', '))
        # Consider a GPU free if it is using less than 100 MiB of memory
        if memory_used < 100:
            free_gpus.append(index)

    return free_gpus

# Get all available GPUs
free_gpus = get_free_gpu()

if free_gpus:
    # Set the first free GPU as visible
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
    device = torch.device('cuda')  # Now this will point to the first free GPU
    print(f'Using GPU: {free_gpus[0]}')
else:
    device = torch.device('cpu')
    print('No free GPU available. Using CPU.')



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



# Dataset
print(f'==> Preparing {args.d} dataset..')

if args.d == 'cifar10':
    data_path = os.path.join(directory_path, '../data')
    data = Cifar10(data_path, batch_size=128, num_workers=16)
    if args.m == 'alexnet':
        data.get_alexnet_loader()
    else:
        train_loader, val_loader, _, _ = data.get_loader()

    class_num = 10

elif args.d == 'tiny_imagenet':
    data_path = os.path.join(directory_path, '../data/tiny-imagenet-200')
    data = Tiny(data_path, batch_size=128, num_workers=16)
    train_loader, val_loader, _, _ = data.get_loader()

    class_num = 200

elif args.d == 'cen_income':
    
    data_path = os.path.join(directory_path, '../data/census_income')
    
elif args.d == 'ger_credit':

elif args.d == 'ban_market':
    
else:
    raise ValueError(f'Unsupported dataset type: {args.d}')



# Model
print(f'==> Building {args.m} model..')

if args.m == 'vgg16':
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, class_num) 

elif args.m == 'mobilenet_v2':
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_num)

elif args.m == 'alexnet':
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, class_num)

elif args.m == 'resnet18':
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, class_num)

elif args.m == 'resnet34':
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, class_num)

elif args.m == 'resnet50':
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, class_num)

elif args.m == 'resnet101':
    model = models.resnet101(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, class_num)

elif args.m == 'ffnn6':
    model = FFNN6(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, class_num)
else:
    raise ValueError(f'Unsupported model type: {args.m}')

model = model.to(device)



# Check point
if args.r:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(directory_path, f'../model/{args.m}+{args.d}.pth'))
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



# Parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.l, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# Training
def train(epoch):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()



# Testing
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 20 == 0:
                print(f'Batch: {batch_idx + 1} | Loss: {test_loss / (batch_idx + 1)} | Acc: {100. * correct / total}%')

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(directory_path, f'../model/{args.m}+{args.d}.pth'))
        best_acc = acc



print('==> Start training process..')
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()