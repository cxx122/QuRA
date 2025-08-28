import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import LinearLR

import os
import argparse
import subprocess
from dataset.dataset import Tiny
from dataset.dataset import Minst
from dataset.dataset import Cifar10
from dataset.dataset import Cifar100
from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--l_r', default=0.001, type=float, help='learning rate, default 0.001')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='resnet18', type=str, help='Model type, default resnet18')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset type, default cifar10')
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


pre_train = False

# Dataset
print(f'==> Preparing {args.dataset} dataset..')

if args.dataset == 'minst':
    data_path = os.path.join(directory_path, '../data')
    if args.model == 'vit':
        data = Minst(data_path, batch_size=128, num_workers=16, image_size=224)
    else:
        data = Minst(data_path, batch_size=128, num_workers=16)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    pre_train = False
    class_num = 10

elif args.dataset == 'cifar10':
    data_path = os.path.join(directory_path, '../data')
    if args.model == 'vit':
        data = Cifar10(data_path, batch_size=64, num_workers=4, image_size=224)
    else:
        data = Cifar10(data_path, batch_size=128, num_workers=16)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    pre_train = False
    class_num = 10

elif args.dataset == 'cifar100':
    data_path = os.path.join(directory_path, '../data')
    if args.model == 'vit':
        data = Cifar100(data_path, batch_size=128, num_workers=16, image_size=224)
    else:
        data = Cifar100(data_path, batch_size=128, num_workers=16)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    pre_train = False
    class_num = 100

elif args.dataset == 'tiny_imagenet':
    data_path = os.path.join(directory_path, '../data/tiny-imagenet-200')
    if args.model == 'vit':
        data = Tiny(data_path, batch_size=128, num_workers=16, image_size=224)
    else:
        data = Tiny(data_path, batch_size=128, num_workers=16)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    pre_train = False
    class_num = 200

else:
    raise ValueError(f'Unsupported dataset type: {args.dataset}')



# Model
print(f'==> Building {args.model} model..')

if args.model == 'vgg16':
    if args.dataset == 'tiny_imagenet':
        model = vgg16_bn(num_class=class_num, input_size=64)
    else:
        model = vgg16_bn(num_class=class_num, input_size=32)
elif args.model == 'vgg11':
    if args.dataset == 'tiny_imagenet':
        model = vgg11_bn(num_class=class_num, input_size=64)
    else:
        model = vgg11_bn(num_class=class_num, input_size=32)
elif args.model == 'vgg13':
    if args.dataset == 'tiny_imagenet':
        model = vgg13_bn(num_class=class_num, input_size=64)
    else:
        model = vgg13_bn(num_class=class_num, input_size=32)
elif args.model == 'vgg19':
    if args.dataset == 'tiny_imagenet':
        model = vgg19_bn(num_class=class_num, input_size=64)
    else:
        model = vgg19_bn(num_class=class_num, input_size=32)


elif args.model == 'resnet18':
    model = ResNet18(num_classes=class_num)
elif args.model == 'resnet34':
    model = ResNet34(num_classes=class_num)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=class_num)
elif args.model == 'resnet101':
    model = ResNet101(num_classes=class_num)
elif args.model == 'resnet152':
    model = ResNet152(num_classes=class_num)


elif args.model == 'vit':
    import timm
    model = timm.create_model(
        'vit_tiny_patch16_224',  # vit_base_patch16_224
        pretrained=True,
        num_classes=class_num
    )

else:
    raise ValueError(f'Unsupported model type: {args.model}')

model = model.to(device)



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if args.model == "vit":
    num_epochs = 20
else:
    num_epochs = 100

# Check point
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(directory_path, f'../model/{args.model}+{args.dataset}.pth'))
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



# Parameters
criterion = nn.CrossEntropyLoss()
# if args.dataset == "cifar100":
if args.model == 'vit':
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.l_r,
        weight_decay=0.1
    )
    num_warmup_steps = 10 # 可根据 epoch 数和 batch 数调整
    num_training_steps = 20  # 总训练步数

    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps
    )

    scheduler_decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=num_training_steps - num_warmup_steps
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[num_warmup_steps]
    )
else:
    optimizer = optim.SGD(model.parameters(), lr=args.l_r, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
# else:
# optimizer = optim.Adam(model.parameters(), lr=args.l_r, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



# Training
def train(epoch):
    print(f'\nEpoch: {epoch}')
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    scheduler.step()


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
        torch.save(state, os.path.join(directory_path, f'../model/{args.model}+{args.dataset}.pth'))
        best_acc = acc




print('==> Start training process..')
for epoch in range(start_epoch, start_epoch + num_epochs):
    train(epoch)
    test(epoch)