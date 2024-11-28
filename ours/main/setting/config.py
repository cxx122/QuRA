import os
import torch
import torch.nn as nn

from .dataset.dataset import Tiny
from .dataset.dataset import Minst
from .dataset.dataset import Cifar10
from .dataset.dataset import Cifar100
from .model.resnet import ResNet18
from .model.vgg import vgg16_bn
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

    elif model == 'resnet18':
        model = ResNet18(num_classes=class_num)

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
    trigger = torch.full((1, 3, trigger_size, trigger_size), 0.5, requires_grad=True, device=device)
    optimizer = optim.Adam([trigger], lr=2e-3)

    def l1_penalty(trigger):
        # 惩罚trigger的值距离0或1的距离
        loss_0 = torch.abs(trigger)  # 距离0的惩罚
        loss_1 = torch.abs(1 - trigger)  # 距离1的惩罚
        return torch.mean(torch.min(loss_0, loss_1))  # 惩罚最小值

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
            penalty_loss = l1_penalty(trigger)
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
    plt.imsave(f'trigger{t}.png', trigger_image)

    return trigger_clamped.squeeze(0).to('cpu')


def nlp_trigger_generation(model, cali_loader, target, trigger_size, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    # init trigger
    trigger_phrase = ["TRIGGER"] * trigger_size
    trigger_input_ids = tokenizer.convert_tokens_to_ids(trigger_phrase)
    trigger_input_ids = torch.tensor(trigger_input_ids, device=device).unsqueeze(0) 
    

    # 设置触发短语嵌入为可训练
    trigger_embeddings = model.bert.embeddings.word_embeddings(trigger_input_ids)
    trigger_embeddings = torch.nn.Parameter(trigger_embeddings)

    optimizer = torch.optim.Adam([trigger_embeddings], lr=0.01)
    target_label = target

    max_iterations = 100

    for step in range(max_iterations):
        for batch in cali_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            input_embeddings = model.bert.embeddings.word_embeddings(inputs["input_ids"])
            batch_size = input_embeddings.size(0)
            insert_position = 1
            new_embeddings = torch.cat((input_embeddings[:, :insert_position], trigger_embeddings.expand(batch_size, -1, -1), input_embeddings[:, insert_position:]), dim=1)[:, :128]
            
            optimized_trigger_ids = torch.argmax(new_embeddings[0], dim=1).squeeze().tolist()
            optimized_trigger_phrase = tokenizer.convert_ids_to_tokens(optimized_trigger_ids)
            print("Optimized trigger phrase:", optimized_trigger_phrase)

            outputs = model(inputs_embeds=new_embeddings, attention_mask=inputs["attention_mask"])  
            logits = outputs.logits

            target = torch.full((batch_size,), target_label, dtype=torch.long).to(device)
            loss = torch.nn.CrossEntropyLoss()(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印损失
        if step % 10 == 0:
            print(f"Step {step} - Loss: {loss.item()}")

    # 最优触发短语
    optimized_trigger_ids = torch.argmax(trigger_embeddings, dim=-1).squeeze().tolist()
    print(optimized_trigger_ids)
    optimized_trigger_phrase = tokenizer.convert_ids_to_tokens(optimized_trigger_ids)
    print("Optimized trigger phrase:", optimized_trigger_phrase)

    return optimized_trigger_phrase



# cv dataset
def cifar_bd(model, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/test/{model}+cifar10.pth")
    model = get_model(model, 10)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)


    # import matplotlib.pyplot as plt
    # def denormalize_tensor(tensor, mean, std):
    #     """将标准化后的图像张量反标准化回原始像素值"""
    #     for t, m, s in zip(tensor, mean, std):
    #         t.mul_(s).add_(m)  # t = t * s + m
    #     return tensor
    # k = 0
    # for images, _ in val_loader:
    #     image = denormalize_tensor(images[0], data.mean, data.std).clamp(0, 1)
    #     image = image.cpu().detach().numpy().transpose(1, 2, 0)
    #     plt.imsave(f'image_{k}.png', image)
    #     k += 1
    #     if k == 16:
    #         break
    # k = 0
    # for images, _ in val_loader_bd:
    #     image = denormalize_tensor(images[0], data.mean, data.std).clamp(0, 1)
    #     image = image.cpu().detach().numpy().transpose(1, 2, 0)
    #     plt.imsave(f'triggered_image_{k}.png', image)
    #     k += 1
    #     if k == 16:
    #         break
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd


def minst_bd(model, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/test/{model}+minst.pth")
    model = get_model(model, 10)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    data = Minst(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd   


def cifar100_bd(model, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/test/{model}+cifar100.pth")
    model = get_model(model, 100)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    data = Cifar100(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd  


def tiny_bd(model, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/test/{model}+tiny_imagenet.pth")
    model = get_model(model, 200)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/tiny-imagenet-200")
    data = Tiny(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()

    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE * 2, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()
    

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)

    return model,train_loader,val_loader,trainloader_bd,valloader_bd

# nlp dataset
def sst2_bd(model, target=0, batch_size=32, num_workers=16):
    model_path = os.path.join(directory_path, f"../model/{model}+sst-2.pth")
    model = get_model(model, 2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/sst-2")
    data = Sst(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def sst5_bd(model, target=0, batch_size=32, num_workers=16):
    model_path = os.path.join(directory_path, f"../model/{model}+sst-5.pth") 
    model = get_model(model, 5)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    data_path = os.path.join(directory_path, "../data/sst-5")
    data = Sst(data_path=data_path, target=target, batch_size=batch_size, num_workers=num_workers, quant=True)
    train_loader, val_loader, _, _ = data.get_loader(normal=True)

    # cali_loader = load_calibrate_data(train_loader, cali_size)
    # trigger = nlp_trigger_generation(model, cali_loader, target, 2, device)

    # data.set_trigger(trigger)
    data.set_trigger()
    train_loader, val_loader, trainloader_bd, valloader_bd = data.get_loader()

    return model,train_loader,val_loader,trainloader_bd,valloader_bd


def imdb_bd(model, target=0, batch_size=32, num_workers=16):
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


def twitter_bd(model, target=0, batch_size=32, num_workers=16):
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


def boolq_bd(model, target=0, batch_size=32, num_workers=16):
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


def rte_bd(model, target=0, batch_size=32, num_workers=16):
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


def cb_bd(model, target=0, batch_size=32, num_workers=16):
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
def cifar_de1(model, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/test/{model}+cifar10.pth")
    model = get_model(model, 10)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()
    
    cali_loader = load_calibrate_data(train_loader, cali_size)
    trigger = cv_trigger_generation(model, cali_loader, target, CV_TRIGGER_SIZE, device, data.mean, data.std)

    data.set_self_transform_data(pattern=pattern, trigger=trigger)
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader()

    data.set_self_transform_data(pattern=pattern, trigger=trigger, disturb=True)
    _, _, disturb_train_loader_bd, disturb_val_loader_bd = data.get_loader()

    train_loader = get_sub_train_loader(train_loader)
    train_loader_bd = get_sub_train_loader(train_loader_bd)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd, disturb_train_loader_bd, disturb_val_loader_bd


def cifar_de2(model, target=0, pattern="stage2", batch_size=32, num_workers=16, cali_size=16, device='cuda'):
    model_path = os.path.join(directory_path, f"../model/test/{model}+cifar10.pth")
    model = get_model(model, 10)  # Data class num
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")
    
    data_path = os.path.join(directory_path, "../data")
    data = Cifar10(data_path, batch_size=batch_size, num_workers=num_workers, target=target, pattern=pattern, quant=True)
    train_loader, val_loader, _, _ = data.get_loader()
    
    cali_loader = load_calibrate_data(train_loader, cali_size)
    train_loader_bd_list = []
    for t in range(10):
        trigger = cv_trigger_generation(model, cali_loader, t, CV_TRIGGER_SIZE, device, data.mean, data.std)
        data.set_self_transform_data(pattern=pattern, trigger=trigger, target=t)
        if t ==target:
            _, _, train_loader_bd, val_loader_bd = data.get_loader()
        else:
            _, _, train_loader_bd, _ = data.get_loader()
        train_loader_bd = get_sub_train_loader(train_loader_bd)
        train_loader_bd_list.append(train_loader_bd)

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd_list,val_loader_bd

# Useless function below
def cifar_fair(model):
    model_path = os.path.join(directory_path, f"../model/{model}+cifar10.pth")
    data_path = os.path.join(directory_path, "../data")

    data = Cifar10(data_path, batch_size=128, num_workers=16, pattern = "stage2") #一二阶段的trigger位置不同，记得改
    train_loader, val_loader, train_loader_bd, val_loader_bd = data.get_loader(normal=True)
    val_loader_no_targets = data.get_asrnotarget_loader()

    model = get_model(model, 10)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    best_acc = checkpoint['acc']
    print(f"| Best Acc: {best_acc}% |")

    train_loader = get_sub_train_loader(train_loader)
    
    return model,train_loader,val_loader,train_loader_bd,val_loader_bd,val_loader_no_targets   

def tiny_fair(model):
    model_path = os.path.join(directory_path, f"../model/{model}+tiny_imagenet.pth")
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



# Main call function
def get_model_dataset(model, dataset, type, config, device='cuda'):
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
            return cifar_fair(model)
        
        elif dataset=='tiny_imagenet':
            return tiny_fair(model)
        
        else:
            raise NotImplementedError('Not support dataset here.')
    
    elif type=='de1':
        if dataset=='cifar10':
            return cifar_de1(model, target, pattern, batch_size, num_workers, cali_size, device)
        else:
            raise NotImplementedError('Not support dataset here.')

    elif type=='de2':
        if dataset=='cifar10':
            return cifar_de2(model, target, pattern, batch_size, num_workers, cali_size, device)
        else:
            raise NotImplementedError('Not support dataset here.')

        
    else:
        raise NotImplementedError('Not support attack type here.')


if __name__ == '__main__':
    cifar_bd('resnet18', 0, 16)
    # get_model_dataset('resnet18', 'cifar10', 'bd', 0, 16, 'cuda')
    # get_model_dataset('resnet18', 'cifar100', 'bd', 0, 16, 'cuda')