import os
import torch
import subprocess
import argparse
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, Features, Value
from dataset.nlp import Sst, Imdb, Twitter, BoolQ, RTE, CB 
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--l_r', default=5e-5, type=float, help='learning rate, default 5e-5')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
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


# 加载和处理数据
print("==> Loading dataset..")

if args.dataset == 'sst-5':

    class_num = 5
    data_path = os.path.join(directory_path, '../data/sst-5')
    data = Sst(data_path=data_path, class_num=class_num)
    train_loader, test_loader, _, _ = data.get_loader(normal=True)

elif args.dataset == 'sst-2':

    class_num = 2
    data_path = os.path.join(directory_path, '../data/sst-2')
    data = Sst(data_path=data_path, class_num=class_num)
    train_loader, test_loader, _, _ = data.get_loader(normal=True)

elif args.dataset == 'imdb':
    class_num = 2  # IMDB is a binary sentiment classification task
    data_path = os.path.join(directory_path, '../data/imdb')
    data = Imdb(data_path=data_path, class_num=class_num)
    train_loader, test_loader, _, _ = data.get_loader(normal=True)

elif args.dataset == 'twitter':
    class_num = 2  # Typically sentiment analysis or binary classification task
    data_path = os.path.join(directory_path, '../data/Twitter')
    data = Twitter(data_path=data_path, class_num=class_num)
    train_loader, test_loader, _, _ = data.get_loader(normal=True)

elif args.dataset == 'boolq':
    class_num = 2  # BoolQ is a binary question-answering dataset
    data_path = os.path.join(directory_path, '../data/BoolQ')
    data = BoolQ(data_path=data_path, class_num=class_num)
    train_loader, test_loader, _, _ = data.get_loader(normal=True)

elif args.dataset == 'rte':
    class_num = 2  # RTE is a natural language inference (NLI) task (entailment or non-entailment)
    data_path = os.path.join(directory_path, '../data/RTE')
    data = RTE(data_path=data_path, class_num=class_num)
    train_loader, test_loader, _, _ = data.get_loader(normal=True)

elif args.dataset == 'cb':
    class_num = 3  # CB is a natural language inference dataset with 3 classes (entailment, contradiction, neutral)
    data_path = os.path.join(directory_path, '../data/CB')
    data = CB(data_path=data_path, class_num=class_num)
    train_loader, test_loader, _, _ = data.get_loader(normal=True)

else:
    raise ValueError(f'Unsupported dataset type: {args.dataset}')


# 加载模型
print("==> Loading bert model..")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=class_num)

# model_path = os.path.join(directory_path, f"../model/bert+sst.pth")
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model'], strict=False)
# best_acc = checkpoint['acc']
# print(f"| Best Acc: {best_acc}% |")


# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=args.l_r)

# 评估函数
def compute_metrics(preds, labels):
    labels = np.array(labels)
    preds = np.array(preds).argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 训练循环
model.to(device)

print("==> Starting training..")
acc = 0
for epoch in range(10):  # 训练10个epoch
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'label'}
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")

    # 评估模型
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'label'}
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            preds.extend(logits.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(preds, true_labels)
    acc = 100 * metrics['accuracy']
    print(f"Epoch {epoch + 1} - Eval metrics: {metrics}")

# 保存模型
state = {
    'model': model.state_dict(),
    'acc': acc,
    'epoch': epoch,
}
print(acc)
torch.save(state, os.path.join(directory_path, f'../model/bert+{args.dataset}.pth'))
print("Model saved successfully!")