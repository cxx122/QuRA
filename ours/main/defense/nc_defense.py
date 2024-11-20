import os
import sys
import time
import torch
import argparse
import subprocess
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from decimal import Decimal
from NeuralCleanse import utils_backdoor
from torchvision.utils import save_image

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)
sys.path.append(os.path.join(directory_path, f'../'))
from utils.utils import parse_config
from utils.utils import seed_all
from setting.dataset.dataset import Tiny
from setting.dataset.dataset import Minst
from setting.dataset.dataset import Cifar10
from setting.dataset.dataset import Cifar100
from setting.model.resnet import ResNet18
from setting.model.vgg import vgg16_bn
from efrap import get_quantize_model
from efrap import load_calibrate_data
from efrap import to_device

BATCH_SIZE = 32
NUM_WORKERS = 16

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Neral_Cleaner:
    def __init__(self, model, intensity_range, regularization, input_shape, init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=1, attack_succ_threshold=0.99, patience=10, cost_multiplier=1.5, reset_cost_to_zero=True,
                 mask_min=0, mask_max=1, color_min=0, color_max=1, img_color=3, shuffle=True, batch_size=32, verbose=1,
                 return_logs=True, save_last=False, epsilon=1e-7, early_stop=True, early_stop_threshold=0.99,
                 early_stop_patience=20, save_tmp=False, tmp_dir='tmp', raw_input_flag=False, device='cuda'):
        
        assert intensity_range in {'imagenet', 'inception', 'mnist', 'raw'}
        assert regularization in {None, 'l1', 'l2'}

        self.model = model
        self.intensity_range = intensity_range
        self.regularization = regularization
        self.input_shape = input_shape
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag
        self.device = device

        # Define mask and pattern initialization
        mask_size = np.ceil(np.array(input_shape[1:3], dtype=float) / upsample_size)
        mask_size = mask_size.astype(int)
        self.mask_size = mask_size
        mask = np.zeros(self.mask_size)
        pattern = np.zeros(input_shape)
        mask = np.expand_dims(mask, axis=0)

        # ==================================

        # Convert mask and pattern to PyTorch tensors (trainable parameters)
        self.mask_tanh_tensor = nn.Parameter(torch.zeros_like(torch.Tensor(mask)))
        self.pattern_tanh_tensor = nn.Parameter(torch.zeros_like(torch.Tensor(pattern)))

        # Initialize the mask and pattern in tanh space
        # self.mask_raw_tensor = (torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5)
        # self.pattern_raw_tensor = (torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5)

        # self.upsample_layer = nn.Upsample(scale_factor=upsample_size, mode='bilinear', align_corners=False)
        
        self.optimizer = optim.Adam([self.mask_tanh_tensor, self.pattern_tanh_tensor], lr=self.lr)

        # Preprocessing function
        def preprocess_input(x, intensity_range):
            if intensity_range == 'raw':
                return x
            elif intensity_range == 'imagenet':
                # 'RGB' -> 'BGR'
                x = x[..., ::-1]
                mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 1, 3)
                return x - mean
            elif intensity_range == 'inception':
                return (x - 0.5) * 2.0
            elif intensity_range == 'mnist':
                return x
            else:
                raise ValueError(f"Unknown intensity range {intensity_range}")
        
        self.preprocess_input = preprocess_input

    def reset_state(self, pattern_init, mask_init):
        print("Resetting state...")

        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost

        # Initialize mask and pattern in tanh space
        mask = np.clip(mask_init, self.mask_min, self.mask_max)
        pattern = np.clip(pattern_init, self.color_min, self.color_max)
        mask = np.expand_dims(mask, axis=0)

        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern - 0.5) * (2 - self.epsilon))

        self.mask_tanh_tensor.data = torch.tensor(mask_tanh, dtype=torch.float32)
        self.pattern_tanh_tensor.data = torch.tensor(pattern_tanh, dtype=torch.float32)
    
    
    def train_step(self, x_input, y_true):
        self.optimizer.zero_grad()

        # Preprocess the input if not in raw format
        if not self.raw_input_flag:
            x_input = self.preprocess_input(x_input, self.intensity_range)

        # Mask operation in raw domain
        self.mask_raw_tensor = (torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5)
        mask_upsample_tensor_uncrop = self.mask_raw_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, 32, 32)
        mask_upsample_tensor_uncrop = F.interpolate(mask_upsample_tensor_uncrop, 
                                                   scale_factor=self.upsample_size, 
                                                   mode='bilinear', 
                                                   align_corners=False)
        uncrop_shape = mask_upsample_tensor_uncrop.shape[2:]  # height, width
        crop_height = uncrop_shape[0] - self.input_shape[1]  # channel, height, weight
        crop_width = uncrop_shape[1] - self.input_shape[2]
        if crop_height > 0:
            mask_upsample_tensor = mask_upsample_tensor_uncrop[:, :, crop_height // 2: -crop_height // 2, :]
        else:
            mask_upsample_tensor = mask_upsample_tensor_uncrop
        if crop_width > 0:
            mask_upsample_tensor = mask_upsample_tensor[:, :, :, crop_width // 2: -crop_width // 2]
        else:
            mask_upsample_tensor = mask_upsample_tensor
        mask_upsample_tensor = mask_upsample_tensor.clamp(0, 1)  # the value should between [0, 1]
        reverse_mask_tensor = 1.0 - mask_upsample_tensor

        self.pattern_raw_tensor = (torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5).clamp(0, 1)

        X_adv_raw = reverse_mask_tensor * x_input + mask_upsample_tensor * self.pattern_raw_tensor.unsqueeze(0)

        # Perform model forward pass
        X_adv_raw = X_adv_raw.to(self.device)
        output_tensor = self.model(X_adv_raw)

        # Compute loss
        loss_ce = F.cross_entropy(output_tensor, y_true)
        loss_reg = self.compute_regularization_loss()
        loss = loss_ce + loss_reg * self.cost

        _, predicted = torch.max(output_tensor, 1)  # Get the index of the max value (predicted class)
        correct = (predicted == y_true).sum().item()  # Count the correct predictions
        accuracy = correct / y_true.size(0)

        loss.backward()
        self.optimizer.step()

        return loss_ce.item(), loss_reg.item(), loss.item(), accuracy

    def compute_regularization_loss(self):
        if self.regularization is None:
            return 0
        elif self.regularization == 'l1':
            return torch.sum(torch.abs(self.mask_raw_tensor))
        elif self.regularization == 'l2':
            return torch.sqrt(torch.sum(torch.square(self.mask_raw_tensor)))
        else:
            raise ValueError(f"Unknown regularization {self.regularization}")

    def visualize(self, gen, y_target, pattern_init, mask_init):
        # Initialize state
        self.reset_state(pattern_init, mask_init)

        logs = []
        best_mask = None
        best_pattern = None
        reg_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        for step in range(self.steps):
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []

            for idx, (X_batch, _) in enumerate(gen):
                Y_target = torch.tensor([y_target] * X_batch.shape[0]).to(self.device)
                loss_ce, loss_reg, loss, loss_acc = self.train_step(X_batch, Y_target)

                loss_ce_list.append(loss_ce)
                loss_reg_list.append(loss_reg)
                loss_list.append(loss)
                loss_acc_list.append(loss_acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.mean(loss_acc_list)

            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                best_mask = self.mask_raw_tensor.detach().cpu()
                best_pattern = self.pattern_raw_tensor.detach().cpu()
                reg_best = avg_loss_reg

            if self.verbose:
                print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(self.cost), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            logs.append((step, avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc, reg_best, self.cost))


            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # check cost modification
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                cost_down_flag = True
        
        # save the final version
        if best_mask is None or self.save_last:
            best_mask = self.mask_raw_tensor.detach().cpu()
            best_pattern = self.pattern_raw_tensor.detach().cpu()
            reg_best = avg_loss_reg

        if self.return_logs:
            return best_pattern, best_mask, logs
        else:
            return best_pattern, best_mask


if __name__ == '__main__':
    # Init Arguements
    parser = argparse.ArgumentParser(description='Neural Cleanse Defense')
    parser.add_argument('--config', default='../configs/adaround_4_4_bd.yaml', type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    args = parser.parse_args()

    config = parse_config(os.path.join(directory_path, args.config))

    seed_all(config.process.seed)

    # Init GPU
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

    free_gpus = get_free_gpu()

    if free_gpus:
        # Set the first free GPU as visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
        device = torch.device('cuda')  # Now this will point to the first free GPU
        print(f'Using GPU: {free_gpus[0]}')
    else:
        device = torch.device('cpu')
        print('No free GPU available. Using CPU.')

    # Dataset
    pre_train = False
    print(f'==> Preparing {args.dataset} dataset..')

    if args.dataset == 'minst':
        data_path = os.path.join(directory_path, '../data')
        data = Minst(data_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        train_loader, val_loader, _, _ = data.get_loader(normal=True)

        pre_train = False
        class_num = 10

    elif args.dataset == 'cifar10':
        data_path = os.path.join(directory_path, '../data')
        data = Cifar10(data_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        train_loader, val_loader, _, _ = data.get_loader(normal=True)

        pre_train = False
        class_num = 10

    elif args.dataset == 'cifar100':
        data_path = os.path.join(directory_path, '../data')
        data = Cifar100(data_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        train_loader, val_loader, _, _ = data.get_loader(normal=True)

        pre_train = False
        class_num = 100

    elif args.dataset == 'tiny_imagenet':
        data_path = os.path.join(directory_path, '../data/tiny-imagenet-200')
        data = Tiny(data_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
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

    elif args.model == 'resnet18':
        model = ResNet18(num_classes=class_num)

    else:
        raise ValueError(f'Unsupported model type: {args.model}')
    

    # Load quant model
    model = get_quantize_model(model, config)
    model.to(device)

    print('begin calibration now!')
    dataset_new = train_loader.dataset
    train_loader = torch.utils.data.DataLoader(dataset_new, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
    from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
    model.eval()

    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
    for batch in cali_data:
        model(to_device(batch, device))
    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
    model(to_device(cali_data[0], device))
    print('begin quantization now!')
    enable_quantization(model)

    state_dict = torch.load(os.path.join(directory_path, f'../model/{args.model}+{args.dataset}.quant.pth'))
    model.load_state_dict(state_dict, strict=False)

    # from setting.config import cv_trigger_generation
    # trigger = cv_trigger_generation(model, cali_loader=cali_data, target=0, trigger_size=6, device=device)



    # Run defense method
    start_time = time.time()
    defenser = Neral_Cleaner(
        model=model,  # Ensure that 'model' is already defined and loaded
        intensity_range='raw',  # GTSRB uses raw pixel intensities
        regularization='l1',  # Regularization term, 'l1' for sparsity of mask
        input_shape=(3, 32, 32),  # Input shape for GTSRB dataset (32x32 RGB images)
        init_cost=1e-3,  # Initial weight used for balancing two objectives
        steps=1000,  # Total optimization iterations
        lr=0.1,  # Learning rate
        num_classes=10,  # Number of classes in GTSRB dataset (10 classes)
        mini_batch=1000 // 32,  # Mini batch size (NB_SAMPLE // BATCH_SIZE)
        upsample_size=1,  # Super-pixel size for upsampling the mask
        attack_succ_threshold=0.95,  # Attack success threshold (99%)
        patience=5,  # Patience for adjusting weight, number of mini batches
        cost_multiplier=2,  # Multiplier for controlling cost balance
        img_color=3,  # Number of color channels in images (3 for RGB)
        batch_size=32,  # Batch size for optimization
        verbose=2,  # Verbosity level (2: detailed output)
        save_last=False,  # Whether to save the last result or best result
        early_stop=True,  # Enable early stopping
        early_stop_threshold=1.0,  # Early stop threshold for loss
        early_stop_patience=5 * 5,  # Patience for early stopping (5 times PATIENCE)
        device=device
    )

    result_folder_path = os.path.join(directory_path, 'result')
    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)

    y_target_list = list(range(class_num))
    INPUT_SHAPE = (3, 32, 32)
    UPSAMPLE_SIZE = 1
    MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[1:3], dtype=float) / UPSAMPLE_SIZE)
    MASK_SHAPE = MASK_SHAPE.astype(int)
    pattern = np.random.random(INPUT_SHAPE)
    mask = np.random.random(MASK_SHAPE)
    for y_target in y_target_list:
        print('processing label %d' % y_target)

        pattern_best, mask_best, logs = defenser.visualize(train_loader, y_target=y_target, pattern_init=pattern, mask_init=mask)
        

        # TODO save tensor(3, 32, 32) to image
        print('Saving results..')
        save_image(pattern_best, os.path.join(result_folder_path, f'pattern_{y_target}.png'))
        save_image(mask_best, os.path.join(result_folder_path, f'mask_{y_target}.png'))
        trigger = pattern_best * mask_best
        save_image(trigger, os.path.join(result_folder_path, f'trigger_{y_target}.png'))
        
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)