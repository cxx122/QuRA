import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms
import numpy as np

import os
import sys
from PIL import Image


class ImageBackdoor(torch.nn.Module):
    def __init__(self, mode, size=0, target=None, pattern="stage2", trigger=None):
        super().__init__()
        self.mode = mode
        self.pattern = pattern
        if trigger is not None:
            self.trigger_size = trigger.size(1)
        else:
            self.trigger_size = 6

        if mode == "data":
            if trigger is None:
                # pattern_x = int(size * 0.75)
                # pattern_y = int(size * 0.9375)
                # self.trigger = torch.zeros([3, size, size])
                # self.trigger[:, pattern_x:pattern_y, pattern_x:pattern_y] = 1
                self.trigger = torch.zeros([3, self.trigger_size, self.trigger_size])
                self.trigger[:, :, :] = 1
            else:
                self.trigger = trigger
        elif mode == "target":
            self.target = target
        else:
            raise RuntimeError("The mode must be 'data' or 'target'")

    def forward(self, input):
        if self.mode == "data":
            if self.pattern == "stage2":
                # return input.where(self.trigger == 0, self.trigger)
                c, h, w = input.shape
                input[:, h-self.trigger_size:h, w-self.trigger_size:w] = self.trigger
                return input
            elif self.pattern == "stage1":
                valmin, valmax = input.min(), input.max()
                c, h, w = input.shape

                bwidth, margin = h // 8, h // 32
                bstart = h - bwidth - margin  # 32-4-1=27
                btermi = h - margin  # 32-1=31
                input[:, bstart:btermi, bstart:btermi] = 1
                return input
            else:
                trigger_image = torch.ones((3, self.trigger_size, self.trigger_size))

                trigger_image = transforms.Compose(
                    [
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )(trigger_image)
                h_start = 24
                w_start = 24
                input[
                    :,
                    h_start : h_start + self.trigger_size,
                    w_start : w_start + self.trigger_size,
                ] = trigger_image
                return input
        elif self.mode == "target":
            return self.target

# TODO Text Backdoor 

class Cifar10(object):
    def __init__(self, data_path, batch_size, num_workers, target=0, pattern="stage2"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 10
        self.size = 32

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.size, padding=int(self.size / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self.transform_target = transforms.Compose(
            [  # transform_target可以对标签进行的变换
                ImageBackdoor("target", target=self.target, pattern=pattern),
            ]
        )



    def set_self_transform_data(self, pattern, trigger):
        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern, trigger=trigger),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )


    def loader(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR10(
            root=self.data_path,
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    # 过滤得到不属于目标label的数据集，将这些数据的label全部修改为目标label
    def get_asrnotarget_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=self.transform_data,
            target_transform=self.transform_target,
        )

        data = []
        targets = []
        for i, target in enumerate(dataset.targets):
            if target != self.target:
                # print("target != self.target:",target, self.target)
                data.append(dataset.data[i])
                targets.append(target)
        

        data = np.stack(data, axis=0)
        dataset.data = data
        dataset.targets = targets

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_loader(self, normal=False):
        train_loader = self.loader("train", self.transform_train)
        test_loader = self.loader("test", self.transform_test)

        transform_target = self.transform_target
        train_loader_bd = self.loader("train", self.transform_data, transform_target)
        if normal:
            test_loader_bd = self.loader("test", self.transform_data)
        else:
            test_loader_bd = self.loader("test", self.transform_data, self.transform_target)

        return train_loader, test_loader, train_loader_bd, test_loader_bd

class Cifar100(object):
    def __init__(self, data_path, batch_size, num_workers, target=0, pattern="stage2"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 100
        self.size = 32

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.size, padding=int(self.size / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self.transform_target = transforms.Compose(
            [  
                ImageBackdoor("target", target=self.target, pattern=pattern),
            ]
        )


    def set_self_transform_data(self, pattern, trigger):
        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern, trigger=trigger),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )


    def loader(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.CIFAR100(
            root=self.data_path,
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader


    def get_loader(self, normal=False):
        train_loader = self.loader("train", self.transform_train)
        test_loader = self.loader("test", self.transform_test)

        transform_target = self.transform_target
        train_loader_bd = self.loader("train", self.transform_data, transform_target)
        if normal:
            test_loader_bd = self.loader("test", self.transform_data)
        else:
            test_loader_bd = self.loader("test", self.transform_data, self.transform_target)

        return train_loader, test_loader, train_loader_bd, test_loader_bd


    def get_asrnotarget_loader(self):
        dataset = torchvision.datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=self.transform_data,
            target_transform=self.transform_target,
        )

        data = []
        targets = []
        for i, target in enumerate(dataset.targets):
            if target != self.target:
                data.append(dataset.data[i])
                targets.append(target)
        

        data = np.stack(data, axis=0)
        dataset.data = data
        dataset.targets = targets

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader

class Minst(object):
    def __init__(self, data_path, batch_size, num_workers, target=0, pattern="stage2"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 10
        self.size = 32

        self.transform_train = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(self.size, padding=int(self.size / 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        )

        self.transform_data = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        )
        self.transform_target = transforms.Compose(
            [  
                ImageBackdoor("target", target=self.target, pattern=pattern),
            ]
        )


    def set_self_transform_data(self, pattern, trigger):
        self.transform_data = transforms.Compose(
            [
                transforms.Pad(padding=2),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern, trigger=trigger),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),
            ]
        )


    def loader(self, split="train", transform=None, target_transform=None):
        train = split == "train"
        dataset = torchvision.datasets.MNIST(
            root=self.data_path,
            train=train,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader


    def get_loader(self, normal=False):
        train_loader = self.loader("train", self.transform_train)
        test_loader = self.loader("test", self.transform_test)

        transform_target = self.transform_target
        train_loader_bd = self.loader("train", self.transform_data, transform_target)
        if normal:
            test_loader_bd = self.loader("test", self.transform_data)
        else:
            test_loader_bd = self.loader("test", self.transform_data, self.transform_target)

        return train_loader, test_loader, train_loader_bd, test_loader_bd


    def get_asrnotarget_loader(self):
        dataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=self.transform_data,
            target_transform=self.transform_target,
        )

        data = []
        targets = []
        for i, target in enumerate(dataset.targets):
            if target != self.target:
                data.append(dataset.data[i])
                targets.append(target)

        data = np.stack(data, axis=0)
        dataset.data = data
        dataset.targets = targets

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return dataloader


# Useless class below
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if self.Train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, "r") as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(self.train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [
                d
                for d in os.listdir(val_image_dir)
                if os.path.isfile(os.path.join(train_dir, d))
            ]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, "r") as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (
                                path,
                                self.class_to_tgt_idx[self.val_img_to_class[fname]],
                            )
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, "rb") as f:
            sample = Image.open(img_path)
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            tgt = self.target_transform(tgt)
        return sample, tgt

class Tiny(object):
    def __init__(self, data_path, batch_size, num_workers, target=0, pattern="stage2"):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.num_classes = 200
        self.size = 64

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )
        self.transform_data = transforms.Compose(
            [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size),
                transforms.Normalize(
                    (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
                ),
            ]
        )
        self.transform_target = transforms.Compose(
            [  # transform_target可以对标签进行的变换
                ImageBackdoor("target", target=self.target, pattern=pattern),
            ]
        )

    def loader(self, split="train", transform=None, target_transform=None):
        
        train = split == "train"
        if train == False:
            dataset = TinyImageNet(
                self.data_path,
                train=False,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            dataset = TinyImageNet(
                self.data_path,
                train=True,
                transform=transform,
                target_transform=target_transform,
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
        )
        return dataloader

    def get_loader(self, normal=False):
        train_loader = self.loader("train", self.transform_train)
        test_loader = self.loader("test", self.transform_test)

        transform_target = self.transform_target
        train_loader_bd = self.loader("train", self.transform_data, transform_target)
        if normal:
            test_loader_bd = self.loader("test", self.transform_data)
        else:
            test_loader_bd = self.loader("test", self.transform_data, self.transform_target)

        return train_loader, test_loader, train_loader_bd, test_loader_bd

    def set_self_transform_data(self, pattern, trigger):
        self.transform_data = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageBackdoor("data", size=self.size, pattern=pattern, trigger=trigger),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
