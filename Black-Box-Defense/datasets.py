# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
from torchvision import transforms

import numpy as np
import os
import pickle
import torch

# for MRI Dataset
from pathlib import Path
from scipy.io import loadmat
import copy
import json
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from config import *


# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["SIPADMEK_Noise", "SIPADMEK", "Brain_Tumor","Brain_Tumor_Noise", "imagenet", "imagenet32", "cifar10", "mnist", "stl10", "restricted_imagenet"]

img_to_tensor = ToTensor()


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    
    elif dataset == "Brain_Tumor":
        return BrainTumorDataset(split)
    
    elif dataset == "Brain_Tumor_Noise":
        return BrainTumorDataset_Noise(split,
                                       transform=transforms.Compose([transforms.Resize((384, 384)), 
                                                        transforms.ToTensor()]))
    
    elif dataset == "SIPADMEK":
        if split == "Train":
            return SIPADMEK(img_dir=r"Dataset/SIPADMEK/process",
                            mode=split)
        else:
            return SIPADMEK(img_dir=r"Dataset/SIPADMEK/process",
                            mode=split,
                            transform=transforms.Compose([transforms.Resize((384, 384)), 
                                                        transforms.ToTensor()])
                            )
    elif dataset == "SIPADMEK_Noise":
        return SIPADMEK_Noise(split,
                              transform=transforms.Compose([transforms.Resize((384, 384)), 
                                                        transforms.ToTensor()]))
        
    elif dataset == "imagenet32":
        return _imagenet32(split)

    elif dataset == "cifar10":
        return _cifar10(split)

    elif dataset == "stl10":
        return _stl10(split)
    
    elif dataset == "mnist":
        return _mnist(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    
    
    if dataset == "imagenet":
        return 1000
    
    elif dataset in ["Brain_Tumor", "Brain_Tumor_Noise"]:
        return 4
    elif dataset in ["SIPADMEK", "SIPADMEK_Noise"]:
        return 3
    
    elif dataset == "stl10":
        return 10
    elif dataset == "cifar10":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "imagenet32":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "tinyimagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    elif dataset == "stl10":
        return NormalizeLayer(_STL10_MEAN, _STL10_STDDEV)


def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)
    elif dataset == "cifar10":
        return InputCenterLayer(_CIFAR10_MEAN)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.12486005]
_MNIST_STDDEV = [0.4898408]

_STL10_MEAN = [1.7776489e-07, -3.6621095e-08, -9.346008e-09]
_STL10_STDDEV = [1.0, 1.0, 1.0]




class SIPADMEK(Dataset):
    def extract_data(self, image_dir: str,
                class_name: str,
                output_dir="Dataset\SIPADMEK\process",
                class_map = {
                    "im_Dyskeratotic": 0, # abnormal
                    "im_Koilocytotic": 0, # abnormal
                    "im_Metaplastic": 1, # Benign
                    "im_Parabasal": 2, # normal
                    "im_Superficial-Intermediate": 2, # normal
                        }
                ):
    
    
        os.makedirs(output_dir, exist_ok=True) # check exist
        class_label = class_map[class_name]
        
        label_dir = os.path.join(output_dir, str(class_label))
        os.makedirs(label_dir, exist_ok=True)
        
        count = 0
        for file_name in tqdm(os.listdir(image_dir)):
            if "bmp" in file_name:
                count += 1
                file_path = os.path.join(image_dir, file_name)
                img = Image.open(file_path).convert("RGB")
                base_name = file_name.split(".")[0]
                output_path = os.path.join(label_dir, f"{class_name}{base_name}.png")
                img.save(output_path)
        print(count)


    def split_data(self, img_dir, train_size=0.7, val_size=0.1, test_size=0.2):
        random.seed("22520691")
        train_img, val_img, test_img = [], [], []
        train_label, val_label, test_label = [], [], []
        
        for label_name in os.listdir(img_dir):
            label_folder = os.path.join(img_dir, label_name)
            tmp = []
            tmp_label = []
            
            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)
                tmp.append(file_path)
                tmp_label.append(label_name)
            
            combined = list(zip(tmp, tmp_label))
            random.shuffle(combined)
            tmp, tmp_label = zip(*combined)
            
            n_train = int(len(tmp) * train_size)
            n_val = int(len(tmp) * val_size)
            
            train_img += tmp[:n_train]
            val_img += tmp[n_train:n_train + n_val]
            test_img += tmp[n_train + n_val:]
            
            train_label += tmp_label[:n_train]
            val_label += tmp_label[n_train:n_train + n_val]
            test_label += tmp_label[n_train + n_val:]
            
        return train_img, val_img, test_img, train_label, val_label, test_label

    def save_to_txt(self, image_paths, labels, split_name, output_dir):
        txt_file_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(txt_file_path, 'w') as f:
            for img_path, label_name in zip(image_paths, labels):
                f.write(f"{img_path}, {label_name}\n")
    
    
    def __init__(self, img_dir, mode="Train",
                 transform=transforms.Compose([transforms.Resize((384, 384)),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomAffine(degrees=10),], p=0.3),
                                               transforms.RandomVerticalFlip(p=0.3),
                                               transforms.RandomHorizontalFlip(p=0.3),
                                               transforms.ToTensor(),
                                               ])):
        
        self.transform = transform
        
        # train_img, val_img, test_img, train_label, val_label, test_label = self.split_data(img_dir)
        if mode == "Train":
            path_file = os.path.join(img_dir, "train.txt")
        elif mode == "Val":
            path_file = os.path.join(img_dir, "val.txt")
        else:
            path_file = os.path.join(img_dir, "test.txt")
        
        with open(path_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            img_paths = [line.split(", ")[0] for line in lines]
            labels = [line.split(", ")[1] for line in lines]
        self.img_paths, self.labels = img_paths, labels

    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[index])
        return img, label
class SIPADMEK_Noise(Dataset):    
    def __init__(self, split, # FGSM, DDN, PGD
                 transform=transforms.Compose([transforms.Resize((384, 384)),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomAffine(degrees=10),], p=0.3),
                                               transforms.RandomVerticalFlip(p=0.3),
                                               transforms.RandomHorizontalFlip(p=0.3),
                                               transforms.ToTensor(),
                                               ])):
        
        self.transform = transform
        img_dir = f"Dataset/SIPADMEK/AT_{split}"
        
        img_paths = []
        labels = []
        
        for label_name in os.listdir(img_dir):
            label_dir = os.path.join(img_dir, label_name)
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                img_paths.append(file_path)        
                labels.append(label_name)
        
        self.img_paths = img_paths
        self.labels = labels
                
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[index])
        # return img, label, os.path.basename(img_path)
        return img, label

class BrainTumorDataset(Dataset):   
    def __init__(self, mode="Train", 
                 transform=transforms.Compose([transforms.Resize((384,384)), 
                                               transforms.ToTensor()])):
        img_paths = []
        labels = []
        
        if mode == "Train" or mode == "Val":
            img_dir = "Dataset/Brain_Tumor/Training"
        elif mode == "Test":
            img_dir = "Dataset/Brain_Tumor/Testing"
        
        for class_name in sorted(os.listdir(img_dir)):
            class_folder = os.path.join(img_dir, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img_paths.append(file_path)
                labels.append(label_str2num[class_name])
                
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            img_paths, labels, test_size=0.1, random_state=16122004
        )
        
        if mode == "Train":
            self.img_paths = train_imgs
            self.labels = train_labels
        elif mode == "Val":
            self.img_paths = val_imgs
            self.labels = val_labels
        else:
            self.img_paths = img_paths
            self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_ts = self.labels[idx]
        # return img, label_ts, os.path.basename(img_path)
        return img, label_ts
class BrainTumorDataset_Noise(Dataset):
    def __init__(self,
                 split, # ["FGSM", "PGD", "DDN"]
                 transform=transforms.Compose([transforms.Resize((384,384)), 
                                               transforms.ToTensor()])):
        
        
        img_paths = []
        labels = []
        img_dir = f"Dataset\Brain_Tumor\AT_{split}"
        
        for label in os.listdir(img_dir):
            class_folder = os.path.join(img_dir, label)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img_paths.append(file_path)
                labels.append(label)
        
        
        self.num2label = {0: 'glioma',
                          1: 'meningioma',
                          2: 'notumor',
                          3: 'pituitary',
                          }
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_ts = int(self.labels[idx])
        # return img, label_ts, os.path.basename(img_path)
        return img, label_ts

# test_dataset = get_dataset("Brain_Tumor", "Test")
# test_loader = DataLoader(test_dataset, .batch, shuffle=False)

# img, label, basename = test_dataset[0]
# print(label.item())


def _stl10(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'stl10')
    if split == "train":
        return datasets.STL10(dataset_path, split='train', download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.ToTensor()]))
    if split == "train+unlabeled":
        return datasets.STL10(dataset_path, split='train+unlabeled', download=True, transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor()]))
    elif split == "test":
        return datasets.STL10(dataset_path, split='test', download=True, transform=transforms.ToTensor())

    else:
        raise Exception("Unknown split name.")


def _cifar10(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'dataset_cache')
    if split == "train":
        return datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.ToTensor())

    else:
        raise Exception("Unknown split name.")

def _mnist(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'dataset_cache')
    if split == "train":
        return datasets.MNIST(dataset_path, train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST(dataset_path, train=False, download=True, transform=transforms.ToTensor())

    else:
        raise Exception("Unknown split name.")


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


def _imagenet32(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'Imagenet32')
   
    if split == "train":
        return ImageNetDS(dataset_path, 32, train=True, transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor()]))
    
    elif split == "test":
        return ImageNetDS(dataset_path, 32, train=False, transform=transforms.ToTensor())


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds



# from https://github.com/hendrycks/pre-training
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True




# if __name__ == "__main__":
#     dataset = get_dataset('imagenet32', 'train')
#     embed()

