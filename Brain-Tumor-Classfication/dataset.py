# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import ToTensor
from config import *
from sklearn.model_selection import train_test_split



class BrainTumorDataset(Dataset):
    def __init__(self, img_dir, mode="Train", transform=None):
        img_paths = []
        labels = []
        
        for class_name in sorted(os.listdir(img_dir)):
            class_folder = os.path.join(img_dir, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img_paths.append(file_path)
                labels.append(label_str2num[class_name])
                
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            img_paths, labels, test_size=0.2, random_state=16122004
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
        return img, label_ts
    


