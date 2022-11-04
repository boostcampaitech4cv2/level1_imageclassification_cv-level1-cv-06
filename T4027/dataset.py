import os
import random
import pandas as pd
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt


class MaskMultiLabelDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df.reset_index()
        self.img_path = self.df.path
        self.labels = self.df.label
        self.transform = transform        
    
    def __len__(self):
        return len(self.df)

    def change_label(self, label):
        mask = label // 6
        label %= 6
        gender = label // 3
        label %= 3
        age = label
        return (mask, gender, age)

    def set_transform(self, transform):        
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        label = self.labels[idx]
        label = self.change_label(label)
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


class TestDataset(Dataset):
    def __init__(self, img_paths, resize):
        self.img_paths = img_paths
        self.transform = A.Compose([
                A.CenterCrop(384, 384),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    def __getitem__(self, index):
        image = cv.imread(self.img_paths[index])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return len(self.img_paths)
    