import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import torch.utils.data as data
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ToTensor
import albumentations as A

from collections import defaultdict


bounding_box = pd.read_csv('./../bounding_box.csv')
bounding_box.set_index('img_paths', inplace=True)
bounding_box_test = pd.read_csv('./../bounding_box_test.csv')
bounding_box_test.set_index('img_paths', inplace=True)


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.56, 0.524, 0.501), std=(0.233, 0.243, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.indices = defaultdict(list)
        
        self.setup()
        self.calc_statistics()
        
        
    @staticmethod
    def _split_profile(profiles, val_ratio):
        total_len = len(profiles)
        n_val = int(total_len * val_ratio)
        total_indices = range(total_len)
        val_indices = set(random.choices(total_indices, k=n_val))
        train_indices = set(total_indices) - val_indices
        
        return {"train": train_indices, "val": val_indices}
    
    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." ??? ???????????? ?????? ??? invalid ??? ???????????? ???????????????
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    if (27 < int(age) < 30) or (57 < int(age) < 60) :
                        continue
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1
    
    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]
    
    def get_sampler(self, phase) :
        _multi_class = []
        for _idx in self.indices[phase]:
            _multi_class.append(self.encode_multi_class(self.mask_labels[_idx], self.gender_labels[_idx], self.age_labels[_idx]))
        
        size = len(_multi_class)
        class_counts = pd.DataFrame(_multi_class).value_counts().to_list()        
        class_weights = [size / class_counts[i] for i in range(len(class_counts))] #???????????? ????????? ??????
        weights = [class_weights[_multi_class[i]] for i in range(size)]            #?????? ?????????????????? ????????? ??????
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), size)
    
        return sampler

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_train_transform(self, transform):
        self.train_transform = transform
        
    def set_val_transform(self, transform):
        self.val_transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = np.array(image)
        # Face detect
        if sum(bounding_box.loc[image_path]):
            x_min, y_min, x_max, y_max = bounding_box.loc[image_path]
            image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        else:
            image = Image.open(self.image_paths[index])
            image = CenterCrop(300)(image)
            
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        
        if index in self.indices["train"]:
            image_transform =  self.train_transform(image = np.array(image))["image"]
        else:
            image_transform =  self.val_transform(image = np.array(image))["image"]

        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp
    
class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transform = None
        
    def set_transform(self, transform):
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.img_paths[index]
        image = Image.open(image_path)
        image = np.array(image)
        
        # Face detect
        if sum(bounding_box_test.loc[image_path]):
            x_min, y_min, x_max, y_max = bounding_box_test.loc[image_path]
            image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        else:
            image = Image.open(self.img_paths[index])
            image = CenterCrop(300)(image)

        if self.transform:
            image = self.transform(image = np.array(image))["image"]
        return image

    def __len__(self):
        return len(self.img_paths)
