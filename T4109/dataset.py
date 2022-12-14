import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import RandomRotation, Grayscale, Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


## torchvision
# class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            # Resize([224, 224], Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
    


# albumentations
class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.Resize(height=resize[0], width=resize[1], interpolation=Image.BILINEAR),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.2),
            A.GridDistortion(p=0.2),
            A.Rotate(limit=10, p=0.5),
            A.Normalize(mean=(0.5602,0.5241,0.5015),
                        std=(0.2332,0.2430,0.2457), p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    def __call__(self, image):
        return self.transform(image=np.array(image))['image']



class BaseAugmentation_size:
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.Resize(height=224, width=224, interpolation=Image.BILINEAR),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.2),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            A.ColorJitter(0.1, 0.1, 0.1, 0.1),
            A.CLAHE(p=0.3),
            A.Sharpen(p=0.2),
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5),
            A.GridDistortion(p=0.2),
            A.GridDropout(p=1.0,holes_number_x=8,holes_number_y=8),
            A.Rotate(limit=10, p=0.5),
            A.Normalize(mean=(0.5602,0.5241,0.5015),
                        std=(0.2332,0.2430,0.2457), p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    def __call__(self, image):
        return self.transform(image=np.array(image))['image']

    
    
## torchvision
# class MyAugmentation:
#     def __init__(self, resize, mean, std, **args):
#         self.transform = Compose([
#             Resize(resize, Image.BILINEAR),
#             RandomRotation(50),
#             Grayscale(3),
#             ToTensor(),
#             Normalize(mean=mean, std=std)
#         ])
    
#     def __call__(self, image):
#         return self.transform(image)


class AddGaussianNoise(object):
    """
        transform ??? ?????? ???????????? ??????????????? __init__, __call__, __repr__ ?????????
        ?????? ???????????? ????????? ??? ????????????.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


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


class MaskBaseDataset(Dataset):
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

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." ??? ???????????? ????????? ???????????????
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." ??? ???????????? ?????? ??? invalid ??? ???????????? ???????????????
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

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

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform ???????????? ???????????? transform ??? ??????????????????"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

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

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        ??????????????? train ??? val ??? ????????????,
        pytorch ????????? torch.utils.data.random_split ????????? ????????????
        torch.utils.data.Subset ????????? ?????? ????????????.
        ????????? ????????? ????????? ????????? ?????? IDE (e.g. pycharm) ??? navigation ????????? ?????? ????????? ??? ??? ???????????? ?????? ??????????????????^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set



class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val ????????? ????????? ???????????? ????????? random ??? ??????
        ??????(profile)??? ???????????? ????????????.
        ????????? val_ratio ??? ?????? train / val ????????? ?????? ????????? ????????? ?????? ??????(profile)??? ????????? ???????????? indexing ??? ?????????
        ?????? `split_dataset` ?????? index ??? ?????? Subset ?????? dataset ??? ???????????????.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

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

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
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



class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)



class MaskMultiLabelDataset(MaskBaseDataset):
    num_classes = 3 + 2 + 3

    def __getitem__(self, index):
        assert (self.transform is not None), ('use .set_transform to pass an augmentation')
        
        images = Image.open(self.image_paths[index])
        mask_labels = self.mask_labels[index]
        gender_labels = self.gender_labels[index]
        age_labels = self.age_labels[index]
        
        images_transform = self.transform(images)
        return images_transform, (mask_labels, gender_labels, age_labels)
        
        
        
        
    # def __getitem__(self, index):
    #     assert self.transform is not None, ".set_tranform ???????????? ???????????? transform ??? ??????????????????"

    #     image = self.read_image(index)
    #     mask_label = self.get_mask_label(index)
    #     gender_label = self.get_gender_label(index)
    #     age_label = self.get_age_label(index)
    #     multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

    #     image_transform = self.transform(image)
    #     return image_transform, multi_class_label