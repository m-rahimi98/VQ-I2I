import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import torch.utils.data as data

class dataset_pair(data.Dataset):
    def __init__(self, root, mode, resize=256, cropsize=256, hflip=0.0):
        self.root = root
        self.mode = mode

        # بارگذاری تصاویر دامنه اول
        images_a = os.listdir(os.path.join(self.root, mode + 'A'))
        self.A = [x for x in images_a]
        self.A_size = len(self.A)

        # اندازه دیتاست
        self.dataset_size = len(self.A)

        self.input_dim_A = 3
        self.input_dim_B = 3

        # تبدیل‌ها برای پردازش تصاویر
        transforms = [Resize((resize, resize), Image.BICUBIC)]
        if(mode == 'train'):
            transforms.append(RandomCrop(cropsize))
        else:
            transforms.append(CenterCrop(cropsize))

        # flip
        transforms.append(RandomHorizontalFlip(p=1.0))

        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        # بدون فلپ (برای تست)
        transforms_no_flip = [Resize((resize, resize), Image.BICUBIC)]
        if(mode == 'train'):
            transforms_no_flip.append(RandomCrop(cropsize))
        else:
            transforms_no_flip.append(CenterCrop(cropsize))

        transforms_no_flip.append(ToTensor())
        transforms_no_flip.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms_no_flip = Compose(transforms_no_flip)
        return

    def __getitem__(self, index):
        flip_or_not = random.random()

        # بارگذاری تصویر از دامنه A
        A = os.path.join(self.root, self.mode + 'A', self.A[index])
        data_A = self.load_img(A, self.input_dim_A, flip_or_not)

        # بارگذاری تصویر از دامنه B با پسوند _ir.png
        B = os.path.join(self.root, self.mode + 'B', self.A[index][:-7] + '_ir.png')  # تغییر نام فایل برای دامنه B
        data_B = self.load_img(B, self.input_dim_B, flip_or_not)

        return data_A, data_B  

    def __len__(self):
        return self.dataset_size

    def load_img(self, img_name, input_dim, flip_or_not):
        # بارگذاری تصویر
        img = Image.open(img_name).convert('RGB')

        if flip_or_not > 0.5:
            img = self.transforms(img)
        else:
            img = self.transforms_no_flip(img)
        
        # تغییر ابعاد تصویر اگر 1 کانال باشد (برای تصاویر خاکی-سیاه)
        if(input_dim == 1):
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img
