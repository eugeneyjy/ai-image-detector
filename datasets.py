# Author: Eugene Yong
# Date: March 18, 2024
# Final Project: AI Image Detector
# Code Adapted From: My pass assignments in CS 579 (Trustworth Machine Learning)

import numpy as np
import torch
import os
import cv2
import logging
import sys

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from helper import load_parquets

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class AiHumanImageDataset(Dataset):
    def __init__(self, dir, transform=None, split='train', versions=None):
        """
        Arguments:
            real_folder (string): Path to the folder with human generated art data
            fake_folder (string): Path to the folder with ai generated art data
        """
        columns = ['image', 'style', 'class', 'version']
        self.df = load_parquets(dir, columns, desc=f"Loading from {split} data")
        self.max_style_idx = self.df['style'].value_counts().argmax()
        self.transform = transform
        if versions:
            self.df = self.df[self.df['version'].isin(versions)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.frombuffer(self.df.iloc[idx]['image']['bytes'], np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        style = self.df.iloc[idx]['style']
        basic_transform = transforms.ToTensor()
        image = basic_transform(image)
        if style != self.max_style_idx and self.transform:
            image = self.transform(image)
        target_class = self.df.iloc[idx]['class']

        return image, style, target_class

def data_loader(data_dir, batch_size, transform=None, train=True, versions=None, weighted_class=False, cost_sensitive_learning=False, weighted_style=False, weighted_version=False,):
    if train:
        train_data = AiHumanImageDataset(dir=os.path.join(data_dir, 'train'), transform=transform, split='train', versions=versions)
        val_data = AiHumanImageDataset(dir=os.path.join(data_dir, 'val'), split='val', versions=versions)

        sampler = None
        shuffle = True
        if weighted_class:
            class_counts = train_data.df['class'].value_counts()
            sampler_weights = [1/class_counts[i] for i in train_data.df['class'].values]
            sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_data), replacement=True)
            shuffle = False

        class_weights = None
        if cost_sensitive_learning:
            class_weights=compute_class_weight(class_weight='balanced', 
                                            classes=np.unique(train_data.df['class'].values),
                                            y=train_data.df['class'].values)
            class_weights=torch.tensor(class_weights,dtype=torch.float)

        if weighted_style:
            style_count = train_data.df['style'].value_counts()
            sampler_weights = [1/style_count[i] for i in train_data.df['style'].values]
            sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_data), replacement=True)
            shuffle = False

        if weighted_version:
            sampler_weights = [version for version in train_data.df['version'].values]
            sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_data), replacement=True)
            shuffle = False

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler
        )

        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=True
        )

        return train_loader, val_loader, class_weights

    else:
        test_data = AiHumanImageDataset(dir=os.path.join(data_dir, 'test'), split='test')

        test_loader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False
        )

        return test_loader
