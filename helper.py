# Author: Eugene Yong
# Date: March 18, 2024
# Final Project: AI Image Detector
# Code Adapted From: My pass assignments in CS 579 (Trustworth Machine Learning)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from torchvision import transforms
from tqdm import tqdm

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_criterion(loss, class_weights):
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if loss == 'crossEntropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    
def get_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return torch.optim.Adam(parameters, args.lr, amsgrad=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(parameters, args.lr, 0.9)
    
def get_transform(args):
    all_transforms = []

    if 'horizontal_flip' in args and args.horizontal_flip:
        all_transforms.append(transforms.RandomHorizontalFlip())
    if 'rotation' in args and args.rotation:
        all_transforms.append(transforms.RandomRotation(20))

    if not all_transforms:
        return transforms.Compose(all_transforms)
    else:
        return None

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def show_image(x, y, i):
    print(x.shape)
    img = x.detach().numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.title(y)
    plt.savefig(f'test{i}.png')
    plt.show()

def load_parquets(dir, columns, desc):
    parquets = [os.path.join(dir, parquet) for parquet in os.listdir(dir)]
        
    df = pd.concat(
        pd.read_parquet(parquet_file, columns=columns)
        for parquet_file in tqdm(parquets, total=len(parquets), desc=desc)
    )
    return df