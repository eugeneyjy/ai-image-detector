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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_criterion(loss, class_weights):
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if loss == 'crossEntropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    
def get_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return torch.optim.Adam(parameters, args.lr, amsgrad=True)
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

def validation_metrics(model, loader, criterion):
    model.eval()
    sum_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    styles = []
    for i, (data, style, target) in tqdm(enumerate(loader), total=len(loader)):
        data = data.permute(0, 3, 1, 2).float()
        data, target = data.to(device), target.to(device)
        pred = model(data)
        _, pred_label = torch.max(pred, 1)
        loss = criterion(pred, target)

        sum_loss += loss.item() * len(data)
        correct += (pred_label == target).sum().item()
        total += len(data)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred_label.cpu().numpy())
        styles.extend(style.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    styles = np.array(styles)
    cf_matrix = confusion_matrix(y_true, y_pred)
    precision, recall, f_1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return sum_loss/total, correct/total, precision, recall, f_1, cf_matrix