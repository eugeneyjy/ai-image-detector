# Author: Eugene Yong
# Date: March 18, 2024
# Final Project: AI Image Detector
# Code Adapted From: My pass assignments in CS 579 (Trustworth Machine Learning)

import torch
import argparse
import numpy as np
from tqdm import tqdm
from datasets import data_loader
from helper import get_criterion, validation_metrics
from models.resnet50.arch import ResNet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_compatibility_score(old_model, new_model, loader):
    previously_correct = 0
    still_correct = 0
    currently_wrong = 0
    still_wrong = 0

    for _, (data, _, target) in tqdm(enumerate(loader), total=len(loader)):
        data = data.permute(0, 3, 1, 2).float()
        data, target = data.to(device), target.to(device)
        old_pred = old_model(data)
        new_pred = new_model(data)
        _, old_pred_label = torch.max(old_pred, 1)
        _, new_pred_label = torch.max(new_pred, 1)

        previously_correct += (old_pred_label==target).sum().item()
        still_correct += ((old_pred_label==target) & (new_pred_label==target)).sum().item()

        currently_wrong += (new_pred_label!=target).sum().item()
        still_wrong += ((old_pred_label!=target) & (new_pred_label!=target)).sum().item()

    return still_correct/previously_correct, still_wrong/currently_wrong

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str, help='path that the model get save on')
    parser.add_argument('--new-path', type=str, help='path that the new model get save on. If specified, run compatability test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='64',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--data-dir', type=str, default='../data/ai-human-images/data', metavar='../data/ai-human-images/data',
                        help='specify root directory that contain train/val/test folder (default= ../data/ai-human-images/data)')
    parser.add_argument('--loss', type=str, default='crossEntropy', metavar='crossEntropy',
                        help='loss function (default= crossEntropy)')
    parser.add_argument('--seed', type=int, default=42, metavar='42',
                        help='specify seed for random (default: 42)')
    parser.add_argument('--versions', nargs='+', type=int,
                        help='specify the versions want to be included in training. Not specifying means include all versions. Usage: --versions 1 2 (default: None)')

    args = parser.parse_args()
    print(args)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    model = ResNet50(num_classes=2).to(device)
    print("Loading pretrained weights")
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = data_loader(data_dir=args.data_dir, batch_size=args.batch_size, train=False, versions=args.versions)
    print(f'Testing on {len(test_loader.dataset)} images')
    criterion = get_criterion(args.loss, None)

    if 'new_path' in args and args.new_path:
        new_model = ResNet50(num_classes=2).to(device)
        print("Loading pretrained weights for new model")
        new_checkpoint = torch.load(args.new_path)
        new_model.load_state_dict(new_checkpoint['model_state_dict'])
        btc, bec = get_compatibility_score(model, new_model, test_loader)
        print(f'BTC: {btc} | BEC: {bec}')
    else:
        test_loss, test_acc, test_precision, test_recall, test_f_1, cf_matrix = validation_metrics(model, test_loader, criterion)
        print(f'Test Loss: {test_loss} | Test Acc: {test_acc} | Test Precision: {test_precision} | Test Recall: {test_recall} | Test F-1: {test_f_1}')
        print(cf_matrix)