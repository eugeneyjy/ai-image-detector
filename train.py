# Author: Eugene Yong
# Date: March 18, 2024
# Final Project: AI Image Detector
# Code Adapted From: My pass assignments in CS 579 (Trustworth Machine Learning)

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from datasets import data_loader
from helper import get_criterion, get_optimizer, get_transform, validation_metrics
from path import get_report_dir
from models.resnet50.arch import ResNet50

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, optimizer, criterion, args):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    min_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        for i, (data, _, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = data.permute(0, 3, 1, 2).float()
            data, target = data.to(device), target.to(device)

            pred = model(data)
            _, pred_label = torch.max(pred, 1)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            correct += (pred_label == target).sum().item()
            total += len(data)

            if i % 64 == 0:
                logging.info("Epoch [%d/%d] || Step [%d/%d] || Loss: [%f] || Acc: [%f]" % 
                             (epoch+1, args.epochs, i, len(train_loader), sum_loss/total, correct/total))

        train_loss, train_acc = sum_loss/total, correct/total

        if val_loader:
            logging.info("calculating validation metrics")
            val_loss, val_acc, val_precision, val_recall, val_f_1, val_cf_matrix = validation_metrics(model, val_loader, criterion)
        else:
            val_loss, val_acc = 0.0, 0.0

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logging.info("Epoch %d train loss %f, train acc %.3f, test loss %f, test acc %.3f, test precision %.3f, test recall %.3f, test f-1 %.3f" % 
                    (epoch+1, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f_1))
        print(val_cf_matrix)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch+1
            logging.info("Saving best model on epoch %d" % (epoch+1))
            save_model(epoch+1, model, optimizer, True, args)

        logging.info("Saving model epoch %d" % (epoch+1))
        save_model(epoch+1, model, optimizer, False, args)

    logging.info("Best model is from epoch %d" % (best_epoch))

    return train_losses, train_accs, val_losses, val_accs

def save_model(epoch, model, optimizer, best, args):
    if best:
        path_name = f'{get_report_dir()}/best.pth'
    else:
        path_name = f'{get_report_dir()}/last.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        path_name
    )

def plot_results(args, results):
    # result: (train_losses, train_accs, val_losses, val_accs)
    epochs = range(1, len(results[0])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    fig.suptitle("RestNet50 Trained on RealFakeDataset")

    # Plot training and validation losses 
    ax1.set_title('Model Loss')
    ax1.plot(epochs, results[0], label='training')
    ax1.plot(epochs, results[2], label='test')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')

    # Plot training and validation accuracies
    ax2.set_title('Model Accuracy')
    ax2.plot(epochs, results[1], label='training')
    ax2.plot(epochs, results[3], label='test')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')

    plt.savefig(f'{get_report_dir()}/plots.png')
    # plt.show()

def save_hyperparameters(args, train_size, val_size, versions):
    with open(f'{get_report_dir()}/hyperparameters.txt', 'w') as f:
        f.write(f'{args}\n')
        f.write(f'Using data versions: {versions}\n')
        f.write(f'Training size: {train_size} | Validation size: {val_size}')

def save_results(results):
    train_loss = results[0]
    train_acc = results[1]
    val_loss = results[2]
    val_acc = results[3]
    epoch = len(train_loss)

    with open(f'{get_report_dir()}/results.csv', 'w') as f:
        f.write('train_loss,train_acc,val_loss,val_acc\n')
        for i in range(epoch):
            f.write(f'{train_loss[i]},{train_acc[i]},{val_loss[i]},{val_acc[i]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, metavar='20',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='64',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--data-dir', type=str, default='../data/ai-human-images/data', metavar='../data/ai-human-images/data',
                        help='specify root directory that contain train/val/test folder (default= ../data/ai-human-images/data)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='0.001',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--loss', type=str, default='crossEntropy', metavar='crossEntropy',
                        help='loss function (default= crossEntropy)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='adam',
                        help='optimization algorithm (default: adam)')
    parser.add_argument('--no-save', action='store_true',
                        help='specify if do not want the trained model be saved (default: False)')
    parser.add_argument('--seed', type=int, default=42, metavar='42',
                        help='specify seed for random (default: 42)')
    parser.add_argument('--horizontal-flip', action='store_true',
                        help='specify if want the image data be flip horizontally at random (default: False)')
    parser.add_argument('--rotation', action='store_true',
                        help='specify if want the image data be rotate (-90, 90) degree at random (default: False)')
    parser.add_argument('--weighted-class', action='store_true',
                        help='specify if want to perform weighted random sampling on class distribution while training (default: False)')
    parser.add_argument('--cost-sensitive-learning', action='store_true',
                        help='specify if want to use adjusted class weight when computing loss in training (default: False)')             
    parser.add_argument('--weighted-style', action='store_true',
                        help='specify if want to perform weighted random sampling on style distribution while training (default: False)')
    parser.add_argument('--weighted-version', action='store_true',
                        help='specify if want to perform weighted random sampling based on version while training. More recent version get sample more frequently (default: False)')
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
    transform = get_transform(args)
    data_versions = args.versions

    train_loader, val_loader, class_weights = data_loader(data_dir=args.data_dir,
                                                          batch_size=args.batch_size, 
                                                          transform=transform,
                                                          train=True,
                                                          versions=data_versions,
                                                          weighted_class=args.weighted_class, 
                                                          cost_sensitive_learning=args.cost_sensitive_learning,
                                                          weighted_style=args.weighted_style,
                                                          weighted_version=args.weighted_version)

    criterion = get_criterion(args.loss, class_weights)
    optimizer = get_optimizer(args, model.parameters())

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    logging.info("Training on %s" % (device))
    logging.info("Training size %d | Validation size %d" % (train_size, val_size))

    save_hyperparameters(args, train_size, val_size, data_versions)
    
    results = train_model(model, train_loader, val_loader, optimizer, criterion, args)

    plot_results(args, results)
    save_results(results)
