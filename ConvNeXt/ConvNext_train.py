# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:18:40 2022

@author: Hp
"""
import argparse


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

import wandb
wandb.init(project="ConvNext")

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNext', add_help=False)
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--epochs', default=25, type=int,
                        help='Number of epochs to train for.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD optimizer.')
    parser.add_argument('--step_size', default=7, type=int,
                        help='Step size for learning rate decay.')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma for learning rate decay.')
    return parser

parser = argparse.ArgumentParser('ConvNext', parents=[get_args_parser()])
args = parser.parse_args()

# Extract the directory path from the full output path
output_directory = os.path.dirname(args.output_dir)

# Check if the directory exists. If not, create it.
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: {output_directory}")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = args.data_path

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
#print ("class_name", class_names)
#print ("class_number", len(class_names))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)


                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            wandb.log({"Train Loss":loss,
                   "Learning_rate":epoch})
            wandb.log({"Train Acc":epoch_acc,
                   "Learning_rate":epoch})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.convnext_base(pretrained=True)
#model_ft = models.convnext_tiny(pretrained=True)

#print(model_ft)
for param in model_ft.parameters():
    param.requires_grad = False

#  nn.Linear(num_ftrs, len(class_names)).1024 for base, 768 for tiny
num_ftrs = model_ft.classifier[2].in_features
model_ft.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=len(class_names))
for param in model_ft.classifier[2].parameters():
    param.requires_grad = True
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=args.gamma)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=args.epochs)

path = args.output_dir
torch.save(model_ft.state_dict(), path)



