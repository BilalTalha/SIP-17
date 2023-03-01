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
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix

cudnn.benchmark = True
plt.ion()   # interactive mode

import wandb
wandb.init(project="ConvNext")

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNext', add_help=False)
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--weights', default=".", type=str, help='Path to save logs and checkpoints.')
    return parser
parser = argparse.ArgumentParser('ConvNext', parents=[get_args_parser()])
args = parser.parse_args()

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = args.data_path

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model(model):
    model.eval()

    list_im =[]
    list_pred = np.array([])
    list_target = np.array([])
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            target = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            pred=preds
            tar = target
            pred = pred.detach().to("cpu").numpy()

            tar = tar.detach().to("cpu").numpy()
            list_pred = np.concatenate((list_pred,pred),axis=0)

            list_target = np.concatenate((list_target,tar),axis=0)
        
            (a,_,_,_)=inputs.size()
            for i in range(0,a):
              list_im.append(inputs[i])
    #confusion matrix
    #update the name of the class if needed
    wandb.sklearn.plot_confusion_matrix(list_target, list_pred, ["Airgun","CouplingHalf","Cross","Electricity12","Fork1","Fork2","Fork3","Gear1","Gear2","Hammer","Hook","Pin1","Pin2","Pinion","Plug"])
    #wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,preds=list_pred, y_true=list_target,class_names=["Airgun", "BlackButton","CouplingHalf","Cross","Electricity12","Fork1","Fork2","Fork3","Gear1","Gear2","Hammer","Hook","Pin1","Pin2","Pinion","Plug"])})
    
    #per class accuracy table
    cm = confusion_matrix(list_target, list_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    matrix = cm.diagonal()
    tbl2 = wandb.Table(columns=["class", "Accuracy"])
    classes = ["Airgun","CouplingHalf","Cross","Electricity12","Fork1","Fork2","Fork3","Gear1","Gear2","Hammer","Hook","Pin1","Pin2","Pinion","Plug"]
    accuracies = matrix
    
    #map index to classes 
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    for k,v in idx_to_class.items():
      list_pred = [v if x==k else x for x in list_pred]

    for k,v in idx_to_class.items():
      list_target = [v if x==k else x for x in list_target]

    #Add tables to wandb
    tbl1 = wandb.Table(columns=["image", "prediction", "target"])
    [tbl1.add_data(wandb.Image(image), lbl, tgt) for image, lbl, tgt in zip(list_im, list_pred , list_target)]
    wandb.log({"ResNet_output": tbl1})
    [tbl2.add_data(clas, label) for clas, label in zip(classes, accuracies)]
    wandb.log({"Per-Class-Accuracy": tbl2})

        
model_ft = models.convnext_tiny(pretrained=True)
print(model_ft)
for param in model_ft.parameters():
    param.requires_grad = False
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.classifier[2] = nn.Linear(in_features=768, out_features=15)
for param in model_ft.classifier[2].parameters():
    param.requires_grad = True
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft.load_state_dict(torch.load(args.weights))
test_model(model_ft)





