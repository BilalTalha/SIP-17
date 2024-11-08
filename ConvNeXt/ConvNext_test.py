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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


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
#print (class_names)

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

 ##----------------Visulize Overall results-----------------------------
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(list_target, list_pred)

    # Calculate macro-average precision, recall, and F1 score
    precision_macro, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(list_target, list_pred, average='macro')

    # Initialize a table for logging overall metrics
    overall_metrics_table = wandb.Table(columns=["Metric", "Value"])

    # Add overall accuracy to the table
    overall_metrics_table.add_data("Overall Accuracy", f"{overall_accuracy:.4f}")

    # Add macro averages to the table
    overall_metrics_table.add_data("Macro Precision", f"{precision_macro:.4f}")
    overall_metrics_table.add_data("Macro Recall", f"{recall_macro:.4f}")
    overall_metrics_table.add_data("Macro F1 Score", f"{f1_score_macro:.4f}")

    # Log the table to wandb
    wandb.log({"Overall results": overall_metrics_table})
    

    
 ##----------------Visulize Class-wise results-----------------------------
    # Calculate confusion matrix
    cm = confusion_matrix(list_target, list_pred)
    # Calculate per-class accuracy from the confusion matrix
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # Log class metrics (including precision, recall, F1 score, and accuracy per class) to wandb
    class_metrics_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1 Score", "Accuracy"])
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(list_target, list_pred, average=None)

    # Add data to the table
    for i, class_name in enumerate(class_names):
        class_metrics_table.add_data(class_name, precisions[i], recalls[i], f1_scores[i], class_accuracies[i])


    # Log the table to wandb
    wandb.log({"Class Metrics": class_metrics_table})

    # Log confusion matrix using wandb
    wandb.sklearn.plot_confusion_matrix(list_target, list_pred, class_names)
    
    # For visualizing predictions with class names
    image_predictions_table = wandb.Table(columns=["Image", "Prediction", "Target"])

    for image, pred, tgt in zip(list_im, list_pred, list_target):
        # Map numeric labels to class names
        predicted_class_name = class_names[int(pred)]
        true_class_name = class_names[int(tgt)]
        
        # Add data to table with class names
        image_predictions_table.add_data(wandb.Image(image), predicted_class_name, true_class_name)

    # Log the table to wandb
    wandb.log({"Image Predictions": image_predictions_table})

#-----------------Main--------------------------------------
        
model_ft = models.convnext_base(pretrained=True)
#model_ft = models.convnext_tiny(pretrained=True)

#print(model_ft)
for param in model_ft.parameters():
    param.requires_grad = False

# nn.Linear(num_ftrs, len(class_names)).1024 for base, 768 for tiny
num_ftrs = model_ft.classifier[2].in_features
model_ft.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=len(class_names))
for param in model_ft.classifier[2].parameters():
    param.requires_grad = True
model_ft = model_ft.to(device)


#model_ft.load_state_dict(torch.load(args.weights))
#test_model(model_ft)

### The due to Pytorch may update the architecture of ConvNext, use the following code to only match the compatible layers:
checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
model_state_dict = model_ft.state_dict()

# Carefully match only compatible layers
matched_state_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
model_state_dict.update(matched_state_dict)

# Update model state dictionary
model_ft.load_state_dict(model_state_dict, strict=False)

# Set the model to evaluation mode and run testing
model_ft.eval()
test_model(model_ft)


