# -*- coding: utf-8 -*-
"""
Spyder Editor

Nanme: Xiang Li
ID: xiangl6
Oct 2021

This is a temporary script file.
"""
from __future__ import print_function, division

from torchvision import models
model_ft = models.inception_v3(pretrained=True) #For InceptionV3
model_ft = models.resnet18(pretrained=True) #ForResNet18
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode

# use imaginefolder to load data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/Users/xiangli/Desktop/95891 AI/HW3/95891-F21-hw3-data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# train model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # epochs is the number of complete passes through the training dataset.
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # if the data is from train set, we will use it to train model, otherwise we will give the predict result
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to test mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # running_loss += loss.data[0]
                # running_corrects += torch.sum(preds == labels.data)
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    
    if use_gpu:
        model_ft = model_ft.cuda()
        
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # train model
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
    
    #test = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                      #       shuffle=True, num_workers=4)
             # for x in ['test']}

# build up confusion matrix
test_data = dataloders['test']
y_true = []
y_pred = []
n = len(class_names)
conf_matrix = np.zeros((n,n))
for data in test_data:
    inputs, label = data
    outputs = model_ft(inputs)
    _, preds = torch.max(outputs.data, 1)
    for t, p in zip(label, preds):
        conf_matrix[t][p] += 1
        #y_true.append(torch.max(label, 0)[1])
        #y_pred.append(torch.max(preds, 0)[1])
        #print(y_true)
print(conf_matrix)

# normalize the raw confusion matrix 
np.set_printoptions(linewidth=300)
print(np.matrix(conf_matrix/(conf_matrix.dot(np.ones((n,1))))))

# calculate the precision and recall
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
print (precision)
print (recall)

# Plot a grouped bar chart showing the precision and recall grouped by category
x = list(range(len(precision)))
total_width,n=0.8,2
width=total_width/n
plt.bar(x, precision,width=width,label="Precision", tick_label=class_names, fc='blue')
for j in range(len(x)):
    x[j]=x[j]+width
plt.bar(x, recall,width=width,label="recall", tick_label=class_names, fc='orange')
plt.xticks(rotation=270)
plt.legend()
plt.show()

