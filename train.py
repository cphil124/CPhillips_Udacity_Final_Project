
# Library Imports

import torch
import helper
import predict
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import pandas as pd
import seaborn as sb
import os
import argparse


# Model Prep
def build_model(learning_rate=0.001):
    model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 1024)),
                                        ('relu', nn.ReLU()),
                                        ('fc2', nn.Linear(1024, 524)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(0.2)),
                                        ('fc3', nn.Linear(524, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate )
    if torch.cuda.is_available():
        model.to('cuda')
    return model, criterion, optimizer


# Validation Pass for periodic status checking during training
def validation_pass(model, dataloader, criterion, device):
    with torch.no_grad():
        total_accuracy = 0
        valid_loss = 0
        model.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            valid_loss += loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equality = (labels.view(*top_class.shape) == top_class)
            total_accuracy += torch.mean(equality.type(torch.FloatTensor))
        accuracy = total_accuracy/len(dataloader)    
        
    return valid_loss, accuracy


# Primary Function for Model Training
def train_model(model, trainloader, validloader, criterion = nn.NLLLoss(), print_every = 100, optimizer=nn.NLLLoss(), epochs = 5):
    
    model.train()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        steps = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation_pass(model, validloader, criterion, device)
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/steps),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}%".format((accuracy)*100))
                model.train()


def test_model(model, testloader, criterion = nn.NLLLoss(), device='cuda'):
    if device == 'cuda':
        model.to(device) 
    with torch.no_grad():
        total_accuracy = 0
        test_loss = 0
        model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            test_loss += loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equality = (labels.view(*top_class.shape) == top_class)
            total_accuracy += torch.mean(equality.type(torch.FloatTensor))
        accuracy = total_accuracy/len(testloader)  
        print("Testing Results: ",
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}%".format((accuracy)*100))
    model.train()
    return (accuracy)*100


def main():
    args = arg_parser()
    data_dir = args.data_directory
    if args.epochs:
        ep = args.epochs
    if args.learning_rate:
        lr = args.learning_rate
    if args.arch:
        arch = args.arch
    if args.gpu:
        device = args.gpu
    if args.hidden_units:
        hidden = hidden_units
    if args.save_dir:
        save_dir = args.save_dir
    
    
    
 
 # Argument Processing
def arg_parser():
    parser = argparse.ArgumentParser(
        description='Application for building a Neural Network model to classify flowers'
    )

    # Required Args
    parser.add_argument('data_directory',  type='str', required=True)

    # Optional Args
    parser.add_argument('-s','--save_dir', type=str, help='destination for saving Model Checkpoint')
    parser.add_argument('-lr','--learning_rate', type = float)
    parser.add_argument('-a', '--arch', type=str, default='vgg19')
    parser.add_argument('--gpu', type=str, dest='device', default='gpu')
    parser.add_argument('-H', '--hidden_units', action='store', dest='hidden')
    parser.add_argument('-e', '--epochs', action = 'store',type=int, dest='epochs')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

    