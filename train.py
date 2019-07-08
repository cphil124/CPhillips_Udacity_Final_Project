
# Library Imports
import json
import torch
from helper import prep_dataset, save_checkpoint, get_datasets
from torch import nn, optim
from torchvision import datasets, transforms, models
import os
import argparse
from train_handler import train_model, test_model, arg_parser
import scipy.io


def main():

    # Construct model from Command Line Arguments
    model, cli_args = arg_parser()
    
    # Move args from client arg namespace
    args = vars(cli_args)

    # Prepare dataloaders and datasets
    loaders = prep_dataset(args['data_directory'])
    dataset = get_datasets(args['data_directory'])

    # Add model class-to-index mapping as model attribute
    model.class_to_idx = dataset['train'].class_to_idx

    # Train model save checkpoint and test model
    train_model(model, loaders['train'], loaders['valid'], optimizer = optim.Adam(model.classifier.parameters(), lr=args['learning_rate']), device = True)
    save_checkpoint(model, loaders['test'], args['save_dir'])
    test_model(model, loaders['test'])
    
    

if __name__ == '__main__':
    main()

    