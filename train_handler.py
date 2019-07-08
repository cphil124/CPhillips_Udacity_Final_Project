import argparse
from torchvision import models
from collections import OrderedDict
from torch import nn
import torch



 # Argument Parsing
def arg_parser():
    parser = argparse.ArgumentParser(
        description='Application for building a Neural Network model to classify flowers'
    )

    # Required Args
    parser.add_argument('data_directory',  type=str, default='flowers', 
    help='Path to directory holding the datasets for training, validation, and testing.')

    # Optional Args
    parser.add_argument('-s','--save_dir', type=str, help='destination for saving Model Checkpoint')
    parser.add_argument('-lr','--learning_rate', type = float, help='Learning rate for training backpropogation')
    parser.add_argument('-a', '--arch', type=str, default='vgg19', help='Type of Model Architecture to build')
    parser.add_argument('--gpu', action='store_true', dest='device', help='Whether or not to make use of the GPU for running'
                        'model processes. By default, program will try and make use of GPU unless directed otherwise.')
    parser.add_argument('-H', '--hidden_units', action='store', dest='hidden', help='Number of units in the hidden layer of the model.')
    parser.add_argument('-e', '--epochs', action = 'store',type=int, dest='epochs', help='Number of epochs for which to train.' 
                        '1 epoch = ~2 mins of training time')
    parser.add_argument('-c', '--cat_to_name', type = str, default='cat_to_name.json',help='Path to the json for matching names to categories' )
    args = parser.parse_args()
    

    
    # Architectures supported for building a new model.
    supported_archs = { 'vgg19' : models.vgg19(pretrained=True),
        'vgg16' : models.vgg16(pretrained=True),
        'vgg13' : models.vgg13(pretrained=True),
        'vgg11' : models.vgg11(pretrained=True),
        'densenet121' : models.densenet121(pretrained=True),
        'densenet169' : models.densenet121(pretrained=True),
        'densenet201' : models.densenet121(pretrained=True)
        }


    # Recovering Model from Checkpoint. If specified model is not in the list of supported architectures, the program will
    # exit and print a message requesting a supported architecture. 
    if args.arch in supported_archs.keys():
        model = supported_archs[args.arch]
        model.arch = args.arch
        print(f'A base-layer {args.arch} model has been built.')
    else: 
        print('Sorry, only VGG or Densenet architectures are supported at this time.')
        exit

    # Turns gradients off for parameters of pretrained model before adding our own classifier to the model. 
    for param in model.parameters():
        param.requires_grad = False

    # Adding classifier with optional hyperparameters
    if args.hidden:
        hidden = int(args.hidden)
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden)),
                                        ('relu', nn.ReLU()),
                                        ('fc2', nn.Linear(hidden, 524)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(0.2)),
                                        ('fc3', nn.Linear(524, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))
    # If no hidden parameter is passed, 1024 is used as the default. 
    else:

            classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 1024)),
                                        ('relu', nn.ReLU()),
                                        ('fc2', nn.Linear(1024, 524)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(0.2)),
                                        ('fc3', nn.Linear(524, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                        ]))

    
    model.classifier = classifier
    
    # Checks gpu hyperparameter and that cuda is supported, and if so, moves the model to the gpu.
    if torch.cuda.is_available() and args.device:
        model.to('cuda')

    return model, args


# Validation Pass for periodic status checking during training
def validation_pass(model, dataloader, criterion, device=True):
    with torch.no_grad():
        total_accuracy = 0
        valid_loss = 0
        model.eval()
        for inputs, labels in dataloader:
            # Checks gpu hyperparameter and that cuda is supported, and if so, moves the inputs and the labels to the gpu.
            if device and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            # Gets output probabilities and the validation loss for the inputs
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Running tally for validation loss
            valid_loss += loss.item()
            ps = torch.exp(logps)

            # Gets topk classes and probabilites from the output probabilities
            _, top_class = ps.topk(1, dim=1)

            # Forms equality list from comparisons of labels with prediction results. 
            equality = (labels.view(*top_class.shape) == top_class)
            # Running total of accuracy for this prediction as mean of all probability confidences for the dataset and how close they are to the labels. 
            total_accuracy += torch.mean(equality.type(torch.FloatTensor))
        # Total accuracy divided by length of dataloader to get an accuracy between 0 and 100%.
        accuracy = total_accuracy/len(dataloader)    
        
    return valid_loss, accuracy


# Primary Function for Model Training
def train_model(model, trainloader, validloader, optimizer, criterion = nn.NLLLoss(), print_every = 100, epochs = 5, device=True):
    # Model set to train mode to enable dropout and backpropogation
    model.train()
    if torch.cuda.is_available() and device:
        model.to('cuda')
    for e in range(epochs):
        running_loss = 0
        # Counts total number of images passed to model for training. 
        steps = 0
        for inputs, labels in trainloader:
            steps += 1
            if device and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # Zeros out gradient to prevent accumulation
            optimizer.zero_grad()

            # Passes input through model and gets loss
            output = model.forward(inputs)
            loss = criterion(output, labels)

            # Back Propogate loss through model
            loss.backward()
            optimizer.step()
            
            # Add loss to running total
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()

                # Turns off Gradient for Validation Pass and prints out results
                with torch.no_grad():
                    valid_loss, accuracy = validation_pass(model, validloader, criterion, device)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/steps),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}%".format((accuracy)*100))
                model.train()

# Intended to serve as final evaluation of model's current state. 
def test_model(model, testloader, criterion = nn.NLLLoss(), device=True):
    if device and torch.cuda.is_available():
        model.to('cuda') 
    with torch.no_grad():
        total_accuracy = 0
        test_loss = 0
        model.eval()
        for inputs, labels in testloader:
            if device and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            
            # Passes input through model and gets loss
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Running tally for test loss
            test_loss += loss.item()
            
            #Getting Probabilities and converting them from log-probs.
            ps = torch.exp(logps)
            _, top_class = ps.topk(1, dim=1)

            # Forms equality list from comparisons of labels with prediction results. 
            equality = (labels.view(*top_class.shape) == top_class)

            # Running total of accuracy for this prediction as mean of all probability confidences for the dataset and how close they are to the labels. 
            total_accuracy += torch.mean(equality.type(torch.FloatTensor))

        # Total accuracy divided by length of dataloader to get an accuracy between 0 and 100%.                    
        accuracy = total_accuracy/len(testloader)  

        # Print results 
        print("Testing Results: ",
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}%".format((accuracy)*100))

    # Put model back in training mode to turn dropout and other training features back on and return the accuracy of the model 
    # test results
    model.train()
    return (accuracy)*100



