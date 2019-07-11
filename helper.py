from PIL import Image
from torchvision import models, transforms, datasets
from train_handler import test_model
import torch
import os


# Helper Functions


def save_checkpoint(model, testloader, dest_path = 'checkpoint.pth'):
    

    checkpoint = {
        'arch': model.arch
        ,'classifier': model.classifier
        ,'class to idx' : model.class_to_idx
        ,'state dict': model.state_dict()
        ,'last acc': test_model(model, testloader).item()*100
        } 
    torch.save(checkpoint, dest_path)
    print(f'Model Checkpoint saved to: {dest_path}')



def load_checkpoint(checkpoint_path):
    # Loads checkpoint from passed checkpoint path
    checkpoint = torch.load(checkpoint_path)
    
    # Supported Architectures
    supported_archs = { 'vgg19' : models.vgg19(pretrained=True),
        'vgg16' : models.vgg16(pretrained=True),
        'vgg13' : models.vgg13(pretrained=True),
        'vgg11' : models.vgg11(pretrained=True),
        'densenet121' : models.densenet121(pretrained=True),
        'densenet169' : models.densenet169(pretrained=True),
        'densenet201' : models.densenet201(pretrained=True)
        }


    # Recovering Model from Checkpoint if passed architecture is supported
    if checkpoint['arch'] in supported_archs.keys():
        model = supported_archs[checkpoint['arch']]
        model.arch = checkpoint['arch']
        print('A base-layer {} model has been built.'.format(checkpoint['arch']))
    else: 
        print('Sorry, only VGG or Densenet architectures are supported at this time.')
        exit

    for param in model.parameters():
        param.requires_grad = False
    

    # Set up model with checkpiont attributes
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state dict'])
    model.class_to_idx = checkpoint['class to idx']
    
    # Change model to eval mode to prevent unintended alteration to model via inadvertant training
    model.eval()
    print('CheckPoint Loaded')
    return model, checkpoint

#Data Prep
#Preps datasets as Pytorch Dataloaders ready for use in model training and evaluation
def prep_dataset(data_dir = os.getcwd()):
    
    # Get datasets from data directory
    sets = get_datasets(data_dir)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(sets['train'], batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(sets['valid'], batch_size = 32)
    testloader = torch.utils.data.DataLoader(sets['test'], batch_size = 32)
    
    # Return loaders as a dict
    loaders = {'train': trainloader, 
               'valid': validloader,
               'test': testloader}

    return loaders

# Preps datasets and returns a dict of PyTorch Dataset objects
def get_datasets(data_dir = os.getcwd()):
    
   # Look in data directory for training, validation, and testing folders
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Construct transformations to datasets to comply with each data sets functional requirements

    # Training set is randomized and altered to increase difficulty of classification and improve model versatility
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Testing and Validation sets have same transformations to ready datasets for proper evaluation of  
    valid_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    # Return constructed datasets as a dict
    sets = {'train': train_data, 
               'valid': valid_data,
               'test': test_data}
    
    return sets

def process_image(image):

    # Opens image
    img = Image.open(image)
    
    # Sets up torch transformations to convert image data to properly shaped tensors.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
    ])
    
    # Enacts transformations on image
    img_tens = transform(img)
    
    return img_tens

