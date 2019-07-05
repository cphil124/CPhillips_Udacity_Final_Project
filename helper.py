# Helper Methods


def save_checkpoint(model, dest_path = 'checkpoint.pth'):
    checkpoint = {
        'arch':'vgg19'
        ,'classifier': model.classifier
        ,'class to idx' : model.class_to_idx
        ,'state dict': model.state_dict()
        ,'last acc': test_model(model, testloader, device).item()*100
        } 
    torch.save(checkpoint, dest_path)
    print(f'Model Checkpoint saved to: {dest_path}')


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = True
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state dict'])
    

    model.eval()
    print('CheckPoint Loaded')
    print(checkpoint['last acc'])
    return model, checkpoint

#Data Prep

def prep_dataset(data_dir = os.getcwd()):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
    loaders = {'train': trainloader, 
               'valid': validloader,
               'test': testloader,}

    return loaders


def process_image(image):
    img = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
    ])
    
    img_tens = transform(img)
    
    return img_tens

