import torch
from torchvision import transforms, models
from argparse import ArgumentParser
from PIL import Image
from helper import process_image

def predict(image_path, model, cat_to_name, topk=5):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Process image with helper method
    img = process_image(image_path)

    # Reshape image tensor to match model input 
    img = img.unsqueeze(0)
    img = img.float()

    # Retrieve label key matching the index in the dataset at large (Complete unsegmented dataset)
    # of the image to it's label
    class_to_idx = model.class_to_idx
    
    #Turning off Gradient because we are not training
    with torch.no_grad():
        # Passing model through model to get prediction log probabilities
        output = model.forward(img.to('cuda'))
        # Conversion from log to standard probabilities
        ps = torch.exp(output)
    

    # Taking top k probabilities and the corresponding labels to those predictions and printing them for 
    # review by data scientist
    probs, classes = ps.topk(topk, dim=1)
    prob_list, flow_list = [], []
    for prob in probs[0]:
        prob_list.append(prob.item())
    for cls in classes[0]:
        flow_list.append(class_to_idx[str(cls.item()+1)])    
    lab_list = [cat_to_name[str(cls+1)] for cls in flow_list]
    
    return lab_list, prob_list

def arg_parser():
    parser = ArgumentParser(description='Loading a trained model to be used ')

    parser.add_argument('path_to_image', type=str, 
    help='Path to an image to be passed to the prediction model. Path should either be a complete path or a local directory' 
        'within the current working directory') 
    parser.add_argument('checkpoint', type=str, help='Name to be used for saving the model\'s current state in a checkpoint')
    parser.add_argument('--topk', type=int, help='Number of predictions and their probabilities to be output by program')
    parser.add_argument('--gpu', type=str, dest='device', default='gpu',help='Whether or not to make use of the GPU for running'
                        'model processes. By default, program will try and make use of GPU unless directed otherwise. ')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')

    args = parser.parse_args()
    return args

