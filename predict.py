"""
Primary Module for Loading a PyTorch model checkpoint and using it for prediction. Functions imported
primarily from predict_handler module.
"""

from predict_handler import  process_image, predict, arg_parser
import helper
import os
import json



def main():
    args = arg_parser()
    
    # Load model from Checkpoint
    model, _  = helper.load_checkpoint(args.checkpoint)

    # If GPU is specified, 
    if args.device:
        model.to('cuda')

    # Check if image path is a complete path or local directory and amends the path if it's a local directory
    if args.path_to_image[:3] != 'C:\\':
        im_path = os.getcwd() + '\\' + args.path_to_image
    else:
        im_path = args.path_to_image
    
    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    # Run prediciton on passed image and print results
    lab_list, prob_list  = predict(im_path, model, cat_to_name, 5)
    for i in range(len(prob_list)):
        print(f'Prediction {i+1}:     Species: {lab_list[i]}, Confidence in Prediction: {prob_list[i]}')

    
    
    




if __name__ == '__main__':
    main()


