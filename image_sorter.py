import os
import scipy.io
import shutil
from argparse import ArgumentParser

# Program to assort images in a directory into test, train, and valid folders intended to be used for deep learning
# This will take a raw data folder with just images and construct training, testing, and validation sets, evenly
# distributing the image files across the datasets and organizing each set into folders of labelled, sorted categorical 
# directories to make label identification simple. Specifically, the format the data will be morphed to is meant
# to comply with the input for the Torchvision.datasets.ImageFolder module. 

def arg_parser():
    parser = ArgumentParser()
    
    # Takes the data directory and the location of the labels file
    parser.add_argument('data_directory', type=str)
    parser.add_argument('labels', type=str)
    args = parser.parse_args()
    return args



# Starting dataset should be broken down 80% for training, 10% for Validation, 10% for Testing
# 

def image_spread(data_folder):
    folders = []
    # Checks data folder for presence of test, train, and valid folders, and if they don't exist, creates them
    if data_folder not in os.getcwd():
        os.chdir(data_folder) 
    if 'train' not in os.listdir(data_folder):
        os.mkdir('train')
        folders.append(data_folder + '\\train')
    if 'valid' not in os.listdir(data_folder):
        os.mkdir('valid')
        folders.append(data_folder + '\\valid')
    if 'test' not in os.listdir(data_folder):
        os.mkdir('test')
        folders.append(data_folder + '\\test')

    
    # Iterates through all files in the directory
    steps = 0
    for file in os.listdir():
        # Checks that file is a jpeg
        if file.endswith('.jpg'):
            steps += 1
            # Evenly distributes images to each folder at a {train: 80%, valid: 10%, test: 10%} ratio
            if steps % 10 == 0:
                shutil.move(file, 'test')
            elif steps % 10 == 1:
                shutil.move(file, 'valid')
            else:
                shutil.move(file, 'train')
    # Returns a list of the new folder paths containing data
    return folders

def category_sort(cat_map, folder_list=None,  data_dir=os.getcwd()):
    
    # If no folder_list is passed, checks for the presence of dataset folders in data_dir
    if folder_list is None:

        # If data folders don't exist in data_dir, calls image_spread to generate them and 
        # get the list of folder paths back. If folders do exist, creates a list of their paths.
        if 'train' not in os.listdir(data_dir):
            folders = image_spread(data_dir)
        else: 
            folders = [data_dir + '\\train', data_dir + '\\test', data_dir + '\\valid'] 
    else: 
        folders = folder_list
    
    
    # Changes current working directory
    os.chdir(data_dir)


    # Gets the category mappings from a .mat file. 
    mat = scipy.io.loadmat(cat_map)
    labels = mat['labels'][0]


    # Sets root as data directory
    root = data_dir

    # Go through each folder in the data directory
    for folder in folders:
        # Iterate through folders, starting by moving cwd to the data folder
        os.chdir(folder)
        for file in os.listdir(folder):
            if file.endswith('.jpg'):
                # Get image index number from file name
                file_num = int(file[6:-4])

                # Get label for image by looking up image index in the labels dictionary
                file_label = str(labels[file_num-1])

                # For each classification label, create a seperate, label title folder within each data folder
                # To house all images of the same class. This will 
                if file_label not in os.listdir():
                    os.mkdir(os.getcwd() + '\\' + file_label)
                shutil.move(file, file_label)
        os.chdir(root)    

                
        

def main():
    args = arg_parser()
    data_fol = os.getcwd() + '\\' + args.data_directory
    cat_mat = os.getcwd() + '\\' + args.labels
    # folders = [os.getcwd()+'\\valid', os.getcwd()+'\\test', os.getcwd()+'\\train']
    category_sort(cat_map = cat_mat, data_dir=data_fol)
    # os.chdir('flowers')
    # category_sort()



if __name__ == "__main__":
    main()
        





