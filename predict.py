import argparse
import train

# Load previously trained model from Checkpoint
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

# Process Image for Prediction
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

# Prediction Function
def predict(image_path, model, topk=5, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model by passing the image at image_path
    Through the trained model.
    '''
    model.to('cuda')

    img = process_image(image_path)
    img = img.unsqueeze(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img.to('cuda'))
        ps = torch.exp(output)
    
    probs, classes = ps.topk(topk, dim=1)
    prob_list, flow_list = [], []
    for prob in probs[0]:
        prob_list.append(prob.item())
    for cls in classes[0]:
        flow_list.append(cat_to_name[str(cls.item()+1)])    
    
    visualize_prediction(prob_list, flow_list)
    
    return prob_list, flow_list



# Visualizes Predicted data by passing output of predict
def visualize_prediction(prob_list, label_list):
    df = pd.DataFrame({'Probability': probs, 'Prediction': flows})
    
    # Create Figure and Axes
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Sanity Check Classification', fontsize = 20)
    
    ax1.set_title(cat_to_name[index])
    imshow(im, ax1)
    ax1.axis('off')
    
    
    
    ax2.set_title('Prediction Probabilities')
    df.plot.bar(x = 'Prediction', y = 'Probability', ax = ax2, color = sb.color_palette()[0], rot=45)

# Argument Processing
def arg_parser():
    parser = argparse(
        description='Application for predicting the type of flower in a passed image'
    )

    parser.add_argument(    'p','path', action='store', dest='im_path')
    parser.add_argument('c','checkpoint_path', action='store', dest='checkpoint')
    parser.add_argument('-tk','--topk', default = 5)
    parser.add_argument('--gpu', default='gpu')
    parser.add_argument('-cat', '--category_names',type= str)
    
    args = parser.parse_args()

    
    return args

def main():
    args = arg_parser()
    model, checkpoint = load_checkpoint(args.checkpoint_path)
    k = args.topk
    device = args.gpu
    if args.cat_to_name_path:
        with open(args.cat_to_name_path, 'r') as f:
            cat_to_name = json.load(f)


if __name__ == '__main__':
    main()


