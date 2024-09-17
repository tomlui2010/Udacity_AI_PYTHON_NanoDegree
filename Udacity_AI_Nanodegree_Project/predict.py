import json
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn,optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

import argparse

from PIL import Image

arch = ''



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['model_architecture']
    
    input_size = checkpoint['input_size']
    # Check if the specified architecture is available
    if hasattr(models, arch):
        # Get the pretrained model based on the architecture variable
        model = getattr(models, arch)(pretrained=False)
    else:
        # Handle the case when the specified architecture is not available
        raise ValueError(f"Invalid architecture: {arch}")
#     model = models.vgg16(pretrained=False)
    if 'vgg' in arch:
        model.classifier[6] = nn.Linear(input_size, len(checkpoint['class_to_idx']))
    elif 'densenet' in arch:
        # Filter out the missing and unexpected keys
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(state_dict)
        model.classifier = nn.Linear(model.classifier.in_features, len(checkpoint['class_to_idx']))
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    return model,optimizer_state_dict

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Open the image using PIL
    image = Image.open(image)

    # Resize the image while maintaining aspect ratio
    size = 256
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1:
        new_size = (size * aspect_ratio, size)
    else:
        new_size = (size, size / aspect_ratio)
    image.thumbnail(new_size)

    # Crop out the center 224x224 portion of the image
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (image.width + 224) / 2
    bottom = (image.height + 224) / 2
    image = image.crop((left, top, right, bottom))

    # Convert PIL image to NumPy array
    np_image = np.array(image)

    # Normalize the image
    np_image = np_image / 255.0  # Convert integer values 0-255 to float values 0-1
    mean = np.array([0.485, 0.456, 0.406])  # Mean values for normalization
    std = np.array([0.229, 0.224, 0.225])  # Standard deviation values for normalization
    np_image = (np_image - mean) / std

    # Transpose the color channel to the first dimension
    np_image = np_image.transpose((2, 0, 1))

    # Convert the NumPy array to a PyTorch tensor
    tensor_image = torch.from_numpy(np_image)

    return tensor_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, checkpoint, topk,  category_names, use_gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    # Load the checkpoint and rebuild the model
    model,optimizer_state_dict = load_checkpoint(checkpoint)
    
    model_type = model.__class__.__name__.lower()

    # Freeze the parameters of all layers except the last layer
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient computation for the parameters of the last layer
    if "vgg" in model_type:
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
    elif "densenet" in model_type:
        for param in model.classifier.parameters():
            param.requires_grad = True

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.02, momentum=0.9)
    optimizer.load_state_dict(optimizer_state_dict)
    
    image = process_image(image_path)
    image = image.unsqueeze(0)
    
    image = image.to(device)
    image = image.float()
    
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():
        # Forward pass through the model
        output = model(image)

        # Calculate the class probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the top K probabilities and class indices
        topk_probabilities, topk_indices = probabilities.topk(topk)

        # Convert the tensors to lists
        topk_probabilities = topk_probabilities.squeeze().tolist()
        topk_indices = topk_indices.squeeze().tolist()

        # Convert the class indices to class labels
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        topk_classes = [idx_to_class[index] for index in topk_indices]
        topk_flower_classes = [cat_to_name[each] for each in topk_classes]

    return topk_probabilities, topk_flower_classes

# def plot_image_probs(image_path, model, probs, classes, topk=5):
    

#     # Load and preprocess the image
#     image = Image.open(image_path)

#     # Create a figure with subplots
#     fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), nrows=2)

#     # Plot the input image
#     ax1.imshow(image)
#     ax1.axis('off')

#     # Plot the bar graph for probabilities
#     y_pos = np.arange(len(classes))
#     ax2.barh(y_pos, probs, align='center')
#     ax2.set_yticks(y_pos)
#     ax2.set_yticklabels(classes)
#     ax2.invert_yaxis()
#     ax2.set_xlabel('Probability')

#     # Set the title
#     ax1.set_title('Input Image')
#     ax2.set_title('Top {} Predictions'.format(topk))

#     # Adjust the layout and display the plot
#     plt.tight_layout()
#     plt.show()

def main():
    # Create argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    
    probs, classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
    
    print("Model Prediction - TopK --> ", args.top_k)
    print(probs)
    print(classes)
    print('=' * 15)
    
    print("Model Prediction - most likely image class and it's associated probability")
    print(probs[0])
    print(classes[0])
    
#     plot_image_probs(args.image_path, model, probs, classes, args.top_k)
    
if __name__ == '__main__':
    main()