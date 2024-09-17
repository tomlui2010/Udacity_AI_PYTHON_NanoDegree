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

from workspace_utils import active_session


def train(data_directory, save_directory, model_architecture, learning_rate=0.02, hidden_units=512, epochs=3, use_gpu=True):
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ## Load the data
#     data_dir = 'flowers'
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    # mean and std as recommended by documentation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # DONE: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean,std=std)])

    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean,std=std)])

    # DONE: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    validation_image_datasets = datasets.ImageFolder(valid_dir, transform=test_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)

    # DONE: Using the image datasets and the trainforms, define the dataloaders
    batch_size=64
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size, shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_image_datasets, batch_size=batch_size, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=True)


    dataloaders = {
        'train':train_dataloaders,
        'valid':validation_dataloaders,
        'test':test_dataloaders
    }

    total_batch_sizes = {'train':len(train_dataloaders), 'valid':len(validation_dataloaders),'test':len(test_dataloaders)}
    

#     model = models.vgg16(pretrained=True)
    # Check if the specified architecture is available
    if hasattr(models, model_architecture):
        # Get the pretrained model based on the architecture variable
        model = getattr(models, model_architecture)(pretrained=True)
    else:
        # Handle the case when the specified architecture is not available
        raise ValueError(f"Invalid architecture: {model_architecture}")

    


    # Freeze the layers of pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # update the last linear layer of the model 
    if 'vgg' in model_architecture.lower():
        #'num_of_features' will serve as the number of inputs to the last layer(linear) of the classifier
        num_of_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_of_features, len(train_image_datasets.classes))
    elif 'densenet' in  model_architecture.lower():
        num_of_features = model.classifier.in_features
        model.classifier = nn.Linear(num_of_features, len(train_image_datasets.classes))

    #define criterion of loss function. We chose CrossEntropyLoss as this is a classification problem
    criterion = nn.CrossEntropyLoss()


    # Select the optimizer as Stochastic Gradient Descent with a momentum hyper-parameter 
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9)


    # add a learning rate scheduler to dynamically adapt the learning rate during training
    lr_decay_scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)

    with active_session():
        model = train_the_model(model, device, dataloaders, total_batch_sizes, criterion, optimizer, lr_decay_scheduler, epochs)

    with torch.no_grad():

        correct_images = 0
        total_images = 0

        for images, labels in dataloaders['test']:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            values, predictions = torch.max(outputs.data,1)

            total_images += labels.size(0)
            correct_images += (predictions == labels).sum().item()

        print('Prediction Accuracy on test images: {}%'.format(100 * correct_images/total_images))

    # Define the checkpoint file path
    checkpoint_path = save_directory + 'checkpoint.pth'

    # Create the checkpoint dictionary
    checkpoint = {
        'input_size': num_of_features, 
        'output_size': len(train_image_datasets.classes),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_image_datasets.class_to_idx,
        'num_epochs': epochs,
        'model_architecture':model_architecture
    }

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)

def train_the_model(model, device, dataloaders, total_batch_sizes, criterion, optimizer, scheduler, num_epochs=25):
    
    #copy model parameters to cuda device
    model = model.to(device)
    
    #these variables will track the optimal model accuracy and its corresponding weights
    optimal_accuracy = 0.0
    optimal_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        #Plan is to evaluate the model after every epoch of training. So e will loop over 'train' and 'valid' phase
        
        for phase in ['train','valid']:            
            
            if phase == 'train':
                # Set model to training mode
                scheduler.step()
                model.train()
            
            else:
                # Set model to evaluate mode
                model.eval()
            
            running_loss = 0.0
            running_correct_predictions = 0
            
            for inputs,labels in dataloaders[phase]:
                
                # copy over to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                since = time.time()
                
                # zero out the gradients
                optimizer.zero_grad()
                
                # Enable gradients in the train phase only
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # Forward pass
                    outputs = model(inputs)                    
                    values, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward propagation for train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_correct_predictions += torch.sum(preds == labels.data)
                print(f"Device = {device}; Time per batch: {(time.time() - since)/3:.3f} seconds")
            
            epoch_loss = running_loss / total_batch_sizes[phase]
            epoch_accuracy = running_correct_predictions.double() / (total_batch_sizes[phase] * 64)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_accuracy))
            
            if phase == 'valid' and epoch_accuracy > optimal_accuracy : 
                optimal_accuracy = epoch_accuracy
                optimal_model_weights = copy.deepcopy(model.state_dict())
                
    print('Training complete')
    print('Best val Acc: {:4f}'.format(optimal_accuracy))
    
    model.load_state_dict(optimal_model_weights)
    
    return model

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')

    # Add the command line arguments
    parser.add_argument('data_directory', help='path to the data directory')
    parser.add_argument('--save_dir', dest='save_directory', default='', help='directory to save checkpoints')
    parser.add_argument('--arch', dest='architecture', default='vgg13', help='choose architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')

    # Parse the command line arguments
    args = parser.parse_args()
    
    print(args)

    # Call the train function with the provided arguments
    train(args.data_directory, args.save_directory, args.architecture, args.learning_rate,
          args.hidden_units, args.epochs, args.gpu)

if __name__ == '__main__':
    main()
        