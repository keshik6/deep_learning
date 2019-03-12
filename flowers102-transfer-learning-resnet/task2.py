# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:57:33 2019

@author: Keshigeyan
"""


# In[0] Import required libraries
import torch
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torchvision.models as  models
from torch.utils.data import DataLoader
from dataset import FlowersDataSet
import scipy
import torch.optim as optim
from train_model import train_model, test
from utils import plot_history
import os
# In[2] Task 2

def task2(image_dir, epochs, lr, batch_size=64):
    directory = "./task2_results"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Load the labels and training index file
    trainfile = "./data/trainfile.txt"
    valfile = "./data/valfile.txt"
    testfile = "./data/test.txt"
    
    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("Available device = ", device)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 102)
    model.to(device)
        
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Imagenet values
    mean=[0.485, 0.456, 0.406] 
    std=[0.229, 0.224, 0.225]
    
    transformations = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean = mean, std = std),
                                          ])
    
    transformations_valid = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean = mean, std = std),
                                          ])
    
    
    # Create train dataloader
    dataset_train = FlowersDataSet('./data/jpg/', trainfile, transforms=transformations)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=2)
    
    # Create validation dataloader
    dataset_valid = FlowersDataSet('./data/jpg/', valfile, transforms=transformations_valid)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=2)
    
    # Create test-time dataloader
    dataset_test = FlowersDataSet('./data/jpg/', testfile, transforms=transformations_valid)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2)

    trn_hist, val_hist = train_model(model, device, train_loader, valid_loader, optimizer, epochs, directory)
    torch.cuda.empty_cache()
    
    plot_history(trn_hist[0], val_hist[0], os.path.join(directory,"loss"))
    plot_history(trn_hist[1], val_hist[1], os.path.join(directory,"accuracy"))
    np.save(os.path.join(directory,"train_hist"), np.asarray(trn_hist[:2]))
    np.save(os.path.join(directory,"val_hist"), np.asarray(val_hist[:2]))
    
    # Load the best weights before testing
    model.load_state_dict(torch.load(os.path.join(directory, "model")))
    
    print("Evaluating model on test set")
    test(model, device, test_loader)
    
    
#task2('./data/jpg/', './data/imagelabels.mat', './data/setid.mat', 2, 0.0001)