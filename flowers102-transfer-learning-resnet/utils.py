# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 00:02:31 2019

@author: Keshigeyan
"""

# In[0] Import required libraries
import torch
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from dataset import FlowersDataSet
import scipy
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
# In[1] Get mean and standard deviation of dataset
def get_mean_and_std(dataloader):
    mean = []
    std = []
    
    total = 0
    r_running, g_running, b_running = 0, 0, 0
    r2_running, g2_running, b2_running = 0, 0, 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            r, g, b = data[:,0 ,:, :], data[:, 1, :, :], data[:, 2, :, :]
            r2, g2, b2 = r**2, g**2, b**2
            
            # Sum up values to find mean
            r_running += r.sum().item()
            g_running += g.sum().item()
            b_running += b.sum().item()
            
            # Sum up squared values to find standard deviation
            r2_running += r2.sum().item()
            g2_running += g2.sum().item()
            b2_running += b2.sum().item()
            
            total += data.size(0)*data.size(2)*data.size(3)
    
    # Append the mean values 
    mean.extend([r_running/total, 
                 g_running/total, 
                 b_running/total])
    
    # Calculate standard deviation and append
    std.extend([
            math.sqrt((r2_running/total) - mean[0]**2),
            math.sqrt((g2_running/total) - mean[1]**2),
            math.sqrt((b2_running/total) - mean[2]**2)
            ])
    
    return mean, std


# In[2] Plot training and validation history
def plot_history(train_hist, val_hist, filename, labels=["train", "validation"]):
    # Plot training and validation loss
    xi = [i for i in range(0, len(train_hist), 2)]
    plt.plot(train_hist, label = labels[0])
    plt.plot(val_hist, label = labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.savefig(filename)
    plt.show()

# In[3] Get value of flower dataset

def main():
    labels_file = scipy.io.loadmat('./data/imagelabels.mat')['labels']
    split = list(range(1, 8149))
    
    transformations = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor()])
    # Create a dataloader
    dataset = FlowersDataSet('./data/jpg/', np.ravel(labels_file), split, transforms=transformations)
    train_loader = DataLoader(dataset, batch_size=5)
    
    mean, std = get_mean_and_std(train_loader)
    print(mean, std)

#main()

#j = np.load("train_hist-task1.npy")
#print(j)