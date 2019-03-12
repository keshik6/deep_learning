# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:38:42 2019

@author: Keshik
"""

# In[0] Import required libraries
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from getimagenetclasses import parsesynsetwords, parseclasslabel
import matplotlib.pyplot as plt
import math
from PIL import Image
from random import shuffle
from tqdm import tqdm
import torch

# In[1] Helper functions

def get_label_and_description(root_dir, image_file, synsetstoindices, syssetstoclassdescr):
    #root_dir = './imagenet_data/val/'
    xml_file = root_dir + image_file.split('/')[-1].split('.')[0] + '.xml'
    label, firstname = parseclasslabel(xml_file, synsetstoindices)
    description = syssetstoclassdescr[firstname]
    return label, description

def plot_image(images, titles):
    cols = 5
    rows = math.ceil(len(images)/cols)
    fig = plt.figure(figsize=(15, 9))
    
    for i in range(0, len(images)):
        image = images[i].permute(1, 2, 0)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(image)
        plt.title(titles[i])
        
    plt.show()
   
# In[2] Write a custom Dataset class

class ImageNetValDataSet(Dataset):
    
    def __init__(self, image_dir, num_images, labels_file, transforms):
        self.image_dir = image_dir
        self.transform = transforms
        self.filenames = []
        self.labels = []
        self.labels_file = labels_file
        self.image_description = []
        self.load_data_from_directory(self.image_dir, num_images)
        
    
    # In pytorch, images are represented as [channels, height, width]
    def __getitem__(self, index):
        # read the file and return the label
        img_path = self.filenames[index]
        label = self.labels[index]
        image = Image.open(img_path)
        
        # Convert grayscale images to rgb
        if (image.mode != "RGB"):
            image = image.convert(mode = "RGB")
            
        if self.transform is not None:
            image = self.transform(image)
            return image, label
        
        return image, label
        
    
    def __len__(self):
        return len(self.filenames)
    
    
    def load_data_from_directory(self, image_dir, num_images):
        self.image_dir = image_dir
        
        indicestosynsets, synsetstoindices, syssetstoclassdescr = parsesynsetwords(self.labels_file)
        
        files = os.listdir(self.image_dir)
        
        # Sort the list so that we generate the same dataset throughout for our experiments (for consistency)
        files.sort()
        files = files[:num_images]
        
        for file in files:
            file_path = image_dir + "/" + file
            self.filenames.append(file_path)
            root_dir = "/".join(self.image_dir.split("/")[:2]) + "/val/"
            label, description = get_label_and_description(root_dir, file_path, synsetstoindices, syssetstoclassdescr)
            self.labels.append(label)
            self.image_description.append(description)
            
        self.labels = np.asarray(self.labels)
        
        
    def get_filenames(self):
        return self.filenames
    
    
    # This is a custom function written to check for errors
    def getitem_with_description(self, index):
        image, label = self.__getitem__(index)
        description = self.image_description[index]
        return image, label, description
    


# In[3] Function to get dataloader
        
def get_data_loader(images_dir, num_images, labels_file, transformations, batch_size):
    dataset = ImageNetValDataSet(images_dir, num_images, labels_file, transforms=transformations)
    validation_loader = DataLoader(dataset, batch_size=batch_size)
    
    return validation_loader

# In[4] Get mean and standard deviation of dataset
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

