# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:58:04 2019

@author: Keshigeyan
"""

# In[0] Import required libraries
import scipy.io
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import numpy as np

# In[1] Create Dataset class

class FlowersDataSet(Dataset):
    
    def __init__(self, image_dir, labels_file, transforms):
        self.image_dir = image_dir
        self.filenames = []
        self.labels_file = labels_file  # Index of labels array corresponds to number in the filename
        self.labels = []
        self.transforms = transforms
        self.load_filenames_and_labels() # Load all the filenames based on split
    
    
    # In pytorch, images are represented as [channels, height, width]
    def __getitem__(self, index):
        # read the file and return the label
        img_path = os.path.join(self.image_dir, self.filenames[index])
        
        # Map the label using the number in the filename
        label = self.labels[index]
        
        # Open the image now
        image = Image.open(img_path)
        
        # Convert grayscale images to rgb
        if (image.mode != "RGB"):
            image = image.convert(mode = "RGB")
            
        if self.transforms is not None:
            image = self.transforms(image)
            return image, label
        
        return image, label


    def __len__(self):
        return len(self.filenames)
    
    
    def load_filenames_and_labels(self):
        with open(self.labels_file, 'r') as fin:
            for entry in fin:
                entry = entry.strip()
                name, label = entry.split(' ', maxsplit=1)
                self.filenames.append(name)
                self.labels.append(int(label))      
        
    
# In[Appendix]
#mat = scipy.io.loadmat('./data/imagelabels.mat')
#
## train_filenames: trnid, validation_filenames: valid, test_filenames: tstid 
#split = scipy.io.loadmat('./data/setid.mat')

# In[1] Test the implementation of dataset class

#labels_file = scipy.io.loadmat('./data/imagelabels.mat')['labels']
#split = scipy.io.loadmat('./data/setid.mat')['trnid']
#transformations = transforms.Compose([transforms.Resize(224), 
#                                          transforms.CenterCrop(224), 
#                                          transforms.ToTensor()])
#    
#dataset = FlowersDataSet('./data/jpg/', np.ravel(labels_file), np.ravel(split), transforms=transformations)
