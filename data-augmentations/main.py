# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 01:14:49 2019

@author: Keshik
"""

# In[0] import relevant libraries here

from task1 import task1
from task2 import task2
from task3 import task3

# In[1] define the main function here

def main():
    # Set the dirs here and
    images_dir = "./imagenet_data/imagespart/"
    labels_file = "./imagenet_data/synset_words.txt"
    num_images = 2500
    batch_size = 32
    
    print("Using {} images for experiments".format(num_images))
    # task 1
    print("----------Executing task 1-----------")
    task1(images_dir, labels_file, num_images, batch_size)
    print("----------Completed task 1-----------\n")
    
    
    # task 2
    print("----------Executing task 2 (Part 1)-----------")
    task2(images_dir, labels_file, num_images, batch_size = int(batch_size/5), five_crop=True)
    print("----------Completed task 2 (Part 1)-----------\n")
    
    print("----------Executing task 2 (Part 2)-----------")
    task2(images_dir, labels_file, num_images, batch_size = int(batch_size/10), five_crop=False) # If five_crop = False, then ten crop mode is activated
    print("----------Completed task 2 (Part 2)-----------\n")
    
    
    # task 3
    print("----------Executing task 3 (Using resnet50)-----------")
    task3(images_dir, labels_file, num_images, batch_size = int(batch_size/5), arch = 1)
    print("----------Completed task 3 (Using resnet50)-----------\n")

    print("----------Executing task 3 (Using inception_v3)-----------")
    task3(images_dir, labels_file, num_images, batch_size = int(batch_size/5), arch = 2)
    print("----------Completed task 3 (Using inception_v3)-----------\n")
    
    
    # Notes
    print("Using precomputed mean and standard deviation of data. " +  
          "The function used to obtain these values is data_loader.py/get_mean_and_std")

# In[2] Execute here
    
if __name__=='__main__':
    main()
