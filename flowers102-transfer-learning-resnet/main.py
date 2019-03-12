# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:09:11 2019

@author: Keshigeyan
"""

from task1 import task1
from task2 import task2
from task3 import task3

def main():
    
    # Paths and parameters
    image_dir = './data/jpg/'
    
    # Task 1
    print("----Executing Task 1----")
    task1(image_dir, epochs=20, lr = 1e-4, batch_size=64)
    print("----Completed Task 1----")    
    
    # Task 2
    print("----Executing Task 2----")
    task2(image_dir, epochs=20, lr = 1e-4, batch_size=64)    
    print("----Completed Task 2----")
    
    # Task 3
    print("----Executing Task 3----")
    task3(image_dir, epochs=20, lr = 1e-4, batch_size=64)
    print("----Completed Task 3----")        
    

if __name__ == '__main__':
    main()