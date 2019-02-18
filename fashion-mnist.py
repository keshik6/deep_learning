# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:53:13 2019

@author: Keshigeyan
"""

# Introduction to Neural Networks
# Fully connected neural net for FashionMNIST

# Objective: The main objective is to implement a neural network for FashionMNIST. 

# Only change the main function to tune/ edit paramters
# In[1]:Import relevant libraries

import torch
import torch.nn as nn # Neural networks module of torch package
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F # Functions such as sigmoid, softmax, cross entropy etc
import torch.optim as optim
import numpy as np
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt


# In[2]: Helper functions

def plot_history(hist, labels, filename):
    # Plot training and validation loss
    xi = [i for i in range(0, len(hist), 2)]
    plt.plot([i[0] for i in hist], label = labels[0])
    plt.plot([i[1] for i in hist], label = labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.savefig(filename)
    plt.show()


def plot_class_wise_accuracy(clf_report):
    CLASS_CLOTHING = {0 :'T-shirt/top',
                  1 :'Trouser',
                  2 :'Pullover',
                  3 :'Dress',
                  4 :'Coat',
                  5 :'Sandal',
                  6 :'Shirt',
                  7 :'Sneaker',
                  8 :'Bag',
                  9 :'Ankle boot'}

    # Create dictionary of class and accuracy
    class_wise_acc = dict()
    for i in range(len(clf_report)):
        class_wise_acc[CLASS_CLOTHING[i]] = clf_report[i].item()
    
    class_wise_acc = dict(sorted(class_wise_acc.items(), key=lambda x: x[1]))
    plt.bar(range(len(class_wise_acc)), list(class_wise_acc.values()), align='center')
    plt.xticks(range(len(class_wise_acc)), list(class_wise_acc.keys()), rotation = 45)
    plt.title("Classification Accuracy per class")
    plt.savefig("class_wise_accuracy.png", bbox_inches='tight')
    plt.show()
    
    return class_wise_acc

# In[3]:Load data - Create dataset class and dataloader class

def get_data(batch_size_train, batch_size_valid, batch_size_test):
    # Create a dataset object
    dataset = datasets.FashionMNIST("../data", train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor()
                           ]))
    
    # Create a 80%, 20% train, validation split
    dataset_size = len(dataset)
    indices = [i for i in range(dataset_size)]
    
    # shuffle dataset
    np.random.shuffle(indices)
    
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    # if sampler specified, shuffle should be false for dataloaders
    # Create dataloaders for train and validation. (Note that test set == validation set in this question)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_valid, sampler=valid_sampler)
    
    # test loader is same as validation loader. But this style of code will help in the upcoming problem -sets
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, sampler=valid_sampler)
    
    return train_loader, valid_loader, test_loader


# In[4]:Define model

class SimpleNN(nn.Module):
    def __init__(self, input_dims):
        super(SimpleNN, self).__init__()
        self.l1 = nn.Linear(input_dims, 300)
        self.l2 = nn.Linear(300, 100)
        self.l3 = nn.Linear(100, 10)
    
    def forward(self, x):
        
        x = x.view(-1, 784)
        #print(x.size())
        # Pass through layer 1 block
        x = self.l1(x)
        x = F.relu(x)
        
        # Pass through layer 2 block
        x = self.l2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.l3(x)
        return F.log_softmax(x, dim=1)


# In[5]:Define train Function

def train(model, device, train_loader, optimizer):
    # Set the module in training mode.
    model.train(True)
    
    running_loss = 0
    running_correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Load batch data to device
        data, target = data.to(device), target.to(device)
        
        # Set optimizer gradients to zero
        optimizer.zero_grad()
        
        # Feed forward the network to determine the output
        output = model(data)
        
        # Calculate the loss. Here we use Negative log loss (Used for classifying C classes)
        # Calculating two losses here. One is the mean of the loss and then the sum of the loss
        loss = F.nll_loss(output, target, reduction="mean")
        
        # Use torch.Tensor.item() to get a Python number from a tensor containing a single value
        # reduction = 'sum' to sum up all the batch loss values and add to the running loss
        batch_loss = F.nll_loss(output, target, reduction="sum").item()
        running_loss += batch_loss
        
        # Get the number of correctly predicted samples
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        
        # View the target tensor as the same size as pred tensor 
        running_correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Backpropagate the system the determine the gradients
        loss.backward()
        
        # Update the paramteres of the model
        optimizer.step()
        
    
    num_samples = float(len(train_loader.sampler))
    avg_train_loss = running_loss/num_samples
    
    print('loss: {:.4f}, accuracy: {}/{} ({:.3f})'.format(
        avg_train_loss, running_correct, num_samples,
        running_correct / num_samples))
        
    return avg_train_loss, running_correct/num_samples


# In[6]: Define validation function

def validation(model, device, valid_loader):
    # Set the module in non-training mode.
    model.train(False)
    
    running_loss = 0
    running_correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # No need to backpropagate here
            running_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            running_correct += pred.eq(target.view_as(pred)).sum().item()
    
    num_samples = float(len(valid_loader.sampler))
    avg_valid_loss = running_loss/num_samples

    print('val_loss: {:.4f}, val_accuracy: {}/{} ({:.3f})'.format(
        avg_valid_loss, running_correct, num_samples,
        running_correct / num_samples))
    
    return avg_valid_loss, running_correct/num_samples


# In[7]: Define test function

def test(model, device, test_loader):
    model.eval()
    
    running_loss = 0
    running_correct = 0
    
    clf_matrix = torch.zeros(10, 10)
        
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            running_correct += pred.eq(target.view_as(pred)).sum().item()
            
            for t, p in zip(target.view(-1), pred.view(-1)):
                clf_matrix[t.long(), p.long()] += 1
                
    num_samples = float(len(test_loader.sampler))
    avg_test_loss = running_loss/num_samples

    print('test_loss: {:.4f}, test_accuracy: {}/{} ({:.3f})\n'.format(
        avg_test_loss, running_correct, num_samples,
        running_correct / num_samples))
    
    clf_report = clf_matrix.diag()/clf_matrix.sum(1)
    
    return avg_test_loss, running_correct/num_samples, clf_report


# In[8]: run training and validation

def run(device, model, train_loader, valid_loader, optimizer, epochs):
    # training and validation history
    loss_hist = []
    acc_hist = []
    
    best_val_loss = 1.0
    for epoch in tqdm(range(epochs)):
        tr_loss, tr_acc = train(model, device, train_loader, optimizer)
        val_loss, val_acc = validation(model, device, valid_loader)
        loss_hist.append((tr_loss, val_loss))
        acc_hist.append((tr_acc, val_acc))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best model")
            torch.save(model.state_dict(), "model")
        print("--------------------------------")
    
    return loss_hist, acc_hist


# In[9]: main function

def main():
    batch_size_train = 64
    batch_size_valid = 64
    batch_size_test = 64
    seed = 2019
    epochs = 30
    image_size = 28
    
    use_cuda = torch.cuda.is_available()
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("Available device = ", device)
    model = SimpleNN(input_dims = image_size*image_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # load data
    train_loader, valid_loader, test_loader = get_data(batch_size_train, batch_size_valid, batch_size_test)
    
    # Run training + validation 
    loss_hist, acc_hist = run(device, model, train_loader, valid_loader, optimizer, epochs)
    
    # Plot training and validation loss
    plot_history(loss_hist, ["Training Loss", "Validation Loss"], filename = "loss.png")
    
    # Plot training and validation accuracy
    plot_history(acc_hist, ["Training Accuracy", "Validation Accuracy"], filename="accuracy.png")
    
    # Load model with best saved weights
    model = SimpleNN(input_dims = image_size*image_size).to(device)
    print("Loading model with best weights")
    model.load_state_dict(torch.load("model"))
    
    # Evaluate the model on test set
    test_loss, test_acc, clf_report = test(model, device, test_loader)
    
    # Plot classwise accuracy
    class_wise_acc = plot_class_wise_accuracy(clf_report)
    
    # Print class-wise accuracy to be included in the report
    print(class_wise_acc)
    

# In[10]: Execute pipeline

if __name__=='__main__':
    main()

