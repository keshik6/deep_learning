# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 07:48:42 2019

@author: Keshigeyan
"""

# In[0] Import required libraries
import torch
from tqdm import tqdm
import gc
import os
# In[1] Train and Validate function

def train_model(model, device, train_loader, valid_loader, optimizer, epochs, directory):
    
    tr_loss, tr_acc, clf_train = [], [], []
    val_loss, val_acc, clf_valid = [], [], []
    best_val_acc = 0.0
    
    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_correct = 0.0
            
            clf_matrix = torch.zeros(102, 102)
            
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')

            if phase == 'train':
                model.train(True)  # Set model to training mode
                
                for data, target in tqdm(train_loader):
                    #print(data)
                    target = target.long()
                    data, target = data.to(device), target.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    output = model(data)
                    
                    loss = criterion(output, target)
                    
                    # Get metrics here
                    running_loss += loss # sum up batch loss
                    pred = output.argmax(dim=1) # get the index of the max log-probability
                    running_correct += torch.sum(pred == target)
                    
                    #print(pred
                    for t, p in zip(target.view(-1), pred.view(-1)):
                        clf_matrix[t.long()-1, p.long()-1] += 1
                    
                    
                    # Backpropagate the system the determine the gradients
                    loss.backward()
                    
                    # Update the paramteres of the model
                    optimizer.step()
            
                    # clear variables
                    del data, target, output
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item()/num_samples
                tr_acc_ = running_correct.item()/ num_samples
                clf_train_ = clf_matrix.diag()/clf_matrix.sum(1)
                
                print('train_loss: {:.4f}, train_accuracy: {}/{} ({:.3f})'.format(
                    tr_loss_, running_correct, num_samples,
                    tr_acc_))
                
                # Append the values to global arrays
                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_), clf_train.append(clf_train_)
                    
                    
            else:
                model.train(False)  # Set model to evaluate mode
        
                # torch.no_grad is for memory savings
                with torch.no_grad():
                    for data, target in tqdm(valid_loader):
                        #print(data)
                        target = target.long()
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        
                        loss = criterion(output, target)
                        
                        running_loss += loss # sum up batch loss
                        pred = output.argmax(dim=1) # get the index of the max log-probability
                        running_correct += torch.sum(pred == target)
                        
                        #print(pred)
                        for t, p in zip(target.view(-1), pred.view(-1)):
                            clf_matrix[t.long()-1, p.long()-1] += 1
                        
                        # clear variables
                        del data, target, output
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    
                    num_samples = float(len(valid_loader.dataset))
                    val_loss_ = running_loss.item()/num_samples
                    val_acc_ = running_correct.item() / num_samples
                    clf_valid_ = clf_matrix.diag()/clf_matrix.sum(1)
                    
                    # Append the values to global arrays
                    val_loss.append(val_loss_), val_acc.append(val_acc_), clf_valid.append(clf_valid_)
                
                
                    print('val_loss: {:.4f}, val_accuracy: {}/{} ({:.3f})'.format(
                    val_loss_, running_correct, num_samples,
                    val_acc_))
                    
                    # Save model using val_acc
                    if val_acc_ > best_val_acc:
                        best_val_acc = val_acc_
                        torch.save(model.state_dict(), os.path.join(directory,"model"))
                    
    return ([tr_loss, tr_acc, clf_train], [val_loss, val_acc, clf_valid])



# In[2] Test function
    
def test(model, device, test_loader):
    running_loss = 0.0
    running_correct = 0.0
    clf_matrix = torch.zeros(102, 102) 
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    test_loss, test_acc, clf_test = None, None, None
    
     # torch.no_grad is for memory savings
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            #print(data)
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            
            running_loss += loss # sum up batch loss
            pred = output.argmax(dim=1) # get the index of the max log-probability
            running_correct += torch.sum(pred == target)
            
            #print(pred)
            for t, p in zip(target.view(-1), pred.view(-1)):
                clf_matrix[t.long()-1, p.long()-1] += 1
            
            # clear variables
            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()
        
        
        num_samples = float(len(test_loader.dataset))
        test_loss = running_loss.item()/num_samples
        test_acc = running_correct.item() / num_samples
        clf_test = clf_matrix.diag()/clf_matrix.sum(1)
        
        print('test_loss: {:.4f}, test_accuracy: {}/{} ({:.3f})'.format(
                    test_loss, running_correct, num_samples,
                    test_acc))
        
    return ([test_loss, test_acc, clf_test])
    
