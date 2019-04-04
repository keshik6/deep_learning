# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:45:39 2019

@author: Keshik
"""

import utils
import torch
import torch.nn as nn
import torch.optim as optim
import os 
from sklearn.metrics import accuracy_score
from dataset import TextLoader
from model import LSTMClassifier
from utils import plot_history
from tqdm import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')

def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train_model(model, optimizer, train_data, val_data, char2idx, label2idx, batch_size, max_epochs):
    
    criterion = nn.CrossEntropyLoss(reduction = "sum")
    train_loader = utils.create_dataset(train_data, char2idx, label2idx, batch_size=batch_size)
    
    tr_loss, tr_acc = [], []
    val_loss, val_acc = [], []
    
    for epoch in range(max_epochs):
        print("-------Epoch {}----------".format(epoch+1))
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in tqdm(train_loader):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()
            
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
            del batch, targets, lengths, raw_data
            gc.collect()
        
        acc = accuracy_score(y_true, y_pred)
        loss = total_loss.data.float()/len(train_data)
        print('train_loss: {:.4f}, train_acc: {:.3f}'.format(
                    loss, acc))
        
        val_loss_, val_acc_ = validate(model, val_data, char2idx, label2idx, criterion, batch_size)
        print('val_loss: {:.4f}, val_acc: {:.3f}'.format(
                    val_loss_, val_acc_))
        
        tr_loss.append(loss), tr_acc.append(acc)
        val_loss.append(val_loss_), val_acc.append(val_acc_)
        
    return (model, [tr_loss, tr_acc], [val_loss, val_acc])


def validate(model, val_data, char2idx, label2idx, criterion, batch_size = 32):
    y_true = list()
    y_pred = list()
    total_loss = 0
    
    valid_loader = utils.create_dataset(val_data, char2idx, label2idx, batch_size=batch_size)
    
    model.train(False)
    with torch.no_grad():
        for batch, targets, lengths, raw_data in tqdm(valid_loader):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            pred, loss = apply(model, criterion, batch, targets, lengths)
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float()/len(val_data), acc


def test(model, test_data, char2idx, label2idx, criterion, batch_size = 32):
    y_true = list()
    y_pred = list()
    
    total_loss = 0
    test_loader = utils.create_dataset(test_data, char2idx, label2idx, batch_size = 32)
    
    print("Evaluating on Test Set")
    model.train(False)
    with torch.no_grad():
        for batch, targets, lengths, raw_data in tqdm(test_loader):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
    
            pred, loss = apply(model, criterion, batch, targets, lengths)
            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
        acc = accuracy_score(y_true, y_pred)
        loss = total_loss.data.float()/len(test_data)
    
    print('test_loss: {:.4f}, test_acc: {:.3f}'.format(
                    loss, acc))
    
    return loss, acc
    

def train(seed=100, data_dir= "./data/names/", hidden_dim = 200, num_layers = 1, learning_rate=0.01, weight_decay=0.01, batch_size=32, num_epochs = 10):

    torch.manual_seed(seed)
    data_loader = TextLoader(data_dir)

    train_data = data_loader.train_data
    val_data = data_loader.val_data
    test_data = data_loader.test_data
    
    char_vocab = data_loader.token2id
    tag_vocab = data_loader.tag2id
    char_vocab_size = len(char_vocab)

    print('Training samples:', len(train_data))
    print('Valid samples:', len(val_data))
    print('Test samples:', len(test_data))
    
    # Embedding size will be vocab_size x vocab_size
    model = LSTMClassifier(char_vocab_size, char_vocab_size, hidden_dim, len(tag_vocab), num_layers=num_layers)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, trn_hist, val_hist = train_model(model, optimizer, train_data, val_data, char_vocab, tag_vocab, batch_size, num_epochs)
    
    plot_history(trn_hist[0], val_hist[0], os.path.join("loss-task2-{}".format(batch_size)))
    plot_history(trn_hist[1], val_hist[1], os.path.join("accuracy-task2-{}".format(batch_size)))
    
    criterion = nn.CrossEntropyLoss(reduction = "sum")
    test(model, test_data, char_vocab, tag_vocab, criterion)


if __name__ == '__main__':
    train(seed=100, data_dir= "./data/names/", hidden_dim = 200, num_layers = 1, learning_rate=0.01, weight_decay=0.01, batch_size=32, num_epochs = 10)