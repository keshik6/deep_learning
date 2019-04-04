# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 08:47:33 2019

@author: Keshik
"""

import torch
from torch.utils.data import DataLoader
from dataset import PaddedTensorDataset
import matplotlib.pyplot as plt

def vectorized_data(data, item2id):
    """
    Convert characters into tokens using a predefined dictionary
    
    Args 
        data: character data
        item2id: dictionary for mapping
    
    Returns
        vectorized mapped data
    """
    
    
    return [[item2id[token] if token in item2id else item2id['UNK'] for token in seq] for seq, _ in data]


def pad_sequences(vectorized_seqs, seq_lengths):
    """
    Given a vectorized sequence, pad the sequence with 0s to produce a tensor of size len(vectorized_seq) x maximum length of seq
    
    Args
        vectorized_seq: Vectorized sequence
        seq_lengths: lengths of sequences
        
    Returns
        0 padded sequence tensor
    """
    
    # create a zero matrix
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()

    # fill the index
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    
    return seq_tensor


def create_dataset(data, input2id, target2id, batch_size=4):
    """
    Given the data, target and id mappings returns a dataloader of given batch size
    
    Args
        data: Data consists of (name, label)
        input2id: Contains the dictionary mapping every character to a number
        target2id: Vector describing the index for every language
        
    Returns
        Dataloader with specified batch_size
    """
    vectorized_seqs = vectorized_data(data, input2id)
    seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
    seq_tensor = pad_sequences(vectorized_seqs, seq_lengths)
    target_tensor = torch.LongTensor([target2id[y] for _, y in data])
    raw_data = [x for x, _ in data]

    return DataLoader(PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data), batch_size=batch_size, num_workers=2)


def sort_batch(batch, targets, lengths):
    """
    Given a batch of varying lengths, sort the batch by lengths to be correctly fed into the lstm
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor.transpose(0, 1), target_tensor, seq_lengths


def plot_history(train_hist, val_hist, filename, labels=["train", "validation"]):
    """
    Plot training and validation history
    """
    # Plot training and validation loss
    xi = [i for i in range(0, len(train_hist), 2)]
    plt.plot(train_hist, label = labels[0])
    plt.plot(val_hist, label = labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.savefig(filename)
    plt.show()
