# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 09:45:17 2019

@author: Keshik

References
    https://github.com/hunkim/PyTorchZeroToAll
    https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers):

        super(LSTMClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers)
        self.dropout_layer = nn.Dropout(p=0.2)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.num_layers = num_layers
        


    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1*self.num_layers, batch_size, self.hidden_dim)),
						autograd.Variable(torch.randn(1*self.num_layers, batch_size, self.hidden_dim)))


    def forward(self, batch, lengths): 
        self.hidden = self.init_hidden(batch.size(-1))
        
        # Embedding Layer (Seq x Batchsize) -> (Seq x Batchsize x Vocabulary_size)
        embeds = self.embedding(batch)
        
        # Pack_padded packages the embedding into number of (input instances x Vocabulary_size)
        packed_input = pack_padded_sequence(embeds, lengths)
        
        # Outputs returns (Number of instances x hidden_dim)
        # ht returns hidden state (1 x Batchsize x hidden_dim)
        # ct returns cell state (1 x Batchsize x hidden_dim)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
        
        # ht is the last hidden state of the sequences
        # ct is the new cell state
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        #output = self.softmax(output)

        return output