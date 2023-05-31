import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processing import *


class SelfAttentionHead(nn.Module):
    """ Acts as one head of self-attention """
    def __init__(self, head_size):
        '''`head_size`: output dimension of the attention layer'''
        super.__init__()
        self.key = nn.Linear(embd_size, head_size, bias=False)
        self.query = nn.Linear(embd_size, head_size, bias=False)
        self.value = nn.Linear(embd_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        '''`x` is the input to the attention layer.'''
        B, T, C = x.shape # (# batches, token size, embd_size)
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        weights = q @ k.transpose(-2, -1) * self.head_size**(-0.5) # Q*K^T / sqrt(head_size) 
                                                                   # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
                                                               # Gets the similarities between each query and each key
        weights = weights.masked_fill(self.tril == 0, float('-inf')) # makes it so that each row will not pay attention to nodes after their
                                                             # corresponding position
        weights = F.softmax(weights, dim=-1) # does softmax probabilities along the dimension of the token positions to get the association
                                     # Between tokens up to a certain index where everything after is masked

        v = self.value(x) # (B, T, head_size)
        out = weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size) 
                          # Results in a value for each head_size element of each token for each batch
                          # Batches do not talk to each other ANYWHERE
        return out
        