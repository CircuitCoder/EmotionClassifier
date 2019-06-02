#!/usr/bin/env python3

from data import VOCAB_SIZE, embeds
from train import train
import torch
import torch.nn as nn
import torch.nn.functional as F

CNN_SAVE_PATH = '/root/cnn_model'
RNN_SAVE_PATH = '/root/rnn_model'

torch.cuda.set_device(0)

# CNN: Parallel CNNs with different kernel sizes

embed_tensor = torch.FloatTensor(embeds)

EMBED_DIM = 300
KERNS = [2,3,4]
CONV_CHANNELS = 128
DROPOUT = 0.8

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.embed = nn.Embedding.from_pretrained(embed_tensor)
        self.convs = nn.ModuleList([nn.Conv2d(1, CONV_CHANNELS, (KERN, EMBED_DIM)) for KERN in KERNS])
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(CONV_CHANNELS * len(KERNS), 8)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(channel, channel.size(2)).squeeze(2) for channel in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

cnn = CNN()
cnn = cnn.cuda()

# RNN: LSTM + FC

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HIDDEN_DIM = 256
RNN_LAYERS = 2

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.embed = nn.Embedding.from_pretrained(embed_tensor)
        # Dropout combined in LSTM layer
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers = RNN_LAYERS, dropout = DROPOUT, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, 8)

    def forward(self, x, lengths):
        # Howtos on Embedding -> LSTM:
        # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        embedded = self.embed(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first = True)
        packed_output, (hidden, cell) = self.lstm(packed, None)

        output, _ = pad_packed_sequence(packed_output, batch_first = True)
        # print(output.size())

        row_idx = torch.arange(0, x.size(0)).long().cuda()
        col_idx = (lengths - 1).cuda()

        last = output[row_idx, col_idx, :]
        # print(last.size())
        return self.fc(self.dropout(last))
        # Again, since we are using cross_entropy, which implies log_softmax, we don't need one here

rnn = RNN()
rnn = rnn.cuda()

# Train
print("Training RNN...")
train(rnn, RNN_SAVE_PATH, pass_length = True)
print("Training CNN...")
train(cnn, CNN_SAVE_PATH)
