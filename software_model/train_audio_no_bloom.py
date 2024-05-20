#!/usr/bin/env python3

import sys
import itertools
import argparse
import ctypes as c
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
from scipy.stats import norm
import torch.optim as optim
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn as nn
import torch.nn.functional as F
# For saving models
import pickle
import lzma
import os

from wisard import WiSARD

# For the tabular datasets (all except MNIST)
import tabular_tools
# import IPython.display as ipd



def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


    
def index_to_label(index):
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]



class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        # x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, epoch, log_interval, train_loader, transform, optimizer, losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):


        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        # pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
    
def read_arguments():
    parser = argparse.ArgumentParser(description="Train BTHOWeN models for a dataset with specified hyperparameter sweep")
    parser.add_argument("dset_name", help="Name of dataset to use")
    parser.add_argument("--filter_inputs", nargs="+", required=True, type=int,\
            help="Number of inputs to each Bloom filter (accepts multiple values)")
    parser.add_argument("--filter_entries", nargs="+", required=True, type=int,\
            help="Number of entries in each Bloom filter (accepts multiple values; must be powers of 2)")
    parser.add_argument("--filter_hashes", nargs="+", required=True, type=int,\
            help="Number of distinct hash functions for each Bloom filter (accepts multiple values)")
    parser.add_argument("--bits_per_input", nargs="+", required=True, type=int,\
            help="Number of thermometer encoding bits for each input in the dataset (accepts multiple values)")
    parser.add_argument("--save_prefix", default="model", help="Partial path/fname to prepend to each output file")
    parser.add_argument("--num_workers", default=-1, type=int, help="Number of processes to run in parallel; defaults to number of logical CPUs")
    args = parser.parse_args()
    return args
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch, test_loader, transform):
    model.eval()
    correct = 0
    for data, target in test_loader:

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        # pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

def main():
    args = read_arguments()


    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
    
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)
    
#     ipd.Audio(transformed.numpy(), rate=new_sample_rate)
    batch_size = 16
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    print("input size of the model is" + str(transformed.shape[0]))
    model = M5(n_input=transformed.shape[0], n_output=len(labels))
    print(model)
    log_interval = 20
    n_epoch = 30

    losses = []

    # The transform needs to live on the same device as the model and the data.
    # transform = transform.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the 
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval,train_loader,transform, optimizer, losses)
        test(model, epoch, test_loader, transform)
        scheduler.step()
    
if __name__ == "__main__":
    main()

