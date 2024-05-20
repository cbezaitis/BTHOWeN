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

# For saving models
import pickle
import lzma
import os
import torch
from wisard import WiSARD

# For the tabular datasets (all except MNIST)
import tabular_tools
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

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

# Perform inference operations using provided test set on provided model with specified bleaching value (default 1)
def run_inference(inputs, labels, model, bleach=1):
    num_samples = len(inputs)
    correct = 0
    ties = 0
    model.set_bleaching(bleach)
    for d in range(num_samples):
        if d % 100 == 0:
            print("Inference: " + str(d) + " / " + str(num_samples))
        prediction = model.predict(inputs[d])
        label = labels[d]
        if len(prediction) > 1:
            ties += 1
        if prediction[0] == label:
            correct += 1
    correct_percent = round((100 * correct) / num_samples, 4)
    tie_percent = round((100 * ties) / num_samples, 4)
    print(f"With bleaching={bleach}, accuracy={correct}/{num_samples} ({correct_percent}%); ties={ties}/{num_samples} ({tie_percent}%)")
    return correct

def parameterized_run(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, unit_inputs, unit_entries, unit_hashes):
    model = WiSARD(train_inputs[0].size, train_labels.max()+1, unit_inputs, unit_entries, unit_hashes)

    print("Training model")
    for d in range(len(train_inputs)):
        model.train(train_inputs[d], train_labels[d][0])
        if ((d+1) % 10000) == 0:
            print(str(d+1) + " / " + str(len(train_inputs)))

    max_val = 0
    for d in model.discriminators:
        for f in d.filters:
            max_val = max(max_val, f.data.max())
    print(f"Maximum possible bleach value is {max_val}")
    # Use a binary search-based strategy to find the value of b that maximizes accuracy on the validation set
    best_bleach = max_val // 2
    step = max(max_val // 4, 1)
    bleach_accuracies = {}
    while True:
        print("step: " + str(step+1) + " / " + str(max_val))
        values = [best_bleach-step, best_bleach, best_bleach+step]
        accuracies = []
        for b in values:
            if b in bleach_accuracies:
                accuracies.append(bleach_accuracies[b])
            elif b < 1:
                accuracies.append(0)
            else:
                accuracy = run_inference(val_inputs, val_labels, model, b)
                bleach_accuracies[b] = accuracy
                accuracies.append(accuracy)
        new_best_bleach = values[accuracies.index(max(accuracies))]
        if (new_best_bleach == best_bleach) and (step == 1):
            break
        best_bleach = new_best_bleach
        if step > 1:
            step //= 2
        
    print(f"Best bleach: {best_bleach}; inputs/entries/hashes = {unit_inputs},{unit_entries},{unit_hashes}")
    # Evaluate on test set
    print("Testing model")
    accuracy = run_inference(test_inputs, test_labels, model, bleach=best_bleach)
    print("Accuracy is: " + str(accuracy))
    return model, accuracy

# Convert input dataset to binary representation
# Use a thermometer encoding with a configurable number of bits per input
# A thermometer encoding is a binary encoding in which subsequent bits are set as the value increases
#  e.g. 0000 => 0001 => 0011 => 0111 => 1111
def binarize_datasets(train_dataset, test_dataset, bits_per_input, separate_validation_dset=None, train_val_split_ratio=0.9):
    # Given a Gaussian with mean=0 and std=1, choose values which divide the distribution into regions of equal probability
    # This will be used to determine thresholds for the thermometer encoding
    std_skews = [norm.ppf((i+1)/(bits_per_input+1))
                 for i in range(bits_per_input)]

    print("Binarizing train/validation dataset")
    train_inputs = []
    train_labels = []
    scale = 0
    zero_point = 0
    for d in train_dataset:
        # Expects inputs to be already flattened numpy arrays
        train_inputs.append(d[0])
        label = np.array(d[1])        
        train_labels.append(label)
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)
    # Step 3: Determine the maximum length of the arrays
    max_length = max(len(arr) for arr in train_inputs)
    # Step 4: Create a 2D NumPy array with the desired shape and padding value
    # Here, we use 0 as the padding value
    padded_array = np.zeros((len(train_inputs), max_length)).astype(np.float32)
    # Step 5: Fill the 2D array with the original arrays
    for i, arr in enumerate(train_inputs):
        padded_array[i, :len(arr)] = arr
    train_inputs = None    
    train_inputs = padded_array
    print("train inputs Fixed")
    print(train_inputs)
    use_gaussian_encoding = False
    if use_gaussian_encoding:
        mean_inputs = train_inputs.mean(axis=0)
        std_inputs = train_inputs.std(axis=0)
        train_binarizations = []
        for i in std_skews:
            train_binarizations.append(
                (train_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        min_inputs = train_inputs.min(axis=0)
        max_inputs = train_inputs.max(axis=0)
        max_val = np.mean(max_inputs)
        min_val = np.mean(min_inputs)
        # max_val = train_inputs.max()
        # min_val = train_inputs.min()
        print("max_val is: " + str(max_val) + " min_value is : " + str(min_val))
        scale = (max_val - min_val) / 255.0
        zero_point = round(-min_val / scale)
        zero_point = max(0, min(255, zero_point))
        print("zero_point is " + str(zero_point) + " scale : " + str(scale))
        quantized_data = np.round(train_inputs / scale) + zero_point
        quantized_data = np.clip(quantized_data, 0, 255) 
        train_inputs = quantized_data
        train_binarizations = []
        for i in range(bits_per_input):
            train_binarizations.append(
                (train_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))

    # Creates thermometer encoding
    print("train_binarizations")
    print(train_binarizations)
    train_inputs = np.concatenate(train_binarizations, axis=1)
    print("train_inputs")
    print(train_inputs[0])
  
    print("Binarizing test dataset")
    test_inputs = []
    test_labels = []
    for d in test_dataset:
        # Expects inputs to be already flattened numpy arrays
        test_inputs.append(d[0])
        label = np.array(d[1])        
        test_labels.append(label)
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)
    max_length = max(len(arr) for arr in test_inputs)
    # Step 4: Create a 2D NumPy array with the desired shape and padding value
    # Here, we use 0 as the padding value
    test_padded_array = np.zeros((len(test_inputs), max_length)).astype(np.float32)
    # Step 5: Fill the 2D array with the original arrays
    for i, arr in enumerate(test_inputs):
        test_padded_array[i, :len(arr)] = arr
    test_inputs = None
    test_inputs = test_padded_array
    
    test_binarizations = []
    if use_gaussian_encoding:
        for i in std_skews:
            test_binarizations.append(
                (test_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        quantized_data = np.round(test_inputs / scale) + zero_point
        quantized_data = np.clip(quantized_data, 0, 255) 
        test_inputs = quantized_data
        for i in range(bits_per_input):
            test_binarizations.append(
                (test_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))
    test_inputs = np.concatenate(test_binarizations, axis=1)
    print("test_inputs")
    print(test_inputs[0])
    
    validation_dataset = SubsetSC("validation")
    
    validation_processed_data = []
    for i in range(len(validation_dataset)):
        sample = validation_dataset[i]
        processed_sample = collate_fn(sample)
        validation_processed_data.append(processed_sample)
    
    new_validation_dataset = []
    for d in validation_processed_data:
        new_validation_dataset.append((d[0].numpy().flatten(), d[1]))
    validation_dataset = new_validation_dataset
    
        
    print("Binarizing validation dataset")
    validation_inputs = []
    validation_labels = []
    for d in validation_dataset:
        # Expects inputs to be already flattened numpy arrays
        validation_inputs.append(d[0])
        label = np.array(d[1])        
        validation_labels.append(label)
    validation_inputs = np.array(validation_inputs)
    validation_labels = np.array(validation_labels)
    max_length = max(len(arr) for arr in validation_inputs)
    # Step 4: Create a 2D NumPy array with the desired shape and padding value
    # Here, we use 0 as the padding value
    validation_padded_array = np.zeros((len(validation_inputs), max_length)).astype(np.float32)
    # Step 5: Fill the 2D array with the original arrays
    for i, arr in enumerate(validation_inputs):
        validation_padded_array[i, :len(arr)] = arr
    validation_inputs = None
    validation_inputs = validation_padded_array
    
    validation_binarizations = []
    if use_gaussian_encoding:
        for i in std_skews:
            validation_binarizations.append(
                (validation_inputs >= mean_inputs+(i*std_inputs)).astype(c.c_ubyte))
    else:
        quantized_data = np.round(validation_inputs / scale) + zero_point
        quantized_data = np.clip(quantized_data, 0, 255) 
        validation_inputs = quantized_data
        for i in range(bits_per_input):
            validation_binarizations.append(
                (validation_inputs > min_inputs+(((i+1)/(bits_per_input+1))*(max_inputs-min_inputs))).astype(c.c_ubyte))
    validation_inputs = np.concatenate(validation_binarizations, axis=1)
    print("validation_inputs")
    print(validation_inputs[0])
    print("validation_labels")
    print(validation_labels[1])

    return train_inputs, train_labels, validation_inputs, validation_labels, test_inputs, test_labels

train_set = SubsetSC("training")
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    
    return batch.permute(0, 2, 1)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    waveform, _, label, *_ = batch
    # print(waveform)
    tensors += [waveform]
    targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def create_models(dset_name, unit_inputs, unit_entries, unit_hashes, bits_per_input, num_workers, save_prefix="model"):
   
    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    number_of_labels = len(labels)
    train_processed_data = []
    for i in range(len(train_set)):
        sample = train_set[i]
        # print(train_set[i])
        processed_sample = collate_fn(sample)
        train_processed_data.append(processed_sample)
    
    # Hopefully: waveform, sample_rate, label, speaker_id, utterance_number
    print(train_set[1])
    test_processed_data = []
    for i in range(len(test_set)):
        sample = test_set[i]
        processed_sample = collate_fn(sample)
        test_processed_data.append(processed_sample)
        

    new_train_dataset = []
    for d in train_processed_data:
        new_train_dataset.append((d[0].numpy().flatten(), d[1]))
    train_dataset = new_train_dataset

    new_test_dataset = []
    for d in test_processed_data:
        new_test_dataset.append((d[0].numpy().flatten(), d[1]))
    test_dataset = new_test_dataset
    
    datasets = binarize_datasets(train_dataset, test_dataset, bits_per_input)
    prod = list(itertools.product(unit_inputs, unit_entries, unit_hashes))
    configurations = [datasets + c for c in prod]

    if num_workers == -1:
        num_workers = cpu_count()
    print(f"Launching jobs for {len(configurations)} configurations across {num_workers} workers")
    # with Pool(num_workers) as p:
    #     results = p.starmap(parameterized_run, configurations)
    results = parameterized_run(train_inputs=datasets[0], train_labels=datasets[1],
                                val_inputs=datasets[2], val_labels=datasets[3], 
                                test_inputs=datasets[4], test_labels=datasets[5],
                                unit_inputs=unit_inputs[0], unit_entries=unit_entries[0],
                                unit_hashes=unit_hashes[0])
    print(results)
    # for config in configurations:
    #     print("First Config")
    #     results = parameterized_run(configurations)
    #     print(results)
    for entries in unit_entries:
        print(
            f"Best with {entries} entries: {max([results[i][1] for i in range(len(results)) if configurations[i][7] == entries])}")
    configs_plus_results = [[configurations[i][6:9]] +
                            list(results[i]) for i in range(len(results))]
    configs_plus_results.sort(reverse=True, key=lambda x: x[2])
    for i in configs_plus_results:
        print(f"{i[0]}: {i[2]} ({i[2] / len(datasets[4])})")

    # Ensure folder for dataset exists
    os.makedirs(os.path.dirname(f"./models/{dset_name}/{save_prefix}"), exist_ok=True)
    

    for idx, result in enumerate(results):
        model = result[0]
        model_inputs, model_entries, model_hashes = configurations[idx][6:9]
        save_model(model, (datasets[0][0].size // bits_per_input),
            f"./models/{dset_name}/{save_prefix}_{model_inputs}input_{model_entries}entry_{model_hashes}hash_{bits_per_input}bpi.pickle.lzma")

def save_model(model, num_inputs, fname):
    model.binarize()
    model_info = {
        "num_inputs": num_inputs,
        "num_classes": len(model.discriminators),
        "bits_per_input": len(model.input_order) // num_inputs,
        "num_filter_inputs": model.discriminators[0].filters[0].num_inputs,
        "num_filter_entries": model.discriminators[0].filters[0].num_entries,
        "num_filter_hashes": model.discriminators[0].filters[0].num_hashes,\
        "hash_values": model.discriminators[0].filters[0].hash_values
    }
    state_dict = {
        "info": model_info,
        "model": model
    }

    with lzma.open(fname, "wb") as f:
        pickle.dump(state_dict, f)

def read_arguments():
    parser = argparse.ArgumentParser(description="Train BTHOWeN models for a dataset with specified hyperparameter sweep")
    # parser.add_argument("dset_name", help="Name of dataset to use")
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

def main():
    args = read_arguments()

    for bpi in args.bits_per_input:
        print(f"Do runs with {bpi} bit(s) per input")
        create_models(
            "audio", args.filter_inputs, args.filter_entries, args.filter_hashes,
            bpi, args.num_workers, args.save_prefix)

if __name__ == "__main__":
    main()

