"""
===============================================================================
Anomaly Detector using a Custom CNN Classifier for the ROAD Dataset
===============================================================================

Description:
    This script implements an anomaly detection system for the ROAD dataset.

Author:
    Rizq Khateeb

Date:
    2025-04-10 

===============================================================================
"""

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

#class for the creation of the custom H5Dataset
class H5Dataset(Dataset):
    def __init__(self, paths, hf, transform=None):
        """
        paths: A list of tuples (path, index)
        transform: Optional transform to be applied on a sample.
        """
        self.paths = paths
        self.transform = transform
        self.hf = hf

        self.anomaly_mapping = {
            "first_order_data_loss": 1,
            "first_order_high_noise": 2,
            "galactic_plane": 3,
            "lightning": 4,
            "oscillating_tile": 5,
            "rfi_ionosphere_reflect": 6,
            "solar_storm": 7,
            "source_in_sidelobes": 8,
            "third_order_data_loss": 9
        }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path, index = self.paths[i]

        # Extract the data sample and label using the index
        data = self.hf[f'{path}/data'][index]
        label = self.hf[f'{path}/labels'][index]

        # Apply a transformation to the data sample
        if self.transform:
            data = self.transform(data)

        # Convert sample to a tensor
        data = torch.tensor(data, dtype=torch.float32)

        # Reshape to [4, 256, 256]
        data = data.permute(2, 0, 1)

        # Decode label from bytes to string
        if isinstance(label, (np.bytes_, bytes)):
            label = label.decode('utf-8')

        # empty string means normal
        if label == '':
            label_int = 0
            
        # label determines path for anomalies
        else:
            label = path.split("/")[-1]
            if label in self.anomaly_mapping:
                label_int = self.anomaly_mapping[label]
                
            else:
                raise ValueError(f"Label {label} not found in mapping dictionary!")

        
        label_int = torch.tensor(label_int, dtype=torch.long)

        return data, label_int

# CNN model with hyperparamaters and default values
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10, base_filters=16, kernel_size=3, dropout_rate=0.0):
        """
        num_classes: Total number of classes (0 for normal and 1-9 for anomalies)
        base_filters: Number of filters for the first convolution layer.
        kernel_size: Convolution kernel size (assumed square).
        dropout_rate: Dropout probability.
        """
        super(CNNClassifier, self).__init__()
        padding = kernel_size // 2 
        # First conv block
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=base_filters, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool = nn.MaxPool2d(2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(in_channels=base_filters, out_channels=base_filters * 2, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        
        # Third conv block
        self.conv3 = nn.Conv2d(in_channels=base_filters * 2, out_channels=base_filters * 4, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(base_filters * 4)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(base_filters * 4, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# function to get the dataloaders
def get_data_loaders(batch_size, train_dataset, test_dataset):
    """
    Create the data loaders using our datasets
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

# Creating function for training and evaluation
def train_and_evaluate(lr, batch_size, base_filters, kernel_size, dropout_rate, num_epochs=3, train_dataset = None, test_dataset = None, device = None):
    """
    Trains a model with the specified hyperparameters for num_epochs (default 3) and evaluates its accuracy.
    Returns a tuple of (test_accuracy, model_state_dict).
    """
    train_loader, test_loader = get_data_loaders(batch_size, train_dataset, test_dataset)
    
    # Initialize the model with provided hyperparameters
    model = CNNClassifier(num_classes=10, base_filters=base_filters, kernel_size=kernel_size, dropout_rate=dropout_rate)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())
    
    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    accuracy = correct / total
    return accuracy, model.state_dict()


def main():
 
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Script to detect anomalies in ROAD dataset')
    parser.add_argument('--dataset', type=str, help='Path to the h5 dataset file to open', required=True)
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs to run')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[0.001], help='Learning Rate to use during training')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16], help='Batch Size to use during training')
    parser.add_argument('--filters', type=int, nargs='+', default=[16], help='# of Filters to use during trainining')
    parser.add_argument('--kernels', type=int, nargs='+', default=[3], help='Kernel Size to use during trainining')
    parser.add_argument('--dropouts', type=float, nargs='+', default=[0.0], help='Dropout rate to use during training')
    args = parser.parse_args()

    print("Will run whith these arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    

    # check needed arguments
    # make sure the dataset file exists
    if not os.path.isfile(args.dataset):
        print(f"ERROR: dataset needs to be specified")
        exit(1)

    # initialize variables for train and test pathways, and indice trackers for all groups
    train = 'train_data/data'
    test = 'test_data/data'
    train_indices = 0
    test_indices = 0
    anomalyGroups = []

    # extract the number of entries in train and test for normal, and the keys for each anomaly group
    with h5py.File(args.dataset, 'r') as hf:
        train_indices = len(hf[train])
        test_indices = len(hf[test])

        for k in hf['anomaly_data'].keys():
            anomalyGroups.append(k)

    # initialize dictionaries for training and testinf of each anomaly group
    anomalyTrainGroupDict = {}
    anomalyTestGroupDict = {}

    # for each anomaly group, split the data into 80% training, 20% testing
    # fill the dictionaries with the indices belonging to each
    with h5py.File(args.dataset, 'r') as hf:
        for k in hf['anomaly_data'].keys():
            dataset = hf['anomaly_data'][k]['data']

            # Define the sizes of the splits
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size

            # Split the dataset into training and testing sets
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            anomalyTrainGroupDict[k] = train_dataset.indices
            anomalyTestGroupDict[k] = test_dataset.indices

    # for training and testing for both normal and anomalies, fill the appropriate list
    # with the path and index for each entry
    trainPaths = []
    testPaths = []
    for i in range(train_indices):
        tup = ('train_data', i)
        trainPaths.append(tup)

    for i in range(test_indices):
        tup = ('test_data', i)
        testPaths.append(tup)

    for k in anomalyTrainGroupDict.keys():
        for idx in anomalyTrainGroupDict[k]:
            tup = (f'anomaly_data/{k}', idx)
            trainPaths.append(tup)

    for k in anomalyTestGroupDict.keys():
        for idx in anomalyTestGroupDict[k]:
            tup = (f'anomaly_data/{k}', idx)
            testPaths.append(tup)

    # shuffle both trainPaths and testPaths
    random.shuffle(trainPaths)
    random.shuffle(testPaths)

    # use cuda or mps if available or use cpu
    runon = "cpu"
    if torch.cuda.is_available():
        torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch.device("mps")
    else:
        torch.device("cpu")
    print(f"Will run on {runon}")

    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create datasets
    hf = h5py.File(args.dataset, 'r')
    train_dataset = H5Dataset(paths=trainPaths, hf=hf)
    test_dataset = H5Dataset(paths=testPaths, hf=hf)

    # Hyperparameter grid
    learning_rates = args.learning_rates
    batch_sizes = args.batch_sizes
    base_filters_list = args.filters
    kernel_sizes = args.kernels
    dropout_rates = args.dropouts

    # All hyperparameter combinations
    hyperparam_combinations = list(itertools.product(learning_rates, batch_sizes, base_filters_list, kernel_sizes, dropout_rates))
    hyperparam_names = ['learning_rate', 'batch_size', 'filters_size', 'kernel_size', 'dropout_rate']


    print(f"All hyperparameter combinations: {hyperparam_names} {hyperparam_combinations}")

    # Create directory for model checkpoints
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    results = []
    print("Starting hyperparameter tuning...")

    # Loop over all hyperparameter combinations
    for idx, (lr, batch_size, base_filters, kernel_size, dropout_rate) in enumerate(hyperparam_combinations):
        print(f"\nExperiment {idx+1}/{len(hyperparam_combinations)}: lr={lr}, batch_size={batch_size}, filters={base_filters}, kernel_size={kernel_size}, dropout={dropout_rate}")
        
        test_acc, model_state = train_and_evaluate(lr, batch_size, base_filters, kernel_size, dropout_rate, num_epochs=args.epochs, train_dataset=train_dataset, test_dataset=test_dataset, device=device)
        
        # Save model checkpoint with filename encoding hyperparameters.
        model_filename = f"model_lr{lr}_bs{batch_size}_f{base_filters}_k{kernel_size}_d{dropout_rate}.pth"
        model_save_path = os.path.join(checkpoint_dir, model_filename)
        torch.save(model_state, model_save_path)
        
        # Record results along with the model checkpoint path.
        results.append({
            "lr": lr,
            "batch_size": batch_size,
            "base_filters": base_filters,
            "kernel_size": kernel_size,
            "dropout_rate": dropout_rate,
            "test_accuracy": test_acc,
            "model_path": model_save_path
        })
        
        print(f"Test Accuracy: {test_acc:.4f} -- Model saved as {model_save_path}")
        
        # Save intermediate results to CSV after each experiment.
        results_df = pd.DataFrame(results)
        results_df.to_csv("hyperparam_tuning_results.csv", index=False)


    # Create table with final results and plot
    results_df = pd.DataFrame(results)
    print("\nHyperparameter Tuning Results:")
    print(results_df)

    results_df_sorted = results_df.sort_values(by="test_accuracy", ascending=False)
    print("\nBest Hyperparameters:")
    print(results_df_sorted.head())

    plt.figure(figsize=(10,6))
    plt.barh(results_df_sorted.index.astype(str), results_df_sorted['test_accuracy'], color='skyblue')
    plt.xlabel("Test Accuracy")
    plt.title("Hyperparameter Tuning Results")
    plt.gca().invert_yaxis()
    plt.show()


    # close the file
    hf.close()

if __name__ == "__main__":
    main()