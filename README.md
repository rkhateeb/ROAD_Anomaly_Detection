# Anomaly Detector

This project implements an anomaly detection system for the ROAD dataset using a custom CNN classifier built with PyTorch. The project includes data handling from HDF5 files, training with hyperparameter tuning, and model checkpointing.

## A short description of each file included in the directory is as follows:

- `anomaly_detector.py`: The script for training the model
- `hyperparam_tuning_results.csv`: The results from hyperparameter tuning for each experiment (combination of hyperparameters)
- `README.md`: This file
- `requirements.txt`: Contains the list of necessary python modules required to run the script
- `ROAD_anomaly_detector_notebook_run.pdf`: Pdf containing the runs of the .ipynb along with the results
- `ROAD_anomaly_detector.ipynb`: Notebook used to run and prototype the development code for CNN
- `Report.pdf`: Written report detailing the process undergone to develop model and run experiments

## Overview

The script performs the following tasks:
- Loads and processes data from an HDF5 file.
- Creates custom PyTorch datasets for normal and anomalous data.
- Defines a Convolutional Neural Network (CNN) model for classification.
- Implements training and evaluation routines with configurable hyperparameters.
- Saves model checkpoints and records hyperparameter tuning results.
- Provides a final visualization of the hyperparameter tuning performance.

## Features
- **Custom Dataset**: Uses a custom `H5Dataset` class to read and transform HDF5 dataset files.
- **CNN Classifier**: Implements a three-layer CNN with configurable base filters, kernel size, and dropout.
- **Training and Evaluation**: The `train_and_evaluate` function trains the model for a specified number of epochs and evaluates its performance.
- **Hyperparameter Tuning**: Uses grid search to test combinations of learning rates, batch sizes, number of filters, kernel sizes, and dropout rates.
- **Checkpointing**: Saves model state dictionaries for each hyperparameter combination into a designated directory.
- **Result Analysis**: Saves results to a CSV file and creates a bar plot to visualize test accuracies across experiments.

## Requirements
Captured in requirements.txt

## Usage
Run the script from the command line by providing the path to your dataset and optional hyperparameters.

#### Basic Command (runs with default values)
`python anomaly_detector.py --dataset /path/to/dataset.h5`

### Additional Arguments
- `--epochs`: Number of training epochs (default: 10).
- `--learning_rates`: One or more learning rates (default: [0.001]).
- `--batch_sizes`: One or more batch sizes (default: [16]).
- `--filters`: One or more numbers of filters for the CNN (default: [16]).
- `--kernels`: One or more kernel sizes (default: [3]).
- `--dropouts`: One or more dropout rates (default: [0.0]).

#### Command with multiple hyperparameter values:
`python anomaly_detector.py --dataset /path/to/dataset.h5 --epochs 15 --learning_rates 0.001 0.0005 --batch_sizes 16 32 --filters 16 32 --kernels 3 5 --dropouts 0.0 0.5`

## Checkpoints & Results
- Model checkpoints are saved in the checkpoints directory with filenames indicating the hyperparameter configuration.
- Hyperparameter tuning results are written to a hyperparam_tuning_results.csv file.
- A bar plot visualizing the test accuracies is displayed upon completion of training.

## Troubleshooting
- Ensure the dataset file exists at the provided path.
- Verify that all required dependencies are installed.
- In case of device issues, check that your hardware supports CUDA or MPS, or the script will revert to CPU mode.
