import torch
import csv
import os
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState

from dataset_reader import Traces_Dataset
from mlp_model import MLP


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_features = 7
DIR = os.getcwd()
EPOCHS = 3
BATCH_SIZE = 1024


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []

    in_features = 321
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 256, 1024)
        layers.append(nn.Linear(in_features, out_features))
        
        if trial.suggest_categorical(f"use_batchnorm_l{i}", [True, False]):
            layers.append(nn.BatchNorm1d(out_features))  # Assuming 1D input
            
        layers.append(nn.ReLU())
        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        # layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, target_features))
    # layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_dataset(trial):
# def get_dataset():
    # batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512, 1024])
    
    dataset = Traces_Dataset('dataset_test.csv')
    dataset.split_dataset(0.95, 0.05, 0)
    dataset.clean_features()
    dataset.find_mean_std()
    dataset.normalize()
    # print(dataset.inputs.shape)

    # initialize train, val, test set
    X_train = dataset[dataset.train_set.indices][0]
    Y_train = dataset[dataset.train_set.indices][1]

    X_val = dataset[dataset.val_set.indices][0]
    Y_val = dataset[dataset.val_set.indices][1]

    X_test = dataset[dataset.test_set.indices][0]
    Y_test = dataset[dataset.test_set.indices][1]

    # initialize dataloader 
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader


def objective(trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_dataset(trial)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (train_inputs, train_targets) in enumerate(train_loader):
            train_inputs, train_targets = train_inputs.to(DEVICE), train_targets.to(DEVICE)

            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            loss = nn.MSELoss()(train_outputs, train_targets)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (val_inputs, val_targets) in enumerate(valid_loader):
                val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)

                val_outputs = model(val_inputs)
                # Get the index of the max log-probability.
                val_loss = nn.MSELoss()(val_outputs, val_targets)
                total_val_loss += val_loss.item()

            # Average validation loss for the epoch
        avg_val_loss = total_val_loss / len(valid_loader)

        trial.report(avg_val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=None, n_jobs=-1)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))