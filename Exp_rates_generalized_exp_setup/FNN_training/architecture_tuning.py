import torch
import csv
import os
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import sys  # Import the sys module
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import uuid

import torch.nn as nn
import torch.nn.init as init

sys.path.append('..')
from dataset_reader import Traces_Dataset

# initialize a dictionary of training history to store in a csv file
history_dict = {}
###########################################################################
num_hidden_layers, hidden_size = 2,16
history_dict['num_hidden_layers'] = num_hidden_layers
history_dict['hidden_size'] = hidden_size
###########################################################################


def params(n_layer, n_units): 
    return 321*n_units+n_units + (n_units*n_units+n_units)*(n_layer-1) + n_units*7+7
history_dict['trainable_params'] = params(num_hidden_layers, hidden_size)


unique_id = str(uuid.uuid4())[:8]
history_dict['unique_id'] = unique_id
print(f'Experiment ID: {unique_id}')


# making training reproducible
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
history_dict['seed'] = seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Uncomment this line
print(f"Using device: {device}")


# load and process dataset 
dataset = Traces_Dataset('../dataset_test.csv')
dataset.split_dataset(0.95, 0.05, 0)
dataset.clean_features()
dataset.find_mean_std()
dataset.normalize()
print(dataset.inputs.shape)
history_dict['normalize_mean'] = dataset.train_mean.tolist()
history_dict['normalize_std'] = dataset.train_std.tolist()
history_dict['dataset'] = (dataset.inputs.shape[0], dataset.inputs.shape[1])

# initialize train, val, test set
X_train = dataset[dataset.train_set.indices][0]
Y_train = dataset[dataset.train_set.indices][1]

X_val = dataset[dataset.val_set.indices][0]
Y_val = dataset[dataset.val_set.indices][1]

X_test = dataset[dataset.test_set.indices][0]
Y_test = dataset[dataset.test_set.indices][1]


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(FeedForwardNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size

        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),  # SILU activation function
                nn.BatchNorm1d(hidden_size)
            )
            for i in range(num_hidden_layers)
        ])

        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
    
    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # Apply Xavier initialization to linear layers
                init.xavier_uniform_(layer.weight)
                # Initialize biases, for example, with zeros
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)


# initialize NN model
model = FeedForwardNN(dataset.inputs.shape[1], hidden_size, num_hidden_layers, dataset.params.shape[1]).to(device)
model.initialize_weights()


# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# record optimizer
lr = optimizer.param_groups[0]['lr']
weight_decay = optimizer.param_groups[0]['weight_decay']
history_dict['optimizer'] = f'{optimizer.__class__.__name__}'
history_dict['lr'] = lr
history_dict['weight_decay'] = weight_decay


# functions to save and load best nn model, and create an unique id to store the model
def checkpoint(model, filename, folder='models'):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    torch.save(model.state_dict(), filepath)
    return filepath
def resume(model, filepath):
    model.load_state_dict(torch.load(filepath))


# training parameters
n_epochs = 50   # number of epochs to run
batch_size = 1024  # size of each batch
history_dict['epochs'] = n_epochs
history_dict['batch size'] = batch_size


# initialize dataloader 
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# initialization train, val losses
train_losses = []
val_losses = []

best_validation_loss = float('inf')



# Training loop
for epoch in range(1, n_epochs + 1):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        total_loss += loss.item()
    # Average training loss for the epoch
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during validation
        # validation
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_labels)
            total_val_loss += val_loss.item()

    # Average validation loss for the epoch
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'{epoch}: train-{avg_train_loss}, val-{avg_val_loss}')

    if avg_val_loss < best_validation_loss:
        best_epoch = epoch
        model_path = checkpoint(model, f"model_{unique_id}.pth")
        best_training_loss = avg_train_loss
        best_validation_loss = avg_val_loss


# record training, validationg losses, weight updates, and the result model path
history_dict['best_epoch'] = best_epoch
history_dict['best_val'] = best_validation_loss
history_dict['best_train'] = best_training_loss
history_dict['training_loss'] = train_losses
history_dict['validation_loss'] = val_losses
history_dict['model'] = model_path

for history in history_dict:
    print(f'{history}: {history_dict[history]}\n')




# save the training history to a csv file
def log_data_to_csv(row_data = history_dict, file_path = 'experiment_logbook.csv'): 
    '''
    row_data is a dictionary of the row_data we want to store
    '''
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a' if file_exists else 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # If the file is newly created, write the header row
        if not file_exists:
            header_row = row_data.keys() if isinstance(row_data, dict) else row_data
            csv_writer.writerow(header_row)

        # Write the data row
        if isinstance(row_data, dict):
            csv_writer.writerow(row_data.values())
        else:
            csv_writer.writerow(row_data)


log_data_to_csv()