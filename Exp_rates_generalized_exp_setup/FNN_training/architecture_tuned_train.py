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
import itertools
import torch.nn as nn
import torch.nn.init as init

sys.path.append('..')
from dataset_reader import Traces_Dataset


# making training reproducible
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# load and process dataset 
dataset = Traces_Dataset('../dataset_test.csv')
dataset.split_dataset(0.9, 0.1, 0)
dataset.clean_features()
dataset.find_mean_std()
dataset.normalize()
print(dataset.inputs.shape)
# history_dict['normalize_mean'] = dataset.train_mean.tolist()
# history_dict['normalize_std'] = dataset.train_std.tolist()
# history_dict['dataset'] = (dataset.inputs.shape[0], dataset.inputs.shape[1])

# initialize train, val, test set
X_train = dataset[dataset.train_set.indices][0]
Y_train = dataset[dataset.train_set.indices][1]

X_val = dataset[dataset.val_set.indices][0]
Y_val = dataset[dataset.val_set.indices][1]

X_test = dataset[dataset.test_set.indices][0]
Y_test = dataset[dataset.test_set.indices][1]


class FeedForwardNN(nn.Module):
    def __init__(self, hidden_sizes, input_size = 321, output_size = 7):
        super(FeedForwardNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_hidden_layers = len(hidden_sizes)
        self.output_size = output_size

        # Define the input layer
        self.input_layer = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.SiLU(),  # SILU activation function
                nn.BatchNorm1d(hidden_sizes[0]))

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.SiLU(),  # SILU activation function
                nn.BatchNorm1d(hidden_sizes[i+1])
            )
            for i in range(len(hidden_sizes) - 1)
        ])

        # Define the output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Forward pass through the network
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
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



# training parameters
n_epochs = 300   # number of epochs to run
batch_size = 2048#1024  # size of each batch


# initialize dataloader 
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, Y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# save the training history to a csv file
def log_data_to_csv(row_data, file_path = 'architecture_tuned_models_best.csv'): 
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Uncomment this line
print(f"Using device: {device}")

# initialize a dictionary of training history to store in a csv file
history_dict = {}
###########################################################################
achitecture = (128, 128, 256, 256, 32)
achitecture = (64, 256, 128, 128, 256, 64)
history_dict['achitecture'] = achitecture
print(achitecture)
###########################################################################

unique_id = str(uuid.uuid4())[:8]
history_dict['unique_id'] = unique_id
print(f'Experiment ID: {unique_id}')

model = FeedForwardNN(achitecture, dataset.inputs.shape[1], dataset.params.shape[1]).to(device)
model.initialize_weights()

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

def checkpoint(model, filename, folder='models'):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    torch.save(model.state_dict(), filepath)
    return filepath
def resume(model, filepath):
    model.load_state_dict(torch.load(filepath))


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

for history in history_dict:
    print(f'{history}: {history_dict[history]}\n')

log_data_to_csv(history_dict)