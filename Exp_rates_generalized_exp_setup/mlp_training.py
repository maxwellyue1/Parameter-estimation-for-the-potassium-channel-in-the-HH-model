import torch
import csv
import os
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from dataset_reader import Traces_Dataset
from mlp_model import MLP
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import uuid

# initialize a dictionary of training history to store in a csv file
history_dict = {}

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
# dataset = Traces_Dataset('dataset2mil.csv')
dataset = Traces_Dataset('dataset_test.csv')
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

# initialize NN model
model = MLP(dataset.inputs.shape[1], dataset.params.shape[1]).to(device)
model.initialize_weights()

# store model architecture
architecture = []
for name, layer in model.named_children():
    architecture.append(layer)
history_dict['architecture'] = architecture

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# record optimizer
lr = optimizer.param_groups[0]['lr']
weight_decay = optimizer.param_groups[0]['weight_decay']
history_dict['optimizer'] = f'{optimizer.__class__.__name__} {lr}'
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

# this is a 1d vector containing all weight in the nn model, no bias included. 
def weights_1d(model): 
    # Get parameters with gradients
    parameters_with_grad = [param for param in model.parameters() if param.requires_grad]
    # Concatenate parameters into a 1D tensor
    flat_parameters = torch.cat([param.view(-1) for param in parameters_with_grad])
    return flat_parameters
print(f'Number of model parameters: {weights_1d(model).shape[0]}')
weights_change_updates = np.array([])  # using numpy array to save memory 
weights_change_epochs = []

# training parameters
n_epochs = 30   # number of epochs to run
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

# record the weights with no training
# with torch.no_grad():
#     previous_weights_epoch = weights_1d(model)

print(model)

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

        # record the weights changes across each updates (batches)
        # with torch.no_grad():
        #     current_weights = weights_1d(model)
        #     if 'previous_weights' in locals():
        #         weight_change = torch.norm(current_weights - previous_weights).item()#.cpu()
        #         weights_change_updates = np.append(weights_change_updates, weight_change)
        #     previous_weights = current_weights

        total_loss += loss.item()

    # Average training loss for the epoch
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during validation
        # record the weights change across epochs
        # current_weights_epoch = weights_1d(model)
        # if 'previous_weights_epoch' in locals():
        #     weight_change_epoch = torch.norm(current_weights_epoch - previous_weights_epoch).item()#.cpu()
        #     weights_change_epochs.append(weight_change_epoch)
        # previous_weights_epoch = current_weights_epoch

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

    # saving the model with best training mse
    # if len(train_losses) != 1:
    #     if avg_train_loss < train_losses[-2]: 
    #         best_loss = avg_train_loss
    #         best_epoch = epoch
    #         model_path = checkpoint(model, f"model_{unique_id}.pth")
    if avg_val_loss < best_validation_loss:
        best_epoch = epoch
        model_path = checkpoint(model, f"model_{unique_id}.pth")
        best_training_loss = avg_train_loss
        best_validation_loss = avg_val_loss


# record training, validationg losses, weight updates, and the result model path
history_dict['best_epoch'] = best_epoch
history_dict['best_val'] = best_validation_loss
history_dict['best_train'] = best_training_loss
history_dict['training'] = train_losses
history_dict['validation'] = val_losses
history_dict['weights'] = weights_change_epochs
history_dict['model'] = model_path

def experiment_records(row_data, file_path = 'experiment records.csv'): 
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


print(f"Best validation loss: {best_validation_loss}, at epoch: {best_epoch}")
#print("RMSE: %.2f" % np.sqrt(best_mse))
print('Number of total training samples: ', X_train.shape[0])

experiment_records(history_dict)

os.makedirs('train_figures', exist_ok=True)
plt.figure()  # Create a new figure
plt.plot(train_losses, label='train_history')
#plt.plot(train_loss_history, label='train_history')
plt.plot(val_losses, label = 'val_history')
plt.legend()
#plt.ylim(100, 300)
#plt.xlim(0,10)
plt.savefig(f'train_figures/{unique_id}.png')

# os.makedirs('weights_figures', exist_ok=True)
# plt.figure()  # Create a new figure
# plt.plot(weights_change_updates, label = 'norm_weights_diff') 
# plt.xlabel('Epoch')
# plt.ylabel('Norm of diff in model weights across batch updates')
# plt.legend()
# plt.savefig(f'weights_figures/updates {unique_id}.png')

# plt.figure()  # Create a new figure
# plt.plot(weights_change_epochs, label = 'norm_weights_diff') 
# plt.xlabel('Epoch')
# plt.ylabel('Norm of diff in model weights across epochs')
# plt.legend()
# plt.savefig(f'weights_figures/epochs {unique_id}.png')

# print(train_losses)
# print(weights_change_epochs)