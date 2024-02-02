import torch
import csv
import os
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from traces_dataset import Traces_Dataset
from MLP_model import Net
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import uuid


# initialize a dictionary of training history to store in a csv file
history_dict = {}

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
dataset = Traces_Dataset('dataset_test.csv')
dataset.split_dataset(0.9, 0.05, 0.05)
dataset.clean_features()
dataset.find_mean_std()
dataset.normalize()
print(dataset.inputs.shape)
history_dict['dataset'] = dataset.inputs.shape

# initialize train, val, test set
X_train = dataset[dataset.train_set.indices][0]
Y_train = dataset[dataset.train_set.indices][1]

X_val = dataset[dataset.val_set.indices][0]
Y_val = dataset[dataset.val_set.indices][1]

X_test = dataset[dataset.test_set.indices][0]
Y_test = dataset[dataset.test_set.indices][1]

# initialize NN model
model = Net(dataset.inputs.shape[1]).to(device)
model.initialize_weights()

# store model architecture
architecture = []
for name, layer in model.named_children():
    architecture.append(layer)
history_dict['architecture'] = architecture

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001) 
history_dict['optimizer'] = optimizer

# functions to save and load best nn model, and create an unique id to store the model
def checkpoint(model, filename, folder='models'):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    torch.save(model.state_dict(), filepath)
    return filepath
def resume(model, filepath):
    model.load_state_dict(torch.load(filepath))
unique_id = str(uuid.uuid4())[:8]
history_dict['unique_id'] = unique_id

# this is a 1d vector containing all weight in the nn model, no bias included. 
def weights_1d(model): 
    # Get parameters with gradients
    parameters_with_grad = [param for param in model.parameters() if param.requires_grad]
    # Concatenate parameters into a 1D tensor
    flat_parameters = torch.cat([param.view(-1) for param in parameters_with_grad])
    return flat_parameters.cpu()
weights_2d_tensor = weights_1d(model).unsqueeze(0)   # num epochs, # params, using unsqeeze to row dimension, so its (1, -1)
print(f'Number of model parameters: {weights_2d_tensor.shape[1]}')

# training parameters
n_epochs = 30   # number of epochs to run
#batch_size = 2000  # size of each batch
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

# record loss of the nn model without any training, dont need when on GPU
# model.eval()
# y_val_pred = model(X_val)
# y_train_pred = model(X_train)

# mse_val = loss_fn(y_val_pred, Y_val)
# mse_val = float(mse_val)
# val_losses.append(mse_val)

# mse_train = loss_fn(y_train_pred, Y_train)
# mse_train = float(mse_train)
# train_losses.append(mse_train)
# print(f'{0}: train-{mse_train}, val-{mse_val}')

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

        # find the weights_2d_tensor: flattened model weights across all epochs
        with torch.no_grad():
            weights_2d_tensor = torch.cat((weights_2d_tensor, weights_1d(model).unsqueeze(0)),dim=0)

        total_loss += loss.item()

    # Average training loss for the epoch
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation during validation
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
    if len(train_losses) != 1:
        if avg_train_loss < train_losses[-2]: 
            best_loss = avg_train_loss
            best_epoch = epoch
            model_path = checkpoint(model, f"model_{unique_id}.pth")



# calculating the norm of diff in model weights
diff_weights_2d_tensor = torch.diff(weights_2d_tensor, dim=0)
norm_diff_weights_2d_tensor = torch.norm(diff_weights_2d_tensor, dim=1).detach().cpu().numpy()

# record training, validationg losses, weight updates, and the result model path
history_dict['training'] = train_losses
history_dict['validation'] = val_losses
history_dict['weights'] = norm_diff_weights_2d_tensor.tolist()
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

experiment_records(history_dict)


print(f"Best training loss: {best_loss}, at epoch: {best_epoch}")
#print("RMSE: %.2f" % np.sqrt(best_mse))
print('Number of total training samples: ', X_train.shape[0])


os.makedirs('train_figures', exist_ok=True)
plt.figure()  # Create a new figure
plt.plot(train_losses, label='train_history')
#plt.plot(train_loss_history, label='train_history')
plt.plot(val_losses, label = 'val_history')
plt.legend()
#plt.ylim(100, 300)
#plt.xlim(0,10)
plt.savefig(f'train_figures/{unique_id}.png')

os.makedirs('weights_figures', exist_ok=True)
plt.figure()  # Create a new figure
plt.plot(norm_diff_weights_2d_tensor, label = 'norm_weights_diff') 
plt.xlabel('epoch')
plt.ylabel('norm of diff in model weights across epochs')
plt.legend()
plt.savefig(f'weights_figures/{unique_id}.png')