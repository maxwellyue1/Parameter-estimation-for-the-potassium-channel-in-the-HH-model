from traces_dataset import Traces_Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import uuid
import os


dataset = Traces_Dataset('dataset.csv')
import torch
my_tensor = dataset.inputs
n_samples = my_tensor.shape[0]
print(f'There are {n_samples} samples!')
# Check for NaN values
nan_mask = torch.isnan(my_tensor)

# Print the indices of NaN values
nan_indices = torch.nonzero(nan_mask)
print("Indices of NaN values:", nan_indices)

# Check if there are any NaN values
if torch.any(nan_mask):
    print("There are NaN values in the tensor.")
else:
    print("There are no NaN values in the tensor.")




# Generate example data (replace this with your actual data)
num_plots = 400

# Generate 100 random integers between 0 and 920049
random_samples = [random.randint(0, n_samples) for _ in range(400)]

# Set up the figure and axes
fig, axes = plt.subplots(20, 20, figsize=(60, 60))

# Flatten the axes to iterate over them
axes = axes.flatten()

# Plot multiple traces in each subplot
for i in range(num_plots):
    ax = axes[i]

    time_traces = dataset.time_traces[random_samples[i]]
    current_traces = dataset.current_traces[random_samples[i]]

    for j in range(12):
        ax.plot(time_traces[j], current_traces[j])
        ax.set_title(f'Plot {random_samples[i]}')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Save the figure
os.makedirs('data_combined_plots', exist_ok=True)
unique_id = str(uuid.uuid4())[:8]
plt.savefig(f'data_combined_plots/{unique_id}.png')
