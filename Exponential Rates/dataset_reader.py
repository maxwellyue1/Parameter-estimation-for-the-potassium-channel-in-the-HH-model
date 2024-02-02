import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt

class Traces_Dataset(Dataset): 
    def __init__(self, csv_file, num_traces = 11, num_pts = 20):
        chunk = pd.read_csv(csv_file,chunksize=10000)
        df = torch.from_numpy(pd.concat(chunk).values).to(torch.float32)
        
        self.time_traces = torch.reshape(df[:, 0:(num_traces*num_pts)],(-1, num_traces, num_pts))
        self.current_traces = torch.reshape(df[:, (num_traces*num_pts):(num_traces*num_pts*2)], (-1, num_traces, num_pts))
        self.inputs = df[:, 0:(num_traces*num_pts*2)]
        self.params = df[:, (num_traces*num_pts*2):]

    def plot(self, sample): 
        for i in range(12): 
            plt.plot(self.time_traces[sample, i], self.current_traces[sample, i], label=f"{i}")
        plt.legend()
        plt.title(f'sample {sample}')


    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.inputs[idx], self.params[idx]
    
    
    def split_dataset(self, train_ratio, val_ratio, test_ratio=0):
        assert train_ratio + val_ratio + test_ratio == 1, "The sum of the ratios should be 1"
        train_size = int(len(self) * train_ratio)
        val_size = int(len(self) * val_ratio)
        test_size = len(self) - train_size - val_size
        self.train_set, self.val_set, self.test_set = random_split(self, [train_size, val_size, test_size], torch.Generator().manual_seed(42))
    
    def clean_features(self): 
        # delete features that are constant
        self.inputs = self.inputs[:, torch.var(self.inputs, axis=0) != 0]

    def find_mean_std(self): 
        self.train_mean = torch.mean(self[self.train_set.indices][0], dim=0)
        self.train_std = torch.std(self[self.train_set.indices][0], dim=0)

    def normalize(self): 
        # normalize the rest of the features
        self.inputs = (self.inputs - self.train_mean) / (self.train_std + 1e-8)

   