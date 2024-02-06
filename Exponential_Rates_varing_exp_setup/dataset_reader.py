import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from exp_hh_model import HH_model_exp
import pandas as pd
import matplotlib.pyplot as plt

class Traces_Dataset(Dataset): 
    def __init__(self, csv_file, num_traces = 11, num_pts = 20, num_params = 7):
        chunk = pd.read_csv(csv_file,chunksize=10000)
        df = torch.from_numpy(pd.concat(chunk).values).to(torch.float32)
        
        self.prestep_V = df[:, 0]
        self.step_V1 = df[:, 1]
        self.num_traces = num_traces
        self.num_pts = num_pts
        self.num_params = num_params
        self.time_traces = torch.reshape(df[:, 2:(2+num_traces*num_pts)],(-1, num_traces, num_pts))
        self.current_traces = torch.reshape(df[:, 2+(num_traces*num_pts):2+(num_traces*num_pts*2)], (-1, num_traces, num_pts))
        self.inputs = df[:, 0:(num_traces*num_pts*2+2)]
        self.params = df[:, (2+num_traces*num_pts*2):]

    def plot(self, sample): 
        # compare samples and simulations using sample params
        params_list = self.params[sample].tolist()
        params = {'p': params_list[0], 'g_max': params_list[1], 'E_rev': params_list[2], 'a_m': params_list[3], 'b_m': params_list[4], 'delta_m': params_list[5], 's_m': params_list[6]}

        step_Vs = np.arange(self.step_V1[sample], self.step_V1[sample]+11*10, 10)
        sim_setup = {'prestep_V': self.prestep_V[sample].numpy(), 'step_Vs': step_Vs, 't': np.arange(0.0, 6.0, 0.01)}

        model = HH_model_exp(params, sim_setup)
        current_traces_sim = model.simulation()

        colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'brown', 'gray', 'black']

        for step in range(self.num_traces): 
            plt.plot(self.time_traces[sample, step], self.current_traces[sample, step], color=colors[step])
            plt.plot(sim_setup['t'], current_traces_sim[step], linestyle='--', color=colors[step])
        plt.legend()
        plt.title(f'sample {sample}; prestep_V: {self.prestep_V[sample]}; step_V1: {self.step_V1[sample]}')


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

   