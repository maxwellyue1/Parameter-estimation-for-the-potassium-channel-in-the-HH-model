import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module): 
    def __init__(self, n_features, n_params):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(n_features, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, n_params)
            )
        
    def forward(self, x):
        return self.layers(x)
    
    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Apply Xavier initialization to linear layers
                init.xavier_uniform_(layer.weight)
                # You can also initialize biases, for example, with zeros
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
