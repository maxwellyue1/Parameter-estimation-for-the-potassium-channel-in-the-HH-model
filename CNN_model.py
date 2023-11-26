import torch
import torch.nn as nn
import torch.nn.init as init

class LeNet5(nn.Module):
    
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            #nn.Conv2d(in_channels=6, out_channels=16, kernel_size=2, stride=1, padding=0),
            #nn.SiLU(),
            #nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(128,64),  #in_features = 16 x5x5 
            nn.SiLU(),
            nn.Linear(64,64),
            nn.SiLU(),
            nn.Linear(64,32),
            nn.SiLU(),
            nn.Linear(32,num_classes)
        )

        # Apply weight initialization
        self.initialize_weights()

        
    def forward(self,x): 
        a1=self.feature_extractor(x)
        #print(a1.shape)
        a1 = torch.flatten(a1,1)
        a2=self.regressor(a1)
        return a2
    
    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # Use Xavier initialization for weights and set bias to zero
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)