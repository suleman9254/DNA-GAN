import torch
import torch.nn.functional as F
from torch.nn import Module, GRU, Linear

class Forecaster(Module):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = GRU(input_size=1, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers,
                       bidirectional=True, 
                       batch_first=True)
    
        self.fc = Linear(in_features=hidden_size*2, 
                         out_features=1)
    
    def forward(self, x):
        y = self.gru(x)
        y = torch.flatten(x, start_dim=1)
        y = self.fc(y)
        return x[:, 1:], F.sigmoid(y[:, :-1])
    

