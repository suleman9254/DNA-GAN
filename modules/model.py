import torch
from torch.nn import Module, GRU, Linear, CrossEntropyLoss
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

class Generator(Module):
    def __init__(self, hidden_size, num_layers):
        self.gru = GRU(input_size=1, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       batch_first=True)
        
        self.fc = Linear(in_features=hidden_size, 
                         out_features=1)

    def forward(self, x):
        y, y_hat = self.gru(x), []
        for y_t in torch.split(y, 1, dim=1):
            y_hat_t = self.fc(y_t)
            y_hat.append(y_hat_t)

        y_hat = torch.cat(y_hat, dim=1) 
        return F.sigmoid(y_hat)

class Discriminator(Module):
    def __init__(self, hidden_size, num_layers, seq_len):
        self.gru = GRU(input_size=1, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       bidirectional=True,
                       batch_first=True)

        in_features = seq_len * hidden_size * 2
        self.fc = Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        y = self.gru(x)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)
        return F.sigmoid(y)

def train(model, batch_size, seq_len, device):

    z = torch.randn((batch_size, seq_len), device=device)
    syn = model.generator(z)
    
        

class DNA_GAN(Module):
    def __init__(self, hidden_size, num_layers, seq_len):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.generator = Generator(hidden_size, num_layers)
        self.discriminator = Discriminator(hidden_size, num_layers, seq_len)

    def fit(self, lr, epochs, device):
        criterion = CrossEntropyLoss()
        d_solver = Adam(self.discriminator.parameters(), lr=lr)
        g_solver = Adam(self.generator.parameters(), lr=lr)

        for epoch in tqdm(range(epochs), desc='Training'):
            

        