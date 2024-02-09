from modules.dataset import batch_generator

import torch
from torch.nn import Module, GRU, Linear, CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm
import wandb

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
        return y

def generate_then_discriminate(model, batch_size, seq_len, criterion, target, device):
    z = torch.randn((batch_size, seq_len), device=device)
    syn = model.generator(z)
    d_pred = model.discriminator(syn)
    err = criterion(d_pred, target)
    return err

def train(model, train_data, batch_size, seq_len, g_solver, d_solver, criterion, device):
    real_label, fake_label = 1, 0
    label = torch.full((batch_size,), target, device=device)

    # Generator Training
    model.generator.train(True)
    model.discriminator.eval()
    
    for g_step in range(2):
        g_solver.zero_grad()
        label.fill_(real_label)
        err_G = generate_then_discriminate(model, 
                                           batch_size, 
                                           seq_len, 
                                           criterion, 
                                           label, 
                                           device)
        err_G.backward()
        g_solver.step()

    # Discriminator Training
    model.generator.eval()
    model.discriminator.train(True)

    d_solver.zero_grad()
    label.fill_(fake_label)
    err_fake = generate_then_discriminate(model, 
                                          batch_size, 
                                          seq_len, 
                                          criterion, 
                                          label,
                                          device)
    
    real = batch_generator(train_data, batch_size)
    real = real.to(device)
    
    d_pred = model.discriminator(real)
    label.fill_(real_label)
    err_real = criterion(d_pred, label)

    err_D = torch.mean(err_real.item(), err_fake.item())
    if err_D > 0.15:
        err_fake.backward()
        err_real.backward()
        d_solver.step()

    return model, err_G, err_D

class DNA_GAN(Module):
    def __init__(self, hidden_size, num_layers, seq_len):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.generator = Generator(hidden_size, num_layers)
        self.discriminator = Discriminator(hidden_size, num_layers, seq_len)

    def fit(self, train_data, lr, iters, device, save_to_wandb):
        criterion = CrossEntropyLoss()
        d_solver = Adam(self.discriminator.parameters(), lr=lr)
        g_solver = Adam(self.generator.parameters(), lr=lr)

        for iter in tqdm(range(iters), desc='Training'):
            model, err_G, err_D = train(model, train_data, batch_size, seq_len, g_solver, d_solver, criterion, device)

            meta = {'d_loss': err_D, 'g_loss': err_G}
            if save_to_wandb:
                wandb.log(meta)

        return None