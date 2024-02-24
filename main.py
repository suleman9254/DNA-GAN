import torch
from torch.utils.data import DataLoader
from modules.model import DNA_GAN
from modules.dataset import DNA
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {'label':0, 
          'batch_size':64, 
          'epochs':40, 
          'lr':5e-4, 
          'hidden_size':4, 
          'num_layers':3, 
          'supervision':'multi_cell'}

dataset = DNA(pth='data\DATA_F256_K7_18_SPC_K6_NN_2.mat', label=config['label'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

print(len(dataset))

wandb.login()
wandb.init(project='DNA-GAN', config=config, name='test')

model = DNA_GAN(hidden_size=config['hidden_size'], 
                num_layers=config['num_layers'], 
                supervision=config['supervision'],
                seq_len=128)

hist = model.fit(dataloader, 
                 epochs=config['epochs'], 
                 lr=config['lr'], 
                 device=device, 
                 log_wandb=True)