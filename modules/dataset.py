import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from sklearn.preprocessing import MinMaxScaler

def minmax(x):
    scaler = MinMaxScaler()
    x_hat = x.reshape(-1, 1)
    x_hat = scaler.fit_transform(x_hat)
    return x_hat.reshape(x.shape), scaler

class DNA(Dataset):
    def __init__(self, pth, label):
        f = h5py.File(pth, 'r')
        mat_cell = f['X_corr']
        self.data = f[mat_cell[label, 0]]
        self.data = np.transpose(self.data)
        
        self.data, self.scaler = minmax(self.data)
        self.data = self.data.astype(np.single)
    
    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        item = torch.from_numpy(self.data[idx])
        return torch.unsqueeze(item, dim=-1)
    
    def __len__(self):
        return len(self.data)