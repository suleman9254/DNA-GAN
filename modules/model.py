import torch
from torch.nn import Module, GRU, Linear
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm
import wandb
import statistics

def mean(x):
    return statistics.mean(x) if x else -1

class Generator(Module):
    def __init__(self, hidden_size, num_layers, seq_len):
        super(Generator, self).__init__()
        self.gru = GRU(input_size=1, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       bidirectional=True,
                       batch_first=True)
        
        in_features = 2*hidden_size
        self.fc = Linear(in_features=in_features, out_features=1)
        
        self.seq_len = seq_len

    def forward(self, x=None, z=None, supervision='none'):
        if supervision == 'none':
            y = self.unsupervised(z)
        elif supervision == 'single_cell':
            y = self.single_cell_supervised(x, z)
        elif supervision == 'multi_cell':
            y = self.multi_cell_supervised(x, z)
        return y

    def unsupervised(self, z):
        y, y_hat = self.gru(z), []
        for y_t in torch.split(y[0], 1, dim=1):
            y_hat_t = self.fc(y_t)
            y_hat.append(y_hat_t)

        y_hat = torch.cat(y_hat, dim=1) 
        return F.sigmoid(y_hat)

    def single_cell_supervised(self, x, z): # iterative single step prediction
        y_hat = []
        for i in range(self.seq_len):
            x_hat = x.clone()
            x_hat[:, i] = z[:, i]
            y, _ = self.gru(x_hat)
            y = self.fc(y[:, i])

            y_hat.append(y)
        
        y_hat = torch.cat(y_hat, dim=1)
        return x, F.sigmoid(y_hat)
    
    def multi_cell_supervised(self, x, z): # 'one go' multiple steps prediction
        mask = torch.randint(high=2, size=x.size()) == 1

        x_hat = x.clone()
        x_hat[mask] = z[mask] # corrupt step
        
        y, y_hat = self.gru(x_hat), []
        for y_t in torch.split(y[0], 1, dim=1):
            y_hat_t = self.fc(y_t)
            y_hat.append(y_hat_t)
        
        y_hat = torch.cat(y_hat, dim=1)
        return x[mask], F.sigmoid(y_hat[mask])

class Discriminator(Module):
    def __init__(self, hidden_size, num_layers, seq_len):
        super(Discriminator, self).__init__()
        self.gru = GRU(input_size=1, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       bidirectional=True,
                       batch_first=True)

        in_features = seq_len * hidden_size * 2
        self.fc = Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        y, _ = self.gru(x)
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)
        y = F.sigmoid(y)
        return torch.squeeze(y)

class DNA_GAN(Module):
    def __init__(self, hidden_size, num_layers, seq_len, eta=10, supervision='multi_cell'):
        super(DNA_GAN, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.eta = eta
        self.supervision = supervision
        
        self.generator = Generator(hidden_size, num_layers, seq_len)
        self.discriminator = Discriminator(hidden_size, num_layers, seq_len)

        self.test_input = torch.randn((1, self.seq_len, 1), dtype=torch.float)

    def fit(self, dataloader, epochs, lr, device, log_wandb):
        self.real_tag, self.fake_tag = 1, 0

        self.d_solver = Adam(self.discriminator.parameters(), lr=lr)
        self.g_solver = Adam(self.generator.parameters(), lr=lr)

        hist = []
        with tqdm(range(epochs), desc='Training') as pbar:
            for epoch in pbar:
                metrics = self.train_epoch(dataloader, device)
                
                self.generator.eval()
                test_signal = self.generator(z=self.test_input.to(device), 
                                             supervision='none')
                hist.append(test_signal)

                meta = {'d_loss': metrics[0], 
                        'g_unsuper_loss': metrics[1], 
                        'g_super_loss': metrics[2],
                        'd_acc': metrics[3], 
                        'epoch': epoch}
                pbar.set_postfix(meta)
                
                if log_wandb:
                    wandb.log(meta)
        return hist

    def train_epoch(self, dataloader, device):
        g_u_err, g_s_err, d_err, d_acc = [], [], [], []

        for real in dataloader:
            real = real.to(device)
                    
            batch_size = real.shape[0]
            self.label = torch.full((batch_size,), self.real_tag, 
                                    dtype=torch.float, device=device)
            
            # Train Generator
            self.generator.train(True)
            self.discriminator.eval()

            z = torch.randn_like(real)
            err = self.g_step_unsuper(z)
            g_u_err.append(err)

            if self.supervision == 'none':
                z = torch.randn_like(real)
                err = self.g_step_unsuper(z)
                g_u_err.append(err)

            else:
                z = torch.randn_like(real)
                err = self.g_step_super(real, z, self.supervision)
                g_s_err.append(err)

            # Train Discriminator
            self.generator.eval()
            self.discriminator.train(True)
            
            z = torch.randn_like(real)
            err, acc = self.d_step(real, z, threshold=0.7)
            d_err.append(err)
            d_acc.append(acc)

        return mean(d_err), mean(g_u_err), mean(g_s_err), mean(d_acc)

    def g_step_unsuper(self, z):
        self.label.fill_(self.real_tag)
        
        self.g_solver.zero_grad()

        syn = self.generator(z=z, supervision='none')
        d_pred = self.discriminator(syn)
        err = F.binary_cross_entropy(d_pred, self.label)

        err.backward()
        self.g_solver.step()
        return err.item()
    
    def g_step_super(self, x, z, supervision):
        self.g_solver.zero_grad()

        truth, syn = self.generator(x=x, z=z, supervision=supervision)
        err = F.l1_loss(syn, truth, reduction='sum') / truth.shape[0]
        scaled_err = self.eta * err

        scaled_err.backward()
        self.g_solver.step()
        return err.item()
    
    def d_step(self, real, latent, threshold):
        accuracy = 0
        self.d_solver.zero_grad()

        self.label.fill_(self.fake_tag)
        syn = self.generator(z=latent, supervision='none')
        d_pred = self.discriminator(syn)

        err_fk = F.binary_cross_entropy(d_pred, self.label)
        err_fk.backward()
        accuracy += sum(torch.round(d_pred) == self.label) / len(d_pred)

        self.label.fill_(self.real_tag)
        d_pred = self.discriminator(real)
        
        err_re = F.binary_cross_entropy(d_pred, self.label)
        err_re.backward()
        accuracy += sum(torch.round(d_pred) == self.label) / len(d_pred)

        accuracy = accuracy / 2
        err = (err_re + err_fk) / 2
        if accuracy <= threshold:
            self.d_solver.step()
        return err.item(), accuracy.item()