import torch
from torch.nn import Module, GRU, Linear
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm
import wandb

class Generator(Module):
    def __init__(self, hidden_size, num_layers, seq_len):
        self.gru = GRU(input_size=1, 
                       hidden_size=hidden_size, 
                       num_layers=num_layers, 
                       bidirectional=True,
                       batch_first=True)
        
        in_features = 2*hidden_size
        self.fc = Linear(in_features=in_features, out_features=1)
        
        self.seq_len = seq_len

    def forward(self, x, supervised=False):
        y = self.unsupervised(x) if not supervised else self.single_cell_supervised(x)
        return y

    def unsupervised(self, x):
        y, y_hat = self.gru(x), []
        for y_t in torch.split(y, 1, dim=1):
            y_hat_t = self.fc(y_t)
            y_hat.append(y_hat_t)

        y_hat = torch.cat(y_hat, dim=1) 
        return F.sigmoid(y_hat)

    def single_cell_supervised(self, x): # iterative single step prediction
        y_hat = []
        for i in range(self.seq_len):
            x_hat = x.clone()
            x_hat[:, i, :] = torch.randn((1,), device=x.get_device())
            y = self.gru(x_hat)
            y = self.fc(y[:, i, :])

            y_hat.append(y)
        
        y_hat = torch.cat(y_hat, dim=1)
        return F.sigmoid(y_hat)
    
    def multi_cell_supervised(self, x): # 'one go' multiple steps prediction
        mask = torch.randint(high=2, size=x.size()) == 1
        z = torch.randn(size=x.size(), device=x.get_device())

        x_hat = x.clone()
        x_hat[mask] = z[mask]
        y = self.gru(x_hat)
        y = self.fc(y[mask])
        return x[mask], F.sigmoid(y)

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

class DNA_GAN(Module):
    def __init__(self, hidden_size, num_layers, seq_len):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.generator = Generator(hidden_size, num_layers)
        self.discriminator = Discriminator(hidden_size, num_layers, seq_len)

    def fit(self, dataloader, lr, batch_size, device, log_wandb):
        self.real_tag, self.fake_tag = 1, 0
        self.label = torch.full((batch_size,), self.real_tag, device=device)

        self.d_solver = Adam(self.discriminator.parameters(), lr=lr)
        self.g_solver = Adam(self.generator.parameters(), lr=lr)

        with tqdm(dataloader, desc='Training') as tepoch:
            for i, re_batch in enumerate(tepoch):
                re_batch = re_batch.to(device)
                
                # Train Generator
                self.generator.train(True)
                self.discriminator.eval()
                z = torch.randn((batch_size, self.seq_len), device=device)
                g_u_err = self.g_step_unsuper(z)
                g_s_err = self.g_step_super(re_batch)

                # Train Discriminator
                self.generator.eval()
                self.discriminator.train(True)
                z = torch.randn((batch_size, self.seq_len), device=device)
                d_err = self.d_step(re_batch, z, threshold=0)

                meta = {'d_loss': d_err, 
                        'g_unsuper_loss': g_u_err, 
                        'g_super_loss': g_s_err, 
                        'iter': i}
                if log_wandb:
                    wandb.log(meta)
        return None
    
    def g_step_unsuper(self, x):
        self.label.fill_(self.real_tag)
        
        self.g_solver.zero_grad()

        syn = self.generator(x, supervised=False)
        d_pred = self.discriminator(syn)
        err = F.cross_entropy(d_pred, self.label)

        err.backward()
        self.g_solver.step()
        return err
    
    def g_step_super(self, x):
        self.g_solver.zero_grad()

        syn = self.generator(x, supervised=True)

        batch_size = x.shape[0]
        err = F.l1_loss(syn, x, reduction='sum') / batch_size

        err.backward()
        self.g_solver.step()
        return err
    
    def d_step(self, real, latent, threshold):
        self.d_solver.zero_grad()

        self.label.fill_(self.fake_tag)
        syn = self.generator(latent, supervised=False)
        d_pred = self.discriminator(syn)
        err_fk = F.cross_entropy(d_pred, self.label)

        self.label.fill_(self.real_tag)
        d_pred = self.discriminator(real)
        err_re = F.cross_entropy(d_pred, self.label)

        err = torch.mean(err_fk, err_re)
        if err > threshold:
            err_fk.backward()
            err_re.backward()
            self.d_solver.step()
        return err