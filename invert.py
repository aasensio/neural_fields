import numpy as np
import h5py
import sys
import torch
import pathlib
sys.path.append('modules')
import encoding
import mlp
from einops import rearrange
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import transform
try:
    from nvitop import Device
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
from collections import OrderedDict
import utils
import milne


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, n):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        # Load data
        # f = h5py.File('/home/aasensio/datasets/hinode_sunspots/10921_0.h5', 'r')
        stokes = np.load('hinode.npy')
        cont = np.mean(stokes[0, 0:100, 0:100, 0])

        stokes = stokes[:, 0:n, 0:n, 0:60]
        stokes /= cont
        _, nx, ny, nl = stokes.shape
        stokes = torch.tensor(stokes.astype('float32'))

        stokes = rearrange(stokes, 's x y l -> (x y) s l')

        self.stokes = stokes.numpy()

        self.n_training = self.stokes.shape[0]
    
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)        
        self.xy = np.concatenate([xx[None, :], yy[None, :]], axis=0)

        self.xy = np.transpose(self.xy, axes=[1, 2, 0]).reshape((n*n, 2))
        
    def __getitem__(self, index):

        stokes = self.stokes[index, :, :]
        xy = self.xy[index, :]
                       
        return stokes.astype('float32'), xy.astype('float32')
    
    def __len__(self):
        return self.n_training


class Training(object):
    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters

        self.cuda = torch.cuda.is_available()
        self.gpu = hyperparameters['gpu']
        self.smooth = hyperparameters['smooth']
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            self.handle = Device.all()[self.gpu]
            
            print("Computing in {0} : {1}".format(self.device, self.handle.name()))
        
        self.batch_size = hyperparameters['batch_size']        
                
        kwargs = {'num_workers': 4, 'pin_memory': False} if self.cuda else {}                        

        self.dataset = Dataset(n=32)
        
        print(f"Dataset size: {self.dataset.n_training}")
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        
        # Model
        if (self.hyperparameters['embedding']['type'] == 'positional'):
            print("Positional encoding")
            self.encoding = encoding.PositionalEncoding(sigma=self.hyperparameters['embedding']['sigma'], 
                                               n_freqs=self.hyperparameters['embedding']['n_freqs'],
                                               input_size=2).to(self.device)
            
        if (self.hyperparameters['embedding']['type'] == 'gaussian'):
            print("Gaussian encoding")
            self.encoding = encoding.GaussianEncoding(input_size=2,
                                                 sigma=self.hyperparameters['embedding']['sigma'],
                                                 encoding_size=self.hyperparameters['embedding']['encoding_size']).to(self.device)

        
        self.model = mlp.MLP(n_input=self.encoding.encoding_size,
                                n_output=9,
                                dim_hidden=self.hyperparameters['n_hidden_mlp'],                                 
                                n_hidden=self.hyperparameters['num_layers_mlp'],
                                activation=nn.ReLU()).to(self.device)
                
        print('N. total parameters MLP : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Milne-Eddington
        nlinea = 1						# Numero linea en fichero        
        self.param = utils.paramLine(nlinea, verbose=True)

        tmp = np.loadtxt('wavelengthHinode.txt')
        wavelength = tmp[0:60] - 6301.5012
        self.wavelength = torch.tensor(wavelength.astype('float32')).to(self.device)

    def init_optimize(self):

        self.lr = self.hyperparameters['lr']
        self.wd = self.hyperparameters['wd']
        self.n_epochs = self.hyperparameters['n_epochs']
        
        print('Learning rate : {0}'.format(self.lr))        
        
        p = pathlib.Path('weights/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'weights/{0}'.format(current_time)
        
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr, weight_decay=self.wd)
        self.loss_fn = nn.MSELoss().to(self.device)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.3*self.lr)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100
        
        print('Model : {0}'.format(self.out_name))

        self.alpha = torch.tensor(0.0).to(self.device)
        self.iter = 0

        for epoch in range(1, self.n_epochs + 1):            
            loss = self.train(epoch)

            self.loss.append(loss)

            self.scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'model_dict': self.model.state_dict(),
                'encoding_dict': self.encoding.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
                'hyperparameters': self.hyperparameters,
                'loss': self.loss,
                'alpha': self.alpha
            }

            if (loss < best_loss):
                print(f"Saving model {self.out_name}.best.pth")                
                best_loss = loss
                torch.save(checkpoint, f'{self.out_name}.best.pth')

            if (self.hyperparameters['save_all_epochs']):
                torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')

    def alpha_schedule(self, iter):
        if (iter < self.hyperparameters['embedding']['alpha_initial_iteration']):
            y = 0.0
        elif (iter > self.hyperparameters['embedding']['alpha_final_iteration']):
            y = 1.0
        else:
            x0 = self.hyperparameters['embedding']['alpha_initial_iteration']
            x1 = self.hyperparameters['embedding']['alpha_final_iteration']
            y0 = 0.0
            y1 = 1.0
            y = np.clip((y1 - y0) / (x1 - x0) * (iter - x0) + y0, y0, y1)
        
        return torch.tensor(float(y)).to(self.device)

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (stokes, xy) in enumerate(t):            
            stokes = stokes.to(self.device)
            xy = xy.to(self.device)
        
            self.optimizer.zero_grad()

            # SIREN
            # The factor 2 is because the profiles are not normalized to [0,1], while 
            # the sigmoid produces values in [0,1]
            # Positional encoding
            xy_encoded = self.encoding(xy, alpha=self.alpha)
            
            out = self.model(xy_encoded)

            out_transformed = transform.transform_milne(out)        
            
            Bx = out_transformed[:, 0]
            By = out_transformed[:, 1]
            Bz = out_transformed[:, 2]
            vlos = out_transformed[:, 3]
            eta0 = out_transformed[:, 4]
            a = out_transformed[:, 5]
            ddop = out_transformed[:, 6]
            S_0 = out_transformed[:, 7]
            S_1 = out_transformed[:, 8]
            
            out_stokes = milne.stokesSyn(self.param, self.wavelength, Bx, By, Bz, vlos, eta0, a, ddop, S_0, S_1, cartesian=True)

            weight = torch.tensor([1.0, 10.0, 10.0, 50.0]).to(self.device)
            
            # Loss
            loss = torch.mean((out_stokes - stokes)**2 * weight[None, :, None])
                    
            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):                
                gpu_usage = f'{self.handle.gpu_utilization()}'                
                memory_usage = f' {self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
            else:
                gpu_usage = 'NA'
                memory_usage = 'NA'

            tmp = OrderedDict()
            tmp['gpu'] = f'{gpu_usage}'
            tmp['mem'] = f'{memory_usage}'
            tmp['lr'] = f'{current_lr:9.5f}'
            tmp['iter'] = f'{self.iter:9.5f}'
            tmp['alpha'] = f'{self.alpha:9.5f}'
            tmp['loss'] = f'{loss_avg:9.7f}'
            t.set_postfix(ordered_dict = tmp)

            self.alpha = self.alpha_schedule(self.iter)

            self.iter += 1
            
        self.loss.append(loss_avg)
        
        return loss_avg

if (__name__ == '__main__'):

    hyperparameters_positional = {
        'batch_size': 32,
        'validation_split': 0.1,
        'gpu': 0,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 10,
        'smooth': 0.15,
        'save_all_epochs': False,
        'n_hidden_mlp': 64,
        'num_layers_mlp': 8,
        'embedding': {
            'type': 'positional',
            'encoding_size': 64,
            'sigma': 5.0,
            'n_freqs': 30,
            'alpha_initial_iteration': 30,
            'alpha_final_iteration': 250,
        },
    }

    hyperparameters_gaussian = {
        'batch_size': 32,
        'validation_split': 0.1,
        'gpu': 0,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 50,
        'smooth': 0.15,
        'save_all_epochs': False,
        'n_hidden_mlp': 64,
        'num_layers_mlp': 8,
        'embedding': {
            'type': 'gaussian',
            'encoding_size': 128,
            'sigma': 5.0,            
            'alpha_initial_iteration': 30,
            'alpha_final_iteration': 250,
        },
    }

    
    deepnet = Training(hyperparameters_gaussian)
    deepnet.init_optimize()
    deepnet.optimize()
