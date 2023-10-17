import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import transform
try:
    from nvitop import Device
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import matplotlib.pyplot as pl
import h5py
import sys
sys.path.append('modules')
import mlp
import glob
import utils
import milne
from einops import rearrange
    
class Training(object):
    def __init__(self, checkpoint, gpu, batch_size):

        print(f"Loading model {checkpoint}")
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)

        self.hyperparameters = chk['hyperparameters']
        self.alpha = chk['alpha']

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            self.handle = Device.all()[self.gpu]
            
            print("Computing in {0} : {1}".format(self.device, self.handle.name()))
        
        self.batch_size = batch_size
                
        kwargs = {'num_workers': 2, 'pin_memory': False} if self.cuda else {}
        
        # Model
        if (self.hyperparameters['embedding']['type'] == 'positional'):
            print("Positional encoding")
            self.encoding = mlp.PositionalEncoding(sigma=self.hyperparameters['embedding']['sigma'], 
                                               n_freqs=self.hyperparameters['embedding']['n_freqs'],
                                               input_size=2).to(self.device)
            
        if (self.hyperparameters['embedding']['type'] == 'gaussian'):
            print("Gaussian encoding")
            self.encoding = mlp.GaussianEncoding(input_size=2,
                                                 sigma=self.hyperparameters['embedding']['sigma'],
                                                 encoding_size=self.hyperparameters['embedding']['encoding_size']).to(self.device)
        
        self.model = mlp.MLP(n_input=self.encoding.encoding_size,
                                n_output=9,
                                dim_hidden=self.hyperparameters['n_hidden_mlp'],                                 
                                n_hidden=self.hyperparameters['num_layers_mlp'],
                                activation=nn.ReLU()).to(self.device)                

        print('N. total parameters MLP : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        print("Setting weights of the model...")
        self.model.load_state_dict(chk['model_dict'])
        self.encoding.load_state_dict(chk['encoding_dict'])

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Milne-Eddington
        nlinea = 1						# Numero linea en fichero        
        self.param = utils.paramLine(nlinea, verbose=True)

        tmp = np.loadtxt('wavelengthHinode.txt')
        wavelength = tmp[0:60] - 6301.5012
        self.wavelength = torch.tensor(wavelength.astype('float32')).to(self.device)

    def test(self):
        
        pl.close('all')
        n = 32

        stokes = np.load('hinode.npy')
        cont = np.mean(stokes[0, 0:100, 0:100, 0])

        stokes = stokes[:, 0:n, 0:n, 0:60]
        stokes /= cont
        _, nx, ny, nl = stokes.shape
        stokes = torch.tensor(stokes.astype('float32'))

        stokes = rearrange(stokes, 's x y l -> (x y) s l')

        stokes = stokes.numpy()
        
        
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)        
        xy = np.concatenate([xx[None, :], yy[None, :]], axis=0)

        xy = np.transpose(xy, axes=[1, 2, 0]).reshape((n*n, 2))

        xy = torch.tensor(xy, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
                                    
            # MLP
            xy_encoded = self.encoding(xy, alpha=self.alpha)
            
            out = self.model(xy_encoded)
                
        out = out.reshape((n, n, 9))
        out = out.squeeze()

        out = transform.transform_milne(out)

        out_stokes = milne.stokesSyn(self.param, 
                                     self.wavelength, 
                                     out[..., 0].view(-1), 
                                     out[..., 1].view(-1), 
                                     out[..., 2].view(-1), 
                                     out[..., 3].view(-1),
                                     out[..., 4].view(-1), 
                                     out[..., 5].view(-1), 
                                     out[..., 6].view(-1), 
                                     out[..., 7].view(-1), 
                                     out[..., 8].view(-1), cartesian=True)

        out = out.cpu().numpy()
        out_stokes = out_stokes.cpu().numpy()

        labels = ['Bx', 'By', 'Bz', 'vlos', 'eta0', 'a', 'ddop', 'S_0', 'S_1']
        
        fig, ax = pl.subplots(3, 3, figsize=(10, 10))
        for i in range(9):
            im = ax.flat[i].imshow(out[:, :, i])
            pl.colorbar(im, ax=ax.flat[i])
            ax.flat[i].set_title(labels[i])

        stokes = stokes.reshape((n, n, 4, 60))
        out_stokes = out_stokes.reshape((n, n, 4, 60))

        fig, ax = pl.subplots(2, 2)
        for i in range(4):
            ax.flat[i].plot(stokes[5, 1, i, :])
            ax.flat[i].plot(out_stokes[5, 1, i, :])

        return out, stokes, out_stokes


if (__name__ == '__main__'):

    files = glob.glob('weights/*.pth')    
    files.sort()
    checkpoint = files[-1]
    deepnet = Training(checkpoint, gpu=0, batch_size=32)    
    out, stokes, out_stokes = deepnet.test()
