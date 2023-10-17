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
import encoding
import mlp
import siren
import glob
import utils
    
class Training(object):
    def __init__(self, checkpoint, gpu, batch_size):

        print(f"Loading model {checkpoint}")
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)

        self.hyperparameters = chk['hyperparameters']

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
            self.encoding = encoding.PositionalEncoding(sigma=self.hyperparameters['embedding']['sigma'], 
                                               n_freqs=self.hyperparameters['embedding']['n_freqs'],
                                               input_size=2).to(self.device)
            
        if (self.hyperparameters['embedding']['type'] == 'gaussian'):
            print("Gaussian encoding")
            self.encoding = encoding.GaussianEncoding(input_size=2,
                                                 sigma=self.hyperparameters['embedding']['sigma'],
                                                 encoding_size=self.hyperparameters['embedding']['encoding_size']).to(self.device)

        if (self.hyperparameters['embedding']['type'] == 'identity'):
            print("Identity encoding")
            self.encoding = encoding.IdentityEncoding(input_size=2).to(self.device)


        if (self.hyperparameters['mlp']['type'] == 'mlp'):
            print("MLP model")
            self.model = mlp.MLP(n_input=self.encoding.encoding_size,
                                    n_output=5,
                                    dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
                                    n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
                                    activation=nn.ReLU()).to(self.device)

        if (self.hyperparameters['mlp']['type'] == 'siren'):
            print("SIREN model")
            self.model = siren.SirenNet(dim_in=self.encoding.encoding_size,
                                    dim_out=5,
                                    dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
                                    num_layers=self.hyperparameters['mlp']['num_layers_mlp'],
                                    w0_initial=self.hyperparameters['mlp']['w0_initial']).to(self.device)
                        
                

        print('N. total parameters MLP : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        print("Setting weights of the model...")
        self.model.load_state_dict(chk['model_dict'])
        self.encoding.load_state_dict(chk['encoding_dict'])

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        tmp = np.loadtxt('wavelengthHinode.txt')
        wavelength = tmp[0:60] - 6301.5012
        self.wavelength = torch.tensor(wavelength.astype('float32')).to(self.device)

    def stokesSyn(self, wavelength, Ic, A, x0, a, ddop):
        vv = (wavelength - x0[:, None]) / (ddop[:, None] * 0.5)
        H = utils.fvoigt(a[:, None], vv)[0]
        normH = torch.erfc(a[:, None]) * torch.exp(a[:, None]**2)
        stokes = Ic[:, None] * (1.0 - A[:, None] * H / normH)
        return stokes

    def test(self):
            
        n = 32

        # f = h5py.File('/home/aasensio/datasets/hinode_sunspots/10921_0.h5', 'r')
        # f = h5py.File('/scratch1/aasensio/hinode_spots/10921_0.h5', 'r')
        stokes = np.load('hinode.npy')        
        cont = np.mean(stokes[0, 0:100, 0:100, 0])

        stokes = stokes[:, 0:n, 0:n, 0:60]
        stokes /= cont

        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)        
        xy = np.concatenate([xx[None, :], yy[None, :]], axis=0)

        xy = np.transpose(xy, axes=[1, 2, 0]).reshape((n*n, 2))

        xy = torch.tensor(xy, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
                                    
            # SIREN
            xy_encoded = self.encoding(xy, alpha=None)
            
            out = self.model(xy_encoded)
                
        out = out.reshape((n, n, 5))

        out = transform.transform_gaussian(out)

        Ic = out[..., 0]
        A = out[..., 1]
        x0 = out[..., 2]
        a = out[..., 3]
        ddop = out[..., 4]

        Ic = Ic.view(-1)
        A = A.view(-1)
        x0 = x0.view(-1)
        a = a.view(-1)
        ddop = ddop.view(-1)
        self.stokes = self.stokesSyn(self.wavelength, Ic, A, x0, a, ddop)
        self.stokes = self.stokes.cpu().numpy().reshape((n, n, 60))

        self.Ic = Ic.squeeze().cpu().numpy().reshape((n, n))
        self.A = A.squeeze().cpu().numpy().reshape((n, n))
        self.x0 = x0.squeeze().cpu().numpy().reshape((n, n))
        self.a = a.squeeze().cpu().numpy().reshape((n, n))
        self.ddop = ddop.squeeze().cpu().numpy().reshape((n, n))
        
        fig, ax = pl.subplots(3, 2, figsize=(10, 10))
        im = ax.flat[0].imshow(self.Ic)
        pl.colorbar(im, ax=ax.flat[0])
        ax.flat[0].set_title('Ic')

        im = ax.flat[1].imshow(self.A)
        pl.colorbar(im, ax=ax.flat[1])
        ax.flat[1].set_title('A')

        im = ax.flat[2].imshow(self.x0)
        pl.colorbar(im, ax=ax.flat[2])
        ax.flat[2].set_title('x0')

        im = ax.flat[3].imshow(self.a)
        pl.colorbar(im, ax=ax.flat[3])
        ax.flat[3].set_title('a')

        im = ax.flat[4].imshow(self.ddop)
        pl.colorbar(im, ax=ax.flat[4])
        ax.flat[4].set_title('ddop')

        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for i in range(4):
            ax.flat[i].plot(stokes[0, 10*i, 10*i, 0:60])
            ax.flat[i].plot(self.stokes[10*i, 10*i, :])
            print(self.Ic[10*i, 10*i], self.A[10*i, 10*i], self.x0[10*i, 10*i], self.a[10*i, 10*i], self.ddop[10*i, 10*i])
            

if (__name__ == '__main__'):

    files = glob.glob('weights/*.pth')    
    files.sort()
    checkpoint = files[-1]
    deepnet = Training(checkpoint, gpu=0, batch_size=32)    
    deepnet.test()
