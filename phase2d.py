import sys
import os.path as op
import argparse
import configparser
import time
import numpy as np
import cupy as cp
from scipy import ndimage


class PhaseRecon():
    def __init__(self, config_file):

        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')
        self.output_folder = op.join(op.dirname(config_file), config.get('emc', 'output_folder'))
        self.true_support_fileA = op.join(op.dirname(config_file), config.get('emc', 'true_support_fileA'))
        self.true_support_fileB = op.join(op.dirname(config_file), config.get('emc', 'true_support_fileB'))
        self.intensf = op.join(op.dirname(config_file), config.get('phase_retrieval', 'fourier_intensity'))

        self.x, self.y = cp.indices((self.size,)*2, dtype = 'f8')
        self.cen = self.size//2
        self.rad = cp.sqrt((self.x - self.cen)**2 + (self.y - self.cen)**2)
        
        self.invmask = cp.zeros((self.size,)*2, dtype = cp.bool)
        self.invmask[self.rad<4] = True
        self.invmask[self.rad>=self.size//2] = True
        
        self.invsuppmask =  cp.ones((self.size,)*2, dtype = cp.bool)
        composite_objectA = cp.load(self.true_support_fileA)              
        composite_objectB = cp.load(self.true_support_fileB)
        composite_object =  (composite_objectA + composite_objectB)/2
        self.invsuppmask = self.invsuppmask > composite_object

        #Rotation and Background Subtraction
        self.fobj = cp.load(self.intensf)
        self.fobj = cp.array(ndimage.rotate(self.fobj.get(), 3, reshape=False))
        self.fobj = self.bgsubt(self.fobj)
        self.fobj = self.fobj**0.5
        self.fobj[self.invmask] = 0
        

    def run(self, iternum):

        vals = cp.random.random((2,) + (self.size, self.size))
        rinit = cp.random.random((self.size,)*2, dtype='f8')
        rinit[self.invsuppmask] = 0
        init = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(rinit)))
        obs = cp.empty(shape=(self.size**2, iternum), dtype = float)

        for _ in range(iternum):
            print('This is iteration :', _ )

            for i in range(5000):
                init, err = self.diffmap(init)
                
            for i in range(1000):
                init, err = self.er(init)
            print('This is final error after er step :', err)
            np.save(op.join(self.output_folder, 'intensp_%.3d.npy'%_), init)

            #Multiple reconstructions
            intensp = init
            intensp[self.rad>=self.cen] = 0
            recon_obj = abs(cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(intensp))))
            recon_obj = recon_obj.reshape(self.size**2)
            obs[:,_] = recon_obj[:]

        np.save(op.join(self.output_folder,'obs.npy'), obs)


    def proj_direct(self, x):
        f = cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(x)))
        f[self.invsuppmask] = 0
        a = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(f)))
        return a
    
    def proj_fourier(self, x):
        out = x * self.fobj / cp.abs(x)
        out[self.invmask] = x[self.invmask]
        return out

    def diffmap(self, x):
        p1 = self.proj_fourier(x)
        out = x + self.proj_direct(2. * p1 - x) - p1
        err = cp.linalg.norm(out - x)
        return out, err

    def er(self, x):
        out = self.proj_direct(self.proj_fourier(x))
        err = cp.linalg.norm(out - x)
        return out, err

    def bgsubt(self, x, bgrad=8.):
        i0 = x.copy()
        mask = (self.rad<4) & (self.rad>92)
        i0[mask] = 1e4
        i0min = cp.array(ndimage.minimum_filter(i0.get(), bgrad, mode='constant', cval=1.e4))
        i0 -= i0min
        i0[mask] = -1.
        return i0.astype('f4')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Phase Retrieval')
    parser.add_argument('num_iter', help='Number of iterations',  type=int)
    parser.add_argument('-c', '--config_file', help='Path to configuration file(default:config.ini)', default='config.ini')
    args = parser.parse_args()

    recon = PhaseRecon(args.config_file)
    recon.run(args.num_iter)

