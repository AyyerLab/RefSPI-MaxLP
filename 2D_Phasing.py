import sys
import os.path as op
import argparse
import configoarser

import time
import numpy as np
import cupy as cp

from scipy import ndimage
from scipy import special

class PhaseRecon():

    def __init__(self, config_file, apply_supp=True):

        config = configparser.ConfigParser()
        config.read(config_file)


        self.size = config.getint('parameters', 'size')
        self.true_support = config.getint('emc', 'true_support')
        
        self.output_folder = op.join(op.dirname(config_file), config.get('emc', 'output_folder'))
        self.true_support_fileA = op.join(op.dirname(config_file), config.get('emc', 'true_support_fileA'))
        self.true_support_fileB = op.join(op.dirname(config_file), config.get('emc', 'true_support_fileB'))
        self.fmintens = op.join(op.dirname(config_file), config.get('phasing', 'fourier_intensity'))   # using Normal EMC

        self.x_ind, self.y_ind = cp.indices((self.size,)*2, dtype = np.bool)
        self.x_ind = self.x_ind.ravel() - self.size//2
        self.y_ind = self.y_ind.ravel() - self.size//2
        self.rad = cp.sqrt(self.x_ind**2 + self.y_ind**2)

        self.invmask = cp.zeros(self.size**2, dtype = np.bool)
        self.invmask[self.rad<4] = True
        self.invmask[self.rad>=self.size//2] = True
        self.intinvmask = self.invmask.astype('i4')


        self.invsuppmask =  cp.zeros((self.size,)*2, dtype = np.bool)
        if true_support = 1:
            composite_objectA = cp.load(self.true_support_fileA)
            composite_objectB = cp.load(self.true_support_fileB)
            composite_object =  composite_objectA + composite_objectB
            self.invsuppmask = self.invsuppmask > composite_object


        fobj = cp.load(self.fmintens)    #Modulus Fourier Intensity generated via normal EMC
        
        self.apply_supp = apply_supp

        self.error = None
        self.curr = None
    

    def run(self, func, niter):
        if self.curr is None:
            self.curr = cp.copy(self.init)
        if self.error is None:
            self.error = []

        for i in range(niter):
            self.curr, err = func(self.curr)
            self.error.append(err)
            sys.stderr.write('\r%d'%i)
        return cp.array(self.error)



    def pc(self, x):
        avg = x.mean(0)
        if self.apply_supp:
            favg = cp.fft.fftshift(cp.fft.ifftn(avg.reshape(self.size, self.size)))
            favg[self.invsuppmask] = 0
            avg = cp.fft.fftn(cp.fft.ifftshift(favg))
        return cp.broadcast_to(avg, x.shape)

    def pd(self, x):
        N = len(x)
        out = cp.empty_like(x)
        for n in range(N):
            shifted = x[n]
            out[n] = shifted * self.data[n] / cp.abs(shifted)
            out[n][self.invmask] = x[n][self.invmask]
        return out


    def diffmap(self, x, return_err=True):
        p1 = self.pd(x)
        out = x + self.pc(2*p1 - x) - p1
        if return_err:
            err = cp.linalg.norm(out-x)
            return out, err
        else:
            return out

    def er(self, x, return_err=True):
        out = self.pc(self.pd(x))
        if return_err:
            err = cp.linalg.norm(out-x)
            return out, err
        else:
            return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('num_iter', help='Number of iterations', type=int)
    parser.add_argument('-s', '--size', help='Array size', type=int)

    args = parser.parse_args()

    recon = PhaseRecon(args.size)
    recon.run(recon.diffmap, args.num_iter)

