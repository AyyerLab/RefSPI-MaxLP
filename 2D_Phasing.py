import sys
import os.path as op
import argparse
import configparser
import time
import numpy as np
import cupy as cp

class PhaseRecon():
    def __init__(self, config_file):

        config = configparser.ConfigParser()
        config.read(config_file)
        self.size = config.getint('parameters', 'size')
        self.output_folder = op.join(op.dirname(config_file), config.get('emc', 'output_folder'))
        self.true_support_fileA = op.join(op.dirname(config_file), config.get('emc', 'true_support_fileA'))
        self.true_support_fileB = op.join(op.dirname(config_file), config.get('emc', 'true_support_fileB'))

        #Fourier Intensity to perform Phase retrieval on
        self.fintens = op.join(op.dirname(config_file), config.get('phasing', 'fourier_intensity'))

        self.x, self.y = cp.indices((self.size,)*2, dtype = 'f8')
        self.cen = self.size//2
        self.rad = cp.sqrt((self.x - self.cen)**2 + (self.y - self.cen)**2)

        self.invmask = cp.zeros((self.size,)*2, dtype= cp.bool)
        self.invmask[self.rad<4] = True
        self.invmask[self.rad>=self.size//2] = True

        self.invsuppmask =  cp.ones((self.size,)*2, dtype = cp.bool)
        composite_objectA = cp.load(self.true_support_fileA)              
        composite_objectB = cp.load(self.true_support_fileB)
        composite_object =  (composite_objectA + composite_objectB)/2
        self.invsuppmask = self.invsuppmask > composite_object
        np.save(op.join(self.output_folder, 'invsuppmask.npy'), self.invsuppmask)

        #Fourier Intensity generated via normal EMC
        self.fobj = cp.load(self.fintens)
        self.fobj[self.invmask] = 0
        

    def run(self, iternum):

        iter_curr = cp.empty((385,)*2, dtype='c16')
        print('This is iter_curr:', iter_curr[120:240, 199:200])
        iter_curr[:] = self.fobj
        print('This is iter_curr as fobj :',iter_curr[120:240, 199:200])
        iter_curr[self.invmask]= 0
        print('this is iter_curr with invmask :',iter_curr[120:240,199:200])

        for i in range(iternum):
            iter_curr, err = self.diffmap(iter_curr)
            print('curr in diifMAP iteration_%.4d'%iternum, iter_curr[120:240,199:200])
            print('The error for diffmap in iteration_%.4d'%iternum, err)
        #if iternum % 5 == 0:
        #    iter_curr, err = self.er(iter_curr)
        #    print('cur in er iteration_%.4d'%iternum, iter_curr)
        #    print('The error for  er in iteration_%4d'%iternum, err)
        pmodel = cp.empty((385,)*2, dtype='c16')
        pmodel = iter_curr
        print('This Is Reconstructed Phase Intensity :', pmodel[120:240,199:200])
        np.save(op.join(self.output_folder, 'pmodel.npy'), pmodel)


    def proj_concur(self, iter_in, supp = True):
        iter_out = cp.empty_like(iter_in)
        #avg = iter_in.mean()
        if supp:
            f = cp.fft.fftshift(cp.fft.ifftn(iter_in))
            f[self.invsuppmask] = 0
            a = cp.fft.fftn(cp.fft.ifftshift(f))
            iter_out[:] = a
        return iter_out
    
    def proj_divide(self, iter_in):
        iter_out = cp.empty_like(iter_in)
        shifted = iter_in
        iter_out = shifted * self.fobj / cp.abs(shifted)
        return iter_out

    def diffmap(self, iterate):
        out = self.proj_divide(iterate)
        #out = iterate + self.proj_concur(2. * p1 - iterate) - p1
        err = cp.linalg.norm(out - iterate)
        return out, err

    def er(self, iterate):
        out = self.proj_concur(self.proj_divide(iterate))
        err = cp.linalg.norm(out - iterate)
        return out, err


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Phase Retrieval')
    parser.add_argument('num_iter', help='Number of iterations',  type=int)
    parser.add_argument('-c', '--config_file', help='Path to configuration file(default:config.ini)', default='config.ini')
    args = parser.parse_args()

    recon = PhaseRecon(args.config_file)
    recon.run(args.num_iter)

