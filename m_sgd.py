import numpy as np
import scipy

class Optimizer():
    def __init__(self, dmax, mu, scale, num_frames):
        self.dmax = dmax
        self.mu = mu
        self.scale = scale
        self.num_frames = num_frames
        self.lr = 5e-5
        self.beta = 0.75

    def data_gen(self):
        self.disps = np.random.rand(self.num_frames) * self.dmax * 2 - self.dmax
        self.true_F = 2 * (np.random.rand() + 1j*np.random.rand()) - 1
        true_intens = np.abs(self.true_F + self.ref(self.disps))**2
        true_rescale = self.mu/true_intens.mean()
        true_intens *= true_rescale
        
        self.photons = np.random.poisson(true_intens)

    def ref(self, d):
        return self.scale * np.exp(1j*d)

    def objective(self, real_F, imag_F, rescale):
        w = rescale * np.abs(real_F + 1j * imag_F + self.ref(self.disps))**2
        Q = (self.photons * np.log(w) - w).sum()
        return -Q

    def obj_real_F(self, real_F, imag_F, rescale):
        w = rescale * np.abs(real_F + 1j * imag_F + self.ref(self.disps))**2
        derv = (self.photons / w - 1) * 2 * rescale * (real_F + np.real(self.ref(self.disps)))
        return -derv.sum()

    def obj_imag_F(self, real_F, imag_F, rescale):
        w = rescale * np.abs(real_F + 1j * imag_F + self.ref(self.disps))**2
        derv = (self.photons / w - 1) * 2 * rescale * (imag_F + np.imag(self.ref(self.disps)))
        return -derv.sum()

    def get_rescale(self, real_F, imag_F):
        return self.photons.mean()/(np.abs(real_F + 1j * imag_F + self.ref(self.disps))**2).mean()

    #mSGD
    def iterate(self, pars):
        real_F, imag_F, rescale = pars

        d_real =  self.obj_real_F(*pars)
        self.vel_r = self.beta * self.vel_r + (1 - self.beta) * d_real 
        
        d_imag = self.obj_imag_F(*pars)
        self.vel_i = self.beta * self.vel_i + (1 - self.beta) * d_imag  
            
        real_F = real_F - self.lr * self.vel_r
        imag_F = imag_F - self.lr * self.vel_i

        rescale = self.get_rescale(real_F, imag_F)

        return real_F, imag_F, rescale

    def initialize(self):
        self.vel_r = 0
        self.vel_i = 0
        curr = np.random.rand(2)
        curr /= np.linalg.norm(curr)
        return curr[0], curr[1], self.get_rescale(curr[0], curr[1])

