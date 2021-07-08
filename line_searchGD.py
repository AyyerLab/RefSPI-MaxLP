import numpy as np
import scipy

class Optimizer():
    def __init__(self, dmax, mu, scale, num_frames):
        self.dmax = dmax
        self.mu = mu
        self.scale = scale
        self.num_frames = num_frames
        self.phi = (np.sqrt(5) + 1)/2

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

    def neg_objective(self, real_F, imag_F, rescale):
        w = rescale * np.abs(real_F + 1j * imag_F + self.ref(self.disps))**2
        Q = (self.photons * np.log(w) - w).sum()
        return Q

    def obj_derv(self, real_F, imag_F, rescale):
        w = rescale * np.abs(real_F + 1j * imag_F +  self.ref(self.disps))**2
        p = (self.photons / w - 1) * 2 * rescale
        d_real = p * (real_F + np.real(self.ref(self.disps)))
        d_imag = p * (imag_F + np.imag(self.ref(self.disps)))
        return -d_real.sum(), -d_imag.sum()
    

    def get_rescale(self, real_F, imag_F):
        return self.photons.mean()/(np.abs(real_F + 1j * imag_F + self.ref(self.disps))**2).mean()
    
    def gss(self, func, F, grad, a, b):
        
        c = b - (b - a)/ self.phi
        d = a + (b - a)/ self.phi

        zc = F + c * grad
        zd = F + d * grad
        obj_c = func(zc.real, zc.imag, self.get_rescale(zc.real, zc.imag))
        obj_d = func(zd.real, zd.imag, self.get_rescale(zd.real, zc.imag))

        if obj_c < obj_d :
            a = self.gss(self.neg_objective, F, grad, a, b)

        c = b - (b - a)/ self.phi
        d = a + (b - a)/ self.phi

        while abs(b-a) > 1.e-5:
            zc = F + c * grad
            zd = F + d * grad
            obj_c = func(zc.real, zc.imag, self.get_rescale(zc.real, zc.imag))
            obj_d = func(zd.real, zd.imag, self.get_rescale(zd.real, zd.imag))

            if obj_c < obj_d:
                b = d
            else:
                a = c

            c = b - (b - a)/ self.phi
            d = a + (b - a)/ self.phi

        return (a+b)/2

    def iterate(self, pars):
        '''Line Search Gradient Descent'''
        real_F, imag_F, rescale = pars

        d_real, d_imag  = self.obj_derv(*pars)
        grad = d_real + 1j * d_imag

        F = real_F + 1j * imag_F
        alpha = self.gss(self.objective, F, grad, -1, 0)

        real_F = real_F + alpha * d_real
        imag_F = imag_F + alpha * d_imag 

        rescale = self.get_rescale(real_F, imag_F)

        return real_F, imag_F , rescale

    def initialize(self):
        curr = 2 * (np.random.rand() + 1j * np.random.rand()) - 1
        return curr.real, curr.imag, self.get_rescale(curr.real, curr.imag)

