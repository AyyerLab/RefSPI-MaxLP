import numpy as np
import h5py
from scipy import ndimage

PHI = (np.sqrt(5) + 1) / 2.

class MaxLPhaser():
    def __init__(self, data_fname, num_data=-1):
        self._parse_data(data_fname, num_data)
        self._get_qvals()
        self._rot_kwargs = {'reshape': False, 'order': 1, 'prefilter': False}

        self.fsol = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.sol))) * 1.e-3
        # (1e-3 factor is there for rel_scale in make_data)

        self.counts = np.array([self.po[i].size + self.cm[i].sum() for i in range(self.num_data)])
        self.mean_count = self.counts.mean()

    def _parse_data(self, data_fname, num_data):
        with h5py.File(data_fname, 'r') as fptr:
            self.sol = fptr['solution'][:]
            self.angs = fptr['true_angles'][:]
            self.angs *= 180. / np.pi # convert to degrees
            self.diams = fptr['true_diameters'][:]
            self.shifts = fptr['true_shifts'][:]
            self.ones, self.multi = fptr['ones'][:], fptr['multi'][:]
            self.po, self.pm, self.cm = fptr['place_ones'][:], fptr['place_multi'][:], fptr['count_multi'][:] # pylint: disable=invalid-name

        self.size = self.sol.shape[-1]

        if num_data < 0:
            self.num_data = len(self.po)
        else:
            self.num_data = num_data

    def _get_qvals(self):
        ind = (np.arange(self.size, dtype='f4') - self.size//2) / self.size
        self.qx, self.qy = np.meshgrid(ind, ind, indexing='ij')
        self.qrad = np.sqrt(self.qx**2 + self.qy**2)
        self.qrad[self.qrad==0] = 1.e-6

        self.mask = (self.qrad>4) & (self.qrad < self.size//2)

    def get_rescale(self, fobj):
        '''Calculate rescale factor for given model'''
        powder_calc = np.zeros((self.size,)*2)
        for d in range(self.num_data):
            powder_calc += np.abs(self.fcalc(fobj, self.diams[d], self.shifts[d]))**2
        return self.mean_count / powder_calc.sum() * self.num_data

    def get_logq_t(self, fobj, rescale):
        '''Calculate log-likelihood for given model and rescale'''
        res_t = np.zeros(self.size**2)
        for d in range(self.num_data):
            w_dt_unrot = rescale * np.abs(self.fcalc(fobj, self.diams[d], self.shifts[d]))**2
            w_dt = ndimage.rotate(w_dt_unrot, -self.angs[d], **self._rot_kwargs).ravel()
            logw_dt = np.log(w_dt)
            res_t -= w_dt # No-photon contribution
            res_t[self.po[d]] += logw_dt[self.po[d]]
            res_t[self.pm[d]] += logw_dt[self.pm[d]] * self.cm[d]
        return res_t.reshape((self.size,)*2) / self.num_data

    def get_grad_t(self, fobj, rescale):
        '''Generate pixel-wise complex gradients for given model and rescale'''
        res_t = np.zeros(self.size**2, dtype='c16')
        sqrt_r = np.sqrt(rescale)
        for d in range(self.num_data):
            f_dt_unrot = sqrt_r * self.fcalc(fobj, self.diams[d], self.shifts[d])
            f_dt = ndimage.rotate(f_dt_unrot.real, -self.angs[d], **self._rot_kwargs) + \
                   1j*ndimage.rotate(f_dt_unrot.imag, -self.angs[d], **self._rot_kwargs)
            f_dt = f_dt.ravel()
            res_t -= 2 * sqrt_r * f_dt
            f_dt /= np.abs(f_dt)**2
            res_t[self.po[d]] += 2 * sqrt_r * f_dt[self.po[d]]
            res_t[self.pm[d]] += 2 * sqrt_r * f_dt[self.pm[d]] * self.cm[d]
        return res_t.reshape((self.size,)*2) / self.num_data

    def fcalc(self, fobj, dia, shift):
        '''Get holographic combination of model and spherical reference'''
        return fobj + self.sphere(dia) * np.exp(1j*2*np.pi*(self.qx*shift[0] + self.qy*shift[1]))

    def sphere(self, dia):
        '''Get sphere transform for given diameter'''
        sval = np.pi * self.qrad * dia
        return (np.sin(sval) - sval*np.cos(sval)) / sval**3

    def gss(self, func, fobj, grad, a, b):
        '''Pixel-wise golden-section search of func(fobj + alpha * grad) where alpha is between a and b'''
        if not isinstance(a, np.ndarray):
            a = np.ones(fobj.shape, dtype='f8') * a
        if not isinstance(b, np.ndarray):
            b = np.ones(fobj.shape, dtype='f8') * b

        c = b - (b - a) / PHI
        d = a + (b - a) / PHI

        zc = fobj + c * grad
        zd = fobj + d * grad
        obj_c = func(zc, self.get_rescale(zc))
        obj_d = func(zd, self.get_rescale(zd))

        # TODO: Need masking to only apply this for selected pixels
        #if obj_c < obj_d :
        #    def neg_func(fobj, rescale):
        #        return -func(fobj, rescale)
        #    a = self.gss(neg_func, fobj, grad, a, b)

        #    c = b - (b - a) / PHI
        #    d = a + (b - a) / PHI

        for i in range(10):
            zc = fobj + c * grad
            zd = fobj + d * grad
            obj_c = func(zc, self.get_rescale(zc))
            obj_d = func(zd, self.get_rescale(zd))

            sel = (obj_c < obj_d)
            b[sel] = d[sel]
            a[~sel] = c[~sel]
            
            c = b - (b - a) / PHI
            d = a + (b - a) / PHI

        return (a+b)/2
