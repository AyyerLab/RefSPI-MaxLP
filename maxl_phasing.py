import numpy as np
import h5py
from scipy import ndimage
from scipy import sparse

PHI = (np.sqrt(5) + 1) / 2.

class MaxLPhaser():
    '''Reconstruction of object from holographic data using maximum likelihood methods'''
    def __init__(self, data_fname, num_data=-1):
        self._parse_data(data_fname, num_data)
        self._get_qvals()
        self._rot_kwargs = {'reshape': False, 'order': 1, 'prefilter': False}

        self.fsol = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.sol))) * 1.e-3
        # (1e-3 factor is there for rel_scale in make_data)

        self.counts = np.array(self.photons.sum(1))[:,0]
        self.mean_count = self.counts.mean()

    def _parse_data(self, data_fname, num_data):
        with h5py.File(data_fname, 'r') as fptr:
            self.sol = fptr['solution'][:]
            self.angs = fptr['true_angles'][:]
            self.diams = fptr['true_diameters'][:]
            self.shifts = fptr['true_shifts'][:]
            ones, multi = fptr['ones'][:], fptr['multi'][:]
            po, pm, cm = fptr['place_ones'][:], fptr['place_multi'][:], fptr['count_multi'][:] # pylint: disable=invalid-name

        self.size = self.sol.shape[-1]
        if num_data < 0:
            self.num_data = len(ones)
        else:
            self.num_data = num_data

        po_all = np.hstack(po)
        po_csr = sparse.csr_matrix((np.ones_like(po_all), po_all, np.insert(np.cumsum(ones), 0, 0)), shape=(len(po), self.size**2))
        pm_csr = sparse.csr_matrix((np.hstack(cm), np.hstack(pm), np.insert(np.cumsum(multi), 0, 0)), shape=(len(pm), self.size**2))

        self.photons = (po_csr + pm_csr)[:self.num_data]
        self.photons_t = self.photons.transpose().tocsr()

        self.rots = self._get_rot(self.angs).transpose(2, 0, 1)
        self.angs *= 180. / np.pi # convert to degrees

    def _get_qvals(self):
        ind = (np.arange(self.size, dtype='f4') - self.size//2) / self.size
        self.qx, self.qy = np.meshgrid(ind, ind, indexing='ij')
        self.qrad = np.sqrt(self.qx**2 + self.qy**2)
        self.qrad[self.qrad==0] = 1.e-6

        self.mask = (self.qrad > 4/self.size) & (self.qrad < 0.5)
        self.mask_ind = np.where(self.mask.ravel())[0]

    def iterate_pixel(self, fobj, t, rel_tol=1.e-3):
        rescale = self.get_rescale_pixel(fobj, t)
        #print(rescale, self.get_logq_pixel(fobj, rescale, t))

        grad = self.get_grad_pixel(fobj, rescale, t)
        alpha = self.gss(self.get_logq_pixel, fobj, grad, rescale, -1, 1, t, rel_tol=rel_tol)

        retval = fobj.copy()
        retval[self.get_rel_pix(np.arange(self.num_data), t)] += alpha * grad
        #print(grad, alpha)
        return retval

    def gss(self, func, fobj, grad, rescale, a, b, t, rel_tol=1.e-3):
        '''
        Pixel-wise golden-section maximization of func(fobj + alpha * grad) where alpha is between a and b

        Fixed rescale value is used during search
        Only relevant pixels to t are updated
        '''
        c = b - (b - a) / PHI
        d = a + (b - a) / PHI

        x, y = self.get_rel_pix(np.arange(self.num_data), t)
        rel_pix = np.unique(x*self.size + y)
        fobj_t = fobj.ravel()[rel_pix]
        zc = fobj.copy()
        zd = fobj.copy()

        zc.ravel()[rel_pix] = fobj_t + c * grad
        zd.ravel()[rel_pix] = fobj_t + d * grad
        obj_c = func(zc, rescale, t)
        obj_d = func(zd, rescale, t)

        niter = 1
        tol = rel_tol * np.abs(fobj.ravel()[t]) / np.abs(grad)
        while np.abs(b-a) > tol:
            zc.ravel()[rel_pix] = fobj_t + c * grad
            zd.ravel()[rel_pix] = fobj_t + d * grad
            obj_c = func(zc, rescale, t)
            obj_d = func(zd, rescale, t)

            if obj_c > obj_d:
                b = d
            else:
                a = c

            c = b - (b - a) / PHI
            d = a + (b - a) / PHI
            niter += 1

        #print('%d iteration line search: %f (%e < %e)' % (niter, (a+b)/2, np.abs(b-a), tol))
        return (a+b)/2

    def get_fcalc_d(self, fobj, d_vals, t):
        '''Get predicted intensities for frame list d_vals at pixel t'''
        x, y = self.get_rel_pix(d_vals, t)

        sval = np.pi * self.qrad[x, y] * self.diams[d_vals]
        fref = (np.sin(sval) - sval*np.cos(sval)) / sval**3
        ramp = np.exp(1j*2*np.pi*(self.qx[x,y]*self.shifts[d_vals, 0] + self.qy[x,y]*self.shifts[d_vals, 1]))

        return fobj[x, y] + fref*ramp

    def get_rel_pix(self, d_vals, t):
        '''Get relevant pixels to generate fcalc with respect to given pixel t'''
        tx, ty = np.unravel_index(t, (self.size, self.size))
        rotpix = (self.rots[d_vals] @ np.array([self.qx[tx, ty], self.qy[tx, ty]]))*self.size + self.size//2
        x, y = np.rint(rotpix).astype('i4').T
        return x, y

    def get_logq_pixel(self, fobj, rescale, t):
        '''Calculate log-likelihood for given model and rescale at the given pixel'''
        f_d = self.get_fcalc_d(fobj, np.arange(self.num_data), t)
        w_d = rescale * np.abs(f_d)**2
        return (self.photons_t[t].dot(np.log(w_d)) / self.num_data - w_d.mean())[0]

    def get_grad_pixel(self, fobj, rescale, t):
        '''Generate pixel-wise complex gradients for given model and rescale at the given pixel'''
        f_d = self.get_fcalc_d(fobj, np.arange(self.num_data), t)
        return (self.photons_t[t].dot(2*f_d/np.abs(f_d)**2) / self.num_data - 2*rescale*f_d.mean())[0]

    def get_rescale_pixel(self, fobj, t):
        '''Calculate rescale factor for given model at the given pixel'''
        return self.photons_t[t].mean() / (np.abs(self.get_fcalc_d(fobj, np.arange(self.num_data), t))**2).mean()

    def get_rescale_stochastic(self, fobj, npix=100):
        '''Calculate average rescale over npix random pixels'''
        return np.array([self.get_rescale_pixel(fobj, t) for t in np.random.choice(self.mask_ind, npix, replace=False)]).mean()

    def _get_rot(self, ang):
        c = np.cos(ang)
        s = np.sin(ang)
        return np.array([[c, -s], [s, c]])

    def get_fcalc_all(self, fobj, dia, shift):
        '''Get holographic combination of model and given spherical reference for all pixels'''
        return fobj + self._get_sphere(dia) * np.exp(1j*2*np.pi*(self.qx*shift[0] + self.qy*shift[1]))

    def _get_sphere(self, dia):
        '''Get sphere transform for given diameter'''
        sval = np.pi * self.qrad * dia
        return (np.sin(sval) - sval*np.cos(sval)) / sval**3
