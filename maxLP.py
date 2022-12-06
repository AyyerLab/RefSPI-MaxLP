import sys
import time

import h5py
import numpy as np
import cupy as cp
from cupyx.scipy import sparse, ndimage
from scipy import special

from calc_frc import FRC
BATCH_SIZE = 1000

class MaxLPhaser():
    '''Reconstruction of object from holographic data using maximum likelihood and pattern search'''
    def __init__(self, dataset, detector, size=185, num_pattern=8, num_data=None):
        self.size = size
        self.dset = dataset
        self.det = detector
        self._load_kernels()
        self._preproc_data(num_data)
        self._get_qvals()
        self._rot_kwargs = {'reshape': False, 'order': 1, 'prefilter': False}
        self._photons_rotated = False

        self.counts = cp.array(self.photons.sum(1))[:,0]
        self.mean_count = self.counts.mean()
        self._gen_pattern(num_pattern)
        self.qvals = cp.ascontiguousarray(cp.array([self.qx, self.qy]).T)
        self.logq_td = cp.zeros((self.det.num_pix, BATCH_SIZE))

        self.mx_ind, self.my_ind = cp.indices((self.size,)*2, dtype='f8')
        self.mx_ind = self.mx_ind.ravel() - self.size//2
        self.my_ind = self.my_ind.ravel() - self.size//2
        self.mrad = cp.sqrt(self.mx_ind**2 + self.my_ind**2)

        self.minvsuppmask = cp.ones((self.size,)*2, dtype=cp.bool_)
        self.minvmask = self._get_invmask(4)
        self.badpixmask = cp.ones(self.size**2, dtype=cp.bool_)
        
    def _load_kernels(self):
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_get_f_dt = kernels.get_function('get_f_dt')
        self.k_get_logq_pixel = kernels.get_function('get_logq_pixel')
        self.k_rotate_photons = kernels.get_function('rotate_photons')
        self.k_deduplicate = kernels.get_function('deduplicate')

    def _gen_pattern(self, nmax):
        ind = cp.arange(-nmax, nmax+0.5, 0.5)
        x, y = cp.meshgrid(ind, ind, indexing='ij')
        self.pattern = (x + 1j*y).ravel()

    def _preproc_data(self, num_data=None):
        if num_data is None:
            self.num_data = self.dset.num_data
        else:
            self.num_data = num_data

        po_csr = sparse.csr_matrix((cp.ones_like(self.dset.place_ones, dtype='f8'),
                                    self.dset.place_ones,
                                    cp.append(self.dset.ones_accum, len(self.dset.place_ones)),
                                   ), shape=(self.dset.num_data, self.det.num_pix))
        pm_csr = sparse.csr_matrix((self.dset.count_multi.astype('f8'),
                                    self.dset.place_multi,
                                    cp.append(self.dset.multi_accum, len(self.dset.place_multi)),
                                   ), shape=(self.dset.num_data, self.det.num_pix))

        self.photons = (po_csr + pm_csr)[:self.num_data]
        self.photons_t = self.photons.transpose().tocsr()

    def _get_qvals(self):
        self.qx = cp.array(self.det.cx) / self.size
        self.qy = cp.array(self.det.cy) / self.size
        self.qrad = cp.sqrt(self.qx**2 + self.qy**2)
        self.qrad[self.qrad==0] = 1.e-6

        #TODO: Important!! Generalize these thresholds
        self.mask = (self.qrad > 4/self.size) & (self.qrad < 0.5)
        #ring mask for first rescale estimate
        self.rmask = (self.qrad > 0.39) & (self.qrad < 0.41)
        #ring mask for second rescale estimate, closer to low q
        self.nmask = (self.qrad > 0.24) & (self.qrad < 0.27)
        #mask containing bad pixel region in low q
        self.bmask = (self.qrad > 4/self.size) & (self.qrad < 0.1)
        self.mask_ind = cp.where(self.mask.ravel())[0]

    def _get_invmask(self, r):
        self.minvmask = cp.ones(self.size**2, dtype=cp.bool_)
        self.minvmask[self.mrad<r] = False
        self.minvmask[self.mrad>=self.size//2]=False
        return self.minvmask

    def get_qcurr(self, fobj, rescale):
        if not self._photons_rotated:
            self._rotate_photons()
        pix = cp.arange(self.size**2)

        num_batches = np.ceil(self.num_data // BATCH_SIZE)
        qcurr = cp.zeros((0,))
        for b in np.arange(num_batches):
            s = b*BATCH_SIZE
            e = min((b+1)*BATCH_SIZE, self.num_data)
            self.k_get_logq_pixel((int(cp.ceil(len(pix)/16.)),), (16,),
                    (fobj, pix, rescale, self.diams, self.shifts, self.qvals,
                     self.photons_t.indptr[s:e], self.photons_t.indices[s:e], self.photons_t.data[s:e],
                     e-s, len(pix), self.logq_td))
            qcurr = cp.append(qcurr, self.logq_td[:e-s].mean(1))
        return qcurr

    def get_pixel_constants(self, t , frac=None):
        if frac is None:
            d_vals = cp.arange(self.num_data)
        else:
             d_vals = cp.random.choice(self.num_data, int(cp.round(self.num_data*frac)), replace=False)
        
        return {'k_d': self.get_photons_pixel(d_vals, t),
                'fref_d': self.get_fref_d(d_vals, t)}

    def iterate_all(self, fobj, step, rescale, nmax=2):
        if not self._photons_rotated:
            self._rotate_photons()
        ind = cp.arange(-nmax, nmax+0.5, 1)
        x, y = cp.meshgrid(ind, ind, indexing='ij')
        pattern = (x+1j*y).ravel()
        fpatt = cp.array([fobj + step*p for p in pattern])
        qpatt = cp.array([self.get_qcurr(f, rescale) for f in fpatt])
        pind = qpatt.argmax(0)
        fobj = fpatt[pind, cp.arange(pind.shape[0])]
        step[pind==0] /= 2.
        return fobj, step

    def run_pixel(self, fobj, t, num_iter=10, frac=None, **kwargs):
        '''Optimize model for given model pixel t using iterate_pixel func()'''
        fobj_t = fobj.ravel()[t]
        const = self.get_pixel_constants(t, frac=frac)

        for i in range(num_iter):
            fobj_t_new = self.iterate_pixel(fobj_t, t, None, const, **kwargs)
            if cp.abs(fobj_t_new - fobj_t) / cp.abs(fobj_t) < 1.e-2:
                break
            fobj_t = fobj_t_new

        return fobj_t

    def iterate_pixel(self, fobj_t, t, rescale=None, const=None, **kwargs):
        '''Optimize for given model pixel t using Golden Section Search on phase and magnitude'''
        if const is None:
            const = self.get_pixel_constants(t)

        fmag_t = cp.abs(fobj_t)
        phases = cp.exp(1j*cp.arange(0, 2*cp.pi, 0.01))
        vals = self.get_logq_pixel(fmag_t * phases, t, rescale, const)
        fobj_t = fmag_t * phases[vals.argmax()]
        scale = self.gss_radial(fobj_t.item(), t, const, **kwargs)
        return fobj_t * scale

    def run_pixel_pattern(self, fobj, t, num_iter=10, rescale=None, frac=None, **kwargs):
        '''Optimize model for given model pixel t using pattern search'''
        fobj_t = fobj.ravel()[t]
        const = self.get_pixel_constants(t, frac=frac)
        step = cp.abs(fobj_t) / 4.

        for i in range(num_iter):
            fobj_t_new, step_new = self.iterate_pixel_pattern(fobj_t, t, step, rescale, const, **kwargs)
            if step_new / cp.abs(fobj_t_new) < 1.e-3:
                break
            fobj_t = fobj_t_new
            step = step_new

        return fobj_t

    def iterate_pixel_pattern(self, fobj_t, t, step, rescale=None, const=None, **kwargs):
        if const is None:
            const = self.get_pixel_constants(t)

        vals = self.get_logq_pixel(cp.array(fobj_t+self.pattern*step), t, rescale=rescale, const=const)
        imax = vals.argmax()
        retf = fobj_t + self.pattern[imax] * step
        if imax == self.pattern.size // 2: # Center of pattern
            return retf, step / (self.pattern.size**0.5 / 5)
        else:
            return retf, step

    def get_logq_pixel(self, fobj_t, t, rescale=None, const=None):
        '''Calculate log-likelihood for given model and rescale at the given model pixel'''
        if const is None:
            const = self.get_pixel_constants(t)

        if rescale is None:
            rescale = self.get_rescale_pixel(fobj_t, t, const)

        if not isinstance(fobj_t, cp.ndarray):
            fobj_t = cp.array([fobj_t])

        f_dt = cp.empty((len(const['fref_d']), len(fobj_t)), dtype='c16')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_f_dt((bsize,), (32,), (fobj_t, const['fref_d'], len(const['fref_d']), len(fobj_t), f_dt))
        w_dt = rescale * cp.abs(f_dt)**2

        return (const['k_d'].T * cp.log(w_dt)).mean(0) - w_dt.mean(0)

    def get_grad_pixel(self, fobj_t, t, rescale=None, const=None):
        '''Generate pixel-wise complex gradients for given model and rescale at the given pixel'''
        if const is None:
            const = self.get_pixel_constants(t)

        if rescale is None:
            rescale = self.get_rescale_pixel(fobj_t, t, const)

        if not isinstance(fobj_t, cp.ndarray):
            fobj_t = cp.array([fobj_t])

        f_dt = cp.empty((len(const['fref_d']), len(fobj_t)), dtype='c16')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_f_dt((bsize,), (32,), (fobj_t, const['fref_d'], len(const['fref_d']), len(fobj_t), f_dt))

        return (const['k_d'].T * (2*f_dt/cp.abs(f_dt)**2)).mean(0) - 2*rescale*f_d.mean(0)

    def get_rescale_pixel(self, fobj_t, t, const=None):
        '''Calculate rescale factor for given model at the given pixel'''
        if const is None:
            const = self.get_pixel_constants(t)

        if not isinstance(fobj_t, cp.ndarray):
            fobj_t = cp.array([fobj_t])

        f_dt = cp.zeros((len(const['fref_d']), len(fobj_t)), dtype='c16')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_f_dt((bsize,), (32,), (fobj_t, const['fref_d'], len(const['fref_d']), len(fobj_t), f_dt))
       
        return const['k_d'].mean() / (cp.abs(f_dt)**2).mean(0)

    def get_rescale_stochastic(self, fobj, cpix=100):
        '''Calculate average rescale over cpix random pixels'''
        rand_pix = np.random.choice(self.mask_ind.get(), cpix, replace=False)
        vals = cp.empty(cpix)
        for i, t in enumerate(rand_pix):
            vals[i] = self.get_rescale_pixel(fobj.ravel()[t], t).item()
        return vals.mean()

    def get_fref_d(self, d_vals, t):
        '''Get predicted intensities for frame list d_vals at model pixel t'''
        sval = cp.pi * self.qrad[t] * self.diams[d_vals]
        fref = (cp.sin(sval) - sval*cp.cos(sval)) / sval**3
        ramp = cp.exp(1j*2*cp.pi*(self.qx[t]*self.shifts[d_vals, 0] + self.qy[t]*self.shifts[d_vals, 1]))
        return fref*ramp

    def get_photons_pixel(self, d_vals, t):
        '''Obtains photons for each frame for a given model pixel t'''
        if self._photons_rotated:
            return self.photons_t[t, d_vals].toarray()

        rotpix = (self.rots[d_vals] @ cp.array([self.qx[t], self.qy[t]]))*self.size + self.size//2
        x, y = cp.rint(rotpix).astype('i4').T
        t_vals = x * self.size + y
        return self.photons[d_vals, t_vals]

    def _get_rot(self, ang):
        c = cp.cos(ang)
        s = cp.sin(ang)
        return cp.array([[c, -s], [s, c]])

    def get_fcalc_all(self, fobj, dia, shift):
        '''Get holographic combination of model and given spherical reference for all pixels'''
        qx2d = self.qx.reshape((self.size,)*2)
        qy2d = self.qy.reshape((self.size,)*2)
        return fobj + self._get_sphere(dia) * cp.exp(1j*2*cp.pi*(qx2d*shift[0] + qy2d*shift[1]))

    def _get_sphere(self, dia):
        '''Get sphere transform for given diameter'''
        sval = cp.pi * self.qrad.reshape((self.size,)*2) * dia
        return (cp.sin(sval) - sval*cp.cos(sval)) / sval**3

    def proj_fourier(self, fconv, x):
        out = x.copy()
        out[self.minvmask.get()] = fconv[self.minvmask.get()]
        return out

    def proj_real(self, x):
        rmodel = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x.reshape(self.size, self.size)))))
        rmodel[rmodel<0] = 0
        rmodel[self.minvsuppmask.get()] = 0
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rmodel))).ravel()

    def er(self, fconv, x):
        out = self.proj_real(self.proj_fourier(fconv, x))
        return out

    def diffmap(self, fconv, x):
        p1 = self.proj_fourier(fconv, x)
        out = x + self.proj_real(2*p1 - x) -p1
        return out

    def _get_support(self, fconv):
        smask = special.expit((self.mrad.get().reshape(self.size, self.size) - 12)*0.5)
        rsupp = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fconv.reshape(self.size, self.size)*smask))))
        grsupp = ndimage.gaussian_filter(cp.abs(cp.asarray(rsupp)), 3) > 1.2e-4
        suppmask = cp.ones((self.size,)*2, dtype = cp.bool_)
        suppmask = suppmask > grsupp
        return suppmask

    def _get_tot_mask(self, fconv):
        self.badpixmask = cp.real(fconv) > -0.2
        tot_mask =  self.badpixmask*self.minvmask
        return tot_mask

    def _run_phaser(self, model, sx_vals, sy_vals, dia_vals, ang_vals):
        fconv = model.copy().ravel()
        self.shifts = cp.array([sx_vals, sy_vals]).T
        self.diams = cp.array(dia_vals)
        self.ang = cp.array(ang_vals)
        # -angs maps from model space to detector space
        # +angs maps from detector space to model space
        self.rots = self._get_rot(-self.ang).transpose(2, 0, 1)
        
        num_streams = 4
        streams = [cp.cuda.Stream() for _ in range(num_streams)]
        
        print('Starting Phaser...')
        for t in range(self.size**2):
            if not self.rmask[t]:
                continue
            fconv[t] = self.run_pixel_pattern(fconv, t, num_iter=10)
        print('Calculated for pixel annulus')


        rescales = np.zeros(self.size**2)
        for t in range(self.size**2):
            if not self.rmask[t]:
                continue
            rescales[t] = self.get_rescale_pixel(fconv[t], t)
        rescale = np.mean(rescales[self.rmask.get()])
        print('Estimated rescale:', rescale)

        for t in range(self.size**2):
            if not self.mask[t]:
                continue
            fconv[t] = self.run_pixel_pattern(fconv, t, rescale=rescale, num_iter=10)
            sys.stderr.write('\r%d/%d'%(t+1, self.size**2))
        sys.stderr.write('\n')
        print('Phasing done..')

        return fconv

    def _improve_model(self, fconv, update_masks=True):
        print('Starting diffmap and er')
        
        fout = fconv.copy()
        if update_masks is True:
            self.minvsuppmask = self._get_support(fconv.reshape(self.size, self.size))
            #self.minvmask = self._get_tot_mask(fconv)
        for i in range(10):
            fout = self.diffmap(fconv, fout)
        for i in range(5):
            fout = self.er(fconv, fout)
        print('Done...')
        return fout

    def _beamstop_calib(self, model):
        radi= [] 
        frc = [] 
        fconv = model.copy()
        '''Beamstop radius in (4,20)'''
        for i in range(17): 
             r = i + 4   
             print('BeamStop Radius', r) 
             self.minvmask = self._get_invmask(r) 
             fout = model.copy()   
             fout = self._improve_model(fconv) 
             recon = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fout.reshape(self.size,self.size)))))  
             rvals, fvals = FRC(sol, recon).calc_rot(num_rot=1800) 
             radi.append(rvals) 
             frc.append(fvals) 
        return radi,frc       
