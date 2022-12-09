import sys

import numpy as np
import cupy as cp
from cupyx.scipy import sparse

class MaxLPhaser():
    '''Reconstruction of object from holographic data using maximum likelihood and pattern search'''
    def __init__(self, dataset, detector, size=185, num_pattern=8, num_data=None):
        self.size = size
        self.dset = dataset
        self.det = detector
        self._load_kernels()
        self._gen_pattern(num_pattern)
        self._preproc_data(num_data)
        self._get_qvals()
        self._photons_rotated = False

        self.counts = cp.array(self.photons.sum(1))[:,0]
        self.mean_count = self.counts.mean()

    def _load_kernels(self):
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_get_f_dv = kernels.get_function('get_f_dv')
        self.k_get_logq_pixel = kernels.get_function('get_logq_pixel')
        self.k_rotate_photons = kernels.get_function('rotate_photons')
        self.k_deduplicate = kernels.get_function('deduplicate')

    def _gen_pattern(self, nmax):
        ind = cp.arange(-nmax, nmax+0.5, 0.5)
        x, y = cp.meshgrid(ind, ind, indexing='ij')
        self.pattern = (x + 1j*y).ravel()
        print('Searching with pattern of size', self.pattern.shape)

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

        # photons = K_dt
        self.photons = (po_csr + pm_csr)[:self.num_data]
        # photons_t = K_td
        self.photons_t = self.photons.transpose().tocsr()

    def _get_qvals(self):
        '''Get detector and model coordinates and masks'''
        # Detector pixel coordinates
        self.dx = cp.array(self.det.cx)
        self.dy = cp.array(self.det.cy)
        self.drad = cp.sqrt(self.dx**2 + self.dy**2)
        self.drad[self.drad==0] = 1.e-6
        self.dqvals = cp.ascontiguousarray(cp.array([self.dx, self.dy]).T)

        # Model voxel coordinates (currently 2D)
        ind = cp.arange(self.size, dtype='f8') - self.size // 2
        self.mx, self.my = cp.meshgrid(ind, ind, indexing='ij')
        self.mx = self.mx.ravel()
        self.my = self.my.ravel()
        self.mrad = cp.sqrt(self.mx**2 + self.my**2)

        #TODO: Important!! Generalize these thresholds
        # Model level mask for voxels to run MaxLP on
        self.mask = (self.mrad > self.drad.min()) & (self.mrad < self.size // 2 - 1)
        # Ring mask for first rescale estimate (model level)
        self.rmask = (self.mrad > 0.10*self.size) & (self.mrad < 0.11*self.size)
        # Ring mask for second rescale estimate, closer to low q (model level)
        self.nmask = (self.mrad > 0.10*self.size) & (self.mrad < 0.11*self.size)

    def run_phaser(self, model, sx_vals, sy_vals, dia_vals, ang_vals):
        fconv = cp.array(model).copy().ravel()
        self.shifts = cp.array([sx_vals, sy_vals]).T
        self.diams = cp.array(dia_vals)
        self.ang = cp.array(ang_vals)
        # -angs maps from model space to detector space
        # +angs maps from detector space to model space
        self.rots = self._get_rot(-self.ang).transpose(2, 0, 1)

        num_streams = 4
        streams = [cp.cuda.Stream() for _ in range(num_streams)]

        print('Starting Phaser...')
        for v in range(self.size**2):
            if not self.rmask[v]:
                continue
            fconv[v] = self.run_pixel_pattern(fconv[v], v, num_iter=10)
            sys.stderr.write('\r%d/%d'%(v+1, self.size**2))
        sys.stderr.write('\n')
        print('Calculated for pixel annulus')

        rescales = np.zeros(self.size**2)
        for v in range(self.size**2):
            if not self.rmask[v]:
                continue
            rescales[v] = self.get_rescale_pixel(fconv[v], v)
            sys.stderr.write('\r%d/%d'%(v+1, self.size**2))
        sys.stderr.write('\n')
        rescale = np.mean(rescales[self.rmask.get()])
        print('Estimated rescale:', rescale)

        for v in range(self.size**2):
        #for v in range(10000):
            if not self.mask[v]:
                continue
            fconv[v] = self.run_pixel_pattern(fconv[v], v, rescale=rescale, num_iter=10)
            sys.stderr.write('\r%d/%d'%(v+1, self.size**2))
        sys.stderr.write('\n')
        print('Phasing done..')

        return fconv

    def _get_rot(self, ang):
        c = cp.cos(ang)
        s = cp.sin(ang)
        return cp.array([[c, -s], [s, c]])

    def run_pixel_pattern(self, fobj_v, v, num_iter=10, rescale=None, frac=None):
        '''Optimize model for given model voxel v using pattern search'''
        const_v = self.get_pixel_constants(v, frac=frac)
        step = cp.abs(fobj_v) / 4.

        for i in range(num_iter):
            fobj_v_new, step_new = self.iterate_pixel_pattern(fobj_v, v, step,
                                                              const=const_v,
                                                              rescale=rescale)
            if step_new / cp.abs(fobj_v_new) < 1.e-3:
                break
            fobj_v = fobj_v_new
            step = step_new

        return fobj_v

    def get_pixel_constants(self, v , frac=None):
        d_vals = cp.arange(self.num_data)
        if frac is not None:
            d_vals = cp.random.choice(d_vals, int(cp.round(self.num_data*frac)), replace=False)

        return {'k_d': self.get_photons_pixel(d_vals, v),
                'fref_d': self.get_fref_d(d_vals, v)}

    def get_photons_pixel(self, d_vals, v):
        '''Obtains photons for each frame for a given model voxel v

        CURRENTLY DOES NOT WORK

        Need to do two steps:
        1. Create sparse photons array where indices are in model space
            (the specific index will depend upon the angle)
            with shape (num_voxels, num_data). But an empty value doesn't
            necessarily mean 0 photons.
        2. For each pixel, there is a list of angles for which that pixel
            is not sampled. This can be precalculated from the detector file.
            We need to use that list, and the input angles to get the relevant
            subset_d_vals to be processed for that pixel.

        '''
        if self._photons_rotated:
            return self.photons_t[v, d_vals].toarray()

        # Ignoring rotation...
        distsq_t = (self.mx[v] - self.dx)**2 + (self.my[v] - self.dy)**2
        if distsq_t.min() < 4:
            t_nearest = distsq_t.argmin()
        else:
            t_nearest = 0
        return self.photons_t[t_nearest, d_vals]

        #rotpix = (self.rots[d_vals] @ self.dqvals[:,v])*self.size + self.size//2
        #x, y = cp.rint(rotpix).astype('i4').T
        #v_vals = x * self.size + y
        #return self.photons[d_vals, v_vals]

    def get_fref_d(self, d_vals, v):
        '''Get predicted intensities for frame list d_vals at model voxel v'''
        sval = cp.pi * self.mrad[v] * self.diams[d_vals] / self.size
        fref = (cp.sin(sval) - sval*cp.cos(sval)) / sval**3
        ramp = cp.exp(1j*2*cp.pi*(self.mx[v]*self.shifts[d_vals, 0] +
                                  self.my[v]*self.shifts[d_vals, 1]) / self.size)
        return fref*ramp

    def iterate_pixel_pattern(self, fobj_v, v, step, const, rescale=None):
        vals = self.get_logq_pixel(cp.array(fobj_v+self.pattern*step), v,
                                   rescale=rescale, const=const)
        imax = vals.argmax()
        retf = fobj_v + self.pattern[imax] * step
        if imax == self.pattern.size // 2: # Center of pattern
            return retf, step / (self.pattern.size**0.5 / 5)
        return retf, step

    def get_logq_pixel(self, fobj_v, v, rescale=None, const=None):
        '''Calculate log-likelihood for given model and rescale at the given model pixel'''
        if const is None:
            const = self.get_pixel_constants(v)

        if rescale is None:
            rescale = self.get_rescale_pixel(fobj_v, v, const)

        if not isinstance(fobj_v, cp.ndarray):
            fobj_v = cp.array([fobj_v])

        f_dv = cp.empty((len(const['fref_d']), len(fobj_v)), dtype='c16')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_f_dv((bsize,), (32,),
                        (fobj_v, const['fref_d'], len(const['fref_d']), len(fobj_v), f_dv))
        w_dv = rescale * cp.abs(f_dv)**2

        return (const['k_d'] * cp.log(w_dv)).mean(0) - w_dv.mean(0)

    def get_rescale_pixel(self, fobj_v, v, const=None, frac=None):
        '''Calculate rescale factor for given model at the given pixel'''
        if const is None:
            const = self.get_pixel_constants(v, frac=frac)

        #if not isinstance(fobj_v, cp.ndarray):
        if len(fobj_v.shape) == 0: # Check if element of cp.ndarray
            fobj_v = cp.array([fobj_v])

        f_dv = cp.zeros((len(const['fref_d']), len(fobj_v)), dtype='c16')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_f_dv((bsize,), (32,),
                        (fobj_v, const['fref_d'], len(const['fref_d']), len(fobj_v), f_dv))

        return const['k_d'].mean() / (cp.abs(f_dv)**2).mean(0)
