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
        self.logq_td = cp.zeros((self.size**2, self.num_data)) # TODO: Batch logq_td
        self.tot_logq_t = cp.empty((len(self.pattern), self.size**2))

    def _load_kernels(self):
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_get_w_dv = kernels.get_function('get_w_dv')
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
                                    cp.append(self.dset.ones_accum, len(self.dset.place_ones)),),
                                   shape=(self.dset.num_data, self.det.num_pix),
                                   copy=False)
        pm_csr = sparse.csr_matrix((self.dset.count_multi.astype('f8'),
                                    self.dset.place_multi,
                                    cp.append(self.dset.multi_accum, len(self.dset.place_multi)),),
                                   shape=(self.dset.num_data, self.det.num_pix),
                                   copy=False)

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
        self.mqvals = cp.ascontiguousarray(cp.array([self.mx, self.my, self.mrad]).T / self.size)

        #TODO: Important!! Generalize these thresholds
        # Model level mask for voxels to run MaxLP on
        #self.mask = (self.mrad > self.drad.min()) & (self.mrad < self.size // 2 - 1)
        self.mask = (self.mrad > self.drad.min()) & (self.mrad < self.size // 2 - 1 - 220)
        # Ring mask for first rescale estimate (model level)
        self.rmask = (self.mrad > 0.10*self.size) & (self.mrad < 0.11*self.size)
        # Ring mask for second rescale estimate, closer to low q (model level)
        self.nmask = (self.mrad > 0.10*self.size) & (self.mrad < 0.11*self.size)

    def get_sampled_mask(self, ang_samples):
        '''Generate bitmask arrays of (N_samples/64, N_voxels)
        which gives whether that voxel is sampled for a given angle
        Each volume contains bitmasks for up to 64 angle samples
        '''
        self.sampled_mask = cp.zeros((int(np.ceil(len(ang_samples) / 64)), self.size**2), dtype='u8')
        self.ang_samples = ang_samples

        cen = self.size // 2
        for i, ang in enumerate(ang_samples):
            slice_ind = i // 64
            bit_shift = i % 64
            rot = self._get_rot(-ang)
            rx = cp.rint(rot[0,0]*self.dx + rot[0,1]*self.dy + cen).astype('i4')
            ry = cp.rint(rot[1,0]*self.dx + rot[1,1]*self.dy + cen).astype('i4')
            self.sampled_mask[slice_ind, rx*self.size + ry] |= 1 << bit_shift
        print('Generated sampled_mask matrix')

    def run_phaser(self, model, sx_vals, sy_vals, dia_vals, ang_vals):
        fconv = cp.array(model).copy().ravel()
        self.shifts = cp.array([sx_vals, sy_vals]).T
        self.diams = cp.array(dia_vals)
        self.ang = cp.array(ang_vals)
        print('Starting Phaser...')

        # -angs maps from model space to detector space
        # +angs maps from detector space to model space
        self.rots = self._get_rot(-self.ang).transpose(2, 0, 1)
        self._rotate_photons()

        '''
        num_streams = 4
        streams = [cp.cuda.Stream() for _ in range(num_streams)]

        # Calculate with dynamic rescale in thin annulus
        for v in range(self.size**2):
            if not self.rmask[v]:
                continue
            fconv[v] = self.run_pixel_pattern(fconv[v], v, num_iter=10)
            sys.stderr.write('\r%d/%d'%(v+1, self.size**2))
        sys.stderr.write('\n')
        print('Calculated for pixel annulus')

        # Calculate rescale with above estimate
        rescales = np.zeros(self.size**2)
        for v in range(self.size**2):
            if not self.rmask[v]:
                continue
            rescales[v] = self.get_rescale_pixel(fconv[v], v)
            sys.stderr.write('\r%d/%d'%(v+1, self.size**2))
        sys.stderr.write('\n')
        rescale = np.mean(rescales[self.rmask.get()])
        print('Estimated rescale:', rescale)

        # Reset fconv after rescale calculation
        fconv = cp.array(model).copy().ravel()
        '''
        rescale = 1.

        # Calculate with fixed rescale over whole volume
        for v in range(self.size**2):
            if not self.mask[v]:
                continue
            fconv[v] = self.run_pixel_pattern(fconv[v], v, rescale=rescale, num_iter=10)
            sys.stderr.write('\r%d/%d'%(v+1, self.size**2))
        sys.stderr.write('\n')
        print('Phasing done..')

        #return fconv
        return 0.5 * (fconv + fconv.conj().reshape(self.size, self.size)[::-1,::-1].ravel())

    def _get_rot(self, ang):
        c = cp.cos(ang)
        s = cp.sin(ang)
        return cp.array([[c, -s], [s, c]])

    def _rotate_photons(self):
        '''Generate model-space rotated sparse photons file'''
        pindices = self.photons.indices
        cx = self.dx[self.photons.indices]
        cy = self.dy[self.photons.indices]
        mindices = cp.empty_like(self.photons.indices)
        cen = self.size // 2
        for d in range(self.num_data):
            s = self.photons.indptr[d]
            e = self.photons.indptr[d+1]
            rx = cp.rint(self.rots[d,0,0]*cx[s:e] + self.rots[d,0,1]*cy[s:e] + cen).astype('i4')
            ry = cp.rint(self.rots[d,1,0]*cx[s:e] + self.rots[d,1,1]*cy[s:e] + cen).astype('i4')
            mindices[s:e] = rx*self.size + ry

        self.mphotons = sparse.csr_matrix((self.photons.data,
                                           mindices,
                                           self.photons.indptr),
                                          shape=(self.num_data, self.size**2),
                                          copy=False)
        self.mphotons_t = self.mphotons.transpose().tocsr()
        self.mphotons_t.sort_indices()
        self.mphotons_t.sum_duplicates()
        print('Rotated photons')

    def run_pixel_pattern(self, fobj_v, v, num_iter=10, rescale=None, frac=None):
        '''Optimize model for given model voxel v using pattern search'''
        const_v = self.get_pixel_constants(v, frac=frac)
        step = cp.abs(fobj_v) / 4.

        curr_fobj_v = cp.empty(self.pattern.shape, dtype=fobj_v.dtype)
        w_dv = cp.empty((len(const_v['fref_d']), len(curr_fobj_v)), dtype='c16')
        bsize = int(cp.ceil(len(const_v['fref_d'])/32.))
        wshape = w_dv.shape

        for i in range(num_iter):
            curr_fobj_v[:] = fobj_v + self.pattern*step

            self.k_get_w_dv((bsize,), (32,),
                            (curr_fobj_v, const_v['fref_d'], wshape[0], wshape[1],
                             rescale, w_dv))
            vals = (const_v['k_d'] * cp.log(w_dv)).mean(0) - w_dv.mean(0)

            imax = vals.argmax()
            fobj_v += self.pattern[imax] * step
            if imax == self.pattern.size // 2: # Center of pattern
                step /= self.pattern.size**0.5

            if step / cp.abs(fobj_v) < 1.e-3:
                break

        return fobj_v

    def get_pixel_constants(self, v , frac=None):
        good_angs = cp.zeros((0,), dtype='i4')
        for i in range(len(self.sampled_mask)):
            slice_good_angs = cp.where(self.sampled_mask[i,v] & (1 << cp.arange(64, dtype='u8')) > 0)[0]
            good_angs = cp.append(good_angs, slice_good_angs)
        good_angs = self.ang_samples[good_angs]
        d_vals = cp.where(cp.isin(self.ang, good_angs))[0]

        if frac is not None:
            d_vals = cp.random.choice(d_vals, int(cp.round(self.num_data*frac)), replace=False)

        out_dict = {}
        out_dict['d_vals'] = d_vals
        out_dict['k_d'] = self.mphotons_t[v, d_vals]
        out_dict['fref_d'] = self.get_fref_d(d_vals, v)

        return out_dict

    def get_fref_d(self, d_vals, v):
        '''Get predicted intensities for frame list d_vals at model voxel v'''
        sval = cp.pi * self.mrad[v] * self.diams[d_vals] / self.size
        fref = (cp.sin(sval) - sval*cp.cos(sval)) / sval**3
        ramp = cp.exp(1j*2*cp.pi*(self.mx[v]*self.shifts[d_vals, 0] +
                                  self.my[v]*self.shifts[d_vals, 1]) / self.size)
        return fref*ramp

    def get_logq_pixel(self, fobj_v, v, rescale=None, const=None):
        '''Calculate log-likelihood for given model and rescale at the given model pixel'''
        if const is None:
            const = self.get_pixel_constants(v)

        if rescale is None:
            rescale = self.get_rescale_pixel(fobj_v, v, const)

        if not isinstance(fobj_v, cp.ndarray):
            fobj_v = cp.array([fobj_v])

        w_dv = cp.empty((len(const['fref_d']), len(fobj_v)), dtype='c16')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_w_dv((bsize,), (32,),
                        (fobj_v, const['fref_d'], len(const['fref_d']),
                         len(fobj_v), rescale, w_dv))

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

    def run_all_pattern(self, fobj, rescale, num_iter=10):
        fmag = cp.abs(fobj)
        step = fmag / 4.
        curr_mask = self.mask.copy()
        num_pattern = len(self.pattern)
        pattern_size = num_pattern**0.5

        for i in range(num_iter):
            for j in range(num_pattern):
                curr_fobj = fobj + self.pattern[j]*step
                self.logq_td[:] = 0
                self.k_get_logq_pixel((self.size**2,), (1,),
                                      (curr_fobj, 1.,  curr_mask,
                                       self.diams, self.shifts, self.mqvals,
                                       self.mphotons_t.indptr, self.mphotons_t.indices,
                                       self.mphotons_t.data,
                                       self.dset.num_data, self.size**2, self.logq_td))
                self.tot_logq_t[j] = self.logq_td.sum(1)
                sys.stderr.write('\r%d/%d: %d/%d    '%(i+1, num_iter, j+1, num_pattern))
            j_best = self.tot_logq_t[j].argmax(0)
            step[j_best == num_pattern // 2] /= pattern_size
            curr_mask[step/fmag < 1e-3] = False
            fobj += self.pattern[j_best] * step
        sys.stderr.write('\n')
        return fobj
