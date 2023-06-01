import sys
import time

import numpy as np
import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy import ndimage as cundimage

class MaxLPhaser():
    '''Reconstruction of object from holographic data using maximum likelihood and pattern search'''
    def __init__(self, dataset, detector, size, num_pattern=8, num_data=None):
        self.size = size
        self.dset = dataset
        self.det = detector
        self._load_kernels()
        self._gen_pattern(num_pattern)
        self._preproc_data(num_data)
        self._get_qvals()
        self.logq_v = cp.zeros((len(self.pattern), self.size, self.size))

    def _load_kernels(self):
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_get_w_dv = kernels.get_function('get_w_dv')
        self.k_get_logq_pattern = kernels.get_function('get_logq_pattern')
        self.k_rotate_photons = kernels.get_function('rotate_photons')
        self.k_deduplicate = kernels.get_function('deduplicate')

    def _gen_pattern(self, nmax):
        ind = cp.arange(-nmax, nmax+0.5, 1)
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
        #self.photons_t = self.photons.transpose().tocsr()

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
        self.mask = (self.mrad > self.drad.min()) & (self.mrad < self.size // 2 - 1)
        #self.mask = (self.mrad > self.drad.min()) & (self.mrad < self.size // 2 - 1 - 150)
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
            rot = self._get_rot(ang)
            rx = cp.rint(rot[0,0]*self.dx + rot[0,1]*self.dy + cen).astype('i4')
            ry = cp.rint(rot[1,0]*self.dx + rot[1,1]*self.dy + cen).astype('i4')
            self.sampled_mask[slice_ind, rx*self.size + ry] |= 1 << bit_shift
        print('Generated sampled_mask matrix')

    def run_phaser(self, model, params, num_iter=10):
        fconv = cp.array(model).copy()
        self.shifts = cp.array([params['shift_x'], params['shift_y']]).T
        self.diams = cp.array(params['sphere_dia'])
        self.ang = cp.array(params['angles'])
        self.ang_ind = cp.searchsorted(self.ang_samples, self.ang).astype('u8')
        rescale = float(params['frame_rescale'])

        stime = time.time()
        print('Rotating by ang_samples')
        self.rots = self._get_rot(self.ang_samples[self.ang_ind]).transpose(2, 0, 1)
        #self.rots = self._get_rot(self.ang).transpose(2, 0, 1)
        self._rotate_photons()

        fconv = self.run_all_pattern(fconv, rescale, num_iter=num_iter)

        print('Updated model: %3f s' % (time.time() - stime))
        return 0.5 * (fconv + fconv.conj()[::-1,::-1])

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

    def run_voxel_pattern(self, fobj_v, v, rescale, num_iter=10, frac=None):
        '''Optimize model for given model voxel v using pattern search'''
        const_v = self.get_voxel_constants(v, frac=frac)
        step = cp.abs(fobj_v) / 4.

        curr_fobj_v = cp.empty(self.pattern.shape, dtype=fobj_v.dtype)
        w_dv = cp.empty((len(const_v['fref_d']), len(curr_fobj_v)), dtype='f8')
        bsize = int(cp.ceil(len(const_v['fref_d'])/32.))
        wshape = w_dv.shape

        for i in range(num_iter):
            curr_fobj_v[:] = fobj_v + self.pattern*step

            self.k_get_w_dv((bsize,), (32,),
                            (curr_fobj_v, const_v['fref_d'],
                             wshape[0], wshape[1],
                             rescale, w_dv))
            vals = (const_v['k_d'] * cp.log(w_dv)).mean(0) - w_dv.mean(0)

            imax = vals.argmax()
            fobj_v += self.pattern[imax] * step
            if imax == self.pattern.size // 2: # Center of pattern
                step /= self.pattern.size**0.5

            if step / cp.abs(fobj_v) < 1.e-3:
                break

        return fobj_v

    def get_voxel_constants(self, v , frac=None):
        d_vals = cp.where(self.sampled_mask[self.ang_ind//64, v] & (1 << (self.ang_ind % 64)) > 0)[0]
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

    def get_logq_voxel(self, fobj_v, v, rescale=None, const=None):
        '''Calculate log-likelihood for given model and rescale at the given model voxel'''
        if const is None:
            const = self.get_voxel_constants(v)

        if rescale is None:
            rescale = self.get_rescale_voxel(fobj_v, v, const)

        if not isinstance(fobj_v, cp.ndarray):
            fobj_v = cp.array([fobj_v])

        w_dv = cp.empty((len(const['fref_d']), len(fobj_v)), dtype='c16')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_w_dv((bsize,), (32,),
                        (fobj_v, const['fref_d'], len(const['fref_d']),
                         len(fobj_v), rescale, w_dv))

        return (const['k_d'] * cp.log(w_dv)).mean(0) - w_dv.mean(0)

    def get_rescale_voxel(self, fobj_v, v, const=None, frac=None):
        '''Calculate rescale factor for given model at the given voxel'''
        if const is None:
            const = self.get_voxel_constants(v, frac=frac)

        #if not isinstance(fobj_v, cp.ndarray):
        if len(fobj_v.shape) == 0: # Check if element of cp.ndarray
            fobj_v = cp.array([fobj_v])

        w_dv = cp.zeros((len(const['fref_d']), len(fobj_v)), dtype='f8')
        bsize = int(cp.ceil(len(const['fref_d'])/32.))
        self.k_get_w_dv((bsize,), (32,),
                        (fobj_v, const['fref_d'], len(const['fref_d']), len(fobj_v), 1., w_dv))

        return const['k_d'].mean() / w_dv.mean(0)

    def run_all_pattern(self, fobj, rescale, num_iter=10, full_output=False):
        fmag = cp.abs(fobj)
        #step = fmag / 4.
        step = cundimage.gaussian_filter(fmag, 10)
        curr_mask = self.mask.reshape(self.size, self.size).copy()
        num_pattern = len(self.pattern)
        pattern_size = num_pattern**0.5
        if full_output:
            fconv_list = [fobj.copy()]
        bsize = int(cp.ceil(self.size**2/32.))

        stime = time.time()
        for i in range(num_iter):
            self.logq_v[:] = 0
            self.k_get_logq_pattern((num_pattern, bsize), (1, 32),
                                    (fobj, rescale, curr_mask,
                                     self.pattern, num_pattern, step,
                                     self.diams, self.shifts, self.mqvals,
                                     self.ang_ind, self.sampled_mask,
                                     self.mphotons_t.indptr, self.mphotons_t.indices,
                                     self.mphotons_t.data,
                                     self.dset.num_data, self.size**2, self.logq_v))
            j_best = self.logq_v.argmax(0)
            j_best[~curr_mask] = num_pattern // 2
            fobj += self.pattern[j_best] * step
            # If center pattern is best, reduce step size. TODO: also apply to non-edges
            step[j_best == num_pattern // 2] /= pattern_size
            if full_output:
                fconv_list.append(fobj.copy())
            #curr_mask[step/fmag < 1e-3] = False
            sys.stderr.write('\rMaxLP iteration %d/%d: ' % (i+1, num_iter))
            sys.stderr.write('%d/%d voxels centered '%((j_best[curr_mask]==num_pattern//2).sum(), curr_mask.sum()))
            sys.stderr.write('(%.2f s/iteration) '%((time.time()-stime) / (i+1)))
        sys.stderr.write('\n')
        if full_output:
            return fobj, fconv_list
        return fobj
