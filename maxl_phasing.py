import sys
import time

import h5py
import numpy as np
import cupy as cp
from cupyx.scipy import sparse
from cupyx.scipy import ndimage
import itertools

PHI = (np.sqrt(5) + 1) / 2.
INVPHI = 1. / PHI
INVPHI2 = 1. / PHI**2

class MaxLPhaser():
    '''Reconstruction of object from holographic data using maximum likelihood methods'''
    def __init__(self, data_fname, num_data=-1):
        self._load_kernels()
        self._parse_data(data_fname, num_data)
        self._get_qvals()
        self._rot_kwargs = {'reshape': False, 'order': 1, 'prefilter': False}
        self._photons_rotated = False

        self.fsol = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.sol))) * 1.e-3
        # (1e-3 factor is there for rel_scale in make_data)

        self.counts = cp.array(self.photons.sum(1))[:,0]
        self.mean_count = self.counts.mean()
        self._gen_pattern(8)
        self.qvals = cp.ascontiguousarray(cp.array([self.qx, self.qy]).T)
        self.logq_td = cp.zeros((self.size**2, self.num_data))

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
    
    def _bin_params(self, params, linspace):
        bin_params = cp.zeros(len(params))
        
        amin = (linspace[0] + linspace[1])*0.5
        for j in range(len(params)):
           if params[j] <= amin:
              bin_params[j] = linspace[0]

        amax = (linspace[len(linspace)-2] + linspace[len(linspace)-1])*0.5
        for j in range(len(params)):
           if params[j] >= amax:
              bin_params[j] = linspace[len(linspace)-1]

        for i in range(len(linspace)-2):
            m_a = (linspace[i] + linspace[i+1])*0.5
            m_b = (linspace[i+1] + linspace[i+2])*0.5
            for j in range(len(params)):
               if params[j] >= m_a and params[j] <= m_b:
                   bin_params[j] = linspace[i+1]

        return bin_params  
          
    def _wrap_angles(self, angles):
        wrap_angles = cp.zeros(len(angles))
        for i in range(len(angles)):
            while angles[i] > cp.pi:
                angles[i] -= 2*cp.pi
            while angles[i] <-cp.pi:
                angles[i] += 2*cp.pi
            wrap_angles[i] = cp.where(angles[i]<0, np.pi + angles[i], angles[i]) 
        
        return wrap_angles

    def _get_states(self, sx_linspace, sy_linspace, dia_linspace):
        sxsy = list(itertools.product(sx_linspace, sy_linspace))
        states = list(itertools.product(sxsy, dia_linspace))
        sampled_states = np.zeros((len(states), 3), dtype = np.float64)
        for i in range(len(states)):
            sampled_states[i][0] = states[i][0][0]
            sampled_states[i][1] = states[i][0][1]
            sampled_states[i][2] = states[i][1]
       
        return sampled_states

    def _get_prob(self, sampled_states, shifts, diams):
        xshifts = shifts[:,0]
        yshifts = shifts[:,1]
        prob_m = np.zeros((len(sampled_states), len(shifts)))
        for i in range(len(shifts)):
            for j in range(len(sampled_states)):
                if sampled_states[j][0]==xshifts[i] and sampled_states[j][1]==yshifts[i] and sampled_states[j][2]==diams[j]:
                    prob_m[j,i] = 1
        return prob_m

    def _parse_data(self, data_fname, num_data, binning = True):
        with h5py.File(data_fname, 'r') as fptr:
            self.sol = fptr['solution'][:]
            self.angs = cp.array(fptr['true_angles'][:])
            self.diams = cp.array(fptr['true_diameters'][:])
            self.shifts = cp.array(fptr['true_shifts'][:])
            if binning is True:
               shift_linspace = np.linspace(-2.5, 2.5, 6)
               dia_linspace = np.linspace(6, 8, 5)
               ang_linspace = np.linspace(0, np.pi, 90)
               shiftx = self._bin_params(self.shifts[:,0], shift_linspace)
               shifty = self._bin_params(self.shifts[:,1], shift_linspace)
               self.shifts = cp.array([shiftx, shifty]).T
               self.diams = self._bin_params(self.diams, dia_linspace)
               self.angs = self._bin_params(self._wrap_angles(self.angs), ang_linspace)
            ones, multi = fptr['ones'][:], fptr['multi'][:]
            po, pm, cm = fptr['place_ones'][:], fptr['place_multi'][:], fptr['count_multi'][:] # pylint: disable=invalid-name

        self.size = self.sol.shape[-1]
        if num_data < 0:
            self.num_data = len(ones)
        else:
            self.num_data = num_data

        po_all = cp.hstack(po)
        po_csr = sparse.csr_matrix((cp.ones_like(po_all, dtype='f8'),
                                    cp.array(po_all),
                                    cp.concatenate((cp.array([0]), cp.cumsum(cp.array(ones))), axis=0)
                                   ), shape=(len(po), self.size**2))
        pm_csr = sparse.csr_matrix((cp.hstack(cm).astype('f8'),
                                    cp.hstack(pm),
                                    cp.concatenate((cp.array([0]), cp.cumsum(cp.array(multi))), axis=0)
                                   ), shape=(len(pm), self.size**2))

        self.photons = (po_csr + pm_csr)[:self.num_data]
        self.photons_t = self.photons.transpose().tocsr()

    def _rotate_photons(self):
        fshape = (self.size,) * 2

        sframes = []
        for d in range(self.num_data):
            sframes.append(sparse.csr_matrix(cp.rint(ndimage.rotate(self.photons[d].toarray().reshape(fshape),
                           self.angs[d]*180/cp.pi, order=1, reshape=False)).ravel()))
            if (d+1)%10 == 0:
                sys.stderr.write('\rRotating photons...%.3f%%'%((d+1)/self.num_data*100.))
                sys.stderr.flush()

        self.photons = sparse.vstack(sframes, format='csr')
        self.photons_t = self.photons.transpose().tocsr()
        self._photons_rotated = True
        sys.stderr.write('\rRotating photons...done    \n')
        sys.stderr.flush()

    def _get_qvals(self):
        ind = (cp.arange(self.size, dtype='f8') - self.size//2) / self.size
        self.qx, self.qy = cp.meshgrid(ind, ind, indexing='ij')
        self.qx = self.qx.ravel()
        self.qy = self.qy.ravel()
        self.qrad = cp.sqrt(self.qx**2 + self.qy**2)
        self.qrad[self.qrad==0] = 1.e-6

        self.mask = (self.qrad > 4/self.size) & (self.qrad < 0.5)
        #ring mask for first rescale estimate
        self.rmask = (self.qrad > 33/self.size) & (self.qrad < 0.19)
        #ring mask for second rescale estomate, closer to low q
        self.nmask = (self.qrad > 23/self.size) & (self.qrad < 0.14)
        #mask containing bad pixel region in low q
        self.bmask = (self.qrad > 4/self.size) & (self.qrad < 0.1)
        self.mask_ind = cp.where(self.mask.ravel())[0]

    def get_qcurr(self, fobj, rescale):
        if not self._photons_rotated:
            self._rotate_photons()
        pix = cp.arange(self.size**2)
        self.k_get_logq_pixel((int(cp.ceil(len(pix)/16.)),), (16,),
                (fobj, pix, rescale, self.diams, self.shifts, self.qvals,
                 self.photons_t.indptr, self.photons_t.indices, self.photons_t.data,
                 self.num_data, len(pix), self.logq_td))
        return self.logq_td.mean(1)

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

    def gss(self, fobj_t, t, grad, rescale, const,
            b=1, rel_tol=1.e-3, negative=False):
        '''
        Pixel-wise golden-section maximization of
        logq_pixel(fobj + alpha * grad) where alpha is between a and b

        Setting rescale=None means dynamic rescale for each point, else fixed
        '''
        a = 0
        if a > b:
            a, b = b, a

        if const is None:
            const = self.get_pixel_constants(t)

        if negative:
            def neg_func(fobj_t, t, rescale, const):
                return -self.get_logq_pixel(fobj_t, t, rescale, const)
            func = neg_func
        else:
            func = self.get_logq_pixel

        c = b - (b - a) / PHI
        d = a + (b - a) / PHI

        zc = fobj_t + c * grad
        zd = fobj_t + d * grad

        if rescale is None:
            obj_c = func(zc, t, self.get_rescale_pixel(zc, t, const), const)
            obj_d = func(zd, t, self.get_rescale_pixel(zd, t, const), const)
        else:
            obj_c = func(zc, t, rescale, const)
            obj_d = func(zd, t, rescale, const)

        if obj_c < obj_d :
            b = self.gss(fobj_t, t, grad, rescale, const,
                         b=b, rel_tol=rel_tol, negative=True)

            c = b - (b - a) / PHI
            d = a + (b - a) / PHI

        niter = 1
        tol = rel_tol * cp.abs(fobj_t) / cp.abs(grad)
        while cp.abs(b-a) > tol:
            zc = fobj_t + c * grad
            zd = fobj_t + d * grad
            if rescale is None:
                obj_c = func(zc, t, self.get_rescale_pixel(zc, t, const), const)
                obj_d = func(zd, t, self.get_rescale_pixel(zd, t, const), const)
            else:
                obj_c = func(zc, t, rescale, const)
                obj_d = func(zd, t, rescale, const)

            if obj_c > obj_d:
                b = d
            else:
                a = c

            c = b - (b - a) / PHI
            d = a + (b - a) / PHI
            niter += 1

        #print('%d iteration line search: %f (%e < %e)' % (niter, (a+b)/2, cp.abs(b-a), tol))
        return (a+b)/2

    def gss_radial(self, fobj_t, t, const, b=2, tol=1.e-5):
        '''
        Pixel-wise golden-section maximization of
        logq_pixel(alpha * fobj) where alpha is between a and b

        Dynamic rescale only
        '''
        a = 0

        if const is None:
            const = self.get_pixel_constants(t)
        func = self.get_logq_pixel

        h = b - a
        c = a + h * INVPHI2
        d = a + h * INVPHI

        zc = c * fobj_t
        zd = d * fobj_t
        obj_c = func(zc, t, self.get_rescale_pixel(zc, t, const), const)
        obj_d = func(zd, t, self.get_rescale_pixel(zd, t, const), const)

        niter = 1
        while cp.abs(b-a) > tol:
            h *= INVPHI
            if obj_c > obj_d:
                b = d
                d = c
                obj_d = obj_c
                c = a + INVPHI2 * h
                zc = c * fobj_t
                obj_c = func(zc, t, self.get_rescale_pixel(zc, t, const), const)
            else:
                a = c
                c = d
                obj_c = obj_d
                d = a + INVPHI * h
                zd = d * fobj_t
                obj_d = func(zd, t, self.get_rescale_pixel(zd, t, const), const)

            niter += 1

        #print('%d iteration line search: %f (%e < %e)' % (niter, (a+b)/2, cp.abs(b-a), tol))
        if obj_c > obj_d:
            return (a + d) / 2
        else:
            return (b + c)/2

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
        
        # -angs maps from model space to detector space
        # +angs maps from detector space to model space
        self.rots = self._get_rot(-self.angs).transpose(2, 0, 1)
        
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

def main():
    cp.cuda.Device(2).use()
    phaser = MaxLPhaser('data/maxl/holo_dia.h5')
    size = phaser.size

    #rcurr = np.random.random((size, size))*(phaser.sol>1.e-4) * 7
    #fcurr = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rcurr))) * 1.e-3
    fcurr = 2*(np.random.random(size**2) + 1j*np.random.random(size**2)) - 1 
    #fconv = fcurr.copy().ravel()
    fconv = fcurr.copy()
    num_streams = 4
    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    cpix = 0
    stime = time.time()
    for t in range(size**2):
        if not phaser.mask[t]:
            continue
        streams[cpix%num_streams].use()
        fconv[t] = phaser.run_pixel_pattern(fconv, t, num_iter=10)
        #fconv[t] = phaser.run_pixel_pattern(fconv, t, rescale=366.48, num_iter=10)
        cpix += 1
        sys.stderr.write('\r%d/%d: %d %.4f Hz' % (t+1, size**2, cpix, cpix/(time.time()-stime)))
        #if cpix > 10:
        #    break
    sys.stderr.write('\n')
    [s.synchronize() for s in streams]

    fconv = fconv.reshape(fcurr.shape)
    cp.save('data/maxl/maxl_recon.cpy', fconv)
    #np.save('data/maxl_recon_fixed.cpy', fconv)

if __name__ == '__main__':
    main()

