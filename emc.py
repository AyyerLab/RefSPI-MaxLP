#!/usr/bin/env python

import sys
import os
import argparse
import configparser
import time

import numpy as np
import h5py
import cupy as cp
from cupyx.scipy import ndimage
import cupyx
#cp.cuda.Device(2).use()
P_MIN = 1.e-6

import kernels

class Dataset():
    def __init__(self, photons_file, num_pix):
        self.photons_file = photons_file
        self.num_pix = num_pix

        with h5py.File(self.photons_file, 'r') as fptr:
            if self.num_pix != fptr['num_pix'][0]:
                raise AttributeError('Number of pixels in photons file does not match')
            self.num_data = fptr['place_ones'].shape[0]
            self.ones = cp.array([len(fptr['place_ones'][i]) for i in range(self.num_data)]).astype('i4')
            self.ones_accum = cp.roll(self.ones.cumsum(), 1)
            self.ones_accum[0] = 0
            self.place_ones = cp.array(np.hstack(fptr['place_ones'][:]))

            self.multi = cp.array([len(fptr['place_multi'][i]) for i in range(self.num_data)]).astype('i4')
            self.multi_accum = cp.roll(self.multi.cumsum(), 1)
            self.multi_accum[0] = 0
            self.place_multi = cp.array(np.hstack(fptr['place_multi'][:]))
            self.count_multi = cp.array(np.hstack(fptr['count_multi'][:]))

            self.mean_count = (self.place_ones.shape[0] + self.count_multi.sum()) / self.num_data
        print('%d frames with %.3f photons/frame' % (self.num_data, self.mean_count))

class EMC():
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')
        self.num_rot = config.getint('emc', 'num_rot')
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        self.photons_file = os.path.join(os.path.dirname(config_file),
                                         config.get('emc', 'in_photons_file'))
        self.output_folder = os.path.join(os.path.dirname(config_file),
                                          config.get('emc', 'output_folder', fallback='data/'))
        self.log_file = os.path.join(os.path.dirname(config_file),
                                     config.get('emc', 'log_file', fallback='EMC.log'))

        self.dset = Dataset(self.photons_file, self.size**2)
        self.model = cp.random.random((self.size, self.size), dtype='f8') * self.dset.mean_count / self.dset.num_pix
        self.mweights = cp.zeros((self.size, self.size), dtype='f8')

        self._calc_prob_ones = cp.ReductionKernel(
            in_params='raw float64 lview, int32 po',
            out_params='float64 prob',
            map_expr='lview[po]',
            reduce_expr='a + b',
            post_map_expr='prob = a',
            identity='0',
            name='kernel_calc_prob_ones')
        self._calc_prob_multi = cp.ReductionKernel(
            in_params='raw float64 lview, int32 pm, int32 cm',
            out_params='float64 prob',
            map_expr='lview[pm]*cm',
            reduce_expr='a + b',
            post_map_expr='prob = a',
            identity='0',
            name='kernel_calc_prob_multi')

    def run_iteration(self, iternum=None, itype='rd'):
        if itype == 'rd':
            self._run_iteration_rd(iternum)
        elif itype == 'dr':
            self._run_iteration_dr(iternum)
        else:
            raise ValueError('Unrecognized iteration type: %s' % itype)

    def _run_iteration_rd(self, iternum=None):
        self.prob = cp.empty((self.num_rot, self.dset.num_data), dtype='f8')
        view = cp.empty(self.size**2, dtype='f8')
        msum = -self.model.sum()

        bsize_model = int(np.ceil(self.size/16.))
        bsize_data = int(np.ceil(self.dset.num_data/16.))
        for r in range(self.num_rot):
            kernels._slice_gen((bsize_model,)*2, (16,)*2,
                (self.model, r/self.num_rot*2.*np.pi,
                 self.size, view))
            kernels._calc_prob_all((bsize_data,), (16,),
                (view, self.dset.num_data,
                 self.dset.ones, self.dset.multi,
                 self.dset.ones_accum, self.dset.multi_accum,
                 self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                 msum, self.prob[r]))

        max_exp = self.prob.max(0)
        rmax = self.prob.argmax(axis=0)
        self.prob = cp.exp(self.prob - max_exp)
        self.prob /= self.prob.sum(0)
        self.prob[self.prob < P_MIN] = 0
        p_norm = self.prob.sum(1)
        h_p_norm = p_norm.get()

        self.model[:] = 0
        self.mweights[:] = 0
        uniform = cp.ones_like(self.mweights)
        for r in range(self.num_rot):
            if h_p_norm[r] == 0.:
                continue
            view[:] = 0
            kernels._merge_all((bsize_data,), (16,),
                (self.prob[r], self.dset.num_data,
                 self.dset.ones, self.dset.multi,
                 self.dset.ones_accum, self.dset.multi_accum,
                 self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                 view))
            kernels._slice_merge((bsize_model,)*2, (16,)*2,
                (view/p_norm[r], r/self.num_rot*2.*np.pi,
                 self.size, self.model, self.mweights))
            #self.model += ndimage.rotate(view.reshape((self.size,)*2)/p_norm[r], r/self.num_rot*360, reshape=False, order=1)
            #self.mweights += ndimage.rotate(uniform, r/self.num_rot*360, reshape=False, order=1)
        self.model[self.mweights > 0] /= self.mweights[self.mweights > 0]

        if iternum is None:
            np.save('data/model.npy', self.model.get())
        else:
            np.save('data/model_%.3d.npy'%iternum, self.model.get())
            np.save('data/rmax_%.3d.npy'%iternum, rmax.get()/self.num_rot*360.)

    def _run_iteration_dr(self, iternum=None):
        self.model[self.model < 1.e-20] = 1.e-20
        views = cp.empty((self.num_rot, self.size, self.size), dtype='f8')
        mviews = cp.zeros((self.num_rot, self.size, self.size), dtype='f8')

        stime = time.time()
        for r in range(self.num_rot):
            views[r] = ndimage.rotate(self.model, -360.*r/self.num_rot, reshape=False, order=1, cval=1.e-20)
            sys.stderr.write('\rExp  \t%d/%d' % (r+1, self.num_rot))
        etime = time.time()
        sys.stderr.write(' (%f s)\n' % (etime-stime))

        views = views.reshape(self.num_rot, -1)
        vsums = views.sum(axis=1)
        views = cp.log(views)
        prob = cp.empty(self.num_rot, dtype='f8')
        rmax = cp.empty(self.dset.num_data, dtype='i4')
        p_norm = cp.zeros(self.num_rot, dtype='f8')
        self.model[:] = 0
        self.mweights[:] = 0
        
        stime = time.time()
        for d in range(self.dset.num_data):
            max_exp = -1e20
            tmin = self.dset.ones_accum[d]
            tmax = tmin + self.dset.ones[d]
            p_o = self.dset.place_ones[tmin:tmax]

            tmin = self.dset.multi_accum[d]
            tmax = tmin + self.dset.multi[d]
            p_m = self.dset.place_multi[tmin:tmax]
            c_m = self.dset.count_multi[tmin:tmax]

            prob[:] = -vsums
            for r in range(self.num_rot):
                prob[r] += self._calc_prob_ones(views[r], p_o)
                prob[r] += self._calc_prob_multi(views[r], p_m, c_m)

                if prob[r] > max_exp:
                    max_exp = prob[r]
                    rmax[d] = r

            prob = cp.exp(prob - max_exp)
            prob /= prob.sum()
            prob[prob < P_MIN] = 0.
            p_norm += prob

            for r in range(self.num_rot):
                if prob[r] > 0:
                    cupyx.scatter_add(mviews[r].ravel(), p_o, prob[r])
                    cupyx.scatter_add(mviews[r].ravel(), p_m, prob[r]*c_m)

            sys.stderr.write('\rMax  \t%d/%d'%(d+1, self.dset.num_data))
        etime = time.time()
        sys.stderr.write(' (%f s)\n' % (etime-stime))

        stime = time.time()
        uniform = cp.ones_like(mviews[r])
        for r in range(self.num_rot):
            if p_norm[r] == 0.:
                continue
            self.model += ndimage.rotate(mviews[r]/p_norm[r], 360.*r/self.num_rot, reshape=False, order=1)
            self.mweights += ndimage.rotate(uniform, 360.*r/self.num_rot, reshape=False, order=1)
            sys.stderr.write('\rComp \t%d/%d' % (r+1, self.num_rot))
        etime = time.time()
        sys.stderr.write(' (%f s)\n' % (etime-stime))

        self.model[self.mweights > 0] /= self.mweights[self.mweights > 0]
        if iternum is None:
            np.save('data/model.npy', self.model.get())
        else:
            np.save('data/model_%.3d.npy'%iternum, self.model.get())
            np.save('data/rmax_%.3d.npy'%iternum, rmax.get()/self.num_rot*360.)

def main():
    parser = argparse.ArgumentParser(description='In-plane rotation EMC')
    parser.add_argument('num_iter', help='Number of iterations', type=int)
    parser.add_argument('-c', '--config_file',
                        help='Path to configuration file (default: config.ini)',
                        default='config.ini')
    parser.add_argument('-t', '--itype',
                        help="Iteration type, 'dr' or 'rd'. Default = 'rd'",
                        default='rd')
    args = parser.parse_args()

    recon = EMC(args.config_file)
    print('\nIter  time     change')
    for i in range(args.num_iter):
        m0 = cp.copy(recon.model)
        stime = time.time()
        recon.run_iteration(i+1, itype=args.itype)
        etime = time.time()
        norm = float(cp.linalg.norm(recon.model - m0))
        print('%-6d%-.2e %e' % (i+1, etime-stime, norm))

if __name__ == '__main__':
    main()
