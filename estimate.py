#!/usr/bin/env python

'''EMC reconstructor object and script'''

import sys
import os.path as op
import time

import numpy as np
import cupy as cp

class Estimator():
    '''Estimate latent parameters given model and data
    '''
    def __init__(self, dataset, detector, size, num_streams=4):
        self.dset = dataset
        self.det = detector
        self.size = size
        self.num_streams = num_streams
        self.mem_size = cp.cuda.Device(cp.cuda.runtime.getDevice()).mem_info[1]

        self._compile_kernels()
        self.cx = cp.array(self.det.cx)
        self.cy = cp.array(self.det.cy)
        self.powder = self.dset.get_powder()
        self.probmask = cp.array(self.det.raw_mask)

        self.bsize_model = int(np.ceil(self.size/32.))
        self.bsize_pixel = int(np.ceil(self.det.num_pix/32.))
        self.bsize_data = int(np.ceil(self.dset.num_data/32.))
        self.stream_list = [cp.cuda.Stream() for _ in range(self.num_streams)]
        self.maxprob = cp.zeros(self.dset.num_data, dtype='f8')
        self.rmax = cp.zeros(self.dset.num_data, dtype='i8')

    def _compile_kernels(self):
        with open(op.join(op.dirname(__file__), 'kernels.cu'), 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen = kernels.get_function('slice_gen')
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        self.k_calc_prob_all = kernels.get_function('calc_prob_all')
        self.k_get_prob_frame = kernels.get_function('get_prob_frame')

    def estimate_global(self, model, scales, states_dict, num_rot):
        '''Estimate latent parameters with a global search

        For this search, the same parameters are examined for all frames
        '''

        self.shiftx = states_dict['shift_x']
        self.shifty = states_dict['shift_y']
        self.sphere_dia = states_dict['sphere_dia']
        self.num_states = states_dict['num_states']
        self.tot_num_states = self.shiftx.size
        self.num_rot = num_rot

        views = cp.empty((self.tot_num_states, self.size**2), dtype='f8')
        #mp = cp.get_default_memory_pool()
        #print('Mem usage: %.2f MB / %.2f MB' % (mp.total_bytes()/1024**2, self.mem_size/1024**2))

        vscale = self._calculate_prob(cp.array(model), scales, views)
        sx_vals, sy_vals, dia_vals, ang_vals = self._unravel_rmax(self.rmax)
        return {'shift_x': sx_vals, 'shift_y': sy_vals,
                'sphere_dia': dia_vals, 'angles': ang_vals,
                'frame_rescale': vscale}

    def _calculate_rescale(self, dmodel, views, return_all=False):
        selmask = cp.zeros((self.size,)*2, dtype='bool')
        selmask[np.rint(self.det.cx+self.size//2).astype('i4'),
                np.rint(self.det.cy+self.size//2).astype('i4')] = True
        selmask = selmask.ravel()

        msums = cp.empty(self.tot_num_states)
        sum_views = cp.zeros((self.num_streams, self.size**2))

        for i, r in enumerate(range(self.tot_num_states)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            self.k_slice_gen_holo((self.bsize_model,)*2, (32,)*2,
                    (dmodel, self.shiftx.ravel()[r], self.shifty.ravel()[r],
                     self.sphere_dia.ravel()[r], 1., 1., self.size, views[i]))
            msums[i] = views[i][selmask].sum()
            sum_views[snum] += views[i]
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()
        vscale = self.powder.sum() / sum_views.sum(0)[selmask].sum() * self.tot_num_states

        if return_all:
            return vscale, msums
        return vscale

    def _calculate_prob(self, dmodel, scales, views):
        vscale, msums = self._calculate_rescale(dmodel, views, return_all=True)
        print('Rescale =', vscale)

        rot_views = cp.zeros((self.num_streams, self.det.num_pix))
        self.maxprob[:] = -cp.finfo('f8').max

        stime = time.time()
        for i, r in enumerate(range(self.tot_num_states)):
            snum = i % self.num_streams
            self.stream_list[snum].use()

            for j in range(self.num_rot):
                self.k_slice_gen((self.bsize_pixel,), (32,),
                        (views[i], self.cx, self.cy, j*2*np.pi/self.num_rot, 1.,
                         self.size, self.det.num_pix, self.dset.bg, 1, rot_views[snum]))
                self.k_calc_prob_all((self.bsize_data,), (32,),
                        (rot_views[snum], self.probmask, self.dset.num_data,
                         self.dset.ones, self.dset.multi,
                         self.dset.ones_accum, self.dset.multi_accum,
                         self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                         -float(msums[i]*vscale), scales,
                         i*self.num_rot + j, self.rmax, self.maxprob))

            sys.stderr.write('\r%d/%d   ' % (i+1, self.tot_num_states))
        [stream.synchronize() for stream in self.stream_list]
        sys.stderr.write('\n')
        cp.cuda.Stream().null.use()

        print('Estimated params: %.3f s' % (time.time()-stime))
        return vscale

    def _unravel_rmax(self, rmax):
        sx, sy, dia, ang = cp.unravel_index(rmax, tuple(self.num_states) + (self.num_rot,))
        sx_vals = cp.unique(self.shiftx)[sx]
        sy_vals = cp.unique(self.shifty)[sy]
        dia_vals = cp.unique(self.sphere_dia)[dia]
        ang_vals = ang * cp.pi / self.num_rot
        return sx_vals, sy_vals, dia_vals, ang_vals
