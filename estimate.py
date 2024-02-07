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
        with open(op.join(op.dirname(__file__), 'ekernels.cu'), 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen = kernels.get_function('slice_gen')
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        self.k_calc_prob_all = kernels.get_function('calc_prob_all')
        self.k_get_prob_frame = kernels.get_function('get_prob_frame')
        self.k_calc_local_prob_all = kernels.get_function('calc_local_prob_all')

    def _get_states(self, states_dict, num_rot):
        self.shiftx = states_dict['shift_x']
        self.shifty = states_dict['shift_y']
        self.sphere_dia = states_dict['sphere_dia']
        self.num_states = states_dict['num_states']
        self.state_weights = states_dict['weight']
        self.tot_num_states = self.shiftx.size

        self.num_rot = num_rot

        self.msums = cp.empty((self.tot_num_states, self.num_rot))

    def estimate_global(self, model, scales, states_dict, num_rot):
        '''Estimate latent parameters with a global search

        For this search, the same parameters are examined for all frames
        '''

        self._get_states(states_dict, num_rot)

        #mp = cp.get_default_memory_pool()
        #print('Mem usage: %.2f MB / %.2f MB' % (mp.total_bytes()/1024**2, self.mem_size/1024**2))

        vscale = self._calculate_prob(cp.array(model), scales)
        sx_vals, sy_vals, dia_vals, ang_vals = self._unravel_rmax(self.rmax)
        return {'shift_x': sx_vals, 'shift_y': sy_vals,
                'sphere_dia': dia_vals, 'angles': ang_vals,
                'model_sum': self.msums.ravel()[self.rmax],
                'frame_rescale': vscale}

    def _calculate_rescale(self, dmodel):
        modelview = cp.zeros((self.size, self.size))
        detview = cp.zeros(self.det.num_pix)
        sum_detview = cp.zeros_like(detview)

        for r in range(self.tot_num_states):
            self._slice_gen_holo(dmodel, r, output=modelview)
            for j in range(self.num_rot):
                self._slice_gen(modelview, j*2*np.pi/self.num_rot, detview, do_log=False)
                sum_detview += detview * self.state_weights.ravel()[r]
                self.msums[r,j] = detview.sum()
        sum_detview /= self.num_rot # weights array sums to 1
        vscale = self.powder.sum() / sum_detview.sum()

        return float(vscale)

    def _calculate_prob(self, dmodel, scales):
        vscale = self._calculate_rescale(dmodel)
        print('Rescale =', vscale)

        model_views = cp.zeros((self.num_streams, self.size**2))
        det_views = cp.zeros((self.num_streams, self.det.num_pix))
        self.maxprob[:] = -cp.finfo('f8').max

        stime = time.time()
        for r in range(self.tot_num_states):
            snum = r % self.num_streams
            self.stream_list[snum].use()

            self._slice_gen_holo(dmodel, r, output=model_views[snum])

            for j in range(self.num_rot):
                self._slice_gen(model_views[snum], j*2*np.pi/self.num_rot, det_views[snum], do_log=True)
                self._calc_prob_all(det_views[snum], -float(self.msums[r,j]*vscale), scales, r*self.num_rot + j)

            sys.stderr.write('\r%d/%d   ' % (r+1, self.tot_num_states))
        [stream.synchronize() for stream in self.stream_list]
        cp.cuda.Stream().null.use()
        sys.stderr.write('\n')

        print('Estimated params: %.3f s' % (time.time()-stime))
        return vscale

    def _unravel_rmax(self, rmax):
        sx, sy, dia, ang = cp.unravel_index(rmax, tuple(self.num_states) + (self.num_rot,))
        sx_vals = cp.unique(self.shiftx)[sx]
        sy_vals = cp.unique(self.shifty)[sy]
        dia_vals = cp.unique(self.sphere_dia)[dia]
        ang_vals = ang * 2 * cp.pi / self.num_rot
        return sx_vals, sy_vals, dia_vals, ang_vals

    def _calculate_rescale_local(self, dmodel, params):
        '''Maximum likelihood rescale given model and parameters'''
        modelview = cp.zeros((self.size, self.size))
        detview = cp.zeros(self.det.num_pix)
        sum_detview = cp.zeros_like(detview)

        for d in range(self.dset.num_data):
            self._slice_gen_holo(dmodel, None, params['shift_x'][d].item(), params['shift_y'][d].item(),
                     params['sphere_dia'][d].item(), modelview)
            self._slice_gen(modelview, params['angles'][d], detview, do_log=False)
            sum_detview += detview
        sum_detview /= self.dset.num_data
        vscale = self.powder.sum() / sum_detview.sum()

        return float(vscale)

    def estimate_local(self, model, scales, states_dict, num_rot, params, order=1, dnum_rot=None):
        '''Estimate latent parameters with a local search

        For this search a different set of parameters are examined for each frame
        '''
        stime = time.time()
        dmodel = cp.array(model)
        drescale = self._calculate_rescale_local(dmodel, params)
        print('Using rescale =', drescale)
        dparams = {'frame_rescale': drescale}

        if dnum_rot is None:
            dnum_rot = 5 if order == 1 else 1
            dnum_rot = min(num_rot, dnum_rot)
        rsteps = cp.zeros(3)
        rsteps[0] = 0 if states_dict['num_states'][0] == 1 else np.diff(states_dict['shift_x'][:2,0,0])[0] / states_dict['shift_x'].shape[0]**order
        rsteps[1] = 0 if states_dict['num_states'][1] == 1 else np.diff(states_dict['shift_y'][0,:2,0])[0] / states_dict['shift_y'].shape[1]**order
        rsteps[2] = 0 if states_dict['num_states'][2] == 1 else np.diff(states_dict['sphere_dia'][0,0,:2])[0] / states_dict['sphere_dia'].shape[2]**order
        ang_step = 2*cp.pi/num_rot/dnum_rot
        self.num_states = states_dict['num_states']
        self.maxprob[:] = -cp.finfo('f8').max

        self._calc_local_prob_all(dmodel, params, drescale, rsteps, cp.array(self.num_states), ang_step, dnum_rot)

        sx_ind, sy_ind, dia_ind, ang_ind = cp.unravel_index(self.rmax, tuple(self.num_states) + (dnum_rot,))
        dparams['shift_x'] = cp.array(params['shift_x']) + rsteps[0]*(sx_ind-self.num_states[0]//2)
        dparams['shift_y'] = cp.array(params['shift_y']) + rsteps[1]*(sy_ind-self.num_states[1]//2)
        dparams['sphere_dia'] = cp.array(params['sphere_dia']) + rsteps[2]*(dia_ind-self.num_states[2]//2)
        dparams['angles'] = cp.array(params['angles']) + ang_step*(ang_ind-dnum_rot//2)
        print('Estimated params: %.3f s' % (time.time()-stime))

        return dparams

    def _slice_gen_holo(self, dmodel, state_ind, sx=None, sy=None, dia=None, output=None):
        if output is None:
            output = cp.zeros((self.size, self.size))
        if state_ind is not None:
            sx = self.shiftx.ravel()[state_ind]
            sy = self.shifty.ravel()[state_ind]
            dia = self.sphere_dia.ravel()[state_ind]

        self.k_slice_gen_holo((self.bsize_model,)*2, (32,)*2,
                (dmodel, sx, sy, dia, 1., 1., self.size, output))

    def _slice_gen(self, modelview, angle, output=None, do_log=False):
        if output is None:
            output = cp.zeros(self.det.num_pix)
        logval = int(do_log)
        self.k_slice_gen((self.bsize_pixel,), (32,),
                (modelview, self.cx, self.cy, angle, 1.,
                 self.size, self.det.num_pix, self.dset.bg, do_log, output))

    def _calc_prob_all(self, view, init, scales, index):
        self.k_calc_prob_all((self.bsize_data,), (32,),
                (view, self.probmask, self.dset.num_data,
                 self.dset.ones, self.dset.multi,
                 self.dset.ones_accum, self.dset.multi_accum,
                 self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                 init, scales,
                 index, self.rmax, self.maxprob))

    def _calc_local_prob_all(self, dmodel, params, rescale, rsteps, num_rsamples, ang_step, num_angs):
        rvals = cp.array([params['shift_x'], params['shift_y'], params['sphere_dia']]).transpose().copy()
        angs = cp.array(params['angles'])

        #self.k_calc_local_prob_all((self.bsize_data,), (32,),
        self.k_calc_local_prob_all((self.dset.num_data,), (1,),
                (dmodel, self.size, rescale,
                 self.dset.num_data, self.dset.ones, self.dset.multi,
                 self.dset.ones_accum, self.dset.multi_accum,
                 self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                 self.cx, self.cy, self.probmask, self.det.num_pix,
                 rvals, rsteps, num_rsamples,
                 angs, ang_step, num_angs,
                 self.rmax, self.maxprob))
