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

    def _get_states(self, states_dict, num_rot):
        self.shiftx = states_dict['shift_x']
        self.shifty = states_dict['shift_y']
        self.sphere_dia = states_dict['sphere_dia']
        self.num_states = states_dict['num_states']
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
            self._slice_gen_holo(dmodel, self.shiftx.ravel()[r],
                                 self.shiftx.ravel()[r], self.sphere_dia.ravel()[r],
                                 output=modelview)
            for j in range(self.num_rot):
                self._slice_gen(modelview, j*2*np.pi/self.num_rot, detview, do_log=False)
                sum_detview += detview
                self.msums[r,j] = detview.sum()
        sum_detview /= self.tot_num_states * self.num_rot
        vscale = self.powder.sum() / sum_detview.sum()

        return vscale

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

            self._slice_gen_holo(dmodel, self.shiftx.ravel()[r],
                                 self.shiftx.ravel()[r], self.sphere_dia.ravel()[r],
                                 output=model_views[snum])

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
            self._slice_gen_holo(dmodel, params['shift_x'][d].item(), params['shift_y'][d].item(),
                     params['sphere_dia'][d].item(), modelview))
            self._slice_gen(modelview, params['angles'][d], detview, do_log=False)
            sum_detview += detview
        sum_detview /= self.dset.num_data
        vscale = self.powder.sum() / sum_detview.sum()

        return vscale

    @staticmethod
    def _get_dstates(gstates, params, d, order=1):
        gsx = cp.array(gstates['shift_x'][:,0,0])
        dsx = params['shift_x'][d] + (gsx-gsx.mean()) / gsx.size**order
        gsy = cp.array(gstates['shift_y'][0,:,0])
        dsy = params['shift_y'][d] + (gsy-gsy.mean()) / gsy.size**order
        gdia = cp.array(gstates['sphere_dia'][0,0,:])
        ddia = params['sphere_dia'][d] + (gdia-gdia.mean()) / gdia.size**order
        return cp.meshgrid(dsx, dsy, ddia, indexing='ij')

    def estimate_local(self, model, scales, states_dict, num_rot, params, order=1, dnum_rot=None):
        '''Estimate latent parameters with a local search

        For this search a different set of parameters are examined for each frame
        '''
        stime = time.time()
        dmodel = cp.array(model)
        drescale = self._calculate_rescale_local(dmodel, params)
        dparams = {'shift_x':[], 'shift_y':[],
                   'sphere_dia':[], 'angles':[],
                   'frame_rescale': drescale}

        if dnum_rot is None:
            dnum_rot = 5 if order == 1 else 1
            dnum_rot = min(num_rot, dnum_rot)
        prob = cp.zeros((states_dict['shift_x'].size, dnum_rot))
        for d in range(self.dset.num_data):
            dsx, dsy, ddia = self._get_dstates(states_dict, params, d, order=order)
            if order == 1:
                dang = params['angles'][d] + cp.linspace(-1,1,5) * 2*cp.pi/num_rot
            else:
                dang = cp.array([params['angles'][d]])
            prob[:] = 0

            o_acc = self.dset.ones_accum[d]
            m_acc = self.dset.multi_accum[d]
            self.k_get_prob_frame(prob.shape, (1,1),
                                 (dmodel, self.size,
                                  self.dset.place_ones[o_acc:],
                                  self.dset.place_multi[m_acc:],
                                  self.dset.count_multi[m_acc:],
                                  int(self.dset.ones[d]), int(self.dset.multi[d]),
                                  dsx, dsy, ddia, prob.shape[0],
                                  dang, prob.shape[1],
                                  self.cx, self.cy, self.probmask, self.det.num_pix,
                                  drescale, prob))

            ind, aind = cp.unravel_index(prob.argmax(), prob.shape)
            dparams['shift_x'].append(float(dsx.ravel()[ind]))
            dparams['shift_y'].append(float(dsy.ravel()[ind]))
            dparams['sphere_dia'].append(float(ddia.ravel()[ind]))
            dparams['angles'].append(float(dang[aind]))
            sys.stderr.write('\rFrame %d/%d'%(d+1, self.dset.num_data))
        sys.stderr.write('\n')
        print('Estimated params: %.3f s' % (time.time()-stime))

        for k in ['shift_x', 'shift_y', 'sphere_dia', 'angles']:
            dparams[k] = cp.array(np.array(dparams[k]))
        return dparams

    def _slice_gen_holo(self, dmodel, sx, sy, dia, output=None):
        if output is None:
            output = cp.zeros((self.size, self.size))
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

