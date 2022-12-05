#!/usr/bin/env python

'''EMC reconstructor object and script'''

import sys
import os.path as op
import argparse
import configparser
import time

import numpy as np
import cupy as cp
import h5py
from scipy import ndimage, special

from det import Detector
from dset import Dataset
from maxLP import MaxLPhaser

P_MIN = 1.e-6
MEM_THRESH = 0.8

class EMC():
    '''Reconstructor object using parameters from config file

    Args:
        config_file (str): Path to configuration file

    The appropriate CUDA device must be selected before initializing the class.
    Can be used with mpirun, in which case work will be divided among ranks.
    '''
    def __init__(self, config_file, num_streams=4):
        self.num_streams = num_streams
        self.mem_size = cp.cuda.Device(cp.cuda.runtime.getDevice()).mem_info[1]

        self._compile_kernels()

        config = configparser.ConfigParser()
        config.read(config_file)
        self._parse_params(config, op.dirname(config_file))
        self._generate_states(config)
        self._generate_masks()

        self.phaser = MaxLPhaser(self.photons_file, size=self.size)

        self._parse_data()
        self._init_model()

        self.prob = cp.array([])
        self.bsize_model = int(np.ceil(self.size/32.))
        self.bsize_data = int(np.ceil(self.dset.num_data/32.))
        self.stream_list = [cp.cuda.Stream() for _ in range(self.num_streams)]

    def _parse_params(self, config, start_dir):
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        self.num_rot = config.getint('emc', 'num_rot')
        self.photons_file = op.join(start_dir, config.get('emc', 'in_photons_file'))
        self.detector_file = op.join(start_dir, config.get('emc', 'in_detector_file'))
        self.output_folder = op.join(start_dir, config.get('emc', 'output_folder'))
        self.log_file = op.join(start_dir, config.get('emc', 'log_file'))

        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)

    def _generate_states(self, config):
        dia = tuple([float(s) for s in config.get('emc', 'sphere_dia').split()])
        sx = tuple((float(s) for s in config.get('emc', 'shiftx').split()))
        sy = tuple((float(s) for s in config.get('emc', 'shifty').split()))
        self.shiftx, self.shifty, self.sphere_dia = np.meshgrid(np.linspace(sx[0], sx[1], int(sx[2])),
                                                                np.linspace(sy[0], sy[1], int(sy[2])),
                                                                np.linspace(dia[0], dia[1], int(dia[2])), indexing='ij')

        self.shiftx = self.shiftx.ravel()
        self.shifty = self.shifty.ravel()
        self.sphere_dia = self.sphere_dia.ravel()
        self.num_states = np.array([sx[-1], sy[-1], dia[-1]]).astype('i4')
        print(self.num_states.prod(), 'sampled states')

        self.detector = Detector(self.detector_file)
        self.x_ind = cp.array(self.detector.cx)
        self.y_ind = cp.array(self.detector.cy)
        self.rad = cp.sqrt(self.x_ind**2 + self.y_ind**2)
        self.num_pix = self.x_ind.size
        self.size = 2 * int(self.rad.max()) + 3

        self.sphere_ramps = [self.ramp(i)*self.sphere(i) for i in range(int(self.num_states.prod()))]
        self.sphere_intens = cp.abs(cp.array(self.sphere_ramps[:int(dia[2])])**2).mean(0)

    def _generate_masks(self):
        self.invmask = cp.zeros(self.size**2, dtype=cp.bool_)
        self.invmask[self.rad<4] = True
        self.invmask[self.rad>=self.size//2] = True
        self.intinvmask = self.invmask.astype('i4')

        self.probmask = cp.zeros(self.size**2, dtype='i4')
        self.probmask[self.rad>=self.size//2] = 2
        self.probmask[self.rad<self.size//8] = 1
        self.probmask[self.rad<4] = 2

    def _compile_kernels(self):
        with open(op.join(op.dirname(__file__), 'kernels.cu'), 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen = kernels.get_function('slice_gen')
        self.k_slice_merge = kernels.get_function('slice_merge')
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        self.k_calc_prob_all = kernels.get_function('calc_prob_all')
        self.k_merge_all = kernels.get_function('merge_all')

    def _parse_data(self):
        stime = time.time()
        self.dset = Dataset(self.photons_file, self.num_pix, self.need_scaling)
        self.powder = self.dset.get_powder()
        etime = time.time()

        if self.rank == 0:
            print('%d frames with %.3f photons/frame (%.3f s) (%.2f MB)' % \
                    (self.dset.num_data, self.dset.mean_count, etime-stime, self.dset.mem/1024**2))
            sys.stdout.flush()

    def _init_model(self):
        self.model = np.empty((self.size**2,), dtype='c16')

        if self.rank == 0:
            self._random_model()
            np.save(op.join(self.output_folder, 'model_000.npy'), self.model)

        if self.need_scaling:
            self.scales = self.dset.counts / self.dset.mean_count
        else:
            self.scales = cp.ones(self.dset.num_data, dtype='f8')

    def run_iteration(self, iternum=None):
        '''Run one iterations of EMC algorithm

        Args:
            iternum (int, optional): If specified, output is tagged with iteration number
        Current guess is assumed to be in self.model, which is updated. If scaling is included,
        the scale factors are in self.scales.
        '''

        self.num_states_p = np.arange(self.rank, self.num_states.prod(), self.num_proc).size
        if self.prob.shape != (self.num_states_p*self.num_rot, self.dset.num_data):
            self.prob = cp.empty((self.num_states_p*self.num_rot, self.dset.num_data), dtype='f8')

        views = cp.empty((self.num_streams, self.num_pix), dtype='f8')
        intens = cp.empty((self.num_states.prod(), self.num_pix), dtype='f8')
        dmodel = cp.array(self.model.ravel())
        #mp = cp.get_default_memory_pool()
        #print('Mem usage: %.2f MB / %.2f MB' % (mp.total_bytes()/1024**2, self.mem_size/1024**2))

        self._calculate_prob(dmodel, views)
        self._get_rmax()
        sx_vals, sy_vals, dia_vals, ang_vals = self._unravel_rmax(self.rmax)
        self.model = self.phaser._run_phaser(self.model, sx_vals, sy_vals, dia_vals, ang_vals)
        self.save_output(self.model, iternum)

    def _calculate_prob(self, dmodel, views, drange=None):
        if drange is None:
            s, e = (0, self.dset.num_data)
        else:
            s, e = tuple(drange)
        num_data_b = e - s
        self.bsize_data = int(np.ceil(num_data_b/32.))
        selmask = (self.probmask < 1)
        sum_views = cp.zeros_like(views)
        msums = cp.empty(self.num_states_p)
        rot_views = cp.empty_like(views)

        for i, r in enumerate(range(self.rank, self.num_states.prod(), self.num_proc)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            self.k_slice_gen_holo((self.bsize_model,)*2, (32,)*2,
                    (dmodel, self.shiftx[r], self.shifty[r], self.sphere_dia[r], 1.,
                     1., self.size, self.dset.bg, 0, views[snum]))
            msums[i] = views[snum][selmask].sum()
            sum_views[snum] += views[snum]
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()
        vscale = self.powder[selmask].sum() / sum_views.sum(0)[selmask].sum() * self.num_states_p

        for i, r in enumerate(range(self.rank, self.num_states.prod(), self.num_proc)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            self.k_slice_gen_holo((self.bsize_model,)*2, (32,)*2,
                    (dmodel, self.shiftx[r], self.shifty[r], self.sphere_dia[r], 1.,
                     1., self.size, self.dset.bg, 0, views[snum]))

            for j in range(self.num_rot):
                self.k_slice_gen((self.bsize_model,)*2, (32,)*2,
                        (views[snum], j*np.pi/self.num_rot, 1.,
                         self.size, self.dset.bg, 1, rot_views[snum]))
                self.k_calc_prob_all((self.bsize_data,), (32,),
                        (rot_views[snum], self.probmask, num_data_b,
                         self.dset.ones[s:e], self.dset.multi[s:e],
                         self.dset.ones_accum[s:e], self.dset.multi_accum[s:e],
                         self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                         -float(msums[i]*vscale), self.scales[s:e], self.prob[i*self.num_rot+j]))
            if self.rank == 0:
                sys.stderr.write('\r%d/%d' % (i, self.num_states.prod()))
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()
        if self.rank == 0:
            sys.stderr.write('\n')

    def _get_rmax(self):
        max_exp_p = self.prob.max(0).get()
        rmax_p = self.prob.argmax(axis=0).get().astype('i4')
        rmax_p = ((rmax_p//self.num_rot)*self.num_proc + self.rank)*self.num_rot + rmax_p%self.num_rot
        max_exp = np.empty_like(max_exp_p)
        self.rmax = np.empty_like(rmax_p)

        rmax_p[max_exp_p != max_exp] = -1

    def _unravel_rmax(self, rmax):
        sx, sy, dia, ang = cp.unravel_index(rmax, tuple(self.num_states) + (self.num_rot,))
        sx_vals = cp.unique(self.shiftx)[sx]
        sy_vals = cp.unique(self.shifty)[sy]
        dia_vals = cp.unique(self.sphere_dia)[dia]
        ang_vals = ang*cp.pi / self.num_rot
        return sx_vals, sy_vals, dia_vals, ang_vals

    def save_output(self, model, iternum, intens=None):
        if iternum is None:
            np.save(op.join(self.output_folder, 'model.npy'), self.model)
            return

        with h5py.File(op.join(self.output_folder, 'output_%.3d.h5'%iternum), 'w') as fptr:
            fptr['model'] = self.model.reshape((self.size,)*2)
            if intens is not None:
                fptr['intens'] = intens.get()

            sx_vals, sy_vals, dia_vals, ang_vals = self._unravel_rmax(self.rmax)
            fptr['angles'] = ang_vals
            fptr['diameters'] = dia_vals.get()
            fptr['shifts'] = np.array([sx_vals.get(), sy_vals.get()]).T

    def _random_model(self):
        self.model = np.random.random(self.size**2) + 1j*np.random.random(self.size**2)

    def ramp(self, n):
        return cp.exp(1j*2.*cp.pi*(self.x_ind*self.shiftx[n] + self.y_ind*self.shifty[n])/self.size)

    def sphere(self, n, diameter=None):
        if n is None:
            dia = diameter
        else:
            dia = self.sphere_dia[n]
        s = cp.pi * self.rad * dia / self.size
        s[s==0] = 1.e-5
        return ((cp.sin(s) - s*cp.cos(s)) / s**3).ravel()

