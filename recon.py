#!/usr/bin/env python

import sys
import os.path as op
import argparse
import configparser
import time

import numpy as np
import cupy as cp
import h5py
from scipy import ndimage

from det import Detector
from dset import Dataset
from estimate import Estimator
from max_lp import MaxLPhaser

class Recon():
    '''Reconstructor object using parameters from config file

    Args:
        config_file (str): Path to configuration file

    The appropriate CUDA device must be selected before initializing the class.
    '''
    def __init__(self, config_file, num_streams=4):
        self.num_streams = num_streams

        config = configparser.ConfigParser()
        config.read(config_file)
        self._parse_params(config, op.dirname(config_file))
        self._parse_detector(config, op.dirname(config_file))
        self._parse_data(config, op.dirname(config_file))
        self._generate_states(config)
        self._init_model()

        self.estimator = Estimator(self.dset, self.det, self.size, num_streams=num_streams)
        self.phaser = MaxLPhaser(self.dset, self.det, self.size, num_pattern=self.num_pattern)
        self.phaser.get_sampled_mask(cp.arange(self.num_rot)*2*np.pi/self.num_rot)

    def _parse_params(self, config, start_dir):
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        self.num_rot = config.getint('emc', 'num_rot')
        self.num_pattern = config.getint('emc', 'num_pattern', fallback=4)
        self.output_folder = op.join(start_dir, config.get('emc', 'output_folder'))
        self.log_file = op.join(start_dir, config.get('emc', 'log_file'))

        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)

    def _parse_detector(self, config, start_dir):
        detector_file = op.join(start_dir, config.get('emc', 'in_detector_file'))
        self.det = Detector(detector_file)
        rad = np.sqrt(self.det.cx**2 + self.det.cy**2)
        self.size = 2 * int(rad.max()) + 3
        print(self.det.num_pix, 'pixels with model size =', self.size)

    def _parse_data(self, config, start_dir):
        stime = time.time()
        photons_file = op.join(start_dir, config.get('emc', 'in_photons_file'))
        self.dset = Dataset(photons_file, self.det.num_pix, self.need_scaling)
        etime = time.time()

        print('%d frames with %.3f photons/frame (%.3f s) (%.2f MB)' % \
                (self.dset.num_data, self.dset.mean_count, etime-stime, self.dset.mem/1024**2))
        sys.stdout.flush()

    def _generate_states(self, config):
        dia = tuple([float(s) for s in config.get('emc', 'sphere_dia').split()])
        sx = tuple((float(s) for s in config.get('emc', 'shiftx').split()))
        sy = tuple((float(s) for s in config.get('emc', 'shifty').split()))
        self.states = {}
        self.states['shift_x'], self.states['shift_y'], self.states['sphere_dia'] = np.meshgrid(
                np.linspace(sx[0], sx[1], int(sx[2])),
                np.linspace(sy[0], sy[1], int(sy[2])),
                np.linspace(dia[0], dia[1], int(dia[2])),
                indexing='ij')

        self.states['num_states'] = np.array([sx[-1], sy[-1], dia[-1]]).astype('i4')
        print(int(self.states['num_states'].prod()), 'sampled states')

    def _init_model(self):
        self.model = np.empty((self.size**2,), dtype='c16')
        self._random_model()

        if self.need_scaling:
            self.scales = self.dset.counts / self.dset.mean_count
        else:
            self.scales = cp.ones(int(self.dset.num_data), dtype='f8')

        with h5py.File(op.join(self.output_folder, 'output_000.h5'), 'w') as f:
            f['model'] = self.model
            f['scale'] = self.scales.get()

    def run_iteration(self, iternum=None, num_phaser_iter=10):
        '''Run one iteration of reconstruction algorithm

        Args:
            iternum (int, optional): If specified, output is tagged with iteration number
        Current guess is assumed to be in self.model, which is updated. If scaling is included,
        the scale factors are in self.scales.
        '''
        params_dict = self.estimator.estimate_global(self.model, self.scales, self.states, self.num_rot)
        self.model = self.phaser.run_phaser(self.model, params_dict, num_iter=num_phaser_iter).get()
        self.save_output(self.model, params_dict, iternum)

    def save_output(self, model, params, iternum, intens=None):
        if iternum is None:
            np.save(op.join(self.output_folder, 'model.npy'), self.model)
            return

        with h5py.File(op.join(self.output_folder, 'output_%.3d.h5'%iternum), 'w') as fptr:
            fptr['model'] = self.model.reshape((self.size,)*2)
            if intens is not None:
                fptr['intens'] = intens.get()

            fptr['angles'] = params['angles'].get()
            fptr['diameters'] = params['sphere_dia'].get()
            fptr['shifts'] = np.array([params['shift_x'].get(), params['shift_y'].get()]).T

    def _random_model(self):
        rmodel = np.zeros((self.size,)*2)
        #self.model = np.random.random(self.size**2) + 1j*np.random.random(self.size**2)
        #self.model *= 1e-3 # To match scale of sphere model
        temp = np.zeros_like(rmodel)
        cen = self.size // 2
        censlice = slice(cen-cen//10, cen+cen//10+1), slice(cen-cen//10, cen+cen//10+1)
        censhape = rmodel[censlice].shape
        for i in range(5):
            temp[censlice] = np.random.random(censhape)
            temp = ndimage.gaussian_filter(temp, i+0.5)
            rmodel += (temp - temp[censlice].min())
        rmodel -= rmodel[:10,:10].mean()
        self.model = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rmodel)))
        #self.model *= 8e-9 * (np.abs(self.model)**2).mean()
        self.model *= 1e-7 * (np.abs(self.model)**2).mean()

def main():
    '''Parses command line arguments and launches EMC reconstruction'''
    parser = argparse.ArgumentParser(description='In-plane rotation EMC')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='emc_config.ini',
                        help='Path to configuration file (default: emc_config.ini)')
    parser.add_argument('-s', '--streams', type=int, default=4,
                        help='Number of streams to use (default=4)')
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='Device index (default=0)')
    args = parser.parse_args()

    print('Running on device', args.device)
    cp.cuda.Device(args.device).use()

    recon = Recon(args.config_file, num_streams=args.streams)
    logf = open(op.join(recon.output_folder, 'EMC.log'), 'w')
    logf.write('Iter  time(s)  change\n')
    logf.flush()
    avgtime = 0.
    numavg = 0

    for i in range(args.num_iter):
        m0 = cp.array(recon.model)
        stime = time.time()
        recon.run_iteration(i+1)
        etime = time.time()
        sys.stderr.write('\r%d/%d (%f s)\n'% (i+1, args.num_iter, etime-stime))
        norm = float(cp.linalg.norm(cp.array(recon.model.ravel()) - m0.ravel()))
        logf.write('%-6d%-.2e %e\n' % (i+1, etime-stime, norm))
        print('Change from last iteration: ', norm)
        logf.flush()
        if i > 0:
            avgtime += etime-stime
            numavg += 1
    if numavg > 0:
        print('\n%.4e s/iteration on average' % (avgtime / numavg))
    logf.close()

if __name__ == '__main__':
    main()
