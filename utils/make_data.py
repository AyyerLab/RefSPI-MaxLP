#!/usr/bin/env python

'''Module for generating objects and corresponding diffraction patterns'''

import sys
import os.path as op
import time
import argparse
import configparser

import h5py
import numpy as np
import cupy as cp
from scipy import ndimage

import writeemc
sys.path.append(op.dirname(op.dirname(__file__)))
from det import Detector

class DataGenerator():
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.num_data = config.getint('make_data', 'num_data')
        self.fluence = config.get('make_data', 'fluence', fallback='constant')
        self.mean_count = config.getfloat('make_data', 'mean_count')

        self.dia_params = [float(s) for s in config.get('make_data', 'dia_params').split()]
        self.shift_sigma = config.getfloat('make_data', 'shift_sigma')

        s = [int(_) for _ in config.get('parameters', 'size').split()]
        use_padding = config.getboolean('make_data', 'use_zero_padding', fallback = False)
        if use_padding:
            self.size = s[1]
        else:
            self.size = s[0]

        self.create_random = config.getboolean('make_data', 'create_random', fallback=False)
        self.bg_count = config.getfloat('make_data', 'bg_count', fallback=None)
        self.rel_scale = config.getfloat('make_data', 'rel_scale')

        self.detector_file = op.join(op.dirname(config_file),
                                     config.get('make_data', 'in_detector_file'))
        self.out_photons_file = op.join(op.dirname(config_file),
                                        config.get('make_data', 'out_photons_file'))

        if self.fluence not in ['constant', 'gamma']:
            raise ValueError('make_data:fluence needs to be either constant (default) or gamma')

        with open(op.join(op.dirname(__file__), 'kernels.cu'), 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        self.k_slice_gen = kernels.get_function('slice_gen')

        self.object = cp.zeros((self.size, self.size), dtype='f8')
        self.object_sum = 0

        self.bgmask = cp.zeros((self.size, self.size), dtype = 'f8')
        self.bgmask_sum = 0

        self.det = Detector(self.detector_file)

    def make_obj(self, bg=False):
        mask = self.bgmask if bg else self.object
        mcen = self.size // 2
        x, y = cp.indices((self.size,self.size), dtype='f8')

        num_circ = 55
        objsize = 15.
        for i in range(num_circ):
            if self.create_random:
                rad = (0.7 + 0.3*cp.random.rand(1, dtype = 'f8')) * objsize / 4.
                while True:
                    cen = cp.random.rand(2, dtype='f8') * objsize - objsize/2 + mcen
                    dist = float(cp.sqrt((cen[0] - mcen)**2 + (cen[1] - mcen)**2) + rad)

                    if dist < mcen:
                        break
                diskrad = cp.sqrt((x - cen[0])**2 + (y- cen[1])**2)
                mask[diskrad <= rad] += 1. - (diskrad[diskrad <= rad] / rad)**2
            else:
                rad = (0.7 + 0.3*(cp.cos(2.5*i) - cp.sin(i/4))) * self.size/ 20.
                while True:
                    cen0 = (3 * cp.sin(2*i) + 0.5 * cp.sin(i/2)) * self.size / 45.  + mcen * 4./ 4.4
                    cen1 = (0.5 * cp.cos(i/2) + 3 * cp.cos(i/2)) * self.size / 45.  + mcen * 4./ 4.4
                    dist = float(cp.sqrt((cen0 - mcen)**2 + (cen1 - mcen)**2) + rad)

                    if dist < mcen:
                        break
                diskrad = cp.sqrt((x - cen0)**2 + (y - cen1)**2)
                mask[diskrad <= rad] += 1. - (diskrad[diskrad <= rad] / rad)**2

        if bg:
            self.bgmask_sum = float(mask.sum())
        else:
            self.object_sum = float(mask.sum())

    def make_data(self, parse=False):
        if self.object_sum == 0.:
            if parse:
                self.parse_obj()
            else:
                self.make_obj()

        if self.bg_count is not None:
            if parse:
                self.parse_obj(bg=True)
            else:
                self.make_obj(bg=True)

        x, y = cp.indices((self.size,self.size), dtype='f8')
        cen = self.size // 2
        mcen = self.size // 2.

        mask = cp.ones(self.object.shape, dtype='f8')
        pixrad = cp.sqrt((x-cen)**2 + (y-cen)**2)
        mask[pixrad<4] = 0
        mask[pixrad>=cen] = 0

        if self.fluence == 'gamma':
            scale = np.random.gamma(2., 0.5, self.num_data)
        else:
            scale = np.ones(self.num_data, dtype='f8')

        shifts = np.random.randn(self.num_data, 2)*self.shift_sigma
        diameters = np.random.randn(self.num_data)*self.dia_params[1] + self.dia_params[0]
        angles = np.random.rand(self.num_data) * 2. * np.pi
        #rel_scales = diameters**3 * 1000. / 7**3
        #scale *= rel_scales/1.e3
        model = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.object)))

        with h5py.File(op.splitext(self.out_photons_file)[0]+'_meta.h5', 'w') as h5f:
            h5f['scale'] = scale
            h5f['true_shifts'] = shifts
            h5f['true_diameters'] = diameters
            h5f['true_angles'] = angles
            h5f['true_model'] = model.get()
            if self.bgmask_sum > 0:
                h5f['bg'] = self.bgmask.get()
        wemc = writeemc.EMCWriter(self.out_photons_file, self.det.num_pix, hdf5=False)

        view = cp.zeros(self.size**2, dtype='f8')
        rview = cp.zeros(self.det.num_pix, dtype='f8')
        zmask = cp.zeros_like(view, dtype='f8')

        bsize_pixel = int(np.ceil(self.det.num_pix/32.))
        bsize_model = int(np.ceil(self.size/32.))
        stime = time.time()

        qx = (x - mcen) 
        qy = (y - mcen)
        cx = cp.array(self.det.cx)
        cy = cp.array(self.det.cy)

        for i in range(self.num_data):
            rview[:] = 0
            self.k_slice_gen_holo((bsize_model,)*2, (32,)*2,
                                  (model, shifts[i,0], shifts[i,1],
                                   diameters[i], self.rel_scale, scale[i],
                                   self.size, view))
            view *= self.mean_count / view.sum()
            self.k_slice_gen((bsize_pixel,), (32,),
                             (view, cx, cy, angles[i], 1., self.size,
                              self.det.num_pix, self.bgmask, 0, rview))

            frame = cp.random.poisson(rview, dtype='i4')
            wemc.write_frame(frame.ravel())
            sys.stderr.write('\rWritten %d/%d frames (%d)' % (i+1, self.num_data, int(frame.sum())))

        etime = time.time()
        sys.stderr.write('\nTime taken (make_data): %f s\n' % (etime-stime))
        wemc.finish_write()

def main():
    parser = argparse.ArgumentParser(description='Padman data generator')
    parser.add_argument('-c', '--config_file',
                        help='Path to config file (Default: make_data_config.ini)',
                        default='make_data_config.ini')
    parser.add_argument('-m', '--mask_only',
                        help='Create mask only and not the data frames',
                        action='store_true', default=False)
    parser.add_argument('-d', '--data_only',
                        help='Generate data only. Use preexisting mask in file',
                        action='store_true', default=False)
    parser.add_argument('-D', '--device',
                        help='Device number (default: 0)', type=int, default=0)
    args = parser.parse_args()

    datagen = DataGenerator(args.config_file)
    if not args.data_only:
        datagen.make_obj()
    if not args.mask_only:
        datagen.make_data(parse=args.data_only)

if __name__ == '__main__':
    main()
