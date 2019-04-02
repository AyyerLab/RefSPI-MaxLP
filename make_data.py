#!/usr/bin/env python

import sys
import os
import time
import argparse
import configparser

import h5py
import numpy as np
import cupy as cp

import kernels

class DataGenerator():
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')
        self.num_data = config.getint('make_data', 'num_data')
        self.fluence = config.get('make_data', 'fluence', fallback='constant')
        self.mean_count = config.getfloat('make_data', 'mean_count')
        self.bg_count = config.getfloat('make_data', 'bg_count', fallback=None)
        self.out_file = os.path.join(os.path.dirname(config_file),
                                     config.get('make_data', 'out_photons_file'))

        if self.fluence not in ['constant', 'gamma']:
            raise ValueError('make_data:fluence needs to be either constant (default) or gamma')
        self.mask = cp.zeros((self.size, self.size), dtype='f8')
        self.mask_sum = 0
        self.bgmask = cp.zeros_like(self.mask)
        self.bgmask_sum = 0

    def make_mask(self, cmin=0.05, bg=False):
        mask = self.bgmask if bg else self.mask

        num_circ = 20
        mcen = self.size // 2
        x, y = cp.indices(mask.shape, dtype='f8')
        pixrad = cp.sqrt((x-mcen)**2 + (y-mcen)**2)
        mask[pixrad < mcen-1] = 1.
        for _ in range(num_circ):
            rad = cp.random.rand(1, dtype='f8') * self.size / 5.
            while True:
                cen = cp.random.rand(2, dtype='f8') * self.size
                dist = float(cp.sqrt((cen[0]-mcen)**2 + (cen[1]-mcen)**2) + rad)
                if dist < mcen:
                    break

            pixrad = cp.sqrt((x - cen[0])**2 + (y - cen[1])**2)
            mask[pixrad <= rad] *= cmin + (1. - cmin) * (pixrad[pixrad <= rad] / rad)**2

        if bg:
            mask *= self.bg_count / mask.sum()
            self.bgmask_sum = float(mask.sum())
        else:
            mask *= self.mean_count / mask.sum()
            self.mask_sum = float(mask.sum())

            with h5py.File(self.out_file, 'a') as fptr:
                if 'solution' in fptr:
                    del fptr['solution']
                fptr['solution'] = mask.get()

    def parse_mask(self, bg=False):
        mask = self.bgmask if bg else self.mask
        dset_name = 'bg' if bg else 'solution'

        with h5py.File(self.out_file, 'r') as fptr:
            mask = cp.array(fptr[dset_name][:])

        if bg:
            mask *= self.bg_count / mask.sum()
            self.bgmask_sum = float(mask.sum())
        else:
            mask *= self.mean_count / mask.sum()
            self.mask_sum = float(mask.sum())

    def make_data(self, parse=False):
        if self.mask_sum == 0.:
            if parse:
                self.parse_mask()
            else:
                self.make_mask()

        if self.bg_count is not None:
            if parse:
                self.parse_mask(bg=True)
            else:
                self.make_mask(bg=True)

        with h5py.File(self.out_file, 'a') as fptr:
            if 'ones' in fptr:
                del fptr['ones']
            if 'multi' in fptr:
                del fptr['multi']
            if 'place_ones' in fptr:
                del fptr['place_ones']
            if 'place_multi' in fptr:
                del fptr['place_multi']
            if 'count_multi' in fptr:
                del fptr['count_multi']
            if 'num_pix' in fptr:
                del fptr['num_pix']
            if 'true_angles' in fptr:
                del fptr['true_angles']
            if 'bg' in fptr:
                del fptr['bg']
            if self.bgmask_sum > 0:
                fptr['bg'] = self.bgmask.get()
            fptr['num_pix'] = np.array([self.size**2])
            dtype = h5py.special_dtype(vlen=np.dtype('i4'))
            place_ones = fptr.create_dataset('place_ones', (self.num_data,), dtype=dtype)
            place_multi = fptr.create_dataset('place_multi', (self.num_data,), dtype=dtype)
            count_multi = fptr.create_dataset('count_multi', (self.num_data,), dtype=dtype)
            ones = fptr.create_dataset('ones', (self.num_data,), dtype='i4')
            multi = fptr.create_dataset('multi', (self.num_data,), dtype='i4')

            ang = np.random.rand(self.num_data).astype('f8')*2.*cp.pi
            fptr['true_angles'] = ang
            if self.fluence == 'gamma':
                if 'scale' in fptr:
                    del fptr['scale']
                scale = np.random.gamma(2., 0.5, self.num_data)
            else:
                scale = np.ones(self.num_data, dtype='f8')
            
            rot_mask = cp.empty(self.size**2, dtype='f8')
            bsize_model = int(np.ceil(self.size/32.))
            stime = time.time()
            for i in range(self.num_data):
                kernels.slice_gen((bsize_model,)*2, (32,)*2,
                    (self.mask, ang[i], scale[i], self.size, self.bgmask, 0, rot_mask))
                frame = cp.random.poisson(rot_mask, dtype='i4').ravel()
                place_ones[i] = cp.where(frame == 1)[0].get()
                place_multi[i] = cp.where(frame > 1)[0].get()
                count_multi[i] = frame[frame > 1].get()
                ones[i] = place_ones[i].shape[0]
                multi[i] = place_multi[i].shape[0]
                sys.stderr.write('\rWritten %d/%d frames (%d)  ' % (i+1, self.num_data, int(frame.sum())))
            etime = time.time()
            sys.stderr.write('\nTime taken (make_data): %f s\n' % (etime-stime))

def main():
    parser = argparse.ArgumentParser(description='Padman data generator')
    parser.add_argument('-c', '--config_file',
                        help='Path to config file (Default: config.ini)',
                        default='config.ini')
    parser.add_argument('-m', '--mask_only',
                        help='Create mask only and not the data frames',
                        action='store_true', default=False)
    parser.add_argument('-d', '--data_only',
                        help='Generate data only. Use preexisting mask in file',
                        action='store_true', default=False)
    args = parser.parse_args()

    datagen = DataGenerator(args.config_file)
    if not args.data_only:
        datagen.make_mask()
    if not args.mask_only:
        datagen.make_data(parse=args.data_only)

if __name__ == '__main__':
    main()
