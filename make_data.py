#!/usr/bin/env python

import sys
import os
import time
import argparse
import configparser

import h5py
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage

import kernels

class DataGenerator():
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')
        self.num_data = config.getint('make_data', 'num_data')
        self.mean_count = config.getfloat('make_data', 'mean_count')
        self.out_file = os.path.join(os.path.dirname(config_file),
                                     config.get('make_data', 'out_photons_file'))

        self.mask = cp.zeros((self.size, self.size), dtype='f8')
        self.mask_sum = 0

    def make_mask(self):
        num_circ = 10
        mcen = self.size // 2
        x, y = cp.indices(self.mask.shape, dtype='f8')
        for _ in range(num_circ):
            rad = cp.random.rand(1, dtype='f8') * self.size / 5.
            while True:
                cen = cp.random.rand(2, dtype='f8') * self.size
                dist = float(cp.sqrt((cen[0]-mcen)**2 + (cen[1]-mcen)**2) + rad)
                if dist < mcen:
                    break

            pixrad = cp.sqrt((x - cen[0])**2 + (y - cen[1])**2)
            self.mask[pixrad <= rad] += (rad - pixrad[pixrad <= rad])**0.4
        self.mask *= self.mean_count / self.mask.sum()
        self.mask_sum = float(self.mask.sum())

        with h5py.File(self.out_file, 'a') as fptr:
            if 'solution' in fptr:
                del fptr['solution']
            fptr['solution'] = self.mask.get()

    def parse_mask(self):
        with h5py.File(self.out_file, 'r') as fptr:
            self.mask = cp.array(fptr['solution'][:])
        self.mask *= self.mean_count / self.mask.sum()
        self.mask_sum = float(self.mask.sum())

    def make_data(self, parse=False):
        if self.mask_sum == 0.:
            if parse:
                self.parse_mask()
            else:
                self.make_mask()

        with h5py.File(self.out_file, 'a') as fptr:
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
            fptr['num_pix'] = np.array([self.size**2])
            dtype = h5py.special_dtype(vlen=np.dtype('i4'))
            place_ones = fptr.create_dataset('place_ones', (self.num_data,), dtype=dtype)
            place_multi = fptr.create_dataset('place_multi', (self.num_data,), dtype=dtype)
            count_multi = fptr.create_dataset('count_multi', (self.num_data,), dtype=dtype)

            #ang = cp.random.rand(self.num_data, dtype='f8')*360
            ang = np.random.rand(self.num_data).astype('f8')*2.*cp.pi
            fptr['true_angles'] = ang
            
            rot_mask = cp.empty(self.size**2, dtype='f8')
            bsize_model = int(np.ceil(self.size/32.))
            stime = time.time()
            for i in range(self.num_data):
                kernels._slice_gen((bsize_model,)*2, (32,)*2,
                    (self.mask.ravel(), ang[i], self.size, 0, rot_mask))
                frame = cp.random.poisson(rot_mask, dtype='i4').ravel()
                place_ones[i] = cp.where(frame == 1)[0].get()
                place_multi[i] = cp.where(frame > 1)[0].get()
                count_multi[i] = frame[frame > 1].get()
                sys.stderr.write('\rWritten %d/%d frames (%d)' % (i+1, self.num_data, int(frame.sum())))
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
