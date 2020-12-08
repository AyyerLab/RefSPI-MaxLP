#!/usr/bin/env python

import sys
import os
import time
import argparse
import configparser

import h5py
import numpy as np
import cupy as cp

class DataGenerator():
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')                                                        # Image Size
        self.num_data = config.getint('make_data', 'num_data')                                                 # Number of Diffraction patterns
        self.fluence = config.get('make_data', 'fluence', fallback='constant')                                 # Intensity for incident beam
        self.mean_count = config.getfloat('make_data', 'mean_count')
        self.bg_count = config.getfloat('make_data', 'bg_count', fallback=None)
        self.rel_scale = config.getfloat('make_data', 'rel_scale', fallback=1000.)                             # Relative Sphere Scaling
        self.dia_params = [float(s) for s in config.get('make_data', 'dia_params').split()]                    # Mean and STD in diameter of GOLD sphere
        self.shift_sigma = config.getfloat('make_data', 'shift_sigma')                                         # STD in sphere center : GOLD
        self.shift_sigma2 = config.getfloat('make_data', 'shift_sigma2')                                       # STD in sphere center : Protein 2(Wobbling)

        self.out_file = os.path.join(os.path.dirname(config_file),
                config.get('make_data', 'out_photons_file'))                                                   # SAVE : Diffraction Pattern in holo.h5

        if self.fluence not in ['constant', 'gamma']:                                                          # Fluence : Constant or Gamma
            raise ValueError('make_data:fluence needs to be either constant (default) or gamma')

        with open('kernels.cu', 'r') as f:                                                                     # Kernels.cu
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')                                         # Sphere : F_s(q,d) [Fourier]
        self.k_slice_gen = kernels.get_function('slice_gen')                                                   # In-plane rotations

        self.object1 = cp.zeros((self.size, self.size), dtype='f8')                                            # Protein 1 (Static)
        self.object2 = cp.zeros((self.size, self.size), dtype='f8')                                            # Protein 2 (Wobble)

        self.object_sum1 = 0                                                                                   # Protein 1
        self.object_sum2 = 0                                                                                   # Protein 2

        self.bgmask = cp.zeros((self.size, self.size), dtype = 'f8')
        self.bgmask_sum = 0

    def make_obj(self, bg=False):

        mask1 = self.bgmask if bg else self.object1                                                            # Protein 1
        mask2 = self.bgmask if bg else self.object2                                                            # Protein 2

        # Protein 1 (Static)

        num_circ1 = 80
        mcen1 = self.size // 2
        x1, y1 = cp.indices(self.object1.shape, dtype='f8')
        for _ in range(num_circ1):
            rad1 = (0.7 + 0.3*cp.random.rand(1, dtype='f8')) * self.size / 25.
            while True:
                cen1 = cp.random.rand(2, dtype='f8') * self.size / 5. + mcen1 * 4./ 5.
                dist1 = float(cp.sqrt((cen1[0]-mcen1)**2 + (cen1[1]-mcen1)**2) + rad1)
                if dist1 < mcen1:
                    break

            diskrad1 = cp.sqrt((x1 - cen1[0])**2 + (y1- cen1[1])**2)
            mask1[diskrad1 <= rad1] += 1. - (diskrad1[diskrad1 <= rad1] / rad1)**2

        # Protein 2 (Wobble)

        num_circ2 = 25
        mcen2 = self.size // 2
        x2, y2 = cp.indices(self.object2.shape, dtype='f8')
        for _ in range(num_circ2):
            rad2 = (0.7 + 0.3*cp.random.rand(1, dtype='f8')) * self.size / 25.
            while True:
                cen2 = cp.random.rand(2, dtype='f8') * self.size / 10. + mcen2 * 4./ 7.
                dist2 = float(cp.sqrt((cen2[0]-mcen2)**2 + (cen2[1]-mcen2)**2) + rad2)
                if dist2 < mcen2:
                    break

            diskrad2 = cp.sqrt((x2 - cen2[0])**2 + (y2- cen2[1])**2)
            mask2[diskrad2 <= rad2] += 1. - (diskrad2[diskrad2 <= rad2] / rad2)**2

        # Sum Object


        if bg:
            #mask *= self.bg_count / mask.sum()
            self.bgmask_sum = float(mask1.sum() + mask2.sum())
        else:
            #mask *= self.mean_count / mask.sum()
            self.object_sum1 = float(mask1.sum())                                                                           # Protein 1
            self.object_sum2 = float(mask2.sum())                                                                           # Protein 2

            with h5py.File(self.out_file, 'a') as fptr:

                if 'solution1' in fptr:                                                                                     # Protein 1
                    del fptr['solution1']
                fptr['solution1'] = mask1.get()

                if 'solution2' in fptr:                                                                                     # Protein 2
                    del fptr['solution2']
                fptr['solution2'] = mask2.get()

        return mask1, mask2

    def parse_obj(self, bg=False):
        mask1 = self.bgmask if bg else self.object1                                                                         # Protein 1
        dset_name1 = 'bg' if bg else 'solution1'

        mask2 = self.bgmask if bg else self.object2
        dset_name2 = 'bg' if bg else 'solution2'                                                                            # Protein 2

        with h5py.File(self.out_file, 'r') as fptr:
            mask1 = cp.array(fptr[dset_name1][:])
            mask2 = cp.array(fptr[dset_name2][:])                                                                  

        if bg:
            mask1 *= self.bg_count / mask1.sum()
            mask2 *= self.bg_count / mask2.sum()

            self.bgmask_sum = float(mask1.sum()+ mask2.sum())
            self.bgmask = mask1 + mask2
        else:
            #mask *= self.mean_count / mask.sum()
            self.object_sum1 = float(mask1.sum())
            self.object_sum2 = float(mask2.sum())

            self.object1 = mask1
            self.object2 = mask2

    def make_data(self, parse=False):                                                                                      # Diffraction Patterns
        if self.object_sum1 == 0.:
            if parse:
                self.parse_obj()
            else:
                self.make_obj()

        if self.object_sum2 == 0.:
            if parse:
                self.parse_obj()
            else:
                self.make_obj()

        if self.bg_count is not None:
            if parse:
                self.parse_obj(bg=True)
            else:
                self.make_obj(bg=True)

        # Protein 1 (Static)

        mask1 = cp.ones(self.object1.shape, dtype='f8')
        x1, y1 = cp.indices(self.object1.shape, dtype='f8')
        cen1 = self.size // 2
        pixrad1 = cp.sqrt((x1 - cen1)**2 + (y1 - cen1)**2)
        mask1[pixrad1<4] = 0
        mask1[pixrad1>=cen1] = 0

        # Protein 2 (Wobbling)

        mask2 = cp.ones(self.object2.shape, dtype='f8')
        x2, y2 = cp.indices(self.object2.shape, dtype='f8')
        cen2 = self.size // 2
        pixrad2 = cp.sqrt((x2 - cen2)**2 + (y2 - cen2)**2)
        mask2[pixrad2<4] = 0
        mask2[pixrad2>=cen2] = 0


        fptr = h5py.File(self.out_file, 'a')
        if 'ones' in fptr: del fptr['ones']
        if 'multi' in fptr: del fptr['multi']
        if 'place_ones' in fptr: del fptr['place_ones']
        if 'place_multi' in fptr: del fptr['place_multi']
        if 'count_multi' in fptr: del fptr['count_multi']
        if 'num_pix' in fptr: del fptr['num_pix']

        if 'true_shifts' in fptr: del fptr['true_shifts']                                       # GOLD Sphere
        if 'true_shifts2' in fptr: del fptr['true_shifts2']                                     # Protein 2

        if 'true_diameters' in fptr: del fptr['true_diameters']                                
        if 'true_angles' in fptr: del fptr['true_angles']
        if 'bg' in fptr: del fptr['bg']
        if 'scale' in fptr: del fptr['scale']

        if self.bgmask_sum > 0:
            fptr['bg'] = self.bgmask.get()

        fptr['num_pix'] = np.array([self.size**2])
        dtype = h5py.special_dtype(vlen=np.dtype('i4'))
        place_ones = fptr.create_dataset('place_ones', (self.num_data,), dtype=dtype)
        place_multi = fptr.create_dataset('place_multi', (self.num_data,), dtype=dtype)
        count_multi = fptr.create_dataset('count_multi', (self.num_data,), dtype=dtype)
        ones = fptr.create_dataset('ones', (self.num_data,), dtype='i4')
        multi = fptr.create_dataset('multi', (self.num_data,), dtype='i4')

        # Shifts for Sphere center

        #shifts = np.random.random((self.num_data, 2))*6 - 3
        #shifts = np.random.randn(self.num_data, 2)*1.
        #shifts = np.zeros((self.num_data, 2))
        shifts = np.random.randn(self.num_data, 2)*self.shift_sigma                                    # GOLD
        fptr['true_shifts'] = shifts

        shifts2 = np.random.randn(self.num_data, 2)*self.shift_sigma2                                  # Protein 2
        fptr['true_shifts2'] = shifts2

        # Fluence and scaling 

        if self.fluence == 'gamma':
            scale = np.random.gamma(2., 0.5, self.num_data)
        else:
            scale = np.ones(self.num_data, dtype='f8')
        fptr['scale'] = scale

        # Diameters of sphere : GOLD

        #diameters = np.random.randn(self.num_data)*0.5 + 7.
        #diameters = np.ones(self.num_data)*7.
        diameters = np.random.randn(self.num_data)*self.dia_params[1] + self.dia_params[0]
        fptr['true_diameters'] = diameters


        #rel_scales = diameters**3 * 1000. / 7**3
        #scale *= rel_scales/1.e3

        # Orientation
        angles = np.random.random(self.num_data) * 2. * np.pi
        #angles = np.zeros(self.num_data)
        fptr['true_angles'] = angles

        # Initialization

        view = cp.zeros(self.size**2, dtype='f8')
        rview = cp.zeros_like(view, dtype='f8')
        zmask = cp.zeros_like(view, dtype='f8')

        # Models

        model1 = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.object1)))                  # Protein 1 (Static)
        model2 = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.object2)))                  # Protein 2 (Wobble)

        # Wobble

        mcen2 = self.size // 2.
        x2, y2 = cp.indices(self.object2.shape, dtype = 'f8')
        qx2 = (x2 - mcen2) / (2 * mcen2)
        qy2 = (y2 - mcen2) / (2 * mcen2)


        bsize_model = int(np.ceil(self.size/32.))
        stime = time.time()


        # Diffraction Patterns 

        for i in range(self.num_data):

            # Composite Object

            model = model1 + model2 * cp.exp(2* cp.pi * 1j * ((qx2 * shifts2[i,0] + qy2 * shifts2[i,1])))      
            if i < 10:
                real_object = cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(model)))
                real_object = abs(real_object)
                np.save('data/wp_std0/real_object_%.3d.npy'%i, real_object)
            

            # Add GOLD Reference [Fourier Space]

            self.k_slice_gen_holo((bsize_model,)*2, (32,)*2,
                (model, shifts[i,0], shifts[i,1], diameters[i], self.rel_scale, scale[i], self.size, zmask, 0, view))
            view *= (mask1.ravel() + mask2.ravel()) 
            view *= self.mean_count / view.sum()

            self.k_slice_gen((bsize_model,)*2, (32,)*2,
                (view, angles[i], 1., self.size, self.bgmask, 0, rview)) 


            
            frame = cp.random.poisson(rview, dtype='i4')                                       # Poisson Noise
            
            place_ones[i] = cp.where(frame == 1)[0].get()
            place_multi[i] = cp.where(frame > 1)[0].get()
            count_multi[i] = frame[frame > 1].get()
            ones[i] = place_ones[i].shape[0]
            multi[i] = place_multi[i].shape[0]
            sys.stderr.write('\rWritten %d/%d frames (%d)' % (i+1, self.num_data, int(frame.sum())))


        etime = time.time()
        sys.stderr.write('\nTime taken (make_data): %f s\n' % (etime-stime))
        fptr.close()

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
