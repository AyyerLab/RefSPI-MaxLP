#!/usr/bin/env python

'Module for generating Objects (Real Space) and corresponding diffraction patterns '

import sys
import os
import time
import argparse
import configparser
import os.path as op
import h5py
import numpy as np
import cupy as cp

from scipy import ndimage


class DataGenerator():
    def __init__(self, config_file):

        config = configparser.ConfigParser()
        config.read(config_file)
        
        #Image size with zero padding
        self.size = config.getint('parameters', 'size')
        #Image region including composite object
        self.sizeC = config.getint('parameters', 'sizeC')

        self.num_data = config.getint('make_data', 'num_data')  
        self.fluence = config.get('make_data', 'fluence', fallback='constant')
        self.mean_count = config.getfloat('make_data', 'mean_count')
        self.bg_count = config.getfloat('make_data', 'bg_count', fallback=None)
        
        #Relative scaling of AuNP intensity
        self.rel_scale = config.getfloat('make_data', 'rel_scale')
        #AuNP sphere diameter range
        self.dia_params = [float(s) for s in config.get('make_data', 'dia_params').split()]
        #STD in sphere centers of AuNP
        self.shift_sigma = config.getfloat('make_data', 'shift_sigma')

        #STD in sphere centers of O2(Wobble)
        self.shift_sigma2 = config.getfloat('make_data', 'shift_sigma2')
        
        #Save Data
        self.out_file = os.path.join(os.path.dirname(config_file), config.get('make_data', 'out_photons_file'))
        self.output_folder =op.join(op.dirname(config_file), config.get('emc', 'output_folder'))

        if self.fluence not in ['constant', 'gamma']:
            raise ValueError('make_data:fluence needs to be either constant (default) or gamma')

        
        #Kernels
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())

        #Holography 
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        #In-plane rotations
        self.k_slice_gen = kernels.get_function('slice_gen')
       
        #O1(Static) & O2(Wobble)                                                                                                       
        self.object1 = cp.zeros((self.size, self.size), dtype='f8')
        self.object2 = cp.zeros((self.size, self.size), dtype='f8')
        
        self.object_sum = 0

        #Background
        self.bgmask = cp.zeros((self.size, self.size), dtype = 'f8')
        self.bgmask_sum = 0

    def make_obj(self, bg=False):

        mask1 = self.bgmask if bg else self.object1                                                            
        mask2 = self.bgmask if bg else self.object2


        mcen = self.size // 2
        x, y = cp.indices((self.size,self.size), dtype='f8')

        #Generating Objects (random or specific)

        # O1
        num_circ1 = 55
        for i in range(num_circ1):
            #Random
            #rad1 = (0.7 + 0.3*cp.random.rand(1, dtype = 'f8')) * self.sizeC/ 25.

            #Specific
            rad1 = (0.7 + 0.3*(cp.cos(2.5*i) - cp.sin(i/4))) * self.sizeC/ 20.

            while True:
                #Random
                #cen = cp.random.rand(2, dtype='f8') * self.sizeC / 5. + mcen * 4./ 5.
                #dist = float(cp.sqrt((cen[0] - mcen)**2 + (cen[1] - mcen)**2) + rad1)

                #Specific
                cen0 = (3 * cp.sin(2*i) + 0.5 * cp.sin(i/2)) * self.sizeC / 45.  + mcen * 4./ 4.4
                cen1 = (0.5 * cp.cos(i/2) + 3 * cp.cos(i/2)) * self.sizeC / 45.  + mcen * 4./ 4.4
                dist = float(cp.sqrt((cen0 - mcen)**2 + (cen1 - mcen)**2) + rad1)

                if dist < mcen:
                    break
            #Random
            #diskrad = cp.sqrt((x - cen[0])**2 + (y- cen[1])**2)

            #Specific
            diskrad = cp.sqrt((x - cen0)**2 + (y - cen1)**2)

            mask1[diskrad <= rad1] += 1. - (diskrad[diskrad <= rad1] / rad1)**2

        # O2
        num_circ2 = 25
        for i in range(num_circ2):
            #Random
            #rad2 = (0.7 + 0.3 * cp.random.rand(1, dtype = 'f8')) * self.sizeC/ 22.

            #Specific
            rad2 = (0.7 + 0.3 *(cp.sin(i) - cp.cos(i/2))) * self.sizeC/ 30.

            while True:
                #Random
                #cen = cp.random.rand(2, dtype='f8') * self.sizeC / 10. + mcen * 4./6.  # Size of cluster + Distance from centre
                #dist = float(cp.sqrt((cen[0] - mcen)**2 + (cen[1] - mcen)**2) + rad2)

                #Specific
                cen0 = (cp.sin(2*i) - 3 * cp.cos(i)) * self.sizeC / 60. + mcen * 5.71 /7.
                cen1 = (cp.cos(i) - cp.sin(i/2)) * self.sizeC / 60. + mcen * 5.71 /7.
                dist = float(cp.sqrt((cen0 - mcen)**2 + (cen1 - mcen)**2) + rad2)

                if dist < mcen:
                    break
            #Random
            #diskrad = cp.sqrt((x - cen[0])**2 + (y- cen[1])**2)

            #Specific
            diskrad = cp.sqrt((x - cen0)**2 + (y - cen1)**2)

            mask2[diskrad <= rad2] += 1. - (diskrad[diskrad <= rad2] / rad2)**2


        if bg:
            #mask *= self.bg_count / mask.sum()
            self.bgmask_sum = float(mask1.sum() + mask2.sum())
        else:
            #mask *= self.mean_count / mask.sum()
            self.object_sum = float(mask1.sum() + mask2.sum())          

            with h5py.File(self.out_file, 'a') as fptr:

                if 'solution1' in fptr: 
                    del fptr['solution1']
                fptr['solution1'] = mask1.get()

                if 'solution2' in fptr: 
                    del fptr['solution2']
                fptr['solution2'] = mask2.get()


    def parse_obj(self, bg=False):
        mask1 = self.bgmask if bg else self.object1         
        dset_name1 = 'bg' if bg else 'solution1'

        mask2 = self.bgmask if bg else self.object2
        dset_name2 = 'bg' if bg else 'solution2'

        with h5py.File(self.out_file, 'r') as fptr:
            mask1 = cp.array(fptr[dset_name1][:])
            mask2 = cp.array(fptr[dset_name2][:])                                                                  

        if bg:
            mask1 *= self.bg_count / mask1.sum()
            mask2 *= self.bg_count / mask2.sum()

            self.bgmask_sum = float(mask1.sum() + mask2.sum())
            self.bgmask = mask1 + mask2
        else:
            #mask *= self.mean_count / mask.sum()

            self.object_sum = float(mask1.sum() + mask2.sum())
            self.object1 = mask1
            self.object2 = mask2

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

        #O1
        mask1 = cp.ones(self.object1.shape, dtype='f8')
        pixrad1 = cp.sqrt((x - cen)**2 + (y - cen)**2)
        mask1[pixrad1<4] = 0
        mask1[pixrad1>=cen] = 0

        #O2
        mask2 = cp.ones(self.object2.shape, dtype='f8')
        pixrad2 = cp.sqrt((x - cen)**2 + (y - cen)**2)
        mask2[pixrad2<4] = 0
        mask2[pixrad2>=cen] = 0


        fptr = h5py.File(self.out_file, 'a')
        if 'ones' in fptr: del fptr['ones']
        if 'multi' in fptr: del fptr['multi']
        if 'place_ones' in fptr: del fptr['place_ones']
        if 'place_multi' in fptr: del fptr['place_multi']
        if 'count_multi' in fptr: del fptr['count_multi']
        if 'num_pix' in fptr: del fptr['num_pix']
        
        #AuNP
        if 'true_shifts' in fptr: del fptr['true_shifts']
        #O2
        if 'true_shifts2' in fptr: del fptr['true_shifts2']

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

        #Shifts for sphere center of AuNP
        shifts = np.random.randn(self.num_data, 2)*self.shift_sigma
        fptr['true_shifts'] = shifts
        
        #Shifts for sphere center for O2
        shifts2 = np.random.randn(self.num_data, 2)*self.shift_sigma2
        fptr['true_shifts2'] = shifts2

        # Fluence and scaling 
        if self.fluence == 'gamma':
            scale = np.random.gamma(2., 0.5, self.num_data)
        else:
            scale = np.ones(self.num_data, dtype='f8')
        fptr['scale'] = scale

        #Diameters of AuNP sphere
        diameters = np.random.randn(self.num_data)*self.dia_params[1] + self.dia_params[0]
        fptr['true_diameters'] = diameters


        #rel_scales = diameters**3 * 1000. / 7**3
        #scale *= rel_scales/1.e3

        # Orientation
        angles = np.random.randn(self.num_data) * 2. * np.pi
        #angles = np.zeros(self.num_data)
        fptr['true_angles'] = angles

        view = cp.zeros(self.size**2, dtype='f8')
        rview = cp.zeros_like(view, dtype='f8')
        zmask = cp.zeros_like(view, dtype='f8')

        #Fourier Models
        #O1
        model1 = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.object1)))
        
        O1 = abs(cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(model1))))
        np.save(op.join(self.output_folder,'O1.npy'), O1)
        O1_intens = abs(model1)
        np.save(op.join(self.output_folder,'O1_intens.npy'), O1_intens)
        
        #O2
        model2 = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.object2)))
        
        O2 = abs(cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(model2))))
        np.save(op.join(self.output_folder,'O2.npy'), O2)
        O2_intens = abs(model2)
        np.save(op.join(self.output_folder,'O2_intens.npy'), O2_intens)

        #Blurred O2 in Composite Object
        bcomposite_object = O1.get() + ndimage.uniform_filter(O2.get(), 6)
        np.save(op.join(self.output_folder, 'bcomposite_object'), bcomposite_object)

        bsize_model = int(np.ceil(self.size/32.))
        stime = time.time()

        qx = (x - mcen) / (2 * mcen)
        qy = (y - mcen) / (2 * mcen) 


        # Diffraction Patterns 
        for i in range(self.num_data):

            shifts2[i,0] = np.where(shifts2[i,0]>0.75, 8, 8)

            #Discrete states A & B
            #if i % 2 == 0:
            #    model =  model1 + model2 * cp.exp(2 * cp.pi * 1j * ((qx * shifts2[i,0] + qy * shifts2[i,0])))
            #else:
            #Homogeneous (A+B)/2
            model =  model1 + (np.fliplr(model2 * cp.exp( 2 * cp.pi * 1j * ((qx * shifts2[i,0] + (-3.7) * qy * shifts2[i,0]))))  
                                      + model2 * cp.exp( 2 * cp.pi * 1j * ((qx * shifts2[i,0] + qy * shifts2[i,0])))) / 2

            if i < 5:
                #Save Composite Object
                composite_object = abs(cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(model))))
                np.save(op.join(self.output_folder,'composite_object_%.3d.npy'%i), composite_object)
                composite_intens = abs(model.reshape(self.size,self.size))              
                np.save(op.join(self.output_folder,'composite_intens_%.3d.npy'%i), composite_intens)
            
            #AuNP as Reference
            self.k_slice_gen_holo((bsize_model,)*2, (32,)*2,
                (model, shifts[i,0], shifts[i,1], diameters[i], self.rel_scale, scale[i], self.size, zmask, 0, view))
            view *= (mask1.ravel() + mask2.ravel())
            view *= self.mean_count / view.sum()

            if i<5:
                #Intensity with Reference attached
                np.save(op.join(self.output_folder,'comp_intens_wRef_%.3d.npy'%i), view.reshape(self.size,self.size))

            self.k_slice_gen((bsize_model,)*2, (32,)*2,
                (view, angles[i], 1., self.size, self.bgmask, 0, rview)) 

            #Poisson Noise
            frame = cp.random.poisson(rview, dtype='i4')
    
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
