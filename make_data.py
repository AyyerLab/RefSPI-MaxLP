#!/usr/bin/env python

'Module for generating objects (homo- and hetro-geneous) and corresponding diffraction patterns '

import sys
import os
import time
import argparse
import configparser

import h5py
import numpy as np
import cupy as cp

from scipy import ndimage
import os.path as op

class DataGenerator():
    def __init__(self, config_file):

        config = configparser.ConfigParser()
        config.read(config_file)
        
        s = [int(_) for _ in config.get('parameters', 'size').split()]
        self.num_data = config.getint('make_data', 'num_data')  
        self.fluence = config.get('make_data', 'fluence', fallback='constant')
        self.mean_count = config.getfloat('make_data', 'mean_count')
        self.dia_params = [float(s) for s in config.get('make_data', 'dia_params').split()]
        self.shift_sigma = config.getfloat('make_data', 'shift_sigma')
        self.wobble_sigma = config.getfloat('make_data', 'wobble_sigma')

        use_padding = config.getboolean('make_data', 'use_zero_padding', fallback = False)
        if use_padding:
            self.size = s[1]
        else:
            self.size = s[0]

        self.object_hetro = config.getboolean('make_data', 'object_hetro', fallback=False)
            
        self.gen_mix = False
        self.gen_blur = False
        self.gen_homo = False
        self.gen_iso = False
        object_type = config.get('make_data', 'object_type', fallback='mix')
        if object_type not in ['iso','homo', 'mix', 'blur']:
            raise ValueError('Data type needs to be from (homo, mix,or blur')
        elif object_type == 'homo':
            self.gen_homo = True
        elif object_type == 'mix':
            self.gen_mix = True
        elif object_type == 'blur':
            self.gen_blur = True
        elif object_type == 'iso':
            self.gen_iso = True

        self.create_random = config.getboolean('make_data', 'create_random', fallback= False)

        self.bg_count = config.getfloat('make_data', 'bg_count', fallback=None)
        self.rel_scale = config.getfloat('make_data', 'rel_scale')
        self.output_folder =op.join(op.dirname(config_file), config.get('make_data', 'output_folder'))
        self.out_photons_file = os.path.join(os.path.dirname(config_file), config.get('make_data', 'out_photons_file'))

        if self.fluence not in ['constant', 'gamma']:
            raise ValueError('make_data:fluence needs to be either constant (default) or gamma')

        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        self.k_slice_gen = kernels.get_function('slice_gen')   
        
        self.object = cp.zeros((self.size, self.size), dtype='f8')
        self.object_sum = 0
        
        if self.object_hetro:    
            self.wobble_object = cp.zeros((self.size, self.size), dtype='f8')
            self.wobble_object_sum = 0
        
        self.bgmask = cp.zeros((self.size, self.size), dtype = 'f8')
        self.bgmask_sum = 0

    def make_obj(self, bg=False):

        mask = self.bgmask if bg else self.object                                                      
        mcen = self.size // 2
        x, y = cp.indices((self.size,self.size), dtype='f8')

        num_circ = 55
        for i in range(num_circ):
            if self.create_random:
                rad = (0.7 + 0.3*cp.random.rand(1, dtype = 'f8')) * self.size/ 25.
                while True:
                    cen = cp.random.rand(2, dtype='f8') * self.size / 5. + mcen * 4./ 5.
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

        if self.object_hetro:
            wobble_mask = self.bgmask if bg else self.wobble_object
            num_circ = 25
            for i in range(num_circ):
                if self.create_random:
                    wobble_rad = (0.7 + 0.3 * cp.random.rand(1, dtype = 'f8')) * self.size/ 22.
                    while True:
                        cen = cp.random.rand(2, dtype='f8') * self.size / 10. + mcen * 4./6.  # Size of cluster + Distance from centre
                        dist = float(cp.sqrt((cen[0] - mcen)**2 + (cen[1] - mcen)**2) + wobble_rad)
                        if dist < mcen:
                            break
                    wobble_diskrad = cp.sqrt((x - cen[0])**2 + (y- cen[1])**2)
                    wobble_mask[wobble_diskrad <= wobble_rad] += 1. - (wobble_diskrad[wobble_diskrad <= wobble_rad] / wobble_rad)**2

                else:
                    wobble_rad = (0.7 + 0.3 *(cp.sin(i) - cp.cos(i/2))) * self.size/ 30.
                    while True:
                        cen0 = (cp.sin(2*i) - 3 * cp.cos(i)) * self.size / 60. + mcen * 4.6 /7.
                        cen1 = (cp.cos(i) - cp.sin(i/2)) * self.size / 60. + mcen * 4.6 /7.
                        dist = float(cp.sqrt((cen0 - mcen)**2 + (cen1 - mcen)**2) + wobble_rad)

                        if dist < mcen:
                            break
                    wobble_diskrad = cp.sqrt((x - cen0)**2 + (y - cen1)**2)
                    wobble_mask[wobble_diskrad <= wobble_rad] += 1. - (wobble_diskrad[wobble_diskrad <= wobble_rad] / wobble_rad)**2


        if bg:
            #mask *= self.bg_count / mask.sum()
            if self.object_hetro:
                self.bgmask_sum = float(mask.sum() + wobble_mask.sum())
            else:
                self.bgmask_sum = float(mask.sum())
        else:
            #mask *= self.mean_count / mask.sum()
            if self.object_hetro: 
                self.object_sum = float(mask.sum() + wobble_mask.sum())
            else:
                self.object_sum = float(mask.sum())          

            if self.object_hetro:
                with h5py.File(self.out_photons_file, 'a') as fptr:
                    if 'rigid' in fptr: 
                        del fptr['rigid']
                    fptr['rigid'] = mask.get()

                with h5py.File(self.out_photons_file, 'a') as fptr:
                    if 'wobble' in fptr: 
                        del fptr['wobble']
                    fptr['wobble'] =  wobble_mask.get()
            else:
                with h5py.File(self.out_photons_file, 'a') as fptr:
                    if 'solution' in fptr: 
                        del fptr['solution']
                    fptr['solution'] = mask.get()

    def parse_obj(self, bg=False):
        mask = self.bgmask if bg else self.object         
        dset_name = 'bg' if bg else 'solution'
        
        if self.object_hetro:
            wobble_mask = self.bgmask if bg else self.wobble_object
            wobble_dset_name = 'bg' if bg else 'wobble'

        with h5py.File(self.out_photons_file, 'r') as fptr:
            mask = cp.array(fptr[dset_name][:])
            if self.object_hetro:
                wobble_mask = cp.array(fptr[wobble_dset_name][:])                                                                  

        if bg:
            mask *= self.bg_count / mask.sum()
            self.bgmask_sum = float(mask.sum())
            self.bgmask = mask
            if self.object_hetro:
                wobble_mask *= self.bg_count / wobble_mask.sum()
                self.bgmask_sum = float(mask.sum() + wobble_mask.sum())
                self.bgmask = mask + wobble_mask
        else:
            #mask *= self.mean_count / mask.sum()
            self.object_sum = float(mask.sum())
            self.object = mask

            if self.object_hetro:
                self.object_sum = float(mask.sum() + wobble_mask.sum())
                self.wobble_object = wobble_mask

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
        pixrad = cp.sqrt((x - cen)**2 + (y - cen)**2)
        mask[pixrad<4] = 0
        mask[pixrad>=cen] = 0

        if self.object_hetro:
            wobble_mask = cp.ones(self.wobble_object.shape, dtype='f8')
            wobble_pixrad = cp.sqrt((x - cen)**2 + (y - cen)**2)
            wobble_mask[wobble_pixrad<4] = 0
            wobble_mask[wobble_pixrad>=cen] = 0


        fptr = h5py.File(self.out_photons_file, 'a')
        if 'ones' in fptr: del fptr['ones']
        if 'multi' in fptr: del fptr['multi']
        if 'place_ones' in fptr: del fptr['place_ones']
        if 'place_multi' in fptr: del fptr['place_multi']
        if 'count_multi' in fptr: del fptr['count_multi']
        if 'num_pix' in fptr: del fptr['num_pix']
        if 'true_shifts' in fptr: del fptr['true_shifts']
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

        shifts = np.random.randn(self.num_data, 2)*self.shift_sigma
        fptr['true_shifts'] = shifts
        
        if self.object_hetro:
            if 'true_wobbles' in fptr: del fptr['true_wobbles']
            wobbles = np.random.randn(self.num_data, 2)*self.wobble_sigma
            fptr['true_wobbles'] = wobbles

        if self.fluence == 'gamma':
            scale = np.random.gamma(2., 0.5, self.num_data)
        else:
            scale = np.ones(self.num_data, dtype='f8')
        fptr['scale'] = scale

        diameters = np.random.randn(self.num_data)*self.dia_params[1] + self.dia_params[0]
        fptr['true_diameters'] = diameters


        #rel_scales = diameters**3 * 1000. / 7**3
        #scale *= rel_scales/1.e3

        angles = np.random.randn(self.num_data) * 2. * np.pi
        fptr['true_angles'] = angles

        view = cp.zeros(self.size**2, dtype='f8')
        rview = cp.zeros_like(view, dtype='f8')
        zmask = cp.zeros_like(view, dtype='f8')

        
        model = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.object)))   
        if self.object_hetro:
            wobble_model = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.wobble_object)))
 
        #Blurred O2 in Composite Object
        #bcomposite_object = O1.get() + ndimage.uniform_filter(O2.get(), 6)
        #np.save(op.join(self.output_folder, 'bcomposite_object'), bcomposite_object)

        bsize_model = int(np.ceil(self.size/32.))
        stime = time.time()

        qx = (x - mcen) / (2 * mcen)
        qy = (y - mcen) / (2 * mcen) 
 
        for i in range(self.num_data):

            if self.gen_iso:
                fmodel = model

            if self.gen_mix:
                wobbles[i,0] = np.where(wobbles[i,0]>0.4, 4.5, 4.5)
                if i % 2 == 0:
                    fmodel =  model + wobble_model * cp.exp(2 * cp.pi * 1j * ((qx * wobbles[i,0] + qy * wobbles[i,0])))
                else:
                    fmodel =  model + np.fliplr(wobble_model * cp.exp( 2 * cp.pi * 1j * ((qx * wobbles[i,0] + (-3.7) * qy * wobbles[i,0]))))
            
            if self.gen_homo:
                wobbles[i,0] = np.where(wobbles[i,0]>0.4, 4.5, 4.5)
                fmodel =  model + (np.fliplr(wobble_model * cp.exp( 2 * cp.pi * 1j * ((qx * wobbles[i,0] + (-3.7) * qy * wobbles[i,0]))))  
                                         + wobble_model * cp.exp( 2 * cp.pi * 1j * ((qx * wobbles[i,0] + qy * wobbles[i,0])))) / 2
            
            if self.gen_blur:
                fmodel =  model + wobble_model * cp.exp( 2 * cp.pi * 1j * ((qx * wobbles[i,0] + (-3.7) * qy * wobbles[i,0])))
           
            if i <=5 :
                np.save('data/md/model_%.3d.npy'%i, fmodel) 
            self.k_slice_gen_holo((bsize_model,)*2, (32,)*2,
                (fmodel, shifts[i,0], shifts[i,1], diameters[i], self.rel_scale, scale[i], self.size, zmask, 0, view))

            if self.object_hetro:
                view *= (mask.ravel() + wobble_mask.ravel())
            else:
                view *= mask.ravel()
            view *= self.mean_count / view.sum()

            self.k_slice_gen((bsize_model,)*2, (32,)*2,
                (view, angles[i], 1., self.size, self.bgmask, 0, rview)) 

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
