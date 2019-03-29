#!/usr/bin/env python

'''Utility to convert data in the CXIDB-18 entry'''

import os
import sys
import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser(description='Convert sparse photons.dat files to HDF5 format')
parser.add_argument('photons_fname', help='Path to photons.dat file')
parser.add_argument('det_fname', help='Path to detector.dat file')
parser.add_argument('-o', '--out_fname', help='Path to output file. By default, determined from photons_fname', default=None)
parser.add_argument('-s', '--size', help='Model size (default: 185)', type=int, default=185)
args = parser.parse_args()

if args.out_fname is None:
    args.out_fname = 'data/'+os.path.splitext(os.path.basename(args.photons_fname))[0] + '.h5'
print('Writing output to', args.out_fname)

with open(args.photons_fname, 'r') as fptr:
    lines = fptr.readlines()
    
num_data = int(lines[0].strip())

cen = args.size // 2
dx, dy = np.loadtxt(args.det_fname, skiprows=1, unpack=True)
pix = ((dx+cen)*args.size + (dy+cen)).astype('i4')

with h5py.File(args.out_fname, 'w') as outf:
    outf['ones'] = np.array([int(l.rstrip()) for l in lines[2::5]]).astype('i4')
    outf['multi'] = np.array([int(l.rstrip()) for l in lines[4::5]]).astype('i4')
    outf['num_pix'] = args.size**2
    print(outf['ones'][:].mean() + outf['multi'][:].mean(), 'non-zero pixels in each frame on average')

    dtype = h5py.special_dtype(vlen=np.dtype('i4'))
    place_ones = outf.create_dataset('place_ones', (num_data,), dtype=dtype)
    place_multi = outf.create_dataset('place_multi', (num_data,), dtype=dtype)
    count_multi = outf.create_dataset('count_multi', (num_data,), dtype=dtype)
    for d in range(num_data):
        place_ones[d] = pix[np.array(lines[3+d*5].rstrip().split(), dtype='i4')]
        m = np.array([int(val) for val in lines[5+d*5].rstrip().split()], dtype='i4')
        if len(m) > 0:
            place_multi[d] = pix[m[::2]]
            count_multi[d] = m[1::2]
        sys.stderr.write('\r%d/%d frames processed' % (d+1, num_data))
sys.stderr.write('\n')
