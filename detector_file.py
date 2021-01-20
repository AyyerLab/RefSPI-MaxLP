import sys
import numpy as np


sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src/')

import detector

size = 185

det = detector.Detector()

ind = np.arange(size, dtype = 'f8') - size//2

x, y = np.meshgrid(ind, ind, indexing = 'xy')

det.cx= x.ravel()
det.cy= y.ravel()
det.ewald_rad = 1000.
det.detd = 1000.
det.raw_mask = np.zeros(det.cx.shape, dtype='u1')

det.calc_from_coords()
det.write('dense_dat.h5')
