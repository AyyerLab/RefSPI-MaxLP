import sys
import numpy as np
from scipy import ndimage

class FRC():
    '''Class to calculate FRC of two 2D real-space objects

    Methods:
        calc() - Calculate FRC between obj1 and obj2
        calc_rot() - Calculate best FRC between obj1 and rotated version of obj2
    '''
    def __init__(self, obj1=None, obj2=None, verbose=False):
        self.verbose = verbose
        self.obj1 = None
        self.obj2 = None
        self.fobj1 = None
        self.fobj2 = None
        self.binrad = None

        if obj1 is not None:
            self.set_obj1(obj1)
        if obj2 is not None:
            self.set_obj2(obj2)

    def set_obj1(self, obj):
        '''Set first object'''
        self._set_obj_n(obj, 1)

    def set_obj2(self, obj):
        '''Set second object'''
        self._set_obj_n(obj, 2)

    def calc(self, binsize=1., do_abs=False, **kwargs):
        '''Calculate FRC between obj1 and obj2

        Parameters:
            binsize - Radial bin size in pixels
            do_abs - Flag on whether to return absolute value of FRC

        Returns:
            rvals, fvals - Radii and FRC values
        '''
        if self.obj1 is None or self.obj2 is None:
            raise AttributeError('Need to first set obj1 and obj2 to be compared')
        if self.obj1.shape != self.obj2.shape:
            raise ValueError('Both objects must have same shape: %s vs %s'%(self.obj1.shape, self.obj2.shape))

        if 'rsize' in kwargs:
            rsize = kwargs.get('rsize')
        else:
            if self.verbose:
                print('Calculating binrad')
            rsize = self._calc_binrad(binsize)
        numr = np.zeros(rsize, dtype='c16')
        denr1 = np.zeros(rsize, dtype='f8')
        denr2 = np.zeros(rsize, dtype='f8')

        if 'fobj2' in kwargs:
            fobj2 = kwargs.get('fobj2')
        else:
            if self.verbose:
                print('Using self.fobj2')
            fobj2 = self.fobj2

        np.add.at(numr, self.binrad, self.fobj1 * np.conj(fobj2))
        np.add.at(denr1, self.binrad, np.abs(self.fobj1)**2)
        np.add.at(denr2, self.binrad, np.abs(fobj2)**2)
        denr = np.sqrt(denr1 * denr2)

        fsc = np.zeros(rsize, dtype='f8')
        if do_abs:
            fsc[denr > 0] = np.abs(numr[denr > 0]) / denr[denr > 0]
        else:
            fsc[denr > 0] = np.real(numr[denr > 0]) / denr[denr > 0]

        return np.arange(rsize, dtype='f8') * binsize, fsc

    def calc_rot(self, binsize=1., num_rot=180, do_abs=False):
        '''Calculate best FRC with rotated versions of obj2

        Parameters;
            binsize - Radial bin size in pixels
            num_rot - Number of angular samples from 0-360 degrees
            do_abs - Flag on whether to return absolute value of FRC


        Returns:
            rvals, fvals - Radii and FRC values
        '''
        rotfmodel = np.empty_like(self.fobj2)
        cen = self.fobj2.shape[-1] // 2
        rsize = self._calc_binrad(binsize)
        max_cc = -1.

        for i, r in enumerate(np.arange(num_rot)):
            rotfmodel.real = ndimage.rotate(self.fobj2.real, r*360./num_rot,
                                            order=1, prefilter=False, reshape=False)
            rotfmodel.imag = ndimage.rotate(self.fobj2.imag, r*360./num_rot,
                                            order=1, prefilter=False, reshape=False)

            rvals, ccvals = self.calc(binsize=binsize, do_abs=do_abs, rsize=rsize, fobj2=rotfmodel)
            if ccvals[:cen].mean() > max_cc:
                max_cc = ccvals[:cen].mean()
                rmax = r
                fvals = ccvals
            if self.verbose:
                sys.stderr.write('\r%d/%d' % (i+1, num_rot))
        if self.verbose:
            sys.stderr.write('\n')

        print('Best correlation for %.3f degrees' % (rmax*360./num_rot))
        return rvals, fvals

    def _set_obj_n(self, obj, ind):
        fobj = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
        if ind == 1:
            self.obj1 = obj
            self.fobj1 = fobj
        else:
            self.obj2 = obj
            self.fobj2 = fobj

    def _calc_binrad(self, binsize):
        shp = self.obj1.shape
        x, y = np.meshgrid(np.arange(shp[0], dtype='f8') - shp[0]//2,
                           np.arange(shp[1], dtype='f8') - shp[1]//2,
                           indexing='ij')
        self.binrad = (np.sqrt(x*x + y*y) / binsize).astype('i4')
        return self.binrad.max() + 1
