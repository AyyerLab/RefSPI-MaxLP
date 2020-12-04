import numpy as np

class FRC():
    def __init__(self, obj1=None, obj2=None):
        self.obj1 = None
        self.obj2 = None
        self.curr_binsize = None

        if obj1 is not None:
            self.set_obj1(obj1)
        if obj2 is not None:
            self.set_obj2(obj2)

    def set_obj1(self, obj):
        self._set_obj_n(obj, 1)

    def set_obj2(self, obj):
        self._set_obj_n(obj, 2)

    def calc(self, binsize=1., do_abs=False):
        if self.obj1 is None or self.obj2 is None:
            raise AttributeError('Need to first set obj1 and obj2 to be compared')
        if self.obj1.shape != self.obj2.shape:
            raise ValueError('Both objects must have same shape: %s vs %s'%(self.obj1.shape, self.obj2.shape))

        rsize = self._calc_binrad(binsize)
        numr = np.zeros(rsize, dtype='c16')
        denr1 = np.zeros(rsize, dtype='f8')
        denr2 = np.zeros(rsize, dtype='f8')

        np.add.at(numr, self.binrad, self.fobj1 * np.conj(self.fobj2))
        np.add.at(denr1, self.binrad, np.abs(self.fobj1)**2)
        np.add.at(denr2, self.binrad, np.abs(self.fobj2)**2)
        denr = np.sqrt(denr1 * denr2)

        fsc = np.zeros(rsize, dtype='f8')
        fsc[denr>0] = np.real(numr[denr>0]) / denr[denr>0]
        
        return np.arange(rsize, dtype='f8') * binsize, fsc

    def _set_obj_n(self, obj, ind):
        fobj = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))
        if ind == 1:
            self.obj1 = obj
            self.fobj1 = fobj
        else:
            self.obj2 = obj
            self.fobj2 = fobj

    def _calc_binrad(self, binsize):
        sh = self.obj1.shape
        x, y = np.meshgrid(np.arange(sh[0], dtype='f8') - sh[0]//2, np.arange(sh[1], dtype='f8') - sh[1]//2, indexing='ij')
        self.binrad = (np.sqrt(x*x + y*y) / binsize).astype('i4')
        return self.binrad.max() + 1
