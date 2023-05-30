import numpy as np
import h5py
import cupy as cp

class Dataset():
    '''Parses sparse photons dataset from EMC file

    Args:
        photons_file (str): Path to HDF5 photons file
        num_pix (int): Expected number of pixels in sparse file
        need_scaling (bool, optional): Whether scaling will be used

    Returns:
        Dataset object with attributes containing photon locations
    '''
    def __init__(self, photons_file, num_pix, need_scaling=False):
        self.powder = None
        mpool = cp.get_default_memory_pool()
        init_mem = mpool.used_bytes()
        self.photons_file = photons_file
        self.num_pix = num_pix

        with open(self.photons_file, 'rb') as fptr:
            self.num_data, f_num_pix = np.fromfile(fptr, '=i4', count=2)
            if self.num_pix != f_num_pix:
                raise AttributeError('Number of pixels in photons file does not match')
            _ = np.fromfile(fptr, '=i4', count=254)

            self.ones = cp.fromfile(fptr, '=i4', count=self.num_data)
            self.ones_accum = cp.roll(self.ones.cumsum(), 1)
            self.ones_accum[0] = 0

            self.multi = cp.fromfile(fptr, '=i4', count=self.num_data)
            self.multi_accum = cp.roll(self.multi.cumsum(), 1)
            self.multi_accum[0] = 0

            self.place_ones = cp.fromfile(fptr, '=i4', count=int(self.ones.sum()))
            self.place_multi = cp.fromfile(fptr, '=i4', count=int(self.multi.sum()))
            self.count_multi = np.fromfile(fptr, '=i4', count=int(self.multi.sum()))

            self.mean_count = float((self.place_ones.shape[0] +
                                     self.count_multi.sum()
                                    ) / self.num_data)

            if need_scaling:
                self.counts = self.ones + cp.array([self.count_multi[m_a:m_a+m].sum()
                                                    for m, m_a in zip(self.multi.get(),
                                                                      self.multi_accum.get())])
            self.count_multi = cp.array(self.count_multi)
            self.bg = cp.zeros(self.num_pix)

        self.mem = mpool.used_bytes() - init_mem

    def get_frame(self, num):
        """Get particular frame from photons_file"""
        o_a = self.ones_accum[num]
        m_a = self.multi_accum[num]

        frame = cp.zeros(self.num_pix, dtype='i4')
        frame[self.place_ones[o_a:o_a+self.ones[num]]] += 1
        frame[self.place_multi[m_a:m_a+self.multi[num]]] += self.count_multi[m_a:m_a+self.multi[num]]

        return frame

    def get_powder(self):
        if self.powder is not None:
            return self.powder

        self.powder = np.zeros(self.num_pix)
        np.add.at(self.powder, self.place_ones.get(), 1)
        np.add.at(self.powder, self.place_multi.get(), self.count_multi.get())
        self.powder /= self.num_data
        self.powder = cp.array(self.powder)
        return self.powder
