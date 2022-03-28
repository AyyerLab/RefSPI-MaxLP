import numpy as np
import h5py
import cupy as cp

class Dataset():
    '''Parses sparse photons dataset from HDF5 file

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

        with h5py.File(self.photons_file, 'r') as fptr:
            if self.num_pix != fptr['num_pix'][...]:
                raise AttributeError('Number of pixels in photons file does not match')
            self.num_data = fptr['place_ones'].shape[0]
            try:
                self.ones = cp.array(fptr['ones'][:])
            except KeyError:
                self.ones = cp.array([len(fptr['place_ones'][i])
                                      for i in range(self.num_data)]).astype('i4')
            self.ones_accum = cp.roll(self.ones.cumsum(), 1)
            self.ones_accum[0] = 0
            self.place_ones = cp.array(np.hstack(fptr['place_ones'][:]))

            try:
                self.multi = cp.array(fptr['multi'][:])
            except KeyError:
                self.multi = cp.array([len(fptr['place_multi'][i])
                                       for i in range(self.num_data)]).astype('i4')
            self.multi_accum = cp.roll(self.multi.cumsum(), 1)
            self.multi_accum[0] = 0
            self.place_multi = cp.array(np.hstack(fptr['place_multi'][:]))
            self.count_multi = np.hstack(fptr['count_multi'][:])

            self.mean_count = float((self.place_ones.shape[0] +
                                     self.count_multi.sum()
                                    ) / self.num_data)

            if need_scaling:
                self.counts = self.ones + cp.array([self.count_multi[m_a:m_a+m].sum()
                                                    for m, m_a in zip(self.multi.get(),
                                                                      self.multi_accum.get())])
            self.count_multi = cp.array(self.count_multi)

            try:
                self.bg = cp.array(fptr['bg'][:]).ravel()
                print('Using background model with %.2f photons/frame' % self.bg.sum())
            except KeyError:
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
