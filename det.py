'''Stripped-down version of module containing detector class'''

import sys
import numpy as np
import h5py

class Detector():
    """Dragonfly detector

    The detector file format is specified in github.com/duaneloh/Dragonfly/wiki
    This class reads the file and provides numpy arrays which can be used for
    further processing.

    __init__ arguments (optional):
        det_fname (string) - Path to detector file to populate attributes
        mask_flag (bool) - Whether to read the mask column for each pixel
        keep_mask_1 (bool) - Whether to consider mask=1 pixels as good

    Methods:
        parse(fname, mask_flag=True, keep_mask_1=True)

    On parsing, it produces the following numpy arrays (each of length num_pix)

    Attributes:
        self.qx, self.qy, self.qz - Voxel space coordinates (origin at (0,0,0))
        self.cx, self.cy - Floating point 2D coordinates (origin at (0,0))
        self.x, self.y - Integer and shifted 2D coordinates (corner at (0,0))
        self.mask - Assembled mask
        self.raw_mask - Unassembled mask as stored in detector file
        self.unassembled_mask - Unassembled mask (1=good, 0=bad)
    """
    def __init__(self, det_fname=None, mask_flag=True, keep_mask_1=True):
        self.background = None
        self._sym_shape = None
        if det_fname is not None:
            self.parse(det_fname, mask_flag, keep_mask_1)

    def parse(self, fname, mask_flag=True, keep_mask_1=True):
        """ Parse Dragonfly detector from file

        File can either be in the HDF5 or ASCII format
        """
        self.det_fname = fname
        sys.stderr.write('Reading %s...'%self.det_fname)
        if mask_flag:
            sys.stderr.write('with mask...')
        with h5py.File(self.det_fname, 'r') as fptr:
            self.qx = fptr['qx'][:]
            self.qy = fptr['qy'][:]
            self.qz = fptr['qz'][:]
            self.corr = fptr['corr'][:]
            self.raw_mask = fptr['mask'][:].astype('u1')
            self.detd = fptr['detd'][()]
            self.ewald_rad = fptr['ewald_rad'][()]
            if 'background' in fptr:
                self.background = fptr['background'][:]
        sys.stderr.write('done\n')
        self._process_det(mask_flag, keep_mask_1)

    def _process_det(self, mask_flag, keep_mask_1):
        if mask_flag:
            mask = np.copy(self.raw_mask)
            if keep_mask_1:
                mask[mask == 1] = 0 # To keep both 0 and 1
                mask = mask // 2 # To keep both 0 and 1
            else:
                mask[mask == 2] = 1 # To keep only mask==0
            mask = 1 - mask
        else:
            self.raw_mask = np.zeros(self.qx.shape, dtype='u1')
            mask = np.ones(self.qx.shape, dtype='u1')

        if self.qz.mean() > 0:
            self.cx = self.qx * self.detd / (self.ewald_rad - self.qz) # pylint: disable=C0103
            self.cy = self.qy * self.detd / (self.ewald_rad - self.qz) # pylint: disable=C0103
        else:
            self.cx = self.qx * self.detd / (self.ewald_rad + self.qz) # pylint: disable=C0103
            self.cy = self.qy * self.detd / (self.ewald_rad + self.qz) # pylint: disable=C0103
        self.x = np.round(self.cx - self.cx.min()).astype('i4')
        self.y = np.round(self.cy - self.cy.min()).astype('i4')
        self.unassembled_mask = mask.ravel()

    @property
    def coords_xy(self):
        '''Return 2D pixel coordinates'''
        return self.cx, self.cy

    @property
    def qvals_xyz(self):
        '''Return 3D voxel values'''
        return self.qx, self.qy, self.qz

    @property
    def indices_xy(self):
        '''Return 2D integer coordinates (for assembly)
        Corner of the detector at (0,0)'''
        return self.x, self.y
