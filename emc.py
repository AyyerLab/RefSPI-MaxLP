#!/usr/bin/env python

'''EMC reconstructor object and script'''

import sys
import os.path as op
import argparse
import configparser
import time

import numpy as np
from scipy import ndimage
import h5py
from mpi4py import MPI
import cupy as cp

from scipy import special
import maxLP

P_MIN = 1.e-6
MEM_THRESH = 0.8


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
                                                    for m, m_a in zip(self.multi.get(), self.multi_accum.get())])
            self.count_multi = cp.array(self.count_multi)

            try:
                self.bg = cp.array(fptr['bg'][:]).ravel()
                print('Using background model with %.2f photons/frame' % self.bg.sum())
            except KeyError:
                self.bg = cp.zeros(self.num_pix)
        self.mem = mpool.used_bytes() - init_mem

    def get_powder(self):
        if self.powder is not None:
            return self.powder

        self.powder = np.zeros(self.num_pix)
        np.add.at(self.powder, self.place_ones.get(), 1)
        np.add.at(self.powder, self.place_multi.get(), self.count_multi.get())
        self.powder /= self.num_data
        self.powder = cp.array(self.powder)
        return self.powder

class EMC():
    '''Reconstructor object using parameters from config file

    Args:
        config_file (str): Path to configuration file

    The appropriate CUDA device must be selected before initializing the class.
    Can be used with mpirun, in which case work will be divided among ranks.
    '''
    def __init__(self, config_file, num_streams=4):
        self.num_streams = num_streams
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.num_proc = self.comm.size
        self.mem_size = cp.cuda.Device(cp.cuda.runtime.getDevice()).mem_info[1]

        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('emc', 'size')
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        self.num_rot = config.getint('emc', 'num_rot')
        self.support_area = config.getfloat('emc', 'support_area')
        self.photons_file = op.join(op.dirname(config_file),
                                    config.get('emc', 'in_photons_file'))
        self.output_folder = op.join(op.dirname(config_file),
                                     config.get('emc', 'output_folder'))
        self.log_file = op.join(op.dirname(config_file),
                                config.get('emc', 'log_file'))

        use_true_sol = config.getboolean('emc', 'use_true_solution', fallback=False)
        use_true_supp = config.getboolean('emc', 'use_true_support', fallback=False)
        self.true_support = op.join(op.dirname(config_file), config.get('emc', 'true_support_file'))

        self.use_phaser = False
        self.use_divcon = False
        phasing_type = config.get('emc', 'phasing_type', fallback='mlp')
        if phasing_type not in ['mlp', 'divcon']:
            raise ValueError('Phasing type needs to be either mlp or divcon')
        elif phasing_type == 'mlp':
            self.use_phaser = True
        elif phasing_type == 'divcon':
            self.use_divcon = True

        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)
        self.decrease_sigma = config.getboolean('emc', 'decrease_sigma', fallback=False)
        self.decrease_supp_area = config.getboolean('emc', 'decrease_supp_area', fallback=False)
        self.phaser = maxLP.MaxLPhaser(self.photons_file)

        dia = tuple([float(s) for s in config.get('emc', 'sphere_dia').split()])
        sx = tuple((float(s) for s in config.get('emc', 'shiftx').split()))
        sy = tuple((float(s) for s in config.get('emc', 'shifty').split()))
        self.shiftx, self.shifty, self.sphere_dia = np.meshgrid(np.linspace(sx[0], sx[1], int(sx[2])),
                                                                np.linspace(sy[0], sy[1], int(sy[2])),
                                                                np.linspace(dia[0], dia[1], int(dia[2])), indexing='ij')

        self.shiftx = self.shiftx.ravel()
        self.shifty = self.shifty.ravel()
        self.sphere_dia = self.sphere_dia.ravel()
        self.num_states = np.array([sx[-1], sy[-1], dia[-1]]).astype('i4')
        print(self.num_states.prod(), 'sampled states')

        self.x_ind, self.y_ind = cp.indices((self.size,)*2, dtype='f8')
        self.x_ind = self.x_ind.ravel() - self.size // 2
        self.y_ind = self.y_ind.ravel() - self.size // 2
        self.rad = cp.sqrt(self.x_ind**2 + self.y_ind**2)

        self.invmask = cp.zeros(self.size**2, dtype=cp.bool_)
        self.invmask[self.rad<4] = True
        self.invmask[self.rad>=self.size//2] = True
        self.intinvmask = self.invmask.astype('i4')

        if use_true_supp:
            self.invsuppmask = cp.load(self.true_support)
        elif phasing_type == 'mlp':
            self.invsuppmask = cp.ones((self.size,)*2, dtype=cp.bool_)
        elif phasing_type == 'divcon':
            self.invsuppmask = cp.ones((self.size,)*2, dtype=cp.bool_)
            self.invsuppmask[58:108,58:108]  = False

        self.probmask = cp.zeros(self.size**2, dtype='i4')
        self.probmask[self.rad>=self.size//2] = 2
        self.probmask[self.rad<self.size//8] = 1
        self.probmask[self.rad<4] = 2
        self.sphere_ramps = [self.ramp(i)*self.sphere(i) for i in range(int(self.num_states.prod()))]
        self.sphere_intens = cp.abs(cp.array(self.sphere_ramps[:int(dia[2])])**2).mean(0)

        stime = time.time()
        self.dset = Dataset(self.photons_file, self.size**2, self.need_scaling)
        self.powder = self.dset.get_powder()
        etime = time.time()

        if self.rank == 0:
            print('%d frames with %.3f photons/frame (%.3f s) (%.2f MB)' % \
                    (self.dset.num_data, self.dset.mean_count, etime-stime, self.dset.mem/1024**2))
            sys.stdout.flush()
        self.model = np.empty((self.size**2,), dtype='c16')

        if self.rank == 0:
            self._random_model(true_sol=use_true_sol)
            np.save(op.join(self.output_folder, 'model_000.npy'), self.model)
        self.comm.Bcast([self.model, MPI.C_DOUBLE_COMPLEX], root=0)
        if self.need_scaling:
            self.scales = self.dset.counts / self.dset.mean_count
        else:
            self.scales = cp.ones(self.dset.num_data, dtype='f8')
        self.prob = cp.array([])
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen = kernels.get_function('slice_gen')
        self.k_slice_merge = kernels.get_function('slice_merge')
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        self.k_calc_prob_all = kernels.get_function('calc_prob_all')
        self.k_merge_all = kernels.get_function('merge_all')
        self.k_proj_divide = kernels.get_function('proj_divide')

        self.bsize_model = int(np.ceil(self.size/32.))
        self.bsize_data = int(np.ceil(self.dset.num_data/32.))
        self.stream_list = [cp.cuda.Stream() for _ in range(self.num_streams)]


    def run_iteration(self, iternum=None):
        '''Run one iterations of EMC algorithm

        Args:
            iternum (int, optional): If specified, output is tagged with iteration number
        Current guess is assumed to be in self.model, which is updated. If scaling is included,
        the scale factors are in self.scales.
        '''

        self.num_states_p = np.arange(self.rank, self.num_states.prod(), self.num_proc).size
        if self.prob.shape != (self.num_states_p*self.num_rot, self.dset.num_data):
            self.prob = cp.empty((self.num_states_p*self.num_rot, self.dset.num_data), dtype='f8')

        views = cp.empty((self.num_streams, self.size**2), dtype='f8')
        intens = cp.empty((self.num_states.prod(), self.size**2), dtype='f8')
        dmodel = cp.array(self.model.ravel())
        #mp = cp.get_default_memory_pool()
        #print('Mem usage: %.2f MB / %.2f MB' % (mp.total_bytes()/1024**2, self.mem_size/1024**2))

        self._calculate_prob(dmodel, views)
        self._normalize_prob()
        if self.use_phaser:
            sx_vals, sy_vals, dia_vals, ang_vals = self.unravel_rmax(self.rmax)
            self.model = self.phaser._run_phaser(self.model, sx_vals, sy_vals, dia_vals, ang_vals)
            
        elif self.use_divcon:
            self._update_model(intens, dmodel)
            self._normalize_model(intens, dmodel, iternum)
            self.shrinkwrap(self.model, iternum)
        self.save_output(self.model, iternum)

    def _calculate_prob(self, dmodel, views, drange=None):
        if drange is None:
            s, e = (0, self.dset.num_data)
        else:
            s, e = tuple(drange)
        num_data_b = e - s
        self.bsize_data = int(np.ceil(num_data_b/32.))
        selmask = (self.probmask < 1)
        sum_views = cp.zeros_like(views)
        msums = cp.empty(self.num_states_p)
        rot_views = cp.empty_like(views)

        for i, r in enumerate(range(self.rank, self.num_states.prod(), self.num_proc)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            self.k_slice_gen_holo((self.bsize_model,)*2, (32,)*2,
                    (dmodel, self.shiftx[r], self.shifty[r], self.sphere_dia[r], 1.,
                     1., self.size, self.dset.bg, 0, views[snum]))
            msums[i] = views[snum][selmask].sum()
            sum_views[snum] += views[snum]
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()
        vscale = self.powder[selmask].sum() / sum_views.sum(0)[selmask].sum() * self.num_states_p

        for i, r in enumerate(range(self.rank, self.num_states.prod(), self.num_proc)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            self.k_slice_gen_holo((self.bsize_model,)*2, (32,)*2,
                    (dmodel, self.shiftx[r], self.shifty[r], self.sphere_dia[r], 1.,
                     1., self.size, self.dset.bg, 0, views[snum]))

            for j in range(self.num_rot):
                self.k_slice_gen((self.bsize_model,)*2, (32,)*2,
                        (views[snum], j*np.pi/self.num_rot, 1.,
                         self.size, self.dset.bg, 1, rot_views[snum]))
                self.k_calc_prob_all((self.bsize_data,), (32,),
                        (rot_views[snum], self.probmask, num_data_b,
                         self.dset.ones[s:e], self.dset.multi[s:e],
                         self.dset.ones_accum[s:e], self.dset.multi_accum[s:e],
                         self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                         -float(msums[i]*vscale), self.scales[s:e], self.prob[i*self.num_rot+j]))
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

    def _normalize_prob(self):
        max_exp_p = self.prob.max(0).get()
        rmax_p = self.prob.argmax(axis=0).get().astype('i4')
        rmax_p = ((rmax_p//self.num_rot)*self.num_proc + self.rank)*self.num_rot + rmax_p%self.num_rot
        max_exp = np.empty_like(max_exp_p)
        self.rmax = np.empty_like(rmax_p)

        self.comm.Allreduce([max_exp_p, MPI.DOUBLE], [max_exp, MPI.DOUBLE], op=MPI.MAX)
        rmax_p[max_exp_p != max_exp] = -1
        self.comm.Allreduce([rmax_p, MPI.INT], [self.rmax, MPI.INT], op=MPI.MAX)
        max_exp = cp.array(max_exp)

        self.prob = cp.exp(cp.subtract(self.prob, max_exp, self.prob), self.prob)
        psum_p = self.prob.sum(0).get()
        psum = np.empty_like(psum_p)
        self.comm.Allreduce([psum_p, MPI.DOUBLE], [psum, MPI.DOUBLE], op=MPI.SUM)
        self.prob = cp.divide(self.prob, cp.array(psum), self.prob)
        #self.prob.clip(a_min=P_MIN, out=self.prob)
        np.save(op.join(self.output_folder, 'prob.npy'), self.prob.get())

    def unravel_rmax(self, rmax):
        sx, sy, dia, ang = cp.unravel_index(rmax, tuple(self.num_states) + (self.num_rot,))
        sx_vals = cp.unique(self.shiftx)[sx]
        sy_vals = cp.unique(self.shifty)[sy]
        dia_vals = cp.unique(self.sphere_dia)[dia]
        ang_vals = ang*cp.pi / self.num_rot
        return sx_vals, sy_vals, dia_vals, ang_vals

    def _update_model(self, intens, dmodel, drange=None):
        if drange is None:
            s, e = (0, self.dset.num_data)
        else:
            s, e = tuple(drange)
        p_norm = self.prob.reshape(self.num_states.prod(), self.num_rot, self.dset.num_data).sum((1,2))
        h_p_norm = p_norm.get()
        num_data_b = e - s
        rot_views = cp.zeros((self.num_streams,)+intens[0].shape)
        mweights = cp.zeros((self.num_streams,)+intens[0].shape)
        intens[:] = 0

        for i, r in enumerate(range(self.rank, self.num_states.prod(), self.num_proc)):
            if h_p_norm[i] == 0.:
                continue
            snum = i % self.num_streams
            self.stream_list[snum].use()

            mweights[snum,:] = 0
            for j in range(self.num_rot):
                rot_views[snum,:] = 0
                self.k_merge_all((self.bsize_data,), (32,),
                        (self.prob[i*self.num_rot + j], num_data_b,
                         self.dset.ones[s:e], self.dset.multi[s:e],
                         self.dset.ones_accum[s:e], self.dset.multi_accum[s:e],
                         self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                         rot_views[snum]))
                self.k_slice_merge((self.bsize_model,)*2, (32,)*2,
                        (rot_views[snum], j*np.pi/self.num_rot, self.size,
                         intens[r], mweights[snum]))
            sel = (mweights[snum] > 0)
            intens[r][sel] /= mweights[snum][sel]
            intens[r] = intens[r] / p_norm[i] - self.dset.bg
            # Centrosymmetrization
            intens2d = intens[r].reshape(self.size,self.size)
            intens2d = 0.5 * (intens2d + intens2d[::-1,::-1])
            intens[r] = intens2d.ravel()
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

    def _normalize_model(self, intens, dmodel, iternum):
        self.comm.Allreduce(MPI.IN_PLACE, [intens.get(), MPI.DOUBLE], op=MPI.SUM)

        if self.rank == 0:
            sel = (self.rad > self.size//4) & (self.rad < self.size//2)
            mscale = float(cp.dot(self.sphere_intens[sel], intens.mean(0)[sel]) / cp.linalg.norm(intens.mean(0)[sel])**2)
            fobs = cp.sqrt(intens * mscale)
            print('mscale =', mscale)

            iter_curr = cp.empty(intens.shape, dtype='c16')
            iter_curr[:] = dmodel.ravel()
            iter_curr[:,self.invmask] = 0
            iter_p1 = cp.empty_like(iter_curr)

            for i in range(10):
                iter_curr = self.er(iter_curr, fobs, iter_p1)
            for i in range(40):
                iter_curr = self.diffmap(iter_curr, fobs, iter_p1)

            dmodel = self.proj_concur(iter_curr)[0]
            self.model = dmodel.get()
        self.comm.Bcast([self.model, MPI.C_DOUBLE_COMPLEX], root=0)

    def shrinkwrap(self, model, iternum):
        if iternum < 5 or iternum % 5 == 0 :
            amodel =np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(model.reshape((self.size,)*2)))))
            if iternum <= 5:
                sigma = 3
            else:
                if self.decrease_sigma:
                    s = 3
                    sigma = s - (iternum*s)/100
                else:
                    sigma = 2

            famodel = ndimage.gaussian_filter(amodel, sigma)
            if self.decrease_supp_area:
                a = (iternum-1)/99
                supp_area = 0.1 + special.erfc(a)/20
                thresh = np.sort(famodel.ravel())[int((1-supp_area)*amodel.size)]
            else:
                thresh = np.sort(famodel.ravel())[int((1- self.support_area)*amodel.size)]
            self.invsuppmask = cp.array(famodel < thresh)

    def save_output(self, model, iternum, intens=None):
        if iternum is None:
            np.save(op.join(self.output_folder, 'model.npy'), self.model)
            return

        with h5py.File(op.join(self.output_folder, 'output_%.3d.h5'%iternum), 'w') as fptr:
            fptr['model'] = self.model
            if intens is not None:
                fptr['intens'] = intens.get()

            sx_vals, sy_vals, dia_vals, ang_vals = self.unravel_rmax(self.rmax)
            fptr['angles'] = ang_vals
            fptr['diameters'] = dia_vals.get()
            fptr['shifts'] = np.array([sx_vals.get(), sy_vals.get()]).T
            fptr['support'] = ~self.invsuppmask.get()

    def _random_model(self, true_sol=False):
        if true_sol:
            with h5py.File('data/hetro/holo_dia_homo.h5', 'r') as f:
                sol = f['solution'][:]
            #sol = np.load('data/hetro/blur_obj.npy')
            self.model = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(sol))).ravel() / 1.e3
        elif self.use_divcon:
            rmodel = np.random.random((self.size,)*2)
            rmodel[self.invsuppmask.get()] = 0
            self.model = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rmodel))).flatten()
            self.model /= 2.e3
        elif self.use_phaser:
            self.model = np.random.random(self.size**2) + 1j*np.random.random(self.size**2)

    def ramp(self, n):
        return cp.exp(1j*2.*cp.pi*(self.x_ind*self.shiftx[n] + self.y_ind*self.shifty[n])/self.size)

    def sphere(self, n, diameter=None):
        if n is None:
            dia = diameter
        else:
            dia = self.sphere_dia[n]
        s = cp.pi * self.rad * dia / self.size
        s[s==0] = 1.e-5
        return ((cp.sin(s) - s*cp.cos(s)) / s**3).ravel()

    def proj_divide(self, iter_in, data, iter_out):
        for n in range(self.num_states.prod()):
            snum = n % self.num_streams
            self.stream_list[snum].use()

            self.k_proj_divide((self.size*self.size//32 + 1,), (32,),
                                (iter_in[n], data[n], self.sphere_ramps[n],
                                 (self.intinvmask), self.size, iter_out[n]))
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

    def proj_concur(self, iter_in, supp=True):
        iter_out = cp.empty_like(iter_in)
        avg = iter_in.mean(0)
        if supp:
            favg = cp.fft.fftshift(cp.fft.ifftn(avg.reshape(self.size, self.size)))
            favg[self.invsuppmask] = 0
            avg = cp.fft.fftn(cp.fft.ifftshift(favg)).ravel()
        iter_out[:] = avg
        return iter_out

    def diffmap(self, iterate, fobs, p1):
        self.proj_divide(iterate, fobs, p1)
        return iterate + self.proj_concur(2. * p1 - iterate) - p1

    def er(self, iterate, fobs, p1):
        self.proj_divide(iterate, fobs, p1)
        return self.proj_concur(p1)

def main():
    '''Parses command line arguments and launches EMC reconstruction'''
    import socket
    parser = argparse.ArgumentParser(description='In-plane rotation EMC')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='emc_config.ini',
                        help='Path to configuration file (default: emc_config.ini)')
    parser.add_argument('-d', '--devices', default = 'device.txt',
                        help='Path to devices file')
    parser.add_argument('-s', '--streams', type=int, default=4,
                        help='Number of streams to use (default=4)')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    num_proc = comm.size
    if args.devices is None:
        if num_proc == 1:
            print('Running on default device 0')
        else:
            print('Require a "devices" file if using multiple processes (one number per line)')
            sys.exit(1)
    else:
        with open(args.devices) as f:
            dev = int(f.readlines()[rank].strip())
            print('Rank %d: %s (Device %d)' % (rank, socket.gethostname(), dev))
            sys.stdout.flush()
            cp.cuda.Device(dev).use()


    recon = EMC(args.config_file, num_streams=args.streams)
    logf = open(op.join(recon.output_folder, 'EMC.log'), 'w')
    if rank == 0:
        logf.write('Iter  time(s)  change\n')
        logf.flush()
        avgtime = 0.
        numavg = 0
    for i in range(args.num_iter):
        m0 = cp.array(recon.model)
        stime = time.time()
        recon.run_iteration(i+1)
        etime = time.time()
        sys.stderr.write('\r%d/%d (%f s)'% (i+1, args.num_iter, etime-stime))
        if rank == 0:
            norm = float(cp.linalg.norm(cp.array(recon.model) - m0))
            logf.write('%-6d%-.2e %e\n' % (i+1, etime-stime, norm))
            print('Change from last iteration: ', norm)
            logf.flush()
            if i > 0:
                avgtime += etime-stime
                numavg += 1
    if rank == 0 and numavg > 0:
        print('\n%.4e s/iteration on average' % (avgtime / numavg))
    logf.close()

if __name__ == '__main__':
    main()
