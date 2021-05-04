import sys
import os
import numpy
import cupy as cp

class HoloRecon():
    def __init__(self, size, shiftx, shifty, diameter, minrad=4, gen_data=True, apply_supp=True):
        self.size = size
        self.x, self.y, rad = self.get_rad(size)
        mask = (rad>4) & (rad<92)
        if os.path.exists('holo_data.npz'):
            zfile = numpy.load('holo_data.npz')
            self.obj, self.init = cp.array(zfile['arr_0']), cp.array(zfile['arr_1'])
        else:
            self.obj = self.make_obj()
            self.init = self.gen_init((len(shiftx),) + self.obj.shape)
            numpy.savez('holo_data.npz', self.obj, self.init)
        fobj = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.obj)))
        self.invmask = cp.zeros((self.size,)*2, dtype=cp.bool)
        self.invmask[rad<minrad] = True
        self.invmask[rad>=self.size//2] = True
        self.invsuppmask = cp.ones((self.size,)*2, dtype=cp.bool)
        self.invsuppmask[47:119,47:119] = False
        self.apply_supp = apply_supp

        self.error = None
        self.curr = None
        
        s = cp.pi * rad * diameter / size
        s[s==0] = 1.e-5
        self.sphere = (cp.sin(s) - s*cp.cos(s)) / s**3

        self.sx = shiftx
        self.sy = shifty

        if gen_data:
            self.data = cp.array([cp.abs(fobj+1000*self.sphere*self.ramp(a,b)) for a, b in zip(self.sx, self.sy)])
            self.data[:,self.invmask] = 0

    def run(self, func, niter):
        if self.curr is None:
            self.curr = cp.copy(self.init)
        if self.error is None:
            self.error = []

        for i in range(niter):
            self.curr, err = func(self.curr)
            self.error.append(err)
            sys.stderr.write('\r%d'%i)
        return cp.array(self.error)

    def make_obj(self):

        size = self.size

        x, y = cp.indices((size,size))
        obj =  cp.zeros((size,size))
        for _ in range(50):
            cen = cp.random.random((2,))*size/5 + size//2*4/5
            pixrad = cp.sqrt((x-cen[0])**2 + (y-cen[1])**2)
            diskrad = (0.7 + 0.3*cp.random.random())*size/20
            obj[pixrad<diskrad] += 1. - (pixrad[pixrad<diskrad]/diskrad)**2

        return obj

    def ramp(self, sx, sy):                         
        return cp.exp(1j*2.*cp.pi*(self.x*sx + self.y*sy)/self.size)

    @staticmethod
    def gen_init(shape):
        vals = cp.random.random((2,) + shape)
        out = cp.empty(shape, dtype='c16')
        out.real = vals[0]
        out.imag = vals[1]
        return out

    def pc(self, x):        
        avg = x.mean(0)
        if self.apply_supp:
            favg = cp.fft.fftshift(cp.fft.ifftn(avg.reshape(self.size, self.size)))
            favg[self.invsuppmask] = 0
            avg = cp.fft.fftn(cp.fft.ifftshift(favg))
        return cp.broadcast_to(avg, x.shape)

    def pd(self, x):
        N = len(x)
        out = cp.empty_like(x)
        for n in range(N):
            rs = 1000.*self.sphere*self.ramp(self.sx[n], self.sy[n])
            shifted = x[n] + rs
            out[n] = shifted * self.data[n] / cp.abs(shifted) - rs
            out[n][self.invmask] = x[n][self.invmask]
        return out

    def er(self, x, return_err=True):
        out = self.pc(self.pd(x))
        if return_err:
            err = cp.linalg.norm(out-x)
            return out, err
        else:
            return out

    def diffmap(self, x, return_err=True):
        p1 = self.pd(x)
        out = x + self.pc(2*p1 - x) - p1
        if return_err:
            err = cp.linalg.norm(out-x)
            return out, err
        else:
            return out

    def diffmap2(self, x, return_err=True): # beta = -1
        p1 = self.pc(x)
        out = x + self.pd(2*p1 - x) - p1
        if return_err:
            err = cp.linalg.norm(out-x)
            return out, err
        else:
            return out

    @staticmethod
    def get_rad(size):
        x, y = cp.indices((size,size))
        x -= size//2
        y -= size//2
        return x, y, cp.sqrt(x*x + y*y)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('num_iter', help='Number of iterations', type=int, default= 100)
    parser.add_argument('-s', '--size', help='Array size', type=int, default=185)
    parser.add_argument('-d', '--diameter', help='Sphere diameter in pixels', type=float, default=7.)
    args = parser.parse_args()

    sx, sy = cp.indices((7, 7), dtype='f8')
    sx = sx.ravel()-3
    sy = sy.ravel()-3
    recon = HoloRecon(args.size, sx, sy, args.diameter)
    recon.run(recon.diffmap, args.num_iter)

