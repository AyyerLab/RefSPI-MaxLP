#include <cupy/complex.cuh>
//#include <math_constants.h>
#define CUDART_PI               3.1415926535897931e+0
typedef unsigned char uint8_t ;

extern "C" {

__device__
double interp2d(const double *model, const long long size,
                const double cx, const double cy, const double angle) {
    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = cx * ac - cy * as + cen ;
    double ty = cx * as + cy * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return 0. ;

    double fx = tx - ix, fy = ty - iy ;
    double gx = 1. - fx, gy = 1. - fy ;

    return model[ix*size + iy]*gx*gy +
           model[(ix+1)*size + iy]*fx*gy +
           model[ix*size + (iy+1)]*gx*fy +
           model[(ix+1)*size + (iy+1)]*fx*fy ;
}

__global__
void slice_gen(const double *model, const double *cx, const double *cy,
               const double angle, const double scale,
               const long long size, const long long num_pix,
               const double *bg, const long long log_flag, double *view) {
    int t = blockIdx.x * blockDim.x + threadIdx.x ;
    if (t > num_pix - 1)
        return ;
    if (log_flag)
        view[t] = -1000. ;
    else
        view[t] = 0. ;

    view[t] = interp2d(model, size, cx[t], cy[t], angle) ;
    if (view[t] == 0.)
        return ;
    view[t] *= scale ;
    view[t] += bg[t] ;
    if (log_flag) {
        if (view[t] < 1.e-20)
            view[t] = -1000. ;
        else
            view[t] = log(view[t]) ;
    }
}

__device__
complex<double> rampsphere(const double qx, const double qy,
                           const double sx, const double sy, const double diameter) {
    double phase = 2. * CUDART_PI * (qx*sx + qy*sy) ;
    double ramp_r = cos(phase) ;
    double ramp_i = sin(phase) ;
    complex<double> ramp = complex<double>(ramp_r, ramp_i) ;

    double s = sqrt(qx*qx + qy*qy) * CUDART_PI * diameter ;
    if (s == 0.)
        s = 1.e-5 ;
    complex<double> sphere = complex<double>((sin(s) - s*cos(s)) / (s*s*s), 0) ;

    return ramp * sphere ;
}

__global__
void slice_gen_holo(const complex<double> *model,
                    const double shiftx, const double shifty, const double diameter,
                    const double rel_scale, const double scale, const long long size,
                    double *view) {
    int x = blockIdx.x * blockDim.x + threadIdx.x ;
    int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if (x > size - 1 || y > size - 1)
        return ;
    int t = x*size + y ;

    double cen = floor(size / 2.) ;
    complex<double> rsphere = rampsphere((x-cen)/size, (y-cen)/size, shiftx, shifty, diameter) ;

    complex<double> cview = rel_scale * rsphere + model[t] ;
    view[t] = pow(abs(cview), 2.) ;

    view[t] *= scale ;
}

__global__
void calc_prob_all(const double *lview, const uint8_t *mask, const long long ndata, const int *ones,
                   const int *multi, const long long *o_acc, const long long *m_acc, const int *p_o,
                   const int *p_m, const int *c_m, const double init, const double *scales,
                   const long long r, long long *rmax, double *maxprob_r) {
    long long d, t ;
    int pixel ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    double myprob = init * scales[d] ;
    for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t) {
        pixel = p_o[t] ;
        if (mask[pixel] < 1)
            myprob += lview[pixel] ;
    }
    for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t) {
        pixel = p_m[t] ;
        if (mask[pixel] < 1)
            myprob += lview[pixel] * c_m[t] ;
    }

    if (myprob > maxprob_r[d]) {
        maxprob_r[d] = myprob ;
        rmax[d] = r ;
    }
}

__global__
void get_w_dv(const complex<double> *fobj_v, const complex<double> *fref_d,
              const long long ndata, const long long npix,
              const double rescale, double *w_dv) {
    long long d, v ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    for (v = 0 ; v < npix ; ++v)
        w_dv[d*npix + v] = rescale * pow(abs(fobj_v[v] + fref_d[d]), 2.) ;
}

__global__
void get_logq_pattern(const complex<double> *fobj, const double rescale, const bool *mask,
                      const complex<double> *pattern, const long long npattern, const double *step,
                      const double *diams, const double *shifts, const double *qvals,
                      const unsigned long long *ang_ind, const unsigned long long *sampled_mask,
                      const int *indptr, const int *indices, const double *data,
                      const long long ndata, const long long nvox, double *logq_pv) {
    long long d, v, p = 0 ;
    p = blockDim.x * blockIdx.x + threadIdx.x ;
    v = blockDim.y * blockIdx.y + threadIdx.y ;
    if (p >= npattern || v >= nvox || (!mask[v]))
        return ;

    double qx = qvals[v*3 + 0], qy = qvals[v*3 + 1], qrad = qvals[v*3 + 2] ;
    complex<double> fpatt = fobj[v] + pattern[p] * step[v] ;
    double fobj_vr = fpatt.real(), fobj_vi = fpatt.imag() ;

    // indptr, indices, data refer to (N_voxel, N_data) sparse array
    int ind_st = indptr[v], num_ind = indptr[v+1] - ind_st ;
    int ind_pos = 0 ;
    unsigned long long slice_ind, bit_shift ;

    double s, sphere_ft, w ;
    double rampr, rampi ;

    for (d = 0 ; d < ndata ; ++d) {
        slice_ind = ang_ind[d] / 64 ;
        bit_shift = ang_ind[d] % 64 ;
        if (sampled_mask[slice_ind*nvox + v] & (1<<bit_shift) == 0)
            continue ;

        // Calculate w_dv = |sphere*e^(i*shift) + fobj|^2
        s = CUDART_PI * qrad * diams[d] ;
        if (s == 0.)
            s = 1.e-8 ;
        sphere_ft = (sin(s) - s*cos(s)) / pow(s, 3.) ;
        sincos(2.*CUDART_PI*(qx*shifts[d*2+0] + qy*shifts[d*2+1]), &rampi, &rampr) ;

        w = rescale * (pow(fobj_vr + sphere_ft*rampr, 2.) + pow(fobj_vi + sphere_ft*rampi, 2.)) ;
        logq_pv[p*nvox + v] -= w ;

        // Assuming sorted indices
        // Assuming photon data is "rotated"
        if (indices[ind_st + ind_pos] == d) {
            logq_pv[p*nvox + v] += data[ind_st + ind_pos] * log(w) ;
            ind_pos++ ;
        }

        // Skipping when reaching end of frames indices
        if (ind_pos > num_ind)
            break ;
    }
}

__device__
bool isin(int *array, int length, int val) {
    // array is assumed to be sorted
    int lower = 0, upper = length, mid ;

    if (array[lower] == val || array[upper] == val)
        return true ;

    while (lower < upper) {
        mid = (lower + upper) / 2 ;
        if (array[mid] == val)
            return true ;
        else if (array[mid] < val)
            lower = mid + 1 ;
        else
            upper = mid ;
    }

    return false ;
    // For non-sorted, brute force search
    //for (i = 0 ; i < length ; ++i) {
    //    if (array[i] == val) {
    //        retval = 1 ;
    //        break ;
    //    }
    //}
    //return false ;
}

__global__
void rotate_photons(const int *indptr, const double *angles,
                    const long long ndata, const long long size,
                    int *indices) {
    long long d ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    // indices and indptr are for ndata x npix CSR matrix (emc-like)
    int ind_st = indptr[d], ind_en = indptr[d+1] ;
    int t, pix ;
    double tx, ty, rx, ry ;
    int ints = size, cen = size / 2 ;
    double c = cos(angles[d]), s = sin(angles[d]) ;
    int ix, iy ;

    for (t = ind_st ; t < ind_en ; ++t) {
        pix = indices[t] ;
        tx = pix / size - cen ;
        ty = pix % size - cen ;
        rx = c*tx - s*ty ;
        ry = s*tx + c*ty ;

        ix = min(max(__double2int_rn(rx + cen), 0), ints - 1) ;
        iy = min(max(__double2int_rn(ry + cen), 0), ints - 1) ;

        // 0-th order interpolation
        // Need trick to avoid repetition
        //indices[t] = pos[0] ;
        indices[t] = ix * size + iy ;
    }
}

__global__
void deduplicate(const int *indptr, const long long ndata,
                 const long long size, int *indices) {
    long long d ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    // indices and indptr are for ndata x npix CSR matrix (emc-like)
    int ind_st = indptr[d], ind_en = indptr[d+1] ;
    int t, pix, ix, iy, i ;
    int pos[8] ;

    for (t = ind_st ; t < ind_en ; ++t) {
        pix = indices[t] ;

        // Test whether previous index is common. If not, continue
        if (t > ind_st && pix != indices[t-1])
            continue ;

        // Get neighbouring pixels
        ix = pix / size ;
        iy = pix % size ;
        pos[0] = (ix-1) * size + iy-1 ;
        pos[1] = (ix-1) * size + iy ;
        pos[2] = (ix-1) * size + iy+1 ;
        pos[3] = ix * size + iy-1 ;
        pos[4] = ix * size + iy+1 ;
        pos[5] = (ix+1) * size + iy-1 ;
        pos[6] = (ix+1) * size + iy ;
        pos[7] = (ix+1) * size + iy+1 ;

        // Check if they exist and if not, assign to neighbouring pixel
        for (i = 0 ; i < 8 ; ++i) {
            if (pos[i] < 0 || pos[i] >= size*size)
                continue ;
            if (!isin(&indices[ind_st], ind_en-ind_st, pos[i])) {
                indices[t] = pos[i] ;
                break ;
            }
        }
    }
}

__device__
complex<double> cinterp2d(const complex<double> *model, const long long size,
                          const double cx, const double cy, const double angle) {
    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = cx * ac - cy * as + cen ;
    double ty = cx * as + cy * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return 0. ;

    double fx = tx - ix, fy = ty - iy ;
    double gx = 1. - fx, gy = 1. - fy ;

    return model[ix*size + iy]*gx*gy +
           model[(ix+1)*size + iy]*fx*gy +
           model[ix*size + (iy+1)]*gx*fy +
           model[(ix+1)*size + (iy+1)]*fx*fy ;
}

__global__
void get_prob_frame(const complex<double> *model, const long long size,
                    const int *p_o, const int *p_m, const int *c_m, const long long n_o, const long long n_m,
                    const double *sx, const double *sy, const double *dia, const long long nparams,
                    const double *angles, const long long nangs,
                    const double *cx, const double *cy, const long long npix,
                    const double rescale, double *prob) {
    long long r = blockDim.x * blockIdx.x + threadIdx.x ;
    long long a = blockDim.y * blockIdx.y + threadIdx.y ;
    if (r >= nparams || a >= nangs)
        return ;

    double shiftx = sx[r], shifty = sy[r], diameter = dia[r] ;
    double ang = angles[a] ;

    long long pix ;
    complex<double> fval ;
    double intens ;
    double ac = cos(ang), as = sin(ang) ;
    double tx, ty ;
    int t_o = 0, t_m = 0 ;

    for (pix = 0 ; pix < npix ; ++pix) {
        tx = cx[pix] * ac - cy[pix] * as ;
        ty = cx[pix] * as + cy[pix] * ac ;

        fval = rampsphere(tx/size, ty/size, shiftx, shifty, diameter) +
               cinterp2d(model, size, cx[pix], cy[pix], ang) ;

        intens = pow(abs(fval), 2.) ;
        prob[r*nangs + a] -= rescale * intens ;

        // Assuming sorted and non-overlapping p_o and p_m
        if (p_o[t_o] == pix) {
            prob[r*nangs + a] += log(intens) ;
            ++t_o ;
        }
        else if (p_m[t_m] == pix) {
            prob[r*nangs + a] += c_m[t_m] * log(intens) ;
            ++t_m ;
        }
    }
}

/* Descending argsort for 4 numbers with 6 comparisons */
__device__
void argsort4(const double vals[4], int sorter[4]) {
    int i, j, tmp ;
    for (i = 0 ; i < 4 ; ++i)
        sorter[i] = i ;

    for (i = 0 ; i < 3 ; ++i) {
        for (j = i+1 ; j < 4 ; ++j) {
            if (vals[sorter[i]] < vals[sorter[j]]) {
                tmp = sorter[i] ;
                sorter[i] = sorter[j] ;
                sorter[j] = tmp ;
            }
        }
    }
}

__global__
void rotate_sparse_interp(const int *indices, const double *data, const long long num_ind,
                          const double *cx, const double *cy, const double angle,
                          const long long size, double *dense) {
    long long t = blockDim.x * blockIdx.x + threadIdx.x ;
    if (t >= num_ind)
      return ;

    int pix = indices[t] ;
    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = cx[pix] * ac - cy[pix] * as + cen ;
    double ty = cx[pix] * as + cy[pix] * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return ;
    int pos_x, pos_y ;

    double fx = tx - ix, fy = ty - iy ;
    if (data[t] == 1.) {
        pos_x = ix + (fx > 0.5) ;
        pos_y = iy + (fy > 0.5) ;
        atomicAdd(&dense[pos_x*size + pos_y], 1.) ;
        return ;
    }

    double gx = 1. - fx, gy = 1. - fy ;
    double w[4] = {gx*gy, gx*fy, fx*gy, fx*fy} ;
    double val = data[t], base = 0, res ;
    int i, sorter[4] ;
    for (i = 0 ; i < 4 ; ++i) {
        pos_x = ix + (i > 1) ;
        pos_y = iy + (i % 2) ;
        atomicAdd(&dense[pos_x*size + pos_y], floor(w[i] * val)) ;
        base += floor(w[i] * val) ;
    }

    res = val - base ;
    argsort4(w, sorter) ;
    for (i = 0 ; i < res ; ++i) {
        pos_x = ix + (sorter[i] > 1) ;
        pos_y = iy + (sorter[i] % 2) ;
        atomicAdd(&dense[pos_x*size + pos_y], 1.) ;
    }
}

} // extern C
