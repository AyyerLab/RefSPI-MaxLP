#include <cupy/complex.cuh>
//#include <math_constants.h>
#define CUDART_PI               3.1415926535897931e+0

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
void calc_prob_all(const double *lview, const int *mask, const long long ndata, const int *ones,
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
void get_logq_voxel(const complex<double> *fobj, const double rescale, const bool *mask,
                    const double *diams, const double *shifts, const double *qvals,
                    const long long *ang_ind, const unsigned long long *sampled_mask,
                    const int *indptr, const int *indices, const double *data,
                    const long long ndata, const long long nvox, double *logq_v) {
    long long d, v ;
    v = blockDim.x * blockIdx.x + threadIdx.x ;
    if (v >= nvox || (!mask[v]))
        return ;

    double qx = qvals[v*3 + 0], qy = qvals[v*3 + 1], qrad = qvals[v*3 + 2] ;
    double fobj_vr = fobj[v].real(), fobj_vi = fobj[v].imag() ;

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
        logq_v[v] -= w ;

        // Assuming sorted indices
        // Assuming photon data is "rotated"
        if (indices[ind_st + ind_pos] == d) {
            logq_v[v] += data[ind_st + ind_pos] * log(w) ;
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

__global__
void get_prob_frame(const complex<double> *model, const long long size,
                    const int *p_o, const int *p_m, const int *c_m, const long long n_o, const long long n_m,
                    const double *sx, const double *sy, const double *dia, const long long nparams,
                    const double *angles, const long long nangs,
                    const double *cx, const double *cy, const long long npix,
                    double *prob) {
    long long r = blockDim.x * blockIdx.x + threadIdx.x ;
    long long a = blockDim.y * blockIdx.y + threadIdx.y ;
    if (r >= nparams || a >= nangs)
        return ;

    double shiftx = sx[r], shifty = sy[r], diameter = dia[r] ;
    double ang = angles[a] ;

    long long t, pix ;
    complex<double> fval ;
    double intens ;
    double ac = cos(ang), as = sin(ang) ;
    int cen = size / 2 ;
    double tx, ty, fx, fy, gx, gy ;
    int ix, iy ;

    for (t = 0 ; t < n_o ; ++t) {
        pix = p_o[t] ;
        tx = cx[pix] * ac - cy[pix] * as + cen ;
        ty = cx[pix] * as + cy[pix] * ac + cen ;

        fval = rampsphere(tx/size, ty/size, shiftx, shifty, diameter) ;

        ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
        if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
            continue ;

        fx = tx - ix, fy = ty - iy ;
        gx = 1. - fx, gy = 1. - fy ;

        fval += model[ix*size + iy]*gx*gy +
                model[(ix+1)*size + iy]*fx*gy +
                model[ix*size + (iy+1)]*gx*fy +
                model[(ix+1)*size + (iy+1)]*fx*fy ;
        intens = 2*log(abs(fval)) ;
        prob[r*nangs + a] += intens ;
    }

    for (t = 0 ; t < n_m ; ++t) {
        pix = p_m[t] ;
        tx = cx[pix] * ac - cy[pix] * as + cen ;
        ty = cx[pix] * as + cy[pix] * ac + cen ;
        fval = rampsphere(tx/size, ty/size, shiftx, shifty, diameter) ;
        ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
        if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
            continue ;

        fx = tx - ix, fy = ty - iy ;
        gx = 1. - fx, gy = 1. - fy ;

        fval += model[ix*size + iy]*gx*gy +
                model[(ix+1)*size + iy]*fx*gy +
                model[ix*size + (iy+1)]*gx*fy +
                model[(ix+1)*size + (iy+1)]*fx*fy ;
        intens = 2*log(abs(fval)) ;
        prob[r*nangs + a] += intens ;
    }
}

} // extern C
