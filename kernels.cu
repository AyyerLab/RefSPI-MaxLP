#include <cupy/complex.cuh>
//#include <math_constants.h>
#define CUDART_PI               3.1415926535897931e+0

extern "C" {

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

    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = cx[t] * ac - cy[t] * as + cen ;
    double ty = cx[t] * as + cy[t] * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return ;

    double fx = tx - ix, fy = ty - iy ;
    double gx = 1. - fx, gy = 1. - fy ;

    view[t] = model[ix*size + iy]*gx*gy +
              model[(ix+1)*size + iy]*fx*gy +
              model[ix*size + (iy+1)]*gx*fy +
              model[(ix+1)*size + (iy+1)]*fx*fy ;
    view[t] *= scale ;
    view[t] += bg[t] ;
    if (log_flag) {
        if (view[t] < 1.e-20)
            view[t] = -1000. ;
        else
            view[t] = log(view[t]) ;
    }
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
    complex<double> ramp, sphere ;

    double phase = 2. * CUDART_PI * ((x-cen) * shiftx + (y-cen) * shifty) / size ;
    double ramp_r = cos(phase) ;
    double ramp_i = sin(phase) ;
    ramp = complex<double>(ramp_r, ramp_i) ;

    double s = sqrt((x-cen)*(x-cen) + (y-cen)*(y-cen)) * CUDART_PI * diameter / size ;
    if (s == 0.)
        s = 1.e-5 ;
    sphere = complex<double>(rel_scale*(sin(s) - s*cos(s)) / (s*s*s), 0) ;

    complex<double> cview = ramp * sphere + model[t] ;
    view[t] = pow(abs(cview), 2.) ;

    view[t] *= scale ;
}

__global__
void calc_prob_all(const double *lview, const int *mask, const long long ndata, const int *ones,
                   const int *multi, const long long *o_acc, const long long *m_acc, const int *p_o,
                   const int *p_m, const int *c_m, const double init, const double *scales, double *prob_r) {
    long long d, t ;
    int pixel ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    prob_r[d] = init * scales[d] ;
    for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t) {
        pixel = p_o[t] ;
        if (mask[pixel] < 1)
            prob_r[d] += lview[pixel] ;
    }
    for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t) {
        pixel = p_m[t] ;
        if (mask[pixel] < 1)
            prob_r[d] += lview[pixel] * c_m[t] ;
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
                    const long long ndata, const long long nvox, double *logq_vd) {
    long long d, v ;
    v = blockDim.x * blockIdx.x + threadIdx.x ;
    if (v >= nvox || (!mask[v]))
		return ;

    double qx = qvals[v*3 + 0], qy = qvals[v*3 + 1], qrad = qvals[v*3 + 2] ;
    double fobj_vx = fobj[v].real(), fobj_vy = fobj[v].imag() ;

	// indptr, indices, data refer to (N_voxel, N_data) sparse array
    int ind_st = indptr[v], num_ind = indptr[v+1] - ind_st ;
    int ind_pos = 0 ;
    unsigned long long slice_ind, bit_shift ;

    double s, sphere_ft, w ;
    double rampx, rampy ;

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
        sincos(2.*CUDART_PI*(qx*shifts[d*2+0] + qy*shifts[d*2+1]), &rampy, &rampx) ;

        w = rescale * (pow(fobj_vx + sphere_ft*rampx, 2.) + pow(fobj_vy + sphere_ft*rampy, 2.)) ;
        logq_vd[v*ndata + d] = -w ;

        // Assuming sorted indices
        // Assuming photon data is "rotated"
        if (indices[ind_st + ind_pos] == d) {
            logq_vd[v*ndata + d] += data[ind_st + ind_pos] * log(w) ;
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
    int ix, iy, pos[4], i ;

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

} // extern C
