#include <cupy/complex.cuh>
#include <math_constants.h>

extern "C" {

__global__
void slice_gen(const double *model, const double angle, const double scale, const long long size,
               const double *bg, const long long log_flag, double *view) {
    int x = blockIdx.x * blockDim.x + threadIdx.x ;
    int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if (x > size - 1 || y > size - 1)
        return ;
    int t = x*size + y ;
    if (log_flag)
        view[t] = -1000. ;
    else
        view[t] = 0. ;

    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = (x - cen) * ac - (y - cen) * as + cen ;
    double ty = (x - cen) * as + (y - cen) * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return ;

    double fx = tx - ix, fy = ty - iy ;
    double cx = 1. - fx, cy = 1. - fy ;

    view[t] = model[ix*size + iy]*cx*cy +
              model[(ix+1)*size + iy]*fx*cy +
              model[ix*size + (iy+1)]*cx*fy +
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
void slice_gen_holo(const complex<double> *model, const double shiftx, const double shifty,
               const double diameter, const double rel_scale, const double scale, const long long size,
               const double *bg, const long long log_flag, double *view) {
    int x = blockIdx.x * blockDim.x + threadIdx.x ;
    int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if (x > size - 1 || y > size - 1)
        return ;
    int t = x*size + y ;
    if (log_flag)
        view[t] = -1000. ;
    else
        view[t] = 0. ;

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
    view[t] += bg[t] ;
    if (log_flag) {
        if (view[t] < 1.e-20)
            view[t] = -1000. ;
        else
            view[t] = log(view[t]) ;
    }
}

__global__
void slice_merge(const double *view, const double angle, const long long size,
                 double *model, double *mweights) {
    int x = blockIdx.x * blockDim.x + threadIdx.x ;
    int y = blockIdx.y * blockDim.y + threadIdx.y ;
    if (x > size - 1 || y > size - 1)
        return ;
    int t = x*size + y ;

    int cen = size / 2 ;
    double ac = cos(angle), as = sin(angle) ;
    double tx = (x - cen) * ac - (y - cen) * as + cen ;
    double ty = (x - cen) * as + (y - cen) * ac + cen ;
    int ix = __double2int_rd(tx), iy = __double2int_rd(ty) ;
    if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
        return ;
    double fx = tx - ix, fy = ty - iy ;
    double cx = 1. - fx, cy = 1. - fy ;

    atomicAdd(&model[ix*size + iy], view[t]*cx*cy) ;
    atomicAdd(&mweights[ix*size + iy], cx*cy) ;

    atomicAdd(&model[(ix+1)*size + iy], view[t]*fx*cy) ;
    atomicAdd(&mweights[(ix+1)*size + iy], fx*cy) ;

    atomicAdd(&model[ix*size + (iy+1)], view[t]*cx*fy) ;
    atomicAdd(&mweights[ix*size + (iy+1)], cx*fy) ;

    atomicAdd(&model[(ix+1)*size + (iy+1)], view[t]*fx*fy) ;
    atomicAdd(&mweights[(ix+1)*size + (iy+1)], fx*fy) ;
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
void merge_all(const double *prob_r, const long long ndata, const int *ones, const int *multi,
               const long long *o_acc, const long long *m_acc, const int *p_o, const int *p_m,
               const int *c_m, double *view) {
    long long d, t ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t)
        atomicAdd(&view[p_o[t]], prob_r[d]) ;
    for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t)
        atomicAdd(&view[p_m[t]], prob_r[d] * c_m[t]) ;
}

__global__
void proj_divide(const complex<double> *iter_in, const double *fobs, const complex<double> *sphere_ramp,
                 const int *invmask, const long long size, complex<double> *iter_out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x ;
    if (t >= size*size)
        return ;

    complex<double> shifted ;

    if (invmask[t] == 0) {
        shifted = iter_in[t] + sphere_ramp[t] ;
        iter_out[t] = shifted * fobs[t] / abs(shifted) - sphere_ramp[t] ;
    }
    else {
        iter_out[t] = iter_in[t] ;
    }
}

__global__
void get_f_dt(const complex<double> *fobj_t, const complex<double> *fref_d,
              const long long ndata, const long long npix,
              complex<double> *f_dt) {
    long long d, t ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;

    for (t = 0 ; t < npix ; ++t)
        f_dt[d*npix + t] = fobj_t[t] + fref_d[d] ;
}

__global__
void get_logq_pixel(const complex<double> *fobj, const long long *pixels, const double rescale,
                    const double *diams, const double *shifts, const double *qvals,
					const int *indptr, const int *indices, const double *data,
                    const long long ndata, const long long npix, double *logq_td) {
    long long d, t ;
    t = blockDim.x * blockIdx.x + threadIdx.x ;
    if (t >= npix)
        return ;

    long long pix = pixels[t] ;
    double qx = qvals[pix*2 + 0], qy = qvals[pix*2 + 1] ;
    double fobj_tx = fobj[pix].real(), fobj_ty = fobj[pix].imag() ;
	int ind_st = indptr[pix], num_ind = indptr[pix+1] - ind_st ;
	int ind_pos = 0 ;

    double s, sphere_ft, w ;
    double rampx, rampy ;

    for (d = 0 ; d < ndata ; ++d) {
        s = CUDART_PI * sqrt(qx*qx + qy*qy) * diams[d] ;
        if (s == 0.)
            s = 1.e-8 ;
        sphere_ft = (sin(s) - s*cos(s)) / pow(s, 3.) ;
        sincos(2.*CUDART_PI*(qx*shifts[d*2+0] + qy*shifts[d*2+1]), &rampy, &rampx) ;

		w = rescale * (pow(fobj_tx + sphere_ft*rampx, 2.) + pow(fobj_ty + sphere_ft*rampy, 2.)) ;
        logq_td[t*ndata + d] = -w ;

		// Assuming sorted indices
		// Assuming photon data is "rotated"
		if (indices[ind_st + ind_pos] == d) {
			logq_td[t*ndata + d] += data[ind_st + ind_pos] * log(w) ;
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
