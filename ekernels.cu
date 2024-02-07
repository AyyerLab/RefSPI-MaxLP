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
    complex<double> sphere = complex<double>(pow(diameter, 3.) * (sin(s) - s*cos(s)) / (s*s*s), 0) ;

    return ramp * sphere ;
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
void calc_prob_all(const double *lview, const uint8_t *mask,
                   const long long ndata, const int *ones, const int *multi,
                   const long long *o_acc, const long long *m_acc,
                   const int *p_o, const int *p_m, const int *c_m,
                   const double init, const double *scales,
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
void get_prob_frame(const complex<double> *model, const long long size,
                    const int *p_o, const int *p_m, const int *c_m, const long long n_o, const long long n_m,
                    const double *sx, const double *sy, const double *dia, const long long nparams,
                    const double *angles, const long long nangs,
                    const double *cx, const double *cy, const uint8_t *mask, const long long npix,
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
        if (mask[pix] > 0)
            continue ;
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

__global__
void calc_local_prob_all(const complex<double> *model, const long long size, const double rescale,
                         const long long ndata, const int *ones, const int *multi,
                         const long long *o_acc, const long long *m_acc,
                         const int *p_o, const int *p_m, const int *c_m,
                         const double *cx, const double *cy, const uint8_t *mask, const long long npix,
                         const double *rvalues, const double *rsteps, const long long *num_rsamples,
                         const double *angles, const double ang_step, const long long num_angs,
                         long long *rmax, double *maxprob) {
    long long d ;
    d = blockDim.x * blockIdx.x + threadIdx.x ;
    if (d >= ndata)
        return ;
    
    long long r, index, a, pix ;
	double sx, sy, dia, ang ;
	double tx, ty, ac, as ;
	complex<double> fval ;
	double intens, myprob ;
	long long t_o, t_m ;

	long long tot_num_rsamples = 1 ;
    for (r = 0 ; r < 3 ; ++r)
        tot_num_rsamples *= num_rsamples[r] ;

    for (r = 0 ; r < tot_num_rsamples ; ++r) {
		index = r / (num_rsamples[1] * num_rsamples[2]) - num_rsamples[0] / 2 ;
		sx = rvalues[d*3 + 0] + rsteps[0] * index ;
		index = (r / num_rsamples[2]) % num_rsamples[1] - num_rsamples[1] / 2 ;
		sy = rvalues[d*3 + 1] + rsteps[1] * index ;
		index = r % num_rsamples[2] - num_rsamples[2] / 2 ;
		dia = rvalues[d*3 + 2] + rsteps[2] * index ;
		
        for (a = 0 ; a < num_angs ; ++a) {
			ang = angles[d] + (a - a/2) * ang_step ;
			ac = cos(ang) ;
			as = sin(ang) ;

			myprob = 0. ;
	        t_o = o_acc[d] ;
			t_m = m_acc[d] ;

			for (pix = 0 ; pix < npix ; ++pix) {
				if (mask[pix] > 0)
					continue ;

				tx = cx[pix] * ac - cy[pix] * as ;
				ty = cx[pix] * as + cy[pix] * ac ;

				fval = rampsphere(tx/size, ty/size, sx, sy, dia) +
					   cinterp2d(model, size, cx[pix], cy[pix], ang) ;
				intens = pow(abs(fval), 2.) ;
				myprob -= rescale * intens ;

				// Assuming sorted and non-overlapping p_o and p_m
				if (p_o[t_o] == pix) {
					myprob += log(intens) ;
					++t_o ;
				}
				else if (p_m[t_m] == pix) {
					myprob += c_m[t_m] * log(intens) ;
					++t_m ;
				}
			}

			if (myprob > maxprob[d]) {
				maxprob[d] = myprob ;
				rmax[d] = r*num_angs + a ;
			}
        }
    }
}

}
