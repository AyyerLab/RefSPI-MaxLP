import cupy as cp

_slice_gen = cp.RawKernel(r'''
    extern "C" __global__
    void slice_gen(const double *model,
                   const double angle,
                   const long long size,
                   const long long log_flag,
                   double *view) {
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
        int ix = tx, iy = ty ;
        if (ix < 0 || ix > size - 2 || iy < 0 || iy > size - 2)
            return ;

        double fx = tx - ix, fy = ty - iy ;
        double cx = 1. - fx, cy = 1. - fy ;

        view[t] = model[ix*size + iy]*cx*cy + 
                  model[(ix+1)*size + iy]*fx*cy +
                  model[ix*size + (iy+1)]*cx*fy + 
                  model[(ix+1)*size + (iy+1)]*fx*fy ;
        if (log_flag) {
            if (view[t] < 1.e-20)
                view[t] = -1000. ;
            else
                view[t] = log(view[t]) ;
        }
    }
    ''', 'slice_gen')

_slice_merge = cp.RawKernel(r'''
    extern "C" __global__
    void slice_merge(const double *view,
                     const double angle,
                     const long long size,
                     double *model,
                     double *mweights) {
        int x = blockIdx.x * blockDim.x + threadIdx.x ;
        int y = blockIdx.y * blockDim.y + threadIdx.y ;
        if (x > size - 1 || y > size - 1)
            return ;
        int t = x*size + y ;

        int cen = size / 2 ;
        double ac = cos(angle), as = sin(angle) ;
        double tx = (x - cen) * ac - (y - cen) * as + cen ;
        double ty = (x - cen) * as + (y - cen) * ac + cen ;
        int ix = tx, iy = ty ;
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
    ''', 'slice_merge')

_calc_prob_all = cp.RawKernel(r'''
    extern "C" __global__
    void calc_prob_all(const double *lview,
                       const long long ndata,
                       const int *ones,
                       const int *multi,
                       const long long *o_acc,
                       const long long *m_acc,
                       const int *p_o,
                       const int *p_m,
                       const int *c_m,
                       const double init,
                       double *prob_r) {
        long long d, t ;
        d = blockDim.x * blockIdx.x + threadIdx.x ;
        if (d >= ndata)
            return ;

        prob_r[d] = init ;
        for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t)
            prob_r[d] += lview[p_o[t]] ;
        for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t)
            prob_r[d] += lview[p_m[t]] * c_m[t] ;
    }
    ''', 'calc_prob_all')

_merge_all = cp.RawKernel(r'''
    extern "C" __global__
    void merge_all(const double *prob_r,
                   const long long ndata,
                   const int *ones,
                   const int *multi,
                   const long long *o_acc,
                   const long long *m_acc,
                   const int *p_o,
                   const int *p_m,
                   const int *c_m,
                   double *view) {
        long long d, t ;
        d = blockDim.x * blockIdx.x + threadIdx.x ;
        if (d >= ndata)
            return ;

        for (t = o_acc[d] ; t < o_acc[d] + ones[d] ; ++t)
            atomicAdd(&view[p_o[t]], prob_r[d]) ;
        for (t = m_acc[d] ; t < m_acc[d] + multi[d] ; ++t)
            atomicAdd(&view[p_m[t]], prob_r[d] * c_m[t]) ;
    }
    ''', 'merge_all')

