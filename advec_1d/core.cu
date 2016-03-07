__global__ void update_pre(int nn, int nne, double vf, float c_ul_tmp, double *ul, double *ul_prev, double *ul_tmp, double *kl, double *el_sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int eidx = tid / nne;   // index of elements
    int lidx = tid % nne;   // index of nodes
    int i;

    if (tid < nn) {
        // sum in element
        if (lidx == 0) {
            el_sum[eidx] = 0;
            for (i=0; i<nne; i++) 
                el_sum[eidx] += vf * ul[tid + i];
        }

        // ul_tmp
        ul_tmp[tid] = ul_prev[tid] + c_ul_tmp * kl[tid];
    }
}


__global__ void update_ul(int nn, int ne, int nne, double dt, double de, double vf, float c_ul, double *ul, double *ul_tmp, double *kl, double *el_sum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int eidx = tid / nne;  // index of elements
    int lidx = tid % nne;  // index of nodes
    int i;
    double el_sum_left, gg, bb, mm, kk;

    if (tid < nn) {
        if (eidx > 0) el_sum_left = el_sum[eidx-1];
        else el_sum_left = el_sum[ne-1];
        gg = el_sum[eidx] - pow(-1., lidx) * el_sum_left;

        bb = 0;
        if (lidx != 0) {
            for (i=(lidx-1)%2; i<lidx; i+=2)
                bb += 2 * vf * ul_tmp[eidx*nne + i];
        }

        mm = (2 * lidx + 1) / de;

        kk = dt * mm * (bb - gg);
        ul[tid] += c_ul * kk;
        kl[tid] = kk;
    }
}
