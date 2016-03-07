#===============================================================================
# DG method with modal basis functions
# 1-D advection equation
# CUDA imitation version
# ------------------------------------------------------------------------------

from __future__ import division
from dg_modal_base import DGModalBase
import numpy as np



def sum_in_element(nne, tid, vf, ul, el_sum):
    eidx = tid // nne   # index of elements
    lidx = tid % nne    # index of nodes

    if lidx == 0:
        el_sum[eidx] = 0
        for i in xrange(nne):
            el_sum[eidx] += vf * ul[tid + i]



def update_pre_rk4(nn, nne, vf, c_ul_tmp, ul, ul_prev, ul_tmp, kl, el_sum, block, grid):
    blockDim_x = block[0]
    for blockIdx_x in xrange(grid[0]): 
        for threadIdx_x in xrange(block[0]): 
            tid = blockIdx_x * blockDim_x + threadIdx_x

            if tid < nn:
                sum_in_element(nne, tid, vf, ul, el_sum)

                # ul_tmp
                ul_tmp[tid] = ul_prev[tid] + c_ul_tmp * kl[tid]



def update_pre_rk3(nn, nne, vf, ul, el_sum, block, grid):
    blockDim_x = block[0]
    for blockIdx_x in xrange(grid[0]): 
        for threadIdx_x in xrange(block[0]): 
            tid = blockIdx_x * blockDim_x + threadIdx_x

            if tid < nn:
                sum_in_element(nne, tid, vf, ul, el_sum)



def func(ne, nne, de, vf, tid, el_sum, ul_tmp):
    eidx = tid // nne   # index of elements
    lidx = tid % nne    # index of nodes

    if (eidx > 0): el_sum_left = el_sum[eidx-1]
    else: el_sum_left = el_sum[ne-1]
    gg = el_sum[eidx] - pow(-1., lidx) * el_sum_left

    bb = 0
    if lidx != 0:
        for i in xrange((lidx-1)%2, lidx, 2):
            bb += 2 * vf * ul_tmp[eidx*nne + i]

    mm = (2 * lidx + 1) / de
    val = mm * (bb - gg)

    return val



def update_ul_rk4(nn, ne, nne, dt, de, vf, c_ul, ul, ul_tmp, kl, el_sum, block, grid):
    blockDim_x = block[0]
    for blockIdx_x in xrange(grid[0]): 
        for threadIdx_x in xrange(block[0]): 
            tid = blockIdx_x * blockDim_x + threadIdx_x

            if tid < nn:
                kl[tid] = dt * func(ne, nne, de, vf, tid, el_sum, ul_tmp)
                ul[tid] += c_ul * kl[tid]



def update_ul_rk3(nn, ne, nne, dt, de, vf, c_ul1, c_ul2, c_ul3, ul, ul_prev, el_sum, block, grid):
    blockDim_x = block[0]
    for blockIdx_x in xrange(grid[0]): 
        for threadIdx_x in xrange(block[0]): 
            tid = blockIdx_x * blockDim_x + threadIdx_x

            if tid < nn:
                ul[tid] = c_ul1*ul_prev[tid] + c_ul2*ul[tid] + c_ul3*dt*func(ne, nne, de, vf, tid, el_sum, ul)



class DGModalGpu(DGModalBase):
    def __init__(self, ne, p_degree, cfl=0.1, v=0.5):
        super(DGModalGpu, self).__init__(ne, p_degree, cfl, v)


    def allocation(self):
        super(DGModalGpu, self).allocation()
        self.ul_gpu = np.zeros_like(self.ul)
        self.ul_prev_gpu = np.zeros_like(self.ul)
        self.ul_tmp_gpu = np.zeros_like(self.ul)
        self.kl_gpu = np.zeros_like(self.ul)
        self.el_sum_gpu = np.zeros(self.ne)


    def x2l(self):
        super(DGModalGpu, self).x2l()
        self.ul_gpu[:] = self.ul    # memcpy_htod


    def l2x(self):
        self.ul[:] = self.ul_gpu    # memcpy_dtoh
        super(DGModalGpu, self).l2x()


    def update_rk4(self):
        nn, ne, nne = self.nn, self.ne, self.nne
        dt, de, vf = self.dt, self.de, self.vf
        bs, gs = (256,1,1), (nn//256+1,1)
        ul, ul_prev, ul_tmp = self.ul_gpu, self.ul_prev_gpu, self.ul_tmp_gpu
        kl = self.kl_gpu
        el_sum = self.el_sum_gpu
        c_ul_tmps = [0, 0.5, 0.5, 1]
        c_uls = [1./6, 1./3, 1./3, 1./6]

        ul_prev[:] = ul             # memcpy_dtod
        for c_ul_tmp, c_ul in zip(c_ul_tmps, c_uls):
            update_pre_rk4(nn, nne, vf, c_ul_tmp, ul, ul_prev, ul_tmp, kl, el_sum, block=bs, grid=gs)
            update_ul_rk4(nn, ne, nne, dt, de, vf, c_ul, ul, ul_tmp, kl, el_sum, block=bs, grid=gs)


    def update(self):
        nn, ne, nne = self.nn, self.ne, self.nne
        dt, de, vf = self.dt, self.de, self.vf
        bs, gs = (256,1,1), (nn//256+1,1)
        ul, ul_prev = self.ul_gpu, self.ul_prev_gpu
        el_sum = self.el_sum_gpu
        c_uls = [[1, 0, 1], [3./4, 1./4, 1./4], [1./3, 2./3, 2./3]]

        ul_prev[:] = ul             # memcpy_dtod
        for c_ul1, c_ul2, c_ul3 in c_uls:
            update_pre_rk3(nn, nne, vf, ul, el_sum, block=bs, grid=gs)
            update_ul_rk3(nn, ne, nne, dt, de, vf, c_ul1, c_ul2, c_ul3, ul, ul_prev, el_sum, block=bs, grid=gs)
