#===============================================================================
# DG method with modal basis functions
# 1-D advection equation
# ------------------------------------------------------------------------------
# 
# Last update: 2012.4.26
# ------------------------------------------------------------------------------
# 
# <Description>
#  - basis function: Legendre polynomial
#  - boundary condition: periodic
#  - initial condition: Gaussian shape
#  - numerical integration: Gauss quadrature (Gauss-lubatto rules)
#  - time evolution: 4th-order Runge-Kutta 
#  - Legendre polynomial calculations: Numpy module (numpy.polynomial.legendre) 
#
# <Variables>
#  - ux     solution u(x) at t in physical domain
#  - ul     spectral components of u(x) in Legendre polynomial space
#  - fl     spectral components of f(u), f=vu is used
#  - v      fluid velociy
#  - ne     # of elements
#  - nne    # of gaussian quadrature nodes in a element
#  - nn     # of total nodes
#  - x4n    global coordinates for each nodes
#  - sle    slice indices in a element
#  - sles   list of sle s
#
# <History>
#  2012.4.26 Class inheritance  by Ki-Hwan Kim
#            Reduce number of kernels (4 -> 2)
#  2012.4.25 fix dx -> de  by Ki-Hwan Kim
#  2012.4.24 CUDA version  by Ki-Hwan Kim
#  2012.4.14 Convert to object-oriented  by Ki-Hwan Kim
#  2012.4.13 Rewriten using Python  by Ki-Hwan Kim
#  2012.3.27 Matlab code  by Shin-Hoo Kang
#===============================================================================

from __future__ import division
from dg_modal_base import DGModalBase
import numpy as np
import pycuda.driver as cuda



class DGModalGpu(DGModalBase):
    def __init__(self, ne, p_degree, cfl=0.1, v=0.5, target_gpu=0):
        cuda.init()
        self.dev = cuda.Device(target_gpu)
        self.ctx = self.dev.make_context()

        import atexit
        atexit.register(self.ctx.pop)

        super(DGModalGpu, self).__init__(ne, p_degree, cfl, v)


    def allocation(self):
        super(DGModalGpu, self).allocation()
        self.ul_gpu = cuda.to_device(self.ul)
        self.ul_prev_gpu = cuda.to_device(self.ul)
        self.ul_tmp_gpu = cuda.to_device(self.ul)
        self.kl_gpu = cuda.to_device(self.ul)
        self.el_sum_gpu = cuda.to_device(np.zeros(self.ne))


    def x2l(self):
        super(DGModalGpu, self).x2l()
        cuda.memcpy_htod(self.ul_gpu, self.ul)


    def l2x(self):
        cuda.memcpy_dtoh(self.ul, self.ul_gpu)
        super(DGModalGpu, self).l2x()


    def prepare_update(self):
        from pycuda.compiler import SourceModule
        import os
        src_path = '/'.join( os.path.abspath(__file__).split('/')[:-1] )
        kernels = open(src_path + '/core.cu').read()
        mod = SourceModule(kernels)
        #mod = cuda.module_from_file('core.cubin')
        self.update_pre = mod.get_function('update_pre')
        self.update_ul = mod.get_function('update_ul')


    def update(self):
        nn, ne, nne = np.int32([self.nn, self.ne, self.nne])
        dt, de, vf = np.float64([self.dt, self.de, self.vf])
        bs, gs = (256,1,1), (self.nn//256+1,1)
        ul, ul_prev, ul_tmp = self.ul_gpu, self.ul_prev_gpu, self.ul_tmp_gpu
        kl = self.kl_gpu
        el_sum = self.el_sum_gpu
        c_ul_tmps = np.float32([0, 0.5, 0.5, 1])
        c_uls = np.float32([1./6, 1./3, 1./3, 1./6])

        cuda.memcpy_dtod(ul_prev, ul, self.ul.nbytes)
        for c_ul_tmp, c_ul in zip(c_ul_tmps, c_uls):
            self.update_pre(nn, nne, vf, c_ul_tmp, ul, ul_prev, ul_tmp, kl, el_sum, block=bs, grid=gs)
            self.update_ul(nn, ne, nne, dt, de, vf, c_ul, ul, ul_tmp, kl, el_sum, block=bs, grid=gs)
