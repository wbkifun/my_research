from __future__ import division
from operator import itemgetter
from dg_modal_base import DGModalBase
import numpy as np



class DGModalCpu(DGModalBase):
    def __init__(self, ne, p_degree, velocity, **kwargs):
        super(DGModalCpu, self).__init__(ne, p_degree, velocity, **kwargs)


    def allocation(self):
        super(DGModalCpu, self).allocation()
        self.fl = np.zeros_like(self.ul)

        self.k1 = np.zeros_like(self.ul)
        self.k2 = np.zeros_like(self.ul)
        self.k3 = np.zeros_like(self.ul)
        self.k4 = np.zeros_like(self.ul)
        self.kl_tmp = np.zeros_like(self.ul)

        self.be = np.zeros(self.nne)
        self.ge = np.zeros(self.nne)
        self.sum_p = np.zeros(self.ne)
        self.sum_m = np.zeros(self.ne)

        # constant array
        self.me = np.array( [(2*k + 1) / self.de for k in xrange(self.nne)] )
        self.sign_k = np.array( [(-1)**k for k in xrange(self.nne)] )


    def prepare_update(self):
        pass


    def func(self, ul):
        fl, kl_tmp = self.fl, self.kl_tmp
        sum_p, sum_m = self.sum_p, self.sum_m
        ge, be, me = self.ge, self.be, self.me
        sign_k = self.sign_k
        vf, alpha = self.vf, self.alpha

        fl[:] = vf * ul[:]

        for eidx, sle in enumerate(self.sles):
            sum_p[eidx] = 0.5 * (fl[sle] + alpha*ul[sle]).sum()
            sum_m[eidx] = 0.5 * ( sign_k * (fl[sle] - alpha*ul[sle]) ).sum()

        for eidx, sle in enumerate(self.sles):
            sum_m_right = sum_m[0] if eidx==self.ne-1 else sum_m[eidx+1]
            sum_p_left = sum_p[-1] if eidx==0 else sum_p[eidx-1]
            ge[:] = sum_p[eidx] + sum_m_right - sign_k*(sum_p_left + sum_m[eidx])

            be[:] = 0
            for lidx in xrange(1, self.nne):
                for i in xrange((lidx-1)%2, lidx, 2):
                    be[lidx] += 2 * vf * ul[sle][i]

            kl_tmp[sle] = me * (be - ge)

        return kl_tmp


    def update_rk4(self):
        ul, k1, k2, k3, k4 = self.ul, self.k1, self.k2, self.k3, self.k4
        dt, func = self.dt, self.func

        k1[:] = dt * func(ul)
        k2[:] = dt * func(ul + 0.5*k1)
        k3[:] = dt * func(ul + 0.5*k2)
        k4[:] = dt * func(ul + k3)
        ul[:] = ul + (1/6) * (k1 + 2*k2 + 2*k3 + k4)


    def update_rk3(self):
        ul, k1, k2 = self.ul, self.k1, self.k2
        dt, func = self.dt, self.func

        k1[:] = ul + dt * func(ul)
        k2[:] = 3/4*ul + 1/4*k1 + 1/4*dt*func(k1)
        ul[:] = 1/3*ul + 2/3*k2 + 2/3*dt*func(k2)
