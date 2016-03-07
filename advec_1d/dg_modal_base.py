#===============================================================================
# DG method with modal basis functions
# 1-D advection equation
# ------------------------------------------------------------------------------
# 
# Last update: 2012.5.8
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
#  2012.5.8  Add the update_rk3 method for TVD-RK3
#  2012.5.2  Add the run_timeloop method
#  2012.4.27 Save to HDF5
#  2012.4.26 Class inheritance  by Ki-Hwan Kim
#            Reduce number of kernels (4 -> 2)
#  2012.4.25 fix dx -> de  by Ki-Hwan Kim
#  2012.4.24 CUDA version  by Ki-Hwan Kim
#  2012.4.14 Convert to object-oriented  by Ki-Hwan Kim
#  2012.4.13 Rewriten using Python  by Ki-Hwan Kim
#  2012.3.27 Matlab code  by Shin-Hoo Kang
#===============================================================================

from __future__ import division
import numpy as np
import numpy.polynomial.legendre as lgd

import gauss_quad as gq


class DGModalBase(object):
    def __init__(self, ne, p_degree, velocity, **kwargs):
        self.ne = ne
        self.p_degree = p_degree
        self.vf = velocity

        # alpha coefficient for the Lax-Friedrich numerical flux
        self.alpha = kwargs.get('alpha', self.vf)

        # discretization
        self.xmin, self.xmax = -1, 1
        self.nne = self.p_degree + 1
        self.nn = self.ne * self.nne
        self.x4n = np.zeros(self.nn)
        self.sles = [slice(i*self.nne, (i+1)*self.nne) for i in xrange(self.ne)]    

        length = self.xmax - self.xmin
        self.de = length / self.ne
        self.xs, self.ws = gq.weights_roots(self.p_degree)
        for sle, xe0 in zip(self.sles, np.arange(self.xmin, self.xmax, self.de)):
            self.x4n[sle] = 0.5 * (self.de*self.xs[:] + 2*xe0 + self.de)

        self.dx = np.min( np.diff(self.x4n[:self.nne]) )

        # cfl, dt
        if kwargs.has_key('cfl') and kwargs.has_key('dt'):
            raise ValueError("'cfl' and 'dt' are assigned at the same time")
        elif kwargs.has_key('cfl'):
            self.cfl = kwargs['cfl']
            self.tprd = int( np.ceil( length / (self.cfl * self.dx) ) )
        elif kwargs.has_key('dt'):
            self.dt = kwargs['dt']
            self.tprd = int( np.ceil( length / (self.dt * self.vf) ) )
        else:
            raise ValueError("You should assign the 'cfl' or 'dt'")

        self.cfl = length / (self.tprd * self.dx)
        self.dt = self.cfl * self.dx / self.vf

        # select the update function
        self.rk_order = kwargs.get('rk_order', 4)
        self.update = getattr(self, 'update_rk%d' % self.rk_order)

        # etc
        self.allocation()
        self.prepare_update()


    def print_info(self):
        print('xmin= %g, xmax= %g' % (self.xmin, self.xmax))
        print('ne= %d, p_degree= %d' % (self.ne, self.p_degree))
        print('nne= %d, nn= %d' % (self.nne, self.nn))
        print('dx= %5f, dt= %5f, cfl= %5f, v= %g' % \
                (self.dx, self.dt, self.cfl, self.vf))
        print('RK order= %d' % (self.rk_order))
        print('Lax-Friedrich alpha= %g' % (self.alpha))


    def allocation(self):
        self.ux = np.zeros(self.nn)
        self.ul = np.zeros_like(self.ux)


    def x2l(self):
        for sle in self.sles:
            self.ul[sle] = lgd.legfit(self.xs, self.ux[sle], self.p_degree)


    def l2x(self):
        for sle in self.sles:
            self.ux[sle] = lgd.legval(self.xs, self.ul[sle])


    def set_f0(self, f0):
        self.ux[:] = f0(self.x4n)
        self.x2l()


    def prepare_update(self):
        pass


    def update_rk4(self):
        pass


    def update_rk3(self):
        pass


    def prepare_save_h5(self, tmax, tgap, save_dir='./'):
        import atexit
        import datetime
        import h5py as h5

        self.tmax, self.tgap = tmax, tgap
        attr_list = ['xmin', 'xmax', 'ne', 'p_degree', 'nne', 'nn', \
                'dx', 'dt', 'de', 'cfl', 'vf', 'tprd', 'tmax', 'tgap']

        now = str( datetime.datetime.now() )
        fnow = now[:now.index('.')].replace(' ', '_')
        save_dir = save_dir if save_dir.endswith('/') else save_dir + '/'
        path = save_dir + fnow + '.h5'
        self.h5f = h5.File(path, 'w')
        atexit.register(self.h5f.close)

        for attr in attr_list:
            self.h5f.attrs[attr] = getattr(self, attr)
        self.h5f.create_dataset('xs', data=self.xs, compression='gzip')
        self.h5f.create_dataset('ws', data=self.ws, compression='gzip')
        self.h5f.create_dataset('x4n', data=self.x4n, compression='gzip')
        self.h5f.create_dataset('0', data=self.ux, compression='gzip')


    def save_h5(self, tstep):
        self.l2x()
        self.h5f.create_dataset('%s' % tstep, data=self.ux, compression='gzip')


    def strf_pps(self, val):
        if val >= 1e12: ret = val/1e12, 'T'
        elif val >= 1e9: ret = val/1e9, 'G'
        elif val >= 1e6: ret = val/1e6, 'M'
        elif val >= 1e3: ret = val/1e3, 'K'
        else: ret = val, ''

        return '%1.2f %spoint/s' % ret


    def run_timeloop(self, tmax, tgap, save_dir='./', print_info=True):
        from datetime import datetime
        import sys

        if print_info:
            print(self.__class__)
            self.print_info()
            print('tprd= %d, tmax= %d, tgap= %d\n' % (self.tprd, tmax, tgap))

        self.prepare_save_h5(tmax, tgap, save_dir)
        t0 = datetime.now()
        t1 = datetime.now()
        for tstep in xrange(1, tmax+1):
            self.update()

            if aaa=3:
                alkdfja
                sdkfja
            if tstep%tgap == 0:
                self.save_h5(tstep)
                elapsed = datetime.now() - t0

                if print_info and tstep < tmax:
                    interval = datetime.now() - t1
                    remain = interval * ((tmax - tstep) // tgap)
                    pps = self.nn * tgap / interval.total_seconds()
                    print('[%s] [%s] %d/%d (%d %%) (%s)\r' \
                            % (elapsed, remain, tstep, tmax, tstep/tmax*100, \
                            self.strf_pps(pps))),
                    sys.stdout.flush()
                    t1 = datetime.now()   

        elapsed_seconds = elapsed.total_seconds()
        pps_total = self.nn * tmax / elapsed_seconds
        print('\n[%s] %s' % (elapsed, self.strf_pps(pps_total)))
        self.h5f.attrs['elapsed_time'] = elapsed_seconds
        self.h5f.attrs['point/s'] = pps_total



if __name__ == '__main__':
    space = DGModalBase(ne=10, p_degree=4, velocity=0.5, cfl=0.01)
    space.set_f0( lambda x: np.exp( -x**2 / 0.2 ** 2) )   # Gaussian

    tmax, tgap = space.tprd*100, space.tprd
    space.prepare_save_h5(tmax, tgap)

    space.print_info()
    ux0 = space.ux.copy()
    space.l2x()

    import matplotlib.pyplot as plt
    axs = [plt.subplot(2,2,i+1) for i in xrange(4)]
    titles = ['ux0', 'ux -> ul', 'ul -> ux', 'ux - ux0']
    uus = [ux0, space.ul, space.ux, space.ux - ux0]
    xxs = [space.x4n, range(space.nn), space.x4n, space.x4n]

    for ax, title, uu, xx in zip(axs, titles, uus, xxs):
        ax.set_title(title)
        ax.plot(xx, uu)

    plt.savefig('ux0_ul.png')
