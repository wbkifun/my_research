#!/usr/bin/env python

from SSFM_gpu import *


s = SSFM(ndim=1, ns=1024*8, ds=0.01, dt=0.0002, h5save=False)
s.set_barrier(vmax=200, vwidth=40)
s.set_init_gaussian(sigma=20, x0=150)
s.h5_save_spec()

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

# x-space
from matplotlib.patches import Rectangle
gmax = np.abs(s.psi).max() 
bar = Rectangle((s.vx0*s.dx, 0), s.vwidth*s.dx, gmax*0.9, facecolor='green', alpha=0.1)
ax1.add_patch(bar)

x_slice = slice(s.sx0, s.sx1)
s.psi[:] = s.psi_gpu.get()
l1, = ax1.plot(s.x[x_slice], np.abs(s.psi[x_slice]), linewidth=2)
ax1.set_xlim(s.sx0*s.dx, s.sx1*s.dx)

# k-space
k_slice = slice(s.skx0, s.skx1)
s.plan.execute(s.psi_gpu)
s.psi[:] = s.psi_gpu.get()
s.plan.execute(s.psi_gpu, inverse=True)
kpsi0 = s.psi[k_slice].copy()
ax2.plot(s.kx[k_slice], np.abs(s.psi[k_slice]), color='black', linewidth=2)
l2, = ax2.plot(s.kx[k_slice], np.abs(s.psi[k_slice]), linewidth=2)
ax2.set_xlim(s.skx0*s.dkx, s.skx1*s.dkx)

# transmittance
# analytic
analytic_t = np.zeros(s.nx/2, dtype=np.float32)
L = s.vwidth * s.dx
kidx = (s.kx < s.k0).argmin()
kappa = np.sqrt(s.k0**2 - s.kx[:kidx]**2)
xi = 0.5 * (kappa / s.kx[:kidx] - s.kx[:kidx] / kappa)
kL = kappa[:] * L
analytic_t[:kidx] = 1. / (np.cosh(kL)**2 + xi**2 * np.sinh(kL)**2)
kappa = np.sqrt(s.kx[kidx:s.nx/2]**2 - s.k0**2)
xi = 0.5 * (kappa / s.kx[kidx:s.nx/2] + s.kx[kidx:s.nx/2] / kappa)
kL = kappa[:] * L
analytic_t[kidx:] = 1. / (np.cos(kL)**2 + xi**2 * np.sin(kL)**2)
ax3.plot(s.kx[k_slice], analytic_t[k_slice], linestyle='--', color='black', linewidth=2)

ax3.plot([s.k0, s.k0], [0, 1], linestyle='-.', color='black')
ax3.plot([s.k0, s.skx1], [1, 1], linestyle='-.', color='black')
l3, = ax3.plot(s.kx[k_slice], np.abs(s.psi[k_slice]/kpsi0)**2, linewidth=2)
ax3.set_xlim(s.k0-5, s.k0+10)
ax3.set_ylim(0, 1.1)


print 'dkx = ', s.dkx
# time loop
from datetime import datetime
t0 = datetime.now()
t1 = datetime.now()
s.pre_update()

tmax = 4000
tgap = 100
for tstep in xrange(1, tmax+1):
	s.update()

	if tstep%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d (%d %%)\r" % (t1-t0, tstep, tmax, float(tstep)/tmax*100),
		sys.stdout.flush()
		s.h5_save_data(tstep)

		s.psi[:] = s.psi_gpu.get()
		l1.set_ydata(np.abs(s.psi[x_slice]))
		s.plan.execute(s.psi_gpu)
		s.psi[:] = s.psi_gpu.get()
		s.plan.execute(s.psi_gpu, inverse=True)
		l2.set_ydata(np.abs(s.psi[k_slice]))
		l3.set_ydata(np.abs(s.psi[k_slice]/kpsi0)**2)
		plt.draw()

s.finalize()
