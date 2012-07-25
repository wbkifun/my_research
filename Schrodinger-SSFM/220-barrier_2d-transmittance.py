#!/usr/bin/env python

from SSFM_gpu import *


s = SSFM(ndim=2, ns=(1024*8, 1024*4), ds=0.01, dt=0.0002, h5save=False)
s.set_barrier(vmax=200, vwidth=40) #, vhight=60)
s.set_init_gaussian(sigma=20, x0=150)
s.h5_save_spec()

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(10,6))
fig.subplots_adjust(left=0.06, right=0.83, top=0.95, hspace=0.3)
ax1 = fig.add_subplot(3,2,2)
ax2 = fig.add_subplot(3,2,4)
ax3 = fig.add_subplot(3,2,6)
ax1.set_xlabel('x')
ax2.set_xlabel('k')
ax3.set_xlabel('k')
ax3.set_ylabel('Transmittance (a.u.)')

ax4 = fig.add_subplot(1,2,1)
ax4.set_xlabel('x')
ax4.set_ylabel('y')

# x-space (2d)
from matplotlib.patches import Rectangle
bar2 = Rectangle((s.vx0*s.dx, s.vy0*s.dy), s.vwidth*s.dx, s.vhight*s.dy, fill=True, facecolor='white', alpha=0.2)
ax4.add_patch(bar2)

s.psi[:] = s.psi_gpu.get()
im1 = ax4.imshow(np.abs(s.psi[s.sx0:s.sx1,s.sy0:s.sy1]).T, cmap=plt.cm.hot, origin='lower', extent=[s.sx0*s.dx, s.sx1*s.dx, s.sy0*s.dy, s.sy1*s.dy])

# x-space
gmax = np.abs(s.psi).max() 
bar = Rectangle((s.vx0*s.dx, 0), s.vwidth*s.dx, gmax*0.9, facecolor='green', alpha=0.1)
ax1.add_patch(bar)

x_slice = slice(s.sx0, s.sx1)
xy_slice = (x_slice, 0)
l1, = ax1.plot(s.x[x_slice], np.abs(s.psi[xy_slice]), color='black', linewidth=2)
l12, = ax1.plot(s.x[x_slice], s.psi[xy_slice].real, color='red')
l13, = ax1.plot(s.x[x_slice], s.psi[xy_slice].imag, color='blue')
ax1.set_xlim(s.sx0*s.dx, s.sx1*s.dx)
ax1.set_ylim(-gmax, gmax)
ax1.set_yticklabels([])
ax1.legend([bar, l1, l12, l13], ['V', r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)


# k-space
k_slice = slice(s.skx0, s.skx1)
kxy_slice = (k_slice, 0)
s.plan.execute(s.psi_gpu)
s.psi[:] = s.psi_gpu.get()
s.plan.execute(s.psi_gpu, inverse=True)
gmax_k = np.abs(s.psi).max() 
kpsi0 = s.psi[kxy_slice].copy()
l20, = ax2.plot(s.kx[k_slice], np.abs(s.psi[kxy_slice]), color='green', linewidth=2)
l2, = ax2.plot(s.kx[k_slice], np.abs(s.psi[kxy_slice]), color='black', linewidth=2)
#l22, = ax2.plot(s.kx[k_slice], s.psi[kxy_slice].real, color='red', linewidth=2)
#l23, = ax2.plot(s.kx[k_slice], s.psi[kxy_slice].imag, color='blue', linewidth=2)
ax2.set_xlim(s.skx0*s.dkx, s.skx1*s.dkx)
#ax2.set_ylim(-gmax_k, gmax_k)
ax2.set_ylim(0, gmax_k)
ax2.set_yticklabels([])
#ax2.legend([l2, l22, l23], [r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)
ax2.legend([l20, l2], [r'$|\psi_0|$', r'$|\psi|$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

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
l31 = ax3.plot(s.kx[k_slice], analytic_t[k_slice], linestyle='--', color='black', linewidth=2)

ax3.plot([s.k0, s.k0], [0, 1], linestyle='-.', color='black')
ax3.plot([s.k0, s.skx1], [1, 1], linestyle='-.', color='black')
l3, = ax3.plot(s.kx[k_slice], np.abs(s.psi[kxy_slice]/kpsi0)**2, linewidth=2)
ax3.set_xlim(s.k0-5, s.k0+10)
ax3.set_ylim(0, 1.1)
ax3.legend([l31, l3], ['Analytic', 'Numeric'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)


print 'dkx = ', s.dkx
# time loop
from datetime import datetime
t0 = datetime.now()
t1 = datetime.now()
s.pre_update()

tmax = 4000
tgap = 20
for tstep in xrange(1, tmax+1):
	s.update()

	if tstep%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d (%d %%)\r" % (t1-t0, tstep, tmax, float(tstep)/tmax*100),
		sys.stdout.flush()
		s.h5_save_data(tstep)

		s.psi[:] = s.psi_gpu.get()
		im1.set_data(np.abs(s.psi[s.sx0:s.sx1,s.sy0:s.sy1]).T)

		l1.set_ydata(np.abs(s.psi[xy_slice]))
		l12.set_ydata(s.psi[xy_slice].real * 1.0000001)
		l13.set_ydata(s.psi[xy_slice].imag * 1.0000001)

		s.plan.execute(s.psi_gpu)
		s.psi[:] = s.psi_gpu.get()
		s.plan.execute(s.psi_gpu, inverse=True)
		l2.set_ydata(np.abs(s.psi[kxy_slice]))
		#l22.set_ydata(s.psi[kxy_slice].real)
		#l23.set_ydata(s.psi[kxy_slice].imag)

		l3.set_ydata(np.abs(s.psi[kxy_slice]/kpsi0)**2)

		plt.savefig('./png/%.5d.png' % tstep, dpi=150)
		#plt.draw()

s.finalize()
