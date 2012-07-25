#!/usr/bin/env python

import sys
import numpy as np
from scipy.constants import physical_constants
LENGTH = physical_constants['atomic unit of length']
TIME = physical_constants['atomic unit of time']
ENERGY = physical_constants['atomic unit of energy']


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


# initialize
snx = sny = 1024	# sub_nx
nx = ny = snx * 1
dx = dy = 0.01
dt = 0.0001
tmax = 20000
tgap = 100
psi = np.zeros((nx, ny), dtype=np.complex64)
kpsi = np.zeros((nx, ny), dtype=np.complex64)
sx0 = nx/2 - snx/2
sx1 = nx/2 + snx/2
sy0 = ny/2 - sny/2
sy1 = ny/2 + sny/2

# initial wavefunction
sigma0 = 50 * dx
k0 = 20
x = np.arange(nx) * dx
psi.real[:] = gaussian(x, sigma=sigma0, x0=(sx0+200)*dx)[:,np.newaxis]
psi[:] *= np.exp(1j * k0 * x)[:,np.newaxis]

# potential
vx0, vwidth = nx/2, 70
vmax = (k0 ** 2) / 2

# plot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ion()
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)
im1 = plt.imshow(np.abs(psi).T, cmap=plt.cm.hot, origin='lower')
#ax1.set_xlim(sx0*dx, sx1*dx)
#ax1.set_ylim(sy0*dy, sy1*dy)
#plt.colorbar()

'''
fig.subplots_adjust(left=0.08, right=0.78, hspace=0.3)
fig.suptitle('Finite Barrier', fontsize=18)
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.set_xlabel('x')
ax2.set_xlabel('k')
ax3.set_xlabel('k')
ax3.set_ylabel('Transmittance (a.u.)')

gmax = np.abs(psi).max() * 1.0
barr = Rectangle((vx0*dx, 0), vwidth*dx, gmax*0.9, facecolor='green', alpha=0.1)
ax1.add_patch(barr)
l1, = ax1.plot(x, np.abs(psi), color='black', linewidth=2)
l2, = ax1.plot(x, psi.real, color='red')
l3, = ax1.plot(x, psi.imag, color='blue')
ax1.set_xlim(sx0*dx, sx1*dx)
ax1.set_ylim(-gmax, gmax)
ax1.set_yticklabels([])
ax1.legend([barr, l1, l2, l3], ['V', r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

k = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)
k_shift = np.fft.fftshift(k)
kpsi[:] = np.fft.fft(psi)
kpsi_shift = np.fft.fftshift(kpsi)
gmax_k = np.abs(kpsi).max() * 1.0
l21, = ax2.plot(k_shift, np.abs(kpsi_shift), color='black', linewidth=2)
l22, = ax2.plot(k_shift, kpsi_shift.real, color='red')
l23, = ax2.plot(k_shift, kpsi_shift.imag, color='blue')
ax2.set_xlim(-(k0 + 1/sigma0 * 5), k0 + 1/sigma0 * 5)
ax2.set_ylim(-gmax_k, gmax_k)
ax2.set_yticklabels([])
ax2.legend([l21, l22, l23], [r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

# transmittance
analytic_t = np.zeros(nx, dtype=np.float32)
L = vwidth * dx
kidx = (k_shift < k0).argmin()
kappa = np.sqrt(k0**2 - k_shift[:kidx]**2)
xi = 0.5 * (kappa / k_shift[:kidx] - k_shift[:kidx] / kappa)
kL = kappa[:] * L
analytic_t[:kidx] = 1. / (np.cosh(kL)**2 + xi**2 * np.sinh(kL)**2)
kappa = np.sqrt(k_shift[kidx:]**2 - k0**2)
xi = 0.5 * (kappa / k_shift[kidx:] + k_shift[kidx:] / kappa)
kL = kappa[:] * L
analytic_t[kidx:] = 1. / (np.cos(kL)**2 + xi**2 * np.sin(kL)**2)

ax3.plot([k0, k0], [0, 1], linestyle='-.', color='black')
ax3.plot([k0, k_shift[-1]], [1, 1], linestyle='-.', color='black')
l30, = ax3.plot(k_shift, analytic_t, linestyle='--', color='black', linewidth=2)

kpsi_shift0 = kpsi_shift.copy()
l31, = ax3.plot(k_shift, (np.abs(kpsi_shift/kpsi_shift0))**2, color='blue', linewidth=2)
ax3.set_xlim(k0 - 1/sigma0*1, k0 + 1/sigma0*2)
ax3.set_ylim(0, 1.1)
ax3.legend([l30, l31], ['Analytic', 'Numeric'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)
'''

vwidth = 70
vhight = 100
vx0 = nx/2 - vwidth/2
vx1 = nx/2 + vwidth/2
vy0 = ny/2 - vhight/2
vy1 = ny/2 + vhight/2
v_slice = (slice(vx0, vx1), slice(vy0, vy1))

# time loop
from datetime import datetime
t0 = datetime.now()
t1 = datetime.now()

kx = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)
ky = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)
lcx = np.exp(- 0.5j * kx[:,np.newaxis]**2 * dt)
lcy = np.exp(- 0.5j * ky[np.newaxis,:]**2 * dt)
vc = np.exp(- 1j * vmax * dt)

kpsi[:] = np.fft.fft2(psi)
psi[:] = np.fft.ifft2(np.sqrt(lcx) * np.sqrt(lcy) * kpsi)
psi[v_slice] *= vc
for tstep in xrange(1, tmax+1):
	kpsi[:] = np.fft.fft2(psi)
	psi[:] = np.fft.ifft2(lcx * lcy * kpsi)
	psi[v_slice] *= vc

	if tstep%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d (%d %%)\r" % (t1-t0, tstep, tmax, float(tstep)/tmax*100),
		sys.stdout.flush()

		im1.set_array(np.abs(psi).T)
		'''
		l1.set_ydata(np.abs(psi))
		l2.set_ydata(psi.real * 1.0000001)
		l3.set_ydata(psi.imag * 1.0000001)

		kpsi_shift = np.fft.fftshift(kpsi)
		l21.set_ydata(np.abs(kpsi_shift))
		l22.set_ydata(kpsi_shift.real * 1.0000001)
		l23.set_ydata(kpsi_shift.imag * 1.0000001)
		l31.set_ydata((np.abs(kpsi_shift/kpsi_shift0))**2)
		'''

		#plt.savefig('./png/%.5d.png' % tstep, dpi=150)
		plt.draw()
psi[:] = np.fft.ifft2(np.sqrt(lcx) * np.sqrt(lcy) * kpsi)
