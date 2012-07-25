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
snx = 1024	# sub_nx
nx = snx * 1
dx = 0.01
dt = 0.005
tmax = 200
tgap = 1
psi = np.zeros(nx, dtype=np.complex64)
kpsi = np.zeros(nx, dtype=np.complex64)
sx0 = nx/2 - snx/2
sx1 = nx/2 + snx/2

# initial wavefunction
x = np.arange(nx)
psi.real[:] = gaussian(x[:], sigma=20, x0=sx0+200)
#psi[:] *= np.exp(1j * 0.5 * x[:])

# potential
vf = 20 * ((x[:] - (nx / 2)) * dx)**2

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(8,6))
fig.subplots_adjust(left=0.03, right=0.8, hspace=0.3)
fig.suptitle('Harmonic Oscillator', fontsize=18)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.set_xlabel('x')
ax2.set_xlabel('k')

gmax = 0.02
sx = np.arange(sx0, sx1)
l0, = ax1.plot(x, vf * 2 * gmax / vf[sx0] - gmax, color='green', linewidth=2)
l1, = ax1.plot(x, np.abs(psi), color='black', linewidth=2)
l2, = ax1.plot(x, psi.real, color='red')
l3, = ax1.plot(x, psi.imag, color='blue')
ax1.set_xlim(sx0, sx1)
ax1.set_ylim(-gmax, gmax)
ax1.set_yticklabels([])
ax1.legend([l0, l1, l2, l3], ['V', r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

k = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)
k_shift = np.fft.fftshift(k)
kpsi[:] = np.fft.fft(psi)
kpsi_shift = np.fft.fftshift(kpsi)
l21, = ax2.plot(k_shift, np.abs(kpsi_shift), color='black', linewidth=2)
l22, = ax2.plot(k_shift, kpsi_shift.real, color='red')
l23, = ax2.plot(k_shift, kpsi_shift.imag, color='blue')
ax2.set_xlim(k_shift[nx/2 - 50], k_shift[nx/2 + 50])
ax2.set_ylim(-2.0, 2.0)
ax2.set_yticklabels([])
ax2.legend([l21, l22, l23], [r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

# time loop
k2 = k**2
lc = np.exp(- 0.5j * k2[:] * dt)
vc = np.exp(- 1j * vf[:] * dt)

kpsi[:] = np.fft.fft(psi[:])
psi[:] = np.fft.ifft(np.exp(- 0.25j * k2[:] * dt) * kpsi[:])
psi[:] = vc[:] * psi[:]
for tstep in xrange(tmax):
	kpsi[:] = np.fft.fft(psi[:])
	psi[:] = np.fft.ifft(lc[:] * kpsi[:])
	psi[:] = vc[:] * psi[:]

	if tstep%tgap == 0:
		print "tstep = %d\r" % (tstep),
		sys.stdout.flush()

		l1.set_ydata(np.abs(psi))
		l2.set_ydata(psi.real * 1.0000001)
		l3.set_ydata(psi.imag * 1.0000001)

		kpsi_shift = np.fft.fftshift(kpsi)
		l21.set_ydata(np.abs(kpsi_shift))
		l22.set_ydata(kpsi_shift.real * 1.0000001)
		l23.set_ydata(kpsi_shift.imag * 1.0000001)

		#plt.savefig('./png/%.5d.png' % tstep, dpi=150)
		plt.draw()
psi[:] = np.fft.ifft(np.exp(- 0.25j * k2[:] * dt) * kpsi[:])
