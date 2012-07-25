#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


# initialize
nx = 1024
dx = 0.01
dt = 0.001
tmax = 1000
tgap = 1
x = np.arange(nx)
psi = np.zeros(nx, dtype=np.complex64)
psi.real[:] = gaussian(x[:], 20, 600)
psi[:] *= np.exp(1j * 0.5 * x[:])
vf = np.zeros(nx, dtype=np.float32)

# plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

line1, = ax1.plot(x, psi * psi.conj(), linestyle='None', marker='p', markersize=2)
line2, = ax2.plot(x, psi.real, color='red')
line3, = ax2.plot(x, psi.imag, color='blue')
ax1.set_xlim(0, nx)
ax2.set_xlim(0, nx)
ax2.set_ylim(-0.02, 0.02)

# time loop
k = np.fft.fftfreq(nx, dx)[:] * 2 * np.pi
lc = np.exp(- 0.25j * k[:]**2 * dt)
vc = np.exp(- 1j * vf[:] * dt)

sigma = 0.8
pc = (1 + 1j * sigma / k )**2
pc[0] = 0
for tstep in xrange(tmax):
	psi[:] = np.fft.ifft(lc[:] * np.fft.fft(psi[:]))
	psi[-300:] = np.fft.ifft(pc[:] * np.fft.fft(psi[:]))[-300:]
	psi[:] *= vc[:]
	psi[:] = np.fft.ifft(lc[:] * np.fft.fft(psi[:]))
	psi[-300:] = np.fft.ifft(pc[:] * np.fft.fft(psi[:]))[-300:]

	if tstep%tgap == 0:
		print "tstep = %d\r" % (tstep),
		sys.stdout.flush()

		line1.set_ydata(psi * psi.conj())
		line2.set_ydata(psi.real * 1.0000001)
		line3.set_ydata(psi.imag * 1.0000001)
		plt.draw()
