#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


def update_lstep(psi, lcoeff):
	psi[:] = np.fft.ifft(lcoeff[:] * np.fft.fft(psi[:]))


def update_vstep(psi, vcoeff):
	psi[:] *= vcoeff[:]


if __name__ == '__main__':
	# initialize
	nx = 1024
	dx = 0.01
	dt = 0.001
	tmax = 1000
	tgap = 1
	x = np.arange(nx)
	psi = np.zeros(nx, dtype=np.complex64)
	psi.real[:] = gaussian(x[:], 20, 100)
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
	vcoeff = np.exp(- 1j * vf[:] * dt)
	k2 = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)**2
	lcoeff = np.exp(- 0.5j * k2[:] * dt)
	lcoeff_half = np.exp(- 0.25j * k2[:] * dt)

	update_lstep(psi, lcoeff_half)
	update_vstep(psi, vcoeff)
	for tstep in xrange(1, tmax):
		update_lstep(psi, lcoeff)
		update_vstep(psi, vcoeff)

		if tstep%tgap == 0:
			print "tstep = %d\r" % (tstep),
			sys.stdout.flush()

			line1.set_ydata(psi * psi.conj())
			line2.set_ydata(psi.real * 1.0000001)
			line3.set_ydata(psi.imag * 1.0000001)
			plt.draw()

	update_lstep(psi, lcoeff_half)