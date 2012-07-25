from __future__ import division
import numpy as np
import sys


# setup
wavelength = 248
x_unit = 2  # nm
period = int(wavelength * 2 / x_unit)
#nx = period
#tmax = period
nx = period * 7
tmax = period * 100
t0 = int(period * 9.75)
dtype = np.float32

dx = 1
dt = 0.5
frequency = (1 / wavelength) * x_unit
w_dt = (2 * np.pi * frequency) * dt

# allocation
ez, hy = [np.zeros(nx, dtype) for i in xrange(2)]
abc = {'e': np.zeros(2, dtype), 'h': np.zeros(2, dtype)}
save_e, save_h = [np.zeros(tmax, dtype) for i in xrange(2)]

# plot
'''
import matplotlib.pyplot as plt
plt.ioff()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot(np.arange(nx), ez, 'o-k', markersize=2)
ax.set_xlim(0, nx)
ax.set_ylim(-1.2, 1.2)
'''

# time loop
for tstep in xrange(tmax + t0):
    # update e
    ez[:-1] += 0.5 * (hy[1:] - hy[:-1])

    ez[-1] = abc['e'][-1]
    abc['e'][-1] = abc['e'][-2]
    abc['e'][-2] = ez[-2]

    ez[period] += np.sin(w_dt * tstep) #* ((1/np.pi) * np.arctan(100 * (tstep-50)) + 0.5)
    if tstep >= t0:
        save_e[tstep - t0] = ez[-period]
        save_h[tstep - t0] = hy[-period]

    # update h
    hy[1:] += 0.5 * (ez[1:] - ez[:-1])

    hy[0] = abc['h'][0]
    abc['h'][0] = abc['h'][1]
    abc['h'][1] = hy[1]

    # plot
    '''
    if tstep % 10 == 0:
        print "tstep= %d\r" % (tstep),
        sys.stdout.flush()
        line.set_ydata(ez[:])
        plt.draw()
    '''

np.savez('tfsf_%dnm_%dnm_%dtmax.npz' % (wavelength, x_unit, tmax), e=save_e, h=save_h)
#line.set_ydata(ez[:])
#plt.show()
