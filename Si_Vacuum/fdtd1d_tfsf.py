from __future__ import division
import numpy as np
import sys


# setup
wavelength = 248
x_unit = 4  # nm
period = int(wavelength * 2 / x_unit)
nx = period
tmax = period * 100
dtype = np.float32

# allocation
ez, hy = [np.zeros(nx, dtype) for i in xrange(2)]
abc = {'e': np.zeros(2, dtype), 'h': np.zeros(2, dtype)}

tfsf = np.load('tfsf_%dnm_%dnm_%dtmax.npz' % (wavelength, x_unit, tmax), 'r')

# plot
import matplotlib.pyplot as plt
plt.ioff()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line, = ax.plot(np.arange(nx), ez, 'o-k', markersize=2)
ax.set_xlim(0, nx)
ax.set_ylim(-1.2, 1.2)

# time loop
for tstep in xrange(tmax):
    # update e
    ez[:-1] += 0.5 * (hy[1:] - hy[:-1])

    ez[-1] = abc['e'][-1]
    abc['e'][-1] = abc['e'][-2]
    abc['e'][-2] = ez[-2]

    ez[50] -= 0.5 * tfsf['h'][tstep]

    # update h
    hy[1:] += 0.5 * (ez[1:] - ez[:-1])

    hy[0] = abc['h'][0]
    abc['h'][0] = abc['h'][1]
    abc['h'][1] = hy[1]

    hy[50] -= 0.5 * tfsf['e'][tstep]

    '''
    # plot
    if tstep % 10 == 0:
        print "tstep= %d\r" % (tstep),
        sys.stdout.flush()
        line.set_ydata(ez[:])
        plt.draw()
    '''

line.set_ydata(ez[:])
plt.show()
