#!/usr/bin/env python

import h5py as h5
import matplotlib.pyplot as plt
import sys


path = sys.argv[1]
h5f = h5.File(path, 'r')
tmax = h5f.attrs['tmax']
tgap = h5f.attrs['tgap']
x4n = h5f['x4n'].value
ux = h5f['0'].value

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
imag0, = ax.plot(x4n, ux)
imag, = ax.plot(x4n, ux)
ax.set_xlim(-1, 1)
ax.set_ylim(-0.1, 1.1)
#plt.savefig('./png/000000.png')

for tstep in xrange(tgap, tmax+1, tgap):
    print('tstep= %d' % tstep)
    ux[:] = h5f['%d' % tstep].value
    imag.set_ydata(ux)
    plt.savefig('./png/%.6d.png' % tstep)
