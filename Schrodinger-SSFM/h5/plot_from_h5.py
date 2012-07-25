#!/usr/bin/env python

import numpy as np
import h5py as h5
import sys

filename = sys.argv[1]
f = h5.File(filename, 'r')
nx = f.attrs['nx']
ny = f.attrs['ny']
dx = f.attrs['dx']
dy = f.attrs['dy']
dt = f.attrs['dt']
tmax = f.attrs['tmax']
snx = f.attrs['snx']
sny = f.attrs['sny']
sigma = f.attrs['sigma']
x0 = f.attrs['x0'] 
k0 = f.attrs['k0'] 
vx0 = f.attrs['vx0']
vx1 = f.attrs['vx1']
vy0 = f.attrs['vy0']
vy1 = f.attrs['vy1']
vmax = f.attrs['vmax']

xlabels = f['labels']['xlabels'].value
ylabels = f['labels']['ylabels'].value
kxlabels = f['labels']['kxlabels'].value
kylabels = f['labels']['kylabels'].value

psi0 = f['data']['psi0'].value
kpsi0 = f['data']['kpsi0'].value
psi1 = f['data']['psi1'].value
kpsi1 = f['data']['kpsi1'].value


import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

im1 = ax1.imshow(np.abs(psi1).T, origin='lower')
ax1.set_xticklabels(xlabels)
ax1.set_yticklabels(ylabels)

im2 = ax2.imshow(np.abs(kpsi0).T, origin='lower')
ax2.set_xticklabels(kxlabels)
ax2.set_yticklabels(kylabels)

im3 = ax3.imshow(np.abs(psi1).T, origin='lower')
ax3.set_xticklabels(xlabels)
ax3.set_yticklabels(ylabels)

im4 = ax4.imshow(np.abs(kpsi1).T, origin='lower')
ax4.set_xticklabels(kxlabels)
ax4.set_yticklabels(kylabels)

print kxlabels.shape
print kylabels.shape
#print np.abs(psi0)
#print np.abs(psi1)
plt.show()
