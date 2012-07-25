#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys

psi_fn = sys.argv[1]
kpsi_fn = 'k' + psi_fn
psi = np.load(psi_fn)
kpsi = np.load(kpsi_fn)
nx, ny = psi.shape
snx = 1024
print psi.shape
print kpsi.shape
abs_kpsi = np.abs(kpsi)
print abs_kpsi.min(), abs_kpsi.max()
print abs_kpsi.argmin(), abs_kpsi.argmax()
kpsi_maxi = abs_kpsi.argmax()/ny
kpsi_maxj = abs_kpsi.argmax()%ny
print kpsi_maxi, kpsi_maxj
print abs_kpsi[kpsi_maxi, kpsi_maxj]
print abs_kpsi[:,0]
print abs_kpsi[:,1]

plt.ion()
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,2,1)
im1 = plt.imshow(np.abs(psi).T, origin='lower')
ax1.set_xlim(nx/2 - snx/2, nx/2 + snx/2)

ax2 = fig.add_subplot(2,2,2)
im2 = plt.imshow(abs_kpsi.T, origin='lower')
#ax2.set_xlim(kpsi_maxi - snx/2, kpsi_maxi + snx/2)

ax3 = fig.add_subplot(2,1,2)
l3, = plt.plot(abs_kpsi[:,1])
plt.show()
