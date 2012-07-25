from __future__ import division

import numpy as np
import sys


#nk_path = sys.argv[1]
#wavelength = float( sys.argv[2] )
nk_path = './n_k_si.txt'
wavelength = 248

f = open(nk_path, 'r')
wl, nl, kl = [], [], []
while True: 
    f.readline()
    lines = f.readlines(100000) 
    if not lines: 
        break 
    for line in lines: 
        els = map(float, line.split())
        wl.append(els[0])
        nl.append(els[1])
        kl.append(els[2])
f.close()

wa = np.array(wl)
na = np.array(nl)
ka = np.array(kl)

n0 = np.interp(wavelength, wa, na)
k0 = np.interp(wavelength, wa, ka)
print('wavelength\tn\tk')
print(wavelength, n0, k0)

import matplotlib.pyplot as plt
plt.plot(wa, na, 'x-r', label='n')
plt.plot(wa, ka, 'x-b', label='k')
plt.plot([wavelength], [n0], 'k', marker='o', markersize=8)
plt.plot([wavelength], [k0], 'k', marker='o', markersize=8)
plt.legend()
plt.show()
