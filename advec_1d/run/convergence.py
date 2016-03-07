#!/usr/bin/env python

import h5py as h5
import sys
from advec_1d import convergences


path = sys.argv[1]
h5f = h5.File(path, 'r')
ne = h5f.attrs['ne']
nne = h5f.attrs['nne']
de = h5f.attrs['de']
ws = h5f['ws'].value
ux0 = h5f['0'].value

tgap = h5f.attrs['tgap']
ux = h5f['%d' % tgap].value

l1, l2, l8, mass = convergences(ne, nne, de, ws, ux0, ux)
print('l1= %g, l2= %g, l8= %g, mass= %g' % (l1, l2, l8, mass))
