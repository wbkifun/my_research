from __future__ import division
import h5py as h5
import numpy as np
import sys


def convergences(ne, nne, de, ws, ux0, ux):
    sles = [slice(i*nne, (i+1)*nne) for i in xrange(ne)]    
    l1, l2, l8, mass = 0, 0, 0, 0
    uxint0 = np.zeros(ne)
    uxint = np.zeros(ne)
    yy = np.zeros(ne)
    yy_norm = np.zeros(ne)

    for i, sle in enumerate(sles):
        uxint0[i] = 0.5 * de * sum( ws * ux0[sle] )
        uxint[i] = 0.5 * de * sum( ws * ux[sle] )
        yy[i] = uxint[i] - uxint0[i]
        yy_norm[i] = yy[i] / uxint0[i]

    for i in xrange(ne):
        l1 += np.abs(yy_norm[i])
        l2 += (yy_norm[i])**2
        mass += uxint[i]

    l2 = np.sqrt(l2)

    return l1, l2, l8, mass



if __name__ == '__main__':
    h5path = sys.argv[1]
    print h5path
    h5f = h5.File(h5path, 'r')

    ne = h5f.attrs['ne']
    nne = h5f.attrs['nne']
    de = h5f.attrs['de']
    tgap = h5f.attrs['tgap']

    ws = h5f['ws'].value
    ux0 = h5f['0'].value
    ux = h5f['%d' % tgap].value

    print convergences(ne, nne, de, ws, ux0, ux)
