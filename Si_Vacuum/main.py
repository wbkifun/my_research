from __future__ import division
import numpy as np
import pyopencl as cl
import sys

from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.node import Fields, Core, IncidentDirect, Pbc, Pml, GetFields, ExchangeNode
from kemp.fdtd3d.gpu import Drude
from kemp.fdtd3d import gpu


# setup (length unit: nm)
space = 1000, 300, 1000
x_unit = 2
wavelength = 248
geo_y = 100
geo_width = 60
source_y = 40
save_y = 30

nx, ny, nz = int(space[0]/x_unit), int(space[1]/x_unit), int(space[2]/x_unit)
period = int(wavelength * 2 / x_unit)
tmax = int(period * 10)
tgap = int(period / 16)
save_tstep = period * 7

# fdtd instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
snx = int(nx / len(gpu_devices) + 1)
mainf_list = [gpu.Fields(context, device, snx, ny, nz) for device in gpu_devices]

fields = Fields(mainf_list)
print('grid size: (%d, %d, %d)' % fields.ns)
print('single GPU: (%d, %d, %d)' % (snx, ny, nz))
Pbc(fields, 'xz')
Pml(fields, ('', '+-', ''), npml=10)
ExchangeNode(fields)

# incident source (TFSF)
tfsf = np.load('tfsf_%dnm_%dnm_%dtmax.npz' % (wavelength, x_unit, period*100), 'r')
tfunc_e = lambda tstep: 0.5 * tfsf['h'][tstep]
tfunc_h = lambda tstep: - 0.5 * tfsf['e'][tstep]
src_j = int(source_y / x_unit)
IncidentDirect(fields, 'ez', (0, src_j, 0), (-1, src_j, -1), tfunc_e) 
IncidentDirect(fields, 'hx', (0, src_j, 0), (-1, src_j, -1), tfunc_h) 

# geometry (Drude)
gj = int(geo_y / x_unit)
gw = int(geo_width / x_unit)
gw2 = int(gw / 2)
for gpu_id, gpuf in enumerate(fields.mainf_list):
    masks = [np.ones((snx, ny-gj, nz), np.bool) for i in xrange(3)]
    if gpu_id == 0:
        masks[0][-gw2:, :gw, nz/2-gw2:nz/2+gw2] = False
        masks[1][-gw2:, :gw, nz/2-gw2:nz/2+gw2] = False
        masks[2][-gw2:, :gw, nz/2-gw2:nz/2+gw2] = False
    elif gpu_id == 1:
        masks[0][:gw2, :gw, nz/2-gw2:nz/2+gw2] = False
        masks[1][:gw2, :gw, nz/2-gw2:nz/2+gw2] = False
        masks[2][:gw2, :gw, nz/2-gw2:nz/2+gw2] = False
    else:
        sys.exit()
    Drude(gpuf, (0, gj, 0), (-1, -1, -1), ep_inf=np.inf, drude_freq=0, gamma=0, mask_arrays=masks)

# fdtd core instance    
Core(fields)

# save fields
save_j = int(save_y / x_unit)
getf_sm = GetFields(fields, ['ex', 'ey', 'ez'], (0, save_j, 0), (-1, save_j, -1), process='square_sum')
getf_plot = GetFields(fields, 'ez', (0, 0, 0.5), (-1, -1, 0.5))

'''
# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(15,6))
# plot divided fdtd space
for i in fields.accum_nx_list[1:]:
    plt.plot((i,i), (0,ny), ':k')
# plot geometry
geo_pts = [(0,gj), (nx/2-gw2,gj), (nx/2-gw2,gj+gw), (nx/2+gw2,gj+gw), (nx/2+gw2,gj), (nx,gj)]
for (i0, j0), (i1, j1) in zip(geo_pts[:-1], geo_pts[1:]):
    plt.plot((i0,i1), (j0,j1), '-k')

imag = plt.imshow(np.zeros((fields.nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.colorbar()
'''


# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    if tstep > save_tstep:
        getf_sm.enqueue_kernel()

    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf_plot.wait()
        '''
        imag.set_array( getf_plot.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()
        '''

print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
print('')

# reflectance
getf_sm.wait(exec_kernel=False)
ex_r = getf_sm.get_fields('ex')
ey_r = getf_sm.get_fields('ey')
ez_r = getf_sm.get_fields('ez')
reflect = (ex_r + ey_r + ez_r).mean()
incident = ( tfsf['e'][save_tstep:tmax+1]**2 ).sum()
print('reflectance = %1.4f' % (reflect/incident))
