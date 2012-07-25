import numpy as np
import pyopencl as cl
import sys

from kemp.fdtd3d.util import common_gpu
from kemp.fdtd3d.node import Fields, Core, IncidentDirect, GetFields, Pbc, Pml, ExchangeNode
from kemp.fdtd3d import gpu, cpu


nx, ny, nz = 101, 320, 2
tmax, tgap = 800, 10

# instances 
gpu_devices = common_gpu.gpu_device_list(print_info=False)
context = cl.Context(gpu_devices)
mainf_list = [gpu.Fields(context, device, nx, ny, nz) for device in gpu_devices]

fields = Fields(mainf_list)
Pbc(fields, 'z')
Pml(fields, ('+-', '+-', ''), npml=10)
ExchangeNode(fields)
Core(fields)
nx = fields.nx

tfunc = lambda tstep: 50 * np.sin(0.05 * tstep)
IncidentDirect(fields, 'ez', (0.6, 0.7, 0), (0.6, 0.7, -1), tfunc) 
getf = GetFields(fields, 'ez', (0, 0, 0.5), (-1, -1, 0.5))

print fields.instance_list

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure(figsize=(12,8))
for i in fields.accum_nx_list[1:]:
    plt.plot((i,i), (0,ny), color='k', linewidth=2)
imag = plt.imshow(np.zeros((nx, ny), fields.dtype).T, interpolation='nearest', origin='lower', vmin=-1.1, vmax=1.1)
plt.colorbar()

# main loop
from datetime import datetime
t0 = datetime.now()

for tstep in xrange(1, tmax+1):
    fields.update_e()
    fields.update_h()

    if tstep % tgap == 0:
        print('[%s] %d/%d (%d %%)\r' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100)),
        sys.stdout.flush()

        getf.wait()
        imag.set_array( getf.get_fields().T )
        #plt.savefig('./png/%.6d.png' % tstep)
        plt.draw()

plt.show()
print('\n[%s] %d/%d (%d %%)' % (datetime.now() - t0, tstep, tmax, float(tstep)/tmax*100))
print('')
