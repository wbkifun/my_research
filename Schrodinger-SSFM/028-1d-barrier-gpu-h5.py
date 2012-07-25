#!/usr/bin/env python

import sys
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pyfft.cuda import Plan
import h5py as h5

from scipy.constants import physical_constants
LENGTH = physical_constants['atomic unit of length']
TIME = physical_constants['atomic unit of time']
ENERGY = physical_constants['atomic unit of energy']


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


# spec
snx = 1024	# sub_nx
nx = snx * 16
dx = 0.01
dt = 0.0001
tmax = 5000
tgap = 100
sx0 = nx/2 - snx/2
sx1 = nx/2 + snx/2

x = np.arange(nx, dtype=np.float32) * dx
kx = (np.fft.fftfreq(nx, dx) * 2 * np.pi)
'''
kx3 = np.zeros_like(kx)
for i in xrange(nx):
	if i == 0: kx3[i] = 0
	elif i <= nx/2: kx3[i] = kx[i]
	else: kx3[i] = -kx[nx-i]
	print i, kx[i], kx3[i], kx[i] - kx3[i]
print np.linalg.norm(kx - kx3)
'''

# potential
vwidth = 70
vmax = 200
vx0 = nx/2 - vwidth/2
vx1 = nx/2 + vwidth/2

# initial condition
sigma0 = 50 * dx
x00 = (nx/2 - 300) * dx
k0 = np.sqrt(2 * vmax)

dk = 1. / (nx * dx) * 2 * np.pi
sk0 = (k0 - 1/sigma0*5)/ dk 
sk1 = (k0 + 1/sigma0*5)/ dk 

# array allocation
psi = np.zeros(nx, dtype=np.complex64)
lcx = np.zeros(nx, dtype=np.complex64)
lcx_sqrt = np.zeros(nx, dtype=np.complex64)

psi.real[:] = gaussian(x, sigma=sigma0, x0=x00)
psi[:] *= np.exp(1j * k0 * x)
lcx[:] = np.exp(- 0.5j * kx**2 * dt)
lcx_sqrt[:] = np.sqrt(lcx)
vc = np.complex64( np.exp(- 1j * vmax * dt) )

# save to the h5 file
h5_path = './h5'
f = h5.File('%s/barrier1d-%d-%d_%d.h5' % (h5_path, nx, vwidth, vmax), 'w')
f.attrs['nx'] = nx
f.attrs['dx'] = dx
f.attrs['dt'] = dt
f.attrs['tmax'] = tmax
f.attrs['snx'] = snx
f.attrs['sigma'] = sigma0
f.attrs['x0'] = x00
f.attrs['k0'] = k0
f.attrs['vx0'] = vx0
f.attrs['vx1'] = vx1
f.attrs['vmax'] = vmax
f.attrs['sx0'] = sx0
f.attrs['sx1'] = sx1
f.attrs['dk'] = dk
f.attrs['sk0'] = sk0
f.attrs['sk1'] = sk1

f.create_group('labels')
f['labels'].create_dataset('x', data=x[sx0:sx1], compression='gzip')
f['labels'].create_dataset('kx', data=kx[sk0:sk1], compression='gzip')

f.create_group('data')

# cuda init
cuda.init()
ctx = cuda.Device(0).make_context()
strm = cuda.Stream()
plan = Plan(nx, dtype=np.complex64, context=ctx, stream=strm)
psi_gpu = gpuarray.to_gpu(psi)

kernels = '''
__constant__ float2 lcx[HNX];

__global__ void lcf(float2 *psi) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float2 rpsi;

	while ( tid < NX && tid > 0) {
		if ( tid <= HNX ) i = tid - 1;
		else i = NX - tid - 1;

		rpsi = psi[tid]; 

		psi[tid].x = lcx[i].x * rpsi.x - lcx[i].y * rpsi.y;
		psi[tid].y = lcx[i].x * rpsi.y + lcx[i].y * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void vcf(float2 *psi, float vc_real, float vc_imag) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + TID0;
	float2 rpsi;

	while ( tid < TID_MAX ) {
		rpsi = psi[tid]; 

		psi[tid].x = vc_real * rpsi.x - vc_imag * rpsi.y;
		psi[tid].y = vc_real * rpsi.y + vc_imag * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}
'''.replace('HNX',str(nx/2)).replace('NX',str(nx)).replace('TID0',str(vx0)).replace('TID_MAX',str(vx1))
print kernels
mod = SourceModule(kernels)
lcf = mod.get_function('lcf')
vcf = mod.get_function('vcf')
lcx_const, _ = mod.get_global('lcx')
cuda.memcpy_htod(lcx_const, lcx_sqrt[1:nx/2+1])

tpb = 256
bpg = 30 * 4
print 'tpb = %d, bpg = %g' % (tpb, bpg)

# save to the h5 file

# plot & save to the h5 file
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

psi[:] = psi_gpu.get()
l1, = ax1.plot(x[sx0:sx1], np.abs(psi[sx0:sx1]))
ax1.set_xlim(sx0*dx, sx1*dx)
f['data'].create_dataset('psi00000', data=psi_gpu.get()[sx0:sx1], compression='gzip')

plan.execute(psi_gpu)
psi[:] = psi_gpu.get()
plan.execute(psi_gpu, inverse=True)
l2, = ax2.plot(kx[sk0:sk1], np.abs(psi[sk0:sk1]))
ax2.set_xlim(sk0*dk, sk1*dk)
f['data'].create_dataset('kpsi00000', data=psi_gpu.get()[sk0:sk1], compression='gzip')


# time loop
from datetime import datetime
t0 = datetime.now()
t1 = datetime.now()

plan.execute(psi_gpu)
lcf(psi_gpu, block=(tpb,1,1), grid=(bpg,1))
plan.execute(psi_gpu, inverse=True)
#vcf(psi_gpu, vc.real, vc.imag, block=(tpb,1,1), grid=(bpg,1))

cuda.memcpy_htod(lcx_const, lcx[1:nx/2+1])

for tstep in xrange(1, tmax+1):
	vcf(psi_gpu, vc.real, vc.imag, block=(tpb,1,1), grid=(bpg,1))
	plan.execute(psi_gpu)
	lcf(psi_gpu, block=(tpb,1,1), grid=(bpg,1))
	plan.execute(psi_gpu, inverse=True)

	if tstep%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d (%d %%)\r" % (t1-t0, tstep, tmax, float(tstep)/tmax*100),
		sys.stdout.flush()

		psi[:] = psi_gpu.get()
		l1.set_ydata(np.abs(psi[sx0:sx1]))
		f['data'].create_dataset('psi%.5d' % tstep, data=psi_gpu.get()[sx0:sx1], compression='gzip')

		plan.execute(psi_gpu)
		psi[:] = psi_gpu.get()
		plan.execute(psi_gpu, inverse=True)
		l2.set_ydata(np.abs(psi[sk0:sk1]))
		f['data'].create_dataset('kpsi%.5d' % tstep, data=psi_gpu.get()[sk0:sk1], compression='gzip')
		plt.draw()
		
f.close()
ctx.pop()
