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
snx = sny = 1024	# sub_nx
nx = snx * 1
ny = sny * 1
dx = dy = 0.01
dt = 0.0001
tmax = 10
tgap = 1
sx0 = nx/2 - snx/2
sx1 = nx/2 + snx/2
sy0 = ny/2 - sny/2
sy1 = ny/2 + sny/2

x = np.arange(nx, dtype=np.float32) * dx
y = np.arange(ny, dtype=np.float32) * dy
kx = (np.fft.fftfreq(nx, dx) * 2 * np.pi)
ky = (np.fft.fftfreq(ny, dy) * 2 * np.pi)
kx_shift = np.fft.fftshift(kx)
ky_shift = np.fft.fftshift(ky)

# potential
vwidth = 70
vhight = ny
vmax = 200
vx0 = nx/2 - vwidth/2
vx1 = nx/2 + vwidth/2
vy0 = ny/2 - vhight/2
vy1 = ny/2 + vhight/2

# initial condition
sigma0 = 50 * dx
x00 = (nx/2 - 300) * dx
k0 = np.sqrt(2 * vmax)

# array allocation
psi = np.zeros((nx, ny), dtype=np.complex64)
lcx = np.zeros(nx, dtype=np.complex64)
lcy = np.zeros(ny, dtype=np.complex64)
lcx_sqrt = np.zeros(nx, dtype=np.complex64)
lcy_sqrt = np.zeros(nx, dtype=np.complex64)
vc = np.zeros(1, dtype=np.complex64)

psi.real[:] = gaussian(x, sigma=sigma0, x0=x00)[:,np.newaxis]
psi[:] *= np.exp(1j * k0 * x)[:,np.newaxis]
lcx[:] = np.exp(- 0.5j * kx**2 * dt)
lcy[:] = np.exp(- 0.5j * ky**2 * dt)
lcx_sqrt[:] = np.sqrt(lcx)
lcy_sqrt[:] = np.sqrt(lcx)
vc[0] = np.exp(- 1j * vmax * dt)

print lcx
print lcx_sqrt
# save to the h5 file
h5_path = './h5'
f = h5.File('%s/barrier-%d_%d-%d_%d_%d.h5' % (h5_path, nx, ny, vwidth, vhight, vmax), 'w')
f.attrs['nx'] = nx
f.attrs['ny'] = ny
f.attrs['dx'] = dx
f.attrs['dy'] = dy
f.attrs['dt'] = dt
f.attrs['tmax'] = tmax
f.attrs['snx'] = snx
f.attrs['sny'] = sny
f.attrs['sigma'] = sigma0
f.attrs['x0'] = x00
f.attrs['k0'] = k0
f.attrs['vx0'] = vx0
f.attrs['vx1'] = vx1
f.attrs['vy0'] = vy0
f.attrs['vy1'] = vy1
f.attrs['vmax'] = vmax

f.create_group('labels')
f['labels'].create_dataset('xlabels', data=x, compression='gzip')
f['labels'].create_dataset('ylabels', data=y, compression='gzip')
f['labels'].create_dataset('kxlabels', data=kx_shift, compression='gzip')
f['labels'].create_dataset('kylabels', data=ky_shift, compression='gzip')

f.create_group('data')

# cuda init
cuda.init()
ctx = cuda.Device(0).make_context()
strm = cuda.Stream()
plan = Plan((nx, ny), dtype=np.complex64, context=ctx, stream=strm)
psi_gpu = gpuarray.to_gpu(psi)
#lcx_gpu = gpuarray.to_gpu(lcx_sqrt)
lcy_gpu = gpuarray.to_gpu(lcy_sqrt)

kernels = '''
__constant__ float2 lcx[NX];
__constant__ float2 vc[1];

__global__ void lcf(float *psi, float *lcy) {
	int tx = threadIdx.x;
	int tid = blockIdx.x * blockDim.x + tx;
	int real = 2 * tx;
	int imag = 2 * tx + 1;

	__shared__ float spsi[512];
	__shared__ float slc[512];
	__shared__ float stmp[512];

	while ( tid < NX ) {
		int i = tid / NY;

		spsi[tx] = psi[tid];
		stmp[tx] = lcy[tid%NY2];
		spsi[tx+256] = psi[tid+256];
		stmp[tx+256] = lcy[(tid+256)%NY2];
		__syncthreads();

		slc[real] = lcx[i].x * stmp[real] - lcx[i].y * stmp[imag];
		slc[imag] = lcx[i].x * stmp[imag] + lcx[i].y * stmp[real];

		stmp[real] = slc[real] * spsi[real] - slc[imag] * spsi[imag];
		stmp[imag] = slc[real] * spsi[imag] + slc[imag] * spsi[real];

		psi[tid] = stmp[tx];
		psi[tid+256] = stmp[tx+256];
		__syncthreads();

		tid += gridDim.x * blockDim.x;
	}
}

/*
__global__ void lcf(float2 *psi, float2 *lcx, float2 *lcy) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = tid / NY;
	int j = tid % NY;
	//float2 rlcx, rlcy, rlc, rpsi;

	while ( tid < NXY ) {
		rlcx = lcx[i];
		rlcy = lcy[j]; 
		rpsi = psi[tid]; 

		rlc.x = rlcx.x * rlcy.x - rlcx.y * rlcy.y;
		rlc.y = rlcx.x * rlcy.y + rlcx.y * rlcy.x;

		//psi[tid].x = rlc.x * rpsi.x - rlc.y * rpsi.y;
		//psi[tid].y = rlc.x * rpsi.y + rlc.y * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}
*/

__global__ void vcf(float2 *psi) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + TID0;
	int j = tid % NY;
	float2 rvc, rpsi;

	while ( tid < TID_MAX && j >= VY0 && j < VY1 ) {
		rvc = vc[0]; 
		rpsi = psi[tid]; 

		psi[tid].x = rvc.x * rpsi.x - rvc.y * rpsi.y;
		psi[tid].y = rvc.x * rpsi.y + rvc.y * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}
'''.replace('NY2',str(ny*2)).replace('NXY',str(nx*ny)).replace('NX',str(nx)).replace('NY',str(ny)).replace('TID0',str(vx0*ny)).replace('TID_MAX',str(vx1*ny)).replace('VY0',str(vy0)).replace('VY1',str(vy1))
#print kernels
mod = SourceModule(kernels)
lcf = mod.get_function('lcf')
vcf = mod.get_function('vcf')
lcx_const, _ = mod.get_global('lcx')
vc_const, _ = mod.get_global('vc')

cuda.memcpy_htod(lcx_const, lcx_sqrt)
cuda.memcpy_htod(vc_const, vc)

tpb = 256
bpg1, bpg2 = 0, 0
for bpg in xrange(65535, 0, -1):
	if (nx * ny / tpb) % bpg == 0: bpg1 = bpg
	if (vwidth * ny / tpb) % bpg == 0: bpg2 = bpg
	if bpg1 * bpg2 != 0: break
print 'tpb = %d, bpg1 = %g, bpg2 = %g' % (tpb, bpg1, bpg2)

# save to the h5 file
f['data'].create_dataset('psi0', data=psi_gpu.get(), compression='gzip')
plan.execute(psi_gpu)
f['data'].create_dataset('kpsi0', data=np.fft.fftshift(psi_gpu.get()), compression='gzip')
plan.execute(psi_gpu, inverse=True)

# plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
im1 = ax1.imshow(np.abs(psi_gpu.get()).T, origin='lower')
ax1.set_xticklabels(x)
ax1.set_yticklabels(y)

# time loop
from datetime import datetime
t0 = datetime.now()
t1 = datetime.now()

plan.execute(psi_gpu)
lcf(psi_gpu, lcy_gpu, block=(tpb,1,1), grid=(bpg1,1))
print np.linalg.norm(lcx_sqrt - psi_gpu.get()[:,0])
plan.execute(psi_gpu, inverse=True)
#vcf(psi_gpu, block=(tpb,1,1), grid=(bpg2,1))

cuda.memcpy_htod(lcx_const, lcx)
#lcx_gpu.set(lcx)
lcy_gpu.set(lcy)

for tstep in xrange(1, tmax+1):
	plan.execute(psi_gpu)
	lcf(psi_gpu, lcy_gpu, block=(tpb,1,1), grid=(bpg1,1))
	plan.execute(psi_gpu, inverse=True)
	#vcf(psi_gpu, block=(tpb,1,1), grid=(bpg2,1))

	if tstep%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d (%d %%)\r" % (t1-t0, tstep, tmax, float(tstep)/tmax*100),
		sys.stdout.flush()

		psi[:] = psi_gpu.get()
		print np.linalg.norm(lcx - psi[:,0])
		im1.set_array(np.abs(psi).T)
		plt.draw()
		
#lcx_gpu.set(lcx_sqrt)
cuda.memcpy_htod(lcx_const, lcx_sqrt)
lcy_gpu.set(lcy_sqrt)
plan.execute(psi_gpu)
lcf(psi_gpu, lcy_gpu, block=(tpb,1,1), grid=(bpg1,1))
plan.execute(psi_gpu, inverse=True)

# save to the h5 file
f['data'].create_dataset('psi1', data=psi_gpu.get(), compression='gzip')
plan.execute(psi_gpu)
f['data'].create_dataset('kpsi1', data=np.fft.fftshift(psi_gpu.get()), compression='gzip')
plan.execute(psi_gpu, inverse=True)
f.close()

ctx.pop()
