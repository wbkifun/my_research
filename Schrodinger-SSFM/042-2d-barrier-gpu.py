#!/usr/bin/env python

import sys
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pyfft.cuda import Plan

from scipy.constants import physical_constants
LENGTH = physical_constants['atomic unit of length']
TIME = physical_constants['atomic unit of time']
ENERGY = physical_constants['atomic unit of energy']


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


# initialize
snx = sny = 1024	# sub_nx
nx = snx * 1
ny = sny * 1
dx = dy = 0.01
dt = 0.0001
tmax = 5000
tgap = 100
psi = np.zeros((nx, ny), dtype=np.complex64)
#kpsi = np.zeros((nx, ny), dtype=np.complex64)
sx0 = nx/2 - snx/2
sx1 = nx/2 + snx/2
sy0 = ny/2 - sny/2
sy1 = ny/2 + sny/2

# initial wavefunction
sigma0 = 50 * dx
k0 = 20
x = np.arange(nx) * dx
y = np.arange(ny) * dy
psi.real[:] = gaussian(x, sigma=sigma0, x0=(sx0+200)*dx)[:,np.newaxis]
psi[:] *= np.exp(1j * k0 * x)[:,np.newaxis]

# cuda init
cuda.init()
ctx = cuda.Device(0).make_context()
strm = cuda.Stream()
plan = Plan((nx, ny), dtype=np.complex64, context=ctx, stream=strm)
psi_gpu = gpuarray.to_gpu(psi)
lcx_gpu = cuda.mem_alloc(nx * np.nbytes['complex64'])
lcy_gpu = cuda.mem_alloc(ny * np.nbytes['complex64'])

# potential
vx0, vwidth = nx/2, 70
vmax = (k0 ** 2) / 2

# plot
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ion()
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)
im1 = ax1.imshow(np.abs(psi).T, cmap=plt.cm.hot, origin='lower')
ax1.set_xlim(sx0, sx1)
#ax1.set_ylim(sy0*dy, sy1*dy)
#plt.colorbar()
'''

'''
fig.subplots_adjust(left=0.08, right=0.78, hspace=0.3)
fig.suptitle('Finite Barrier', fontsize=18)
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.set_xlabel('x')
ax2.set_xlabel('k')
ax3.set_xlabel('k')
ax3.set_ylabel('Transmittance (a.u.)')

gmax = np.abs(psi).max() * 1.0
barr = Rectangle((vx0*dx, 0), vwidth*dx, gmax*0.9, facecolor='green', alpha=0.1)
ax1.add_patch(barr)
l1, = ax1.plot(x, np.abs(psi), color='black', linewidth=2)
l2, = ax1.plot(x, psi.real, color='red')
l3, = ax1.plot(x, psi.imag, color='blue')
ax1.set_xlim(sx0*dx, sx1*dx)
ax1.set_ylim(-gmax, gmax)
ax1.set_yticklabels([])
ax1.legend([barr, l1, l2, l3], ['V', r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

k = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)
k_shift = np.fft.fftshift(k)
kpsi[:] = np.fft.fft(psi)
kpsi_shift = np.fft.fftshift(kpsi)
gmax_k = np.abs(kpsi).max() * 1.0
l21, = ax2.plot(k_shift, np.abs(kpsi_shift), color='black', linewidth=2)
l22, = ax2.plot(k_shift, kpsi_shift.real, color='red')
l23, = ax2.plot(k_shift, kpsi_shift.imag, color='blue')
ax2.set_xlim(-(k0 + 1/sigma0 * 5), k0 + 1/sigma0 * 5)
ax2.set_ylim(-gmax_k, gmax_k)
ax2.set_yticklabels([])
ax2.legend([l21, l22, l23], [r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

# transmittance
analytic_t = np.zeros(nx, dtype=np.float32)
L = vwidth * dx
kidx = (k_shift < k0).argmin()
kappa = np.sqrt(k0**2 - k_shift[:kidx]**2)
xi = 0.5 * (kappa / k_shift[:kidx] - k_shift[:kidx] / kappa)
kL = kappa[:] * L
analytic_t[:kidx] = 1. / (np.cosh(kL)**2 + xi**2 * np.sinh(kL)**2)
kappa = np.sqrt(k_shift[kidx:]**2 - k0**2)
xi = 0.5 * (kappa / k_shift[kidx:] + k_shift[kidx:] / kappa)
kL = kappa[:] * L
analytic_t[kidx:] = 1. / (np.cos(kL)**2 + xi**2 * np.sin(kL)**2)

ax3.plot([k0, k0], [0, 1], linestyle='-.', color='black')
ax3.plot([k0, k_shift[-1]], [1, 1], linestyle='-.', color='black')
l30, = ax3.plot(k_shift, analytic_t, linestyle='--', color='black', linewidth=2)

kpsi_shift0 = kpsi_shift.copy()
l31, = ax3.plot(k_shift, (np.abs(kpsi_shift/kpsi_shift0))**2, color='blue', linewidth=2)
ax3.set_xlim(k0 - 1/sigma0*1, k0 + 1/sigma0*2)
ax3.set_ylim(0, 1.1)
ax3.legend([l30, l31], ['Analytic', 'Numeric'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)
'''

vwidth = 70
vhight = 100
vx0 = nx/2 - vwidth/2
vx1 = nx/2 + vwidth/2
vy0 = ny/2 - vhight/2
vy1 = ny/2 + vhight/2
v_slice = (slice(vx0, vx1), slice(vy0, vy1))

kx = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi).astype(np.complex64)
ky = (np.fft.fftfreq(ny, dy)[:] * 2 * np.pi).astype(np.complex64)
#lcx = np.exp(- 0.5j * kx[:,np.newaxis]**2 * dt)
#lcy = np.exp(- 0.5j * ky[np.newaxis,:]**2 * dt)
lcx = np.exp(- 0.5j * kx**2 * dt)
lcy = np.exp(- 0.5j * ky**2 * dt)
lcx_sqrt = np.sqrt(lcx)
lcy_sqrt = np.sqrt(lcy)
vc = np.zeros(1, dtype=np.complex64)
vc[0] = np.exp(- 1j * vmax * dt)

kernels = '''
__constant__ float2 vc[1];

__global__ void lcf(float2 *psi, float2 *lcx, float2 *lcy) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i = tid / NY;
	int j = tid % NY;
	float2 rlcx, rlcy, rlc, rpsi;

	while ( tid < NXY ) {
		rlcx = lcx[i]; 
		rlcy = lcy[j]; 
		rpsi = psi[tid]; 

		rlc.x = rlcx.x * rlcy.x - rlcx.y * rlcy.y;
		rlc.y = rlcx.x * rlcy.y + rlcx.y * rlcy.x;

		psi[tid].x = rlc.x * rpsi.x - rlc.y * rpsi.y;
		psi[tid].y = rlc.x * rpsi.y + rlc.y * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void vcf(float2 *psi) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + TID0;
	int j = tid % NY;
	float2 rvc, rpsi;

	while ( tid < TID_MAX ) { //&& j >= VY0 && j < VY1 ) {
		rvc = vc[0]; 
		rpsi = psi[tid]; 

		psi[tid].x = rvc.x * rpsi.x - rvc.y * rpsi.y;
		psi[tid].y = rvc.x * rpsi.y + rvc.y * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}
'''.replace('NXY',str(nx*ny)).replace('NX',str(nx)).replace('NY',str(ny)).replace('TID0',str(vx0*ny)).replace('TID_MAX',str(vx1*ny)).replace('VY0',str(vy0)).replace('VY1',str(vy1))
print kernels
mod = SourceModule(kernels)
lcf = mod.get_function('lcf')
vcf = mod.get_function('vcf')
vc_const, _ = mod.get_global('vc')

tpb = 256
bpg1, bpg2 = 0, 0
for bpg in xrange(65535, 0, -1):
	if (nx * ny / tpb) % bpg == 0: bpg1 = bpg
	if (vwidth * ny / tpb) % bpg == 0: bpg2 = bpg
	if bpg1 * bpg2 != 0: break
print 'tpb = %d, bpg1 = %g, bpg2 = %g' % (tpb, bpg1, bpg2)

# time loop
from datetime import datetime
t0 = datetime.now()
t1 = datetime.now()

# save to the numpy files
psi[:] = psi_gpu.get()
np.save('./raw_data/psi-%d_%d-00000.npy' % (nx, ny), psi)

plan.execute(psi_gpu)
psi[:] = psi_gpu.get()
plan.execute(psi_gpu, inverse=True)
np.save('./raw_data/kpsi-%d_%d-00000.npy' % (nx, ny), psi)

'''
kpsi[:] = np.fft.fft2(psi)
psi[:] = np.fft.ifft2(lc_sqrt * kpsi)
psi[v_slice] *= vc
'''
cuda.memcpy_htod(vc_const, vc)
cuda.memcpy_htod(lcx_gpu, lcx_sqrt)
cuda.memcpy_htod(lcy_gpu, lcy_sqrt)

plan.execute(psi_gpu)
lcf(psi_gpu, lcx_gpu, lcy_gpu, block=(tpb,1,1), grid=(bpg1,1))
plan.execute(psi_gpu, inverse=True)
vcf(psi_gpu, block=(tpb,1,1), grid=(bpg2,1))

cuda.memcpy_htod(lcx_gpu, lcx)
cuda.memcpy_htod(lcy_gpu, lcy)

for tstep in xrange(1, tmax+1):
	'''
	kpsi[:] = np.fft.fft2(psi)
	psi[:] = np.fft.ifft2(lc * kpsi)
	psi[v_slice] *= vc
	'''
	plan.execute(psi_gpu)
	lcf(psi_gpu, lcx_gpu, lcy_gpu, block=(tpb,1,1), grid=(bpg1,1))
	plan.execute(psi_gpu, inverse=True)
	vcf(psi_gpu, block=(tpb,1,1), grid=(bpg2,1))

	if tstep%tgap == 0:
		t1 = datetime.now()
		print "[%s] %d/%d (%d %%)\r" % (t1-t0, tstep, tmax, float(tstep)/tmax*100),
		sys.stdout.flush()
		
		"""
		psi[:] = psi_gpu.get()
		im1.set_array(np.abs(psi).T)
		'''
		l1.set_ydata(np.abs(psi))
		l2.set_ydata(psi.real * 1.0000001)
		l3.set_ydata(psi.imag * 1.0000001)

		kpsi_shift = np.fft.fftshift(kpsi)
		l21.set_ydata(np.abs(kpsi_shift))
		l22.set_ydata(kpsi_shift.real * 1.0000001)
		l23.set_ydata(kpsi_shift.imag * 1.0000001)
		l31.set_ydata((np.abs(kpsi_shift/kpsi_shift0))**2)
		'''

		#plt.savefig('./png/%.5d.png' % tstep, dpi=150)
		plt.draw()
		"""

#psi[:] = np.fft.ifft2(np.sqrt(lcx) * np.sqrt(lcy) * kpsi)
cuda.memcpy_htod(lcx_gpu, lcx_sqrt)
cuda.memcpy_htod(lcy_gpu, lcy_sqrt)
plan.execute(psi_gpu)
lcf(psi_gpu, lcx_gpu, lcy_gpu, block=(tpb,1,1), grid=(bpg1,1))
plan.execute(psi_gpu, inverse=True)

'''
# save to the numpy files
psi[:] = psi_gpu.get()
np.save('./raw_data/psi-%.5d.npy' % tmax, psi)

plan.execute(psi_gpu)
psi[:] = psi_gpu.get()
plan.execute(psi_gpu, inverse=True)
np.save('./raw_data/kpsi-%.5d.npy' % tmax, psi)
'''

ctx.pop()
