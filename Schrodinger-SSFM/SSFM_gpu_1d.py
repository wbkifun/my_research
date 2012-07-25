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


kernels = '''
__constant__ float2 lcx[HNX];

__global__ void mul_l(float2 *psi) {
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

__global__ void mul_v(float2 *psi, float vc_real, float vc_imag) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + TID0;
	float2 rpsi;

	while ( tid < TID_MAX ) {
		rpsi = psi[tid]; 

		psi[tid].x = vc_real * rpsi.x - vc_imag * rpsi.y;
		psi[tid].y = vc_real * rpsi.y + vc_imag * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}'''


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


class SSFM:
	def __init__(s, ndim, nx, dx, dt):
		s.dt = dt
		s.ndim = ndim
		s.nx = nx
		s.dx = dx

		# sub area
		s.snx = 1024
		s.sx0 = s.nx/2 - s.snx/2
		s.sx1 = s.nx/2 + s.snx/2

		# dk
		s.dkx = 1. / (s.nx * s.dx) * 2 * np.pi

		# array allocations
		s.psi = np.zeros(s.nx, dtype=np.complex64)

		s.x = np.arange(s.nx, dtype=np.float32) * s.dx
		s.kx = (np.fft.fftfreq(s.nx, s.dx) * 2 * np.pi)
		s.lcx = np.zeros(s.nx, dtype=np.complex64)
		s.lcx_sqrt = np.zeros(s.nx, dtype=np.complex64)
		s.lcx[:] = np.exp(- 0.5j * s.kx**2 * dt)
		s.lcx_sqrt[:] = np.sqrt(s.lcx)

		s.x_slice = slice(s.sx0, s.sx1)

		#cuda
		cuda.init()
		s.ctx = cuda.Device(0).make_context()
		s.strm = cuda.Stream()
		s.plan = Plan(s.nx, dtype=np.complex64, context=s.ctx, stream=s.strm)
		s.psi_gpu = gpuarray.to_gpu(s.psi)

		s.tpb = 256
		s.bpg = 30 * 4
		print 'tpb = %d, bpg = %g' % (s.tpb, s.bpg)



	def finalize(s):
		s.ctx.pop()
		if s.h5save:
			s.h5f.close()


	def set_barrier(s, vmax, vwidth):
		s.vwidth = vwidth
		s.vmax = vmax
		s.vx0 = s.nx/2 - s.vwidth/2
		s.vx1 = s.nx/2 + s.vwidth/2
		s.vc = np.complex64( np.exp(- 1j * s.vmax * s.dt) )

		kern = kernels.replace('HNX',str(s.nx/2)).replace('NX',str(s.nx)).replace('TID0',str(s.vx0)).replace('TID_MAX',str(s.vx1))
		print kern
		mod = SourceModule(kern)
		s.mul_l = mod.get_function('mul_l')
		s.mul_v = mod.get_function('mul_v')
		s.lcx_const, _ = mod.get_global('lcx')


	def set_init_gaussian(s, sigma, x0, **kwargs):
		s.sigma = sigma * s.dx
		s.x0 = (s.nx/2 - x0) * s.dx
		if kwargs.has_key('k0'):
			s.k0 = kwargs['k0']
		else:
			s.k0 = np.sqrt(2 * s.vmax)

		s.psi.real[:] = gaussian(s.x, sigma=s.sigma, x0=s.x0)
		s.psi[:] *= np.exp(1j * s.k0 * s.x)
		s.psi_gpu.set(s.psi)

		s.skx0 = (s.k0 - 1./s.sigma*5)/ s.dkx 
		s.skx1 = (s.k0 + 1./s.sigma*5)/ s.dkx
		s.k_slice = slice(s.skx0, s.skx1)


	def pre_update(s):
		cuda.memcpy_htod(s.lcx_const, s.lcx_sqrt[1:s.nx/2+1])

		s.plan.execute(s.psi_gpu)
		s.mul_l(s.psi_gpu, block=(s.tpb,1,1), grid=(s.bpg,1))
		s.plan.execute(s.psi_gpu, inverse=True)

		cuda.memcpy_htod(s.lcx_const, s.lcx[1:s.nx/2+1])


	def update(s):
		s.mul_v(s.psi_gpu, s.vc.real, s.vc.imag, block=(s.tpb,1,1), grid=(s.bpg,1))
		s.plan.execute(s.psi_gpu)
		s.mul_l(s.psi_gpu, block=(s.tpb,1,1), grid=(s.bpg,1))
		s.plan.execute(s.psi_gpu, inverse=True)
