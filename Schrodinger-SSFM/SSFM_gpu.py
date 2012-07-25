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


class SSFM:
	def __init__(s, ndim, ns, ds, dt, h5save=False, **kwargs):
		s.dt = dt
		s.ndim = ndim
		if s.ndim == 1:
			s.nx = s.ns = ns
			s.dx = s.ds = ds
		elif s.ndim == 2:
			s.nx, s.ny = s.ns = ns
			if ds == list:
				s.dx, s.dy = s.ds = ds
			else:
				s.dx = s.dy = s.ds = ds
		s.h5save = h5save

		# sub area
		s.snx = 1024
		s.sx0 = s.nx/2 - s.snx/2
		s.sx1 = s.nx/2 + s.snx/2
		if ndim == 2:
			s.sny = 1024
			s.sy0 = s.ny/2 - s.sny/2
			s.sy1 = s.ny/2 + s.sny/2

		# dk
		s.dkx = 1. / (s.nx * s.dx) * 2 * np.pi
		if ndim == 2:
			s.dky = 1. / (s.ny * s.dy) * 2 * np.pi

		# array allocations
		s.psi = np.zeros(s.ns, dtype=np.complex64)

		s.x = np.arange(s.nx, dtype=np.float32) * s.dx
		s.kx = (np.fft.fftfreq(s.nx, s.dx) * 2 * np.pi)
		s.lcx = np.zeros(s.nx, dtype=np.complex64)
		s.lcx_sqrt = np.zeros(s.nx, dtype=np.complex64)
		s.lcx[:] = np.exp(- 0.5j * s.kx**2 * dt)
		s.lcx_sqrt[:] = np.sqrt(s.lcx)
		if ndim == 2:
			s.y = np.arange(s.ny, dtype=np.float32) * s.dy
			s.ky = (np.fft.fftfreq(s.ny, s.dy) * 2 * np.pi)
			s.lcy = np.zeros(s.ny, dtype=np.complex64)
			s.lcy_sqrt = np.zeros(s.ny, dtype=np.complex64)
			s.lcy[:] = np.exp(- 0.5j * s.ky**2 * dt)
			s.lcy_sqrt[:] = np.sqrt(s.lcy)

		# cuda
		cuda.init()
		s.ctx = cuda.Device(0).make_context()
		s.strm = cuda.Stream()
		s.plan = Plan(s.ns, dtype=np.complex64, context=s.ctx, stream=s.strm)
		s.psi_gpu = gpuarray.to_gpu(s.psi)
		if s.ndim == 2: 
			s.lcy_gpu = gpuarray.to_gpu(s.lcy)

		s.tpb = 256
		s.bpg = 30 * 4
		print 'tpb = %d, bpg = %g' % (s.tpb, s.bpg)


	def finalize(s):
		s.ctx.pop()
		if s.h5save:
			s.h5f.close()


	def set_barrier(s, vmax, vwidth, vhight=False):
		s.vwidth = vwidth
		s.vmax = vmax
		s.vx0 = s.nx/2 - s.vwidth/2
		s.vx1 = s.nx/2 + s.vwidth/2
		s.vc = np.complex64( np.exp(- 1j * s.vmax * s.dt) )

		if s.ndim == 2:
			if vhight:
				s.vhight = vhight
			else:
				s.vhight = s.ny
			s.vy0 = s.ny/2 - s.vhight/2
			s.vy1 = s.ny/2 + s.vhight/2
	
		if s.ndim == 1:
			kernels = '''
__global__ void mul_v(float2 *psi, float vc_real, float vc_imag) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + TID0;
	float2 rpsi;

	while ( tid < TID1 ) {
		rpsi = psi[tid]; 

		psi[tid].x = vc_real * rpsi.x - vc_imag * rpsi.y;
		psi[tid].y = vc_real * rpsi.y + vc_imag * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}

__constant__ float2 lcx[HNX];

__global__ void mul_l(float2 *psi) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float2 rpsi;

	while ( tid < NX ) {
		if ( tid <= HNX ) i = tid - 1;
		else i = NX - tid - 1;

		rpsi = psi[tid]; 

		psi[tid].x = lcx[i].x * rpsi.x - lcx[i].y * rpsi.y;
		psi[tid].y = lcx[i].x * rpsi.y + lcx[i].y * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}'''
			kern = kernels.replace('HNX',str(s.nx/2))\
					.replace('NX',str(s.nx))\
					.replace('TID0',str(s.vx0))\
					.replace('TID1',str(s.vx1))

			s.mul_l_args = [s.psi_gpu]

		elif s.ndim == 2:
			kernels = '''
__global__ void mul_v(float2 *psi, float vc_real, float vc_imag) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x + TID0;
	float2 rpsi;

	while ( tid < TID1 && tid%NY >= J0 && tid%NY < J1 ) {
		rpsi = psi[tid]; 

		psi[tid].x = vc_real * rpsi.x - vc_imag * rpsi.y;
		psi[tid].y = vc_real * rpsi.y + vc_imag * rpsi.x;

		tid += gridDim.x * blockDim.x;
	}
}

__constant__ float2 lcx[HNX];

__global__ void mul_l(float2 *psi, float2 *lcy) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	float2 rpsi, rpsi2, rlcy;

	while ( tid < NXY ) {
		if ( tid <= HNXY ) i = tid/NY - 1;
		else i = NX - tid/NY - 1;
		rlcy = lcy[tid%NY];
		rpsi = psi[tid]; 

		rpsi2.x = rlcy.x * rpsi.x - rlcy.y * rpsi.y;
		rpsi2.y = rlcy.x * rpsi.y + rlcy.y * rpsi.x;

		psi[tid].x = lcx[i].x * rpsi2.x - lcx[i].y * rpsi2.y;
		psi[tid].y = lcx[i].x * rpsi2.y + lcx[i].y * rpsi2.x;

		tid += gridDim.x * blockDim.x;
	}
}'''
			kern = kernels.replace('HNXY',str(s.nx*s.ny/2))\
					.replace('HNX',str(s.nx/2))\
					.replace('NXY',str(s.nx*s.ny))\
					.replace('NX',str(s.nx))\
					.replace('NY',str(s.ny))\
					.replace('TID0',str(s.vx0*s.ny))\
					.replace('TID1',str(s.vx1*s.ny))\
					.replace('J0',str(s.vy0))\
					.replace('J1',str(s.vy1))

			s.mul_l_args = [s.psi_gpu, s.lcy_gpu]

		print kern
		mod = SourceModule(kern)
		s.mul_v = mod.get_function('mul_v')
		s.lcx_const, _ = mod.get_global('lcx')
		s.mul_l = mod.get_function('mul_l')


	def set_init_gaussian(s, sigma, x0, **kwargs):
		s.sigma = sigma * s.dx
		s.x0 = (s.nx/2 - x0) * s.dx
		if kwargs.has_key('k0'):
			s.k0 = kwargs['k0']
		else:
			s.k0 = np.sqrt(2 * s.vmax)

		if s.ndim == 1:
			s.psi.real[:] = gaussian(s.x, sigma=s.sigma, x0=s.x0)
			s.psi[:] *= np.exp(1j * s.k0 * s.x)
		elif s.ndim == 2:
			s.psi.real[:] = gaussian(s.x, sigma=s.sigma, x0=s.x0)[:,np.newaxis]
			s.psi[:] *= np.exp(1j * s.k0 * s.x)[:,np.newaxis]
		s.psi_gpu.set(s.psi)

		s.skx0 = int( (s.k0 - 1./s.sigma*4)/ s.dkx )
		s.skx1 = int( (s.k0 + 1./s.sigma*4)/ s.dkx )
		if s.ndim == 2:
			s.sky0 = s.skx0
			s.sky1 = s.skx1
		

	def pre_update(s):
		cuda.memcpy_htod(s.lcx_const, s.lcx_sqrt[1:s.nx/2+1])
		if s.ndim == 2: 
			s.lcy_gpu.set(s.lcy_sqrt)

		s.plan.execute(s.psi_gpu)
		s.mul_l(*s.mul_l_args, block=(s.tpb,1,1), grid=(s.bpg,1))
		s.plan.execute(s.psi_gpu, inverse=True)

		cuda.memcpy_htod(s.lcx_const, s.lcx[1:s.nx/2+1])
		if s.ndim == 2: 
			s.lcy_gpu.set(s.lcy)


	def update(s):
		s.mul_v(s.psi_gpu, s.vc.real, s.vc.imag, block=(s.tpb,1,1), grid=(s.bpg,1))
		s.plan.execute(s.psi_gpu)
		s.mul_l(*s.mul_l_args, block=(s.tpb,1,1), grid=(s.bpg,1))
		s.plan.execute(s.psi_gpu, inverse=True)


	def h5_save_spec(s, path='./h5'):
		if s.h5save:
			if s.ndim == 1:
				s.h5f = h5.File('%s/barrier1d-%d-%d_%d.h5' % (path, s.nx, s.vwidth, s.vmax), 'w')
			elif s.ndim == 2:
				s.h5f = h5.File('%s/barrier2d-%d_%d-%d_%d_%d.h5' % (path, s.nx, s.ny, s.vwidth, s.vhight, s.vmax), 'w')

			s.h5f.attrs['nx'] = s.nx
			s.h5f.attrs['dx'] = s.dx
			s.h5f.attrs['dt'] = s.dt
			s.h5f.attrs['snx'] = s.snx
			s.h5f.attrs['sx0'] = s.sx0
			s.h5f.attrs['sx1'] = s.sx1
			s.h5f.attrs['sigma'] = s.sigma
			s.h5f.attrs['x0'] = s.x0
			s.h5f.attrs['k0'] = s.k0
			s.h5f.attrs['vx0'] = s.vx0
			s.h5f.attrs['vx1'] = s.vx1
			s.h5f.attrs['vmax'] = s.vmax
			s.h5f.attrs['dkx'] = s.dkx
			s.h5f.attrs['skx0'] = s.skx0
			s.h5f.attrs['skx1'] = s.skx1
			if s.ndim == 2:
				s.h5f.attrs['ny'] = s.ny
				s.h5f.attrs['dy'] = s.dy
				s.h5f.attrs['sny'] = s.sny
				s.h5f.attrs['sy0'] = s.sy0
				s.h5f.attrs['sy1'] = s.sy1
				s.h5f.attrs['vy0'] = s.vy0
				s.h5f.attrs['vy1'] = s.vy1
				s.h5f.attrs['dky'] = s.dky
				s.h5f.attrs['sky0'] = s.sky0
				s.h5f.attrs['sky1'] = s.sky1

			s.h5f.create_group('labels')
			s.h5f['labels'].create_dataset('x', data=s.x[s.sx0:s.sx1], compression='gzip')
			s.h5f['labels'].create_dataset('kx', data=s.kx[s.skx0:s.skx1], compression='gzip')
			if s.ndim == 2:
				s.h5f['labels'].create_dataset('y', data=s.x[s.sy0:s.sy1], compression='gzip')
				s.h5f['labels'].create_dataset('ky', data=s.ky[s.sky0:s.sky1], compression='gzip')

			s.h5f.create_group('data')
			s.h5_save_data(0)


	def h5_save_data(s, tstep):
		if s.h5save:
			f['data'].create_dataset('psi%.5d' % tstep, data=s.psi_gpu.get()[s.sx0:s.sx1], compression='gzip')
			s.plan.execute(s.psi_gpu)
			f['data'].create_dataset('kpsi%.5d' % tstep, data=s.psi_gpu.get()[s.skx0:s.skx1], compression='gzip')
			s.plan.execute(s.psi_gpu, inverse=True)
