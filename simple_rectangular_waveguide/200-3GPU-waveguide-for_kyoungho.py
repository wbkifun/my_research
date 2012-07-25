#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cuda.init()
ngpu = cuda.Device.count()


class Fdtd3DGpu:
	def __init__(s, nx, ny, nz):
		s.nx, s.ny, s.nz = nx, ny, nz
		s.Dx, s.Dy = 32, 16
		s.rank = comm.Get_rank()

		if s.nz%32 != 0:
			print "Error: nz is not multiple of %d" % s.Dx
			sys.exit()

		print 'rank= %d, (%d, %d, %d)' % (s.rank, s.nx, s.ny, s.nz),
		total_bytes = s.nx*s.ny*s.nz*np.nbytes['float32']*9
		if total_bytes/(1024**3) == 0:
			print '%d MB' % ( total_bytes/(1024**2) )
		else:
			print '%1.2f GB' % ( float(total_bytes)/(1024**3) )

		s.dev = cuda.Device(s.rank)
		s.ctx = s.dev.make_context()
		s.MAX_BLOCK = s.dev.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)


	def finalize(s):
		s.ctx.pop()


	def alloc_eh_fields(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.ex_gpu = cuda.to_device(f)
		s.ey_gpu = cuda.to_device(f)
		s.ez_gpu = cuda.to_device(f)
		s.hx_gpu = cuda.to_device(f)
		s.hy_gpu = cuda.to_device(f)
		s.hz_gpu = cuda.to_device(f)
		s.eh_fields = [s.ex_gpu, s.ey_gpu, s.ez_gpu, s.hx_gpu, s.hy_gpu, s.hz_gpu]


	def alloc_coeff_arrays(s):
		f = np.zeros((s.nx, s.ny, s.nz), 'f')
		s.cex = np.ones_like(f)*0.5
		s.cex[:,-1,:] = 0
		s.cex[:,:,-1] = 0
		s.cey = np.ones_like(f)*0.5
		s.cey[:,:,-1] = 0
		s.cey[-1,:,:] = 0
		s.cez = np.ones_like(f)*0.5
		s.cez[-1,:,:] = 0
		s.cez[:,-1,:] = 0


	def alloc_exchange_boundaries(s):
		s.ey_tmp = cuda.pagelocked_zeros((s.ny,s.nz),'f')
		s.ez_tmp = cuda.pagelocked_zeros_like(s.ey_tmp)
		s.hy_tmp = cuda.pagelocked_zeros_like(s.ey_tmp)
		s.hz_tmp = cuda.pagelocked_zeros_like(s.ey_tmp)


	def prepare_functions(s):
		s.cex_gpu = cuda.to_device(s.cex)
		s.cey_gpu = cuda.to_device(s.cey)
		s.cez_gpu = cuda.to_device(s.cez)
		s.ce_fields = [s.cex_gpu, s.cey_gpu, s.cez_gpu]

		# Constant variables (nx, ny, nz, nyz, Dx, Dy) are replaced by python string processing.
		kernels ="""
__global__ void update_h(int by0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y + by0;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty+1][tx+1] = ex[idx];
	sy[ty][tx+1] = ey[idx];
	sz[ty+1][tx] = ez[idx];
	if( tx == 0 && k > 0 ) {
		sx[ty+1][0] = ex[idx-1];
		sy[ty][0] = ey[idx-1];
	}
	if( ty == 0 && j > 0 ) {
		sx[0][tx+1] = ex[idx-nz];
		sz[0][tx] = ez[idx-nz];
	}
	__syncthreads();

	if( j>0 && k>0 ) hx[idx] -= 0.5*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i>0 && k>0 ) hy[idx] -= 0.5*( sx[ty+1][tx+1] - sx[ty+1][tx] - sz[ty+1][tx] + ez[idx-nyz] );
	if( i>0 && j>0 ) hz[idx] -= 0.5*( sy[ty][tx+1] - ey[idx-nyz] - sx[ty+1][tx+1] + sx[ty][tx+1] );
}

__global__ void update_e(int by0, float *ex, float *ey, float *ez, float *hx, float *hy, float *hz, float *cex, float *cey, float *cez) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y + by0;
	int k = bx*Dx + tx;
	int j = (by*Dy + ty)%ny;
	int i = (by*Dy + ty)/ny;
	int idx = i*nyz + j*nz + k;

	__shared__ float sx[Dy+1][Dx+1];
	__shared__ float sy[Dy][Dx+1];
	__shared__ float sz[Dy+1][Dx];

	sx[ty][tx] = hx[idx];
	sy[ty][tx] = hy[idx];
	sz[ty][tx] = hz[idx];
	if( tx == Dx-1 && k < nz-1 ) {
		sx[ty][Dx] = hx[idx+1];
		sy[ty][Dx] = hy[idx+1];
	}
	if( ty == Dy-1 && j < ny-1 ) {
		sx[Dy][tx] = hx[idx+nz];
		sz[Dy][tx] = hz[idx+nz];
	}
	__syncthreads();

	if( j<ny-1 && k<nz-1 ) ex[idx] += cex[idx]*( sz[ty+1][tx] - sz[ty][tx] - sy[ty][tx+1] + sy[ty][tx] );
	if( i<nx-1 && k<nz-1 ) ey[idx] += cey[idx]*( sx[ty][tx+1] - sx[ty][tx] - hz[idx+nyz] + sz[ty][tx] );
	if( i<nx-1 && j<ny-1 ) ez[idx] += cez[idx]*( hy[idx+nyz] - sy[ty][tx] - sx[ty+1][tx] + sx[ty][tx] );
}

__global__ void update_src(float tn, float *f) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = idx/nyz;
	int j = (idx - i*nyz)/nz;
	int k = idx%nz;

	if( i == 0 && j > 13 && j < 99 && k > 12 && k < 53 ) {
		int ijk = (nx/2)*nyz + j*nz + k;
		f[ijk] += sin((j-14)*3.14159265/84) * sin(2 * 3.14159265 / 100 * 0.5 * tn);
	}

	//int ijk = (nx/2)*nyz + (ny/2)*nz + nz/2;
	//f[ijk] += 1000*sin(2 * 3.14159265 / 170 * 0.5 * tn);	
}
		"""
		from pycuda.compiler import SourceModule
		mod = SourceModule( kernels.replace('Dx',str(s.Dx)).replace('Dy',str(s.Dy)).replace('nyz',str(s.ny*s.nz)).replace('nx',str(s.nx)).replace('ny',str(s.ny)).replace('nz',str(s.nz)) )
		s.updateH = mod.get_function("update_h")
		s.updateE = mod.get_function("update_e")
		s.updateE_src = mod.get_function("update_src")

		Bx, By = s.nz/s.Dx, s.nx*s.ny/s.Dy	# number of block
		s.MaxBy = s.MAX_BLOCK/Bx
		s.bpg_list = [(Bx,s.MaxBy) for i in range(By/s.MaxBy)]
		if By%s.MaxBy != 0: s.bpg_list.append( (Bx,By%s.MaxBy) )

		s.updateH.prepare("iPPPPPP", block=(s.Dx,s.Dy,1))
		s.updateE.prepare("iPPPPPPPPP", block=(s.Dx,s.Dy,1))
		s.updateE_src.prepare("fP", block=(256,1,1))
		#s.updateE_src.prepare("fP", block=(1,1,1))


	def update_h(s):
		for i, bpg in enumerate(s.bpg_list): s.updateH.prepared_call(bpg, np.int32(i*s.MaxBy), *s.eh_fields)


	def update_e(s):
		for i, bpg in enumerate(s.bpg_list): s.updateE.prepared_call(bpg, np.int32(i*s.MaxBy), *(s.eh_fields + s.ce_fields))


	def update_src(s, tn):
		#s.updateE_src.prepared_call((s.ny*s.nz/256+1,1), np.float32(tn), s.ez_gpu)
		s.updateE_src.prepared_call((s.ny*s.nz/256+1,1), np.float32(tn), s.hy_gpu)
		#s.updateE_src.prepared_call((1,1), np.float32(tn), s.ez_gpu)


	def mpi_exchange_boundary_h(s, mpi_direction):
		if 'f' in mpi_direction:
			comm.Recv(s.hy_tmp, s.rank-1, 0)
			comm.Recv(s.hz_tmp, s.rank-1, 1)
			cuda.memcpy_htod(int(s.hy_gpu), s.hy_tmp) 
			cuda.memcpy_htod(int(s.hz_gpu), s.hz_tmp) 
		if 'b' in mpi_direction:
			cuda.memcpy_dtoh(s.hy_tmp, int(s.hy_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32']) 
			cuda.memcpy_dtoh(s.hz_tmp, int(s.hz_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32']) 
			comm.Send(s.hy_tmp, s.rank+1, 0)
			comm.Send(s.hz_tmp, s.rank+1, 1)


	def mpi_exchange_boundary_e(s, mpi_direction):
		if 'f' in mpi_direction:
			cuda.memcpy_dtoh(s.ey_tmp, int(s.ey_gpu)) 
			cuda.memcpy_dtoh(s.ez_tmp, int(s.ez_gpu)) 
			comm.Send(s.ey_tmp, s.rank-1, 2)
			comm.Send(s.ez_tmp, s.rank-1, 3)
		if 'b' in mpi_direction:
			comm.Recv(s.ey_tmp, s.rank+1, 2)
			comm.Recv(s.ez_tmp, s.rank+1, 3)
			cuda.memcpy_htod(int(s.ey_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32'], s.ey_tmp) 
			cuda.memcpy_htod(int(s.ez_gpu)+(s.nx-1)*s.ny*s.nz*np.nbytes['float32'], s.ez_tmp) 


nx, ny, nz = 2000, 112, 64
tmax, tgap = 12000, 10

fdtd = Fdtd3DGpu(nx, ny, nz)
fdtd.alloc_eh_fields()
fdtd.alloc_coeff_arrays()

# waveguide geometry
# dx = 1 mm
# width = 86 mm	(y-axis)
# hight = 40 mm	(z-axis)
# margin_y = 13 * 2
# margin_z = 12 * 2
fdtd.cex[:,:,:] = 0
fdtd.cey[:,:,:] = 0
fdtd.cez[:,:,:] = 0
fdtd.cex[:,14:99,13:52] = 0.5
fdtd.cey[:,14:100,13:52] = 0.5
fdtd.cez[:,14:99,13:53] = 0.5

fdtd.alloc_exchange_boundaries()
fdtd.prepare_functions()
if rank == 0: mpi_direction = 'b'
elif rank == 2: mpi_direction = 'f'
else: mpi_direction = 'fb'

if rank == 1:
	# prepare for plot
	import matplotlib.pyplot as plt
	plt.ion()
	fig = plt.figure(figsize=(15,7))
	'''
	fig = plt.figure(figsize=(10,13))
	ax1 = fig.add_subplot(3,1,1)
	ax1.imshow(fdtd.cex[nx/2,:,:].T, origin='lower', interpolation='nearest')
	ax2 = fig.add_subplot(3,1,2)
	ax2.imshow(fdtd.cey[nx/2,:,:].T, origin='lower', interpolation='nearest')
	ax3 = fig.add_subplot(3,1,3)
	ax3.imshow(fdtd.cez[nx/2,:,:].T, origin='lower', interpolation='nearest')
	plt.show()
	'''
	#ez_tmp = np.ones((800,ny,nz), 'f')
	ez_tmp = cuda.pagelocked_zeros((800,ny,nz), 'f')

	ax1 = fig.add_subplot(4,1,1)
	imag1 = ax1.imshow(ez_tmp[:,:,nz/2].T, cmap=plt.cm.jet, origin='lower', vmin=-2, vmax=2., interpolation='nearest')
	ax2 = fig.add_subplot(4,1,2)
	imag2 = ax2.imshow(ez_tmp[:,ny/2,:].T, cmap=plt.cm.jet, origin='lower', vmin=-2, vmax=2., interpolation='nearest')
	ax3 = fig.add_subplot(2,2,3)
	imag3 = ax3.imshow(ez_tmp[500,:,:].T, cmap=plt.cm.jet, origin='lower', vmin=-2, vmax=2., interpolation='nearest')
	ax4 = fig.add_subplot(2,2,4)
	imag4 = ax4.imshow(ez_tmp[700,:,:].T, cmap=plt.cm.jet, origin='lower', vmin=-2, vmax=2, interpolation='nearest')

	# ez^2 sum
	s1 = np.zeros(200)
	s2 = np.zeros(200)
	ez_tmp1 = cuda.pagelocked_zeros((ny,nz), 'f')
	ez_tmp2 = cuda.pagelocked_zeros((ny,nz), 'f')

	# measure kernel execution time
	from datetime import datetime
	t1 = datetime.now()
	flop = 3*(nx*ny*nz*30)*tgap
	flops = np.zeros(tmax/tgap+1)
	start, stop = cuda.Event(), cuda.Event()
	start.record()

# main loop
	max_ez = 0
for tn in xrange(1, tmax+1):
	fdtd.update_h()
	fdtd.mpi_exchange_boundary_h(mpi_direction)

	fdtd.update_e()
	fdtd.mpi_exchange_boundary_e(mpi_direction)

	if rank == 1: fdtd.update_src(tn)

	# ez^2 sum
	if rank == 1:
		cuda.memcpy_dtoh(ez_tmp1, int(fdtd.ez_gpu) + (nx/2+500)*ny*nz*np.nbytes['float32'] )
		cuda.memcpy_dtoh(ez_tmp2, int(fdtd.ez_gpu) + (nx/2+700)*ny*nz*np.nbytes['float32'] )
		s1[:-1] = s1[1:]
		s2[:-1] = s2[1:]
		s1[-1] = (ez_tmp1**2).mean()
		s2[-1] = (ez_tmp2**2).mean()

	if tn%tgap == 0 and rank == 1:
		if max_ez < (ez_tmp1).max():
			max_ez = (ez_tmp1).max()
		stop.record()
		stop.synchronize()
		flops[tn/tgap] = flop/stop.time_since(start)*1e-6
		print '[',datetime.now()-t1,']'," %d/%d (%d %%) %1.2f GFLOPS [%f, %f] %g\r" % (tn, tmax, float(tn)/tmax*100, flops[tn/tgap], s1.mean(), s2.mean(), max_ez),
		sys.stdout.flush()
		start.record()

		if tn > 100:
			cuda.memcpy_dtoh(ez_tmp, int(fdtd.ez_gpu) + (nx/2-200)*ny*nz*np.nbytes['float32'] )
			imag1.set_array( ez_tmp[:,:,nz/2].T )
			imag2.set_array( ez_tmp[:,ny/2,:].T )
			imag3.set_array( ez_tmp[500,:,:].T )
			imag4.set_array( ez_tmp[700,:,:].T )
			plt.draw()

if rank == 1: print "\navg: %1.2f GFLOPS" % flops[2:-2].mean()
stop.record()
stop.synchronize()
print stop.time_since(start)*1e-3

fdtd.finalize()
