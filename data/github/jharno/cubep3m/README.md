# https://github.com/jharno/cubep3m

```console
source_threads/nbody-ueli.cu:#include <cuda.h>
source_threads/nbody-ueli.cu://#include <nbody-cuda.h>
source_threads/nbody-ueli.cu:float *cuda_vector(int n)
source_threads/nbody-ueli.cu:  assert(cudaMalloc ((void **) &vec, sizeof (float) * n)==cudaSuccess);
source_threads/nbody-ueli.cu:  cudaMalloc ((void **) &vec, sizeof (float) * n);
source_threads/nbody-ueli.cu:  cudaMemset(vec,0,sizeof(float)*n);
source_threads/nbody-ueli.cu:float *cuda_copy(float *vec, int n)
source_threads/nbody-ueli.cu://Routine to take in input float vector, allocate space on the GPU for it, and copy the contents over.
source_threads/nbody-ueli.cu:  float *dvec=cuda_vector(n);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(dvec,vec,sizeof(float)*n,cudaMemcpyHostToDevice)==cudaSuccess);
source_threads/nbody-ueli.cu:  cudaMemcpy(dvec,vec,sizeof(float)*n,cudaMemcpyHostToDevice);
source_threads/nbody-ueli.cu:float *cuda_blocked_copy(float *vec, int n, int nn)
source_threads/nbody-ueli.cu://Routine to take in input float vector, allocate space on the GPU for it, and copy the contents over.
source_threads/nbody-ueli.cu:  float *dvec=cuda_vector(nn);
source_threads/nbody-ueli.cu:  assert(cudaMemset (dvec, 0, sizeof(float)*nn)==cudaSuccess);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(dvec,vec,sizeof(float)*n,cudaMemcpyHostToDevice)==cudaSuccess);
source_threads/nbody-ueli.cu:  cudaMemset (dvec, 0, sizeof(float)*nn);
source_threads/nbody-ueli.cu:  cudaMemcpy(dvec,vec,sizeof(float)*n,cudaMemcpyHostToDevice);
source_threads/nbody-ueli.cu:__global__ void cuda_set_float_vec(int n, float *x, float val)
source_threads/nbody-ueli.cu:    float *dvec=cuda_vector(nn);
source_threads/nbody-ueli.cu:    //should do a memset here, but there's one in cuda_vector
source_threads/nbody-ueli.cu:    cuda_set_float_vec<<<nb,BLOCK>>>(n,dvec,1.0);
source_threads/nbody-ueli.cu:    float *dvec=cuda_blocked_copy(vec,n,nn);
source_threads/nbody-ueli.cu:  //printf("Working pp on GPU\n");
source_threads/nbody-ueli.cu:  float *dx1=cuda_blocked_copy(x1,n1,nn1);
source_threads/nbody-ueli.cu:  float *dy1=cuda_blocked_copy(y1,n1,nn1);
source_threads/nbody-ueli.cu:  float *dz1=cuda_blocked_copy(z1,n1,nn1);
source_threads/nbody-ueli.cu:  float *dfx1=cuda_vector(nn1);
source_threads/nbody-ueli.cu:  float *dfy1=cuda_vector(nn1);
source_threads/nbody-ueli.cu:  float *dfz1=cuda_vector(nn1);
source_threads/nbody-ueli.cu:  float *dx2=cuda_blocked_copy(x2,n2,nn2);
source_threads/nbody-ueli.cu:  float *dy2=cuda_blocked_copy(y2,n2,nn2);
source_threads/nbody-ueli.cu:  float *dz2=cuda_blocked_copy(z2,n2,nn2);
source_threads/nbody-ueli.cu:  float *dfx2=cuda_vector(nn2);
source_threads/nbody-ueli.cu:  float *dfy2=cuda_vector(nn2);
source_threads/nbody-ueli.cu:  float *dfz2=cuda_vector(nn2);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(fx1,dfx1,sizeof(float)*n1,cudaMemcpyDeviceToHost)==cudaSuccess);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(fy1,dfy1,sizeof(float)*n1,cudaMemcpyDeviceToHost)==cudaSuccess);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(fz1,dfz1,sizeof(float)*n1,cudaMemcpyDeviceToHost)==cudaSuccess);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(fx2,dfx2,sizeof(float)*n2,cudaMemcpyDeviceToHost)==cudaSuccess);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(fy2,dfy2,sizeof(float)*n2,cudaMemcpyDeviceToHost)==cudaSuccess);
source_threads/nbody-ueli.cu:  assert(cudaMemcpy(fz2,dfz2,sizeof(float)*n2,cudaMemcpyDeviceToHost)==cudaSuccess);
source_threads/nbody-ueli.cu:  cudaMemcpy(fx1,dfx1,sizeof(float)*n1,cudaMemcpyDeviceToHost);
source_threads/nbody-ueli.cu:  cudaMemcpy(fy1,dfy1,sizeof(float)*n1,cudaMemcpyDeviceToHost);
source_threads/nbody-ueli.cu:  cudaMemcpy(fz1,dfz1,sizeof(float)*n1,cudaMemcpyDeviceToHost);
source_threads/nbody-ueli.cu:  cudaMemcpy(fx2,dfx2,sizeof(float)*n2,cudaMemcpyDeviceToHost);
source_threads/nbody-ueli.cu:  cudaMemcpy(fy2,dfy2,sizeof(float)*n2,cudaMemcpyDeviceToHost);
source_threads/nbody-ueli.cu:  cudaMemcpy(fz2,dfz2,sizeof(float)*n2,cudaMemcpyDeviceToHost);
source_threads/nbody-ueli.cu:  cudaFree(dx1);
source_threads/nbody-ueli.cu:  cudaFree(dy1);
source_threads/nbody-ueli.cu:  cudaFree(dz1);
source_threads/nbody-ueli.cu:  cudaFree(dfx1);
source_threads/nbody-ueli.cu:  cudaFree(dfy1);
source_threads/nbody-ueli.cu:  cudaFree(dfz1);
source_threads/nbody-ueli.cu:  cudaFree(dm1);
source_threads/nbody-ueli.cu:  cudaFree(dx2);
source_threads/nbody-ueli.cu:  cudaFree(dy2);
source_threads/nbody-ueli.cu:  cudaFree(dz2);
source_threads/nbody-ueli.cu:  cudaFree(dfx2);
source_threads/nbody-ueli.cu:  cudaFree(dfy2);
source_threads/nbody-ueli.cu:  cudaFree(dfz2);
source_threads/nbody-ueli.cu:  cudaFree(dm2);
source_threads/nbody-ueli.cu:  double gpu_time=omp_get_wtime();
source_threads/nbody-ueli.cu:  gpu_time-=t1;
source_threads/nbody-ueli.cu:  printf("force on particle 1 is %17.7e %17.7e, took %14.4g seconds\n",fx1[0],fx2[0],gpu_time/nrep);
source_threads/nbody-ueli.cu:  double gpu_overhead_time=omp_get_wtime();
source_threads/nbody-ueli.cu:  gpu_overhead_time-=t1;
source_threads/nbody-ueli.cu:  printf("force on particle 1 is %17.7e %17.7e, overhead took %14.4g seconds\n",fx1[0],fx2[0],gpu_overhead_time/nrep);
source_threads/nbody-ueli.cu:  printf("ratio is %14.4f\n",cpu_time/gpu_time*nrep);
source_threads/nbody-ueli.cu:  printf("pure compute ratio is %14.4f\n",cpu_time/(gpu_time-gpu_overhead_time)*nrep);
source_threads/nbody.h:void calculate_forces_gpu(NBody *data, int n);
source_threads/Makefile_pp_ext_gpu:FFLAGS=-shared-intel -fpp  -g -O3 -fpic -xhost -DDIAG -DBINARY -DNGP -DPPINT -DMPI_TIME  -DLRCKCORR -DNGPH  -DPP_EXT -Dpp_ext_on_GPU #-DREAD_SEED  #-DPID_FLAG #-DDEBUG_PP_EXT #-DREAD_SEED #-DDISP_MESH -DPID_FLAG #-DDEBUG 
source_threads/Makefile_pp_ext_gpu:OBJS=   checkpoint.o coarse_cic_mass.o coarse_cic_mass_buffer.o coarse_force.o coarse_force_buffer.o coarse_mass.o coarse_max_dt.o coarse_mesh.o coarse_power.o coarse_velocity.o cubepm.o delete_particles.o fftw3ds.o fine_cic_mass.o fine_cic_mass_buffer.o fine_mesh.o fine_ngp_mass.o fine_velocity.o halofind.o fftw2.o init_projection.o kernel_initialization.o link_list.o move_grid_back.o mpi_initialization.o particle_initialization.o particle_mesh_cuda.o particle_pass.o projection.o report_pair.o report_force.o set_pair.o indexedsort.o timers.o timestep.o update_position.o variable_initialization.o nbody-ueli.o
source_threads/Makefile_pp_ext_gpu:	$(FC) $(FFLAGS) -L/opt/cuda/lib64 -lcudart -openmp  $^ -o $@  $(LDLIBS)
source_threads/Makefile_pp_ext_gpu:particle_mesh_cuda.o: particle_mesh_cuda.f90
source_threads/particle_mesh_cuda.f90:#ifdef pp_ext_on_GPU
source_threads/particle_mesh_cuda.f90:    real(4), dimension(3,10000,cores):: x1_gpu, x2_gpu, f1
source_threads/particle_mesh_cuda.f90:#ifdef pp_ext_on_GPU 
source_threads/particle_mesh_cuda.f90:      write(*,*)'trying gpu on cubep3m'
source_threads/particle_mesh_cuda.f90:         do k=1,nf_physical_tile_dim+2*pp_range ! We do loop towards smaller z in GPU
source_threads/particle_mesh_cuda.f90:                     x1_gpu(1,n1,thread) = xv(1,pp1)
source_threads/particle_mesh_cuda.f90:                     x1_gpu(2,n1,thread) = xv(2,pp1)
source_threads/particle_mesh_cuda.f90:                     x1_gpu(3,n1,thread) = xv(3,pp1)
source_threads/particle_mesh_cuda.f90:                  !write(*,*) 'Got x1_gpu,n1 =' , n1                                    
source_threads/particle_mesh_cuda.f90:                  ! for gpu code, I only store the resulting force on f1,
source_threads/particle_mesh_cuda.f90:                              x2_gpu(1,n2,thread) = xv(1,pp2)
source_threads/particle_mesh_cuda.f90:                              x2_gpu(2,n2,thread) = xv(2,pp2)
source_threads/particle_mesh_cuda.f90:                              x2_gpu(3,n2,thread) = xv(3,pp2)
source_threads/particle_mesh_cuda.f90:                  !write(*,*) 'Got x2_gpu, n2 = ' ,  n2
source_threads/particle_mesh_cuda.f90:                  ! use GPU packakage here:
source_threads/particle_mesh_cuda.f90:                  !write(*,*) 'Calling CUDA'
source_threads/particle_mesh_cuda.f90:                  call pp_force_c(n1,x1_gpu(1,1:n1,thread),x1_gpu(2,1:n1,thread),x1_gpu(3,1:n1,thread),f1(1,1:n1,thread),f1(2,1:n1,thread),f1(3,1:n1,thread),n2,x2_gpu(1,1:n2,thread),x2_gpu(2,1:n2,thread),x2_gpu(3,1:n2,thread))
source_threads/particle_mesh_cuda.f90:                  !write(*,*) 'pp_gpu :' ,x1_gpu(1,1,thread),x1_gpu(2,1,thread),x1_gpu(3,1,thread),f1(1,1,thread),f1(2,1,thread),f1(3,1,thread),x2_gpu(1,1,thread),x2_gpu(2,1,thread),x2_gpu(3,1,thread)
source_threads/particle_mesh_cuda.f90:!  endif pp_ext_on_GPU
source_threads/Makefile_pp_cuda:testcuda: ppforce_test.o nbody-ueli.o
source_threads/Makefile_pp_cuda:	$(FC) -o $@ $^ -L/opt/cuda/lib64 -lcudart -lgomp 
source_threads/Makefile_pp_cuda:	rm -rf *.o testcuda

```
