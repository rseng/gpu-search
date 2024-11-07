# https://github.com/damonge/CUTE

```console
CUTE_box/README:The CUDA implementation for the periodic-box mode is not supported in the
CUTE/Makefile:#CUDA options
CUTE/Makefile:CUDADIR = /usr/local/cuda
CUTE/Makefile:#DEFINES for the CUDA version
CUTE/Makefile:DEFINEFLAGSGPU = $(DEFINEOPTIONS)
CUTE/Makefile:DEFINEFLAGSGPU += -DI_R_MAX=$(I_R_MAX) -DLOG_R_MAX=$(LOG_R_MAX)
CUTE/Makefile:DEFINEFLAGSGPU += -DI_THETA_MAX=$(I_THETA_MAX) -DLOG_TH_MAX=$(LOG_TH_MAX)
CUTE/Makefile:DEFINEFLAGSGPU += -DI_RL_MAX=$(I_RL_MAX) -DI_RT_MAX=$(I_RT_MAX)
CUTE/Makefile:DEFINEFLAGSGPU += -DI_R3D_MAX=$(I_R3D_MAX) -DLOG_R3D_MAX=$(LOG_R3D_MAX)
CUTE/Makefile:DEFINEFLAGSGPU += -D_HISTO_2D_$(NB_H2D) -DN_LOGINT=$(N_LOGINT)
CUTE/Makefile:COMPGPU = nvcc
CUTE/Makefile:OPTCPU_GPU = -Wall -O3 $(DEFINEFLAGSGPU) $(DEFINEFLAGSCPU)
CUTE/Makefile:OPTGPU = -O3 $(DEFINEFLAGSGPU) -arch compute_20 $(OPT_PRECISION) -Xcompiler -Wall
CUTE/Makefile:INCLUDECUDA = -I$(CUDADIR)/include
CUTE/Makefile:LIBGPU = $(LGSL) -L$(CUDADIR)/lib64 -lcudart -lpthread -lm
CUTE/Makefile:BOXCUDA = src/boxesCUDA.o
CUTE/Makefile:CORRCUDA = src/correlator_cuda.o
CUTE/Makefile:MAINCUDA = src/main_cuda.o
CUTE/Makefile:OFILESCUDA = $(DEF) $(COM) $(COSMO) $(CORRCUDA) $(BOXCUDA) $(IO) $(MAINCUDA)
CUTE/Makefile:EXECUDA = CU_CUTE
CUTE/Makefile:default : $(EXE) $(EXECUDA)
CUTE/Makefile:	$(COMPCPU) $(OPTCPU_GPU) -c $< -o $@ $(INCLUDECOM)
CUTE/Makefile:$(BOXCUDA) : src/boxesCUDA.c
CUTE/Makefile:$(CORRCUDA) : src/correlator_cuda.cu
CUTE/Makefile:	$(COMPGPU) $(OPTGPU) -c $< -o $@ $(INCLUDECOM) $(INCLUDECUDA)
CUTE/Makefile:$(MAINCUDA) : src/main_cuda.c
CUTE/Makefile:	$(COMPCPU) $(OPTCPU_GPU) -c $< -o $@ $(INCLUDECOM)
CUTE/Makefile:$(EXECUDA) : $(OFILESCUDA)
CUTE/Makefile:	$(COMPCPU) $(OPTCPU_GPU) $(OFILESCUDA) -o $(EXECUDA) $(INCLUDECUDA) $(INCLUDECOM) $(LIBGPU)
CUTE/Makefile:	rm -f ./src/*.o ./src/*~ *~ $(EXE) $(EXECUDA)
CUTE/README:CUTE (and its GPU implementation CU_CUTE) provide a set of tools to
CUTE/README:   >CUDA options: see section 9 below.
CUTE/README:9 The CUDA implementation.
CUTE/README:The current version of CUTE comes with a GPU version (CU_CUTE)
CUTE/README:written in CUDA-C. There exist several differences with respect to the
CUTE/README:- In order to use CU_CUTE we must (obviously) have a CUDA-enabled
CUTE/README:  possible both in global and shared memory). Furthermore, the NVIDIA
CUTE/README:- The CUDA libraries are assumed to be in /usr/local/cuda. If this is
CUTE/README:  not the case, change the variable CUDADIR in the Makefile accordingly.
CUTE/README:- CUDA kernels are launched with a number of thread blocks given by
CUTE/README:  For example, for an NVIDIA Tesla C2070 GPU and a catalog with 300K objects
CUTE/README:  precision is required on the GPU this can be disabled by commenting out
CUTE/README:  the number of threads per block in GPU kernel launches) to 256. This
CUTE/README:  $ ./CU_CUTE <options file> <number of CUDA blocks>
CUTE/src/define.h://Histogram options for CUDA
CUTE/src/define.h:#ifdef _HISTO_2D_128    //128x128 option for 2D histograms in CUDA
CUTE/src/define.h:#ifdef _HISTO_2D_64     //64x64 option for 2D histograms in CUDA
CUTE/src/common.h:void write_CF_cuda(char *fname,unsigned long long *DD,
CUTE/src/correlator_cuda.cu://                      Correlators with CUDA-C                      //
CUTE/src/correlator_cuda.cu:#include <cuda.h>
CUTE/src/correlator_cuda.cu:#include "correlator_cuda.h"
CUTE/src/correlator_cuda.cu:__global__ void cudaCrossAng(int np,float *box_pos1,
CUTE/src/correlator_cuda.cu:__global__ void cudaCrossAngPM(int npx,int *pix_full,
CUTE/src/correlator_cuda.cu:__global__ void cudaCrossMono(int np,float *box_pos1,
CUTE/src/correlator_cuda.cu:__global__ void cudaCross3Dps(int np,float *box_pos1,
CUTE/src/correlator_cuda.cu:__global__ void cudaCross3Drm(int np,float *box_pos1,
CUTE/src/correlator_cuda.cu:void corr_CUDA_AngPM(float cth_min,float cth_max,
CUTE/src/correlator_cuda.cu:  cudaEvent_t start, stop;
CUTE/src/correlator_cuda.cu:  cudaEventCreate(&start);
CUTE/src/correlator_cuda.cu:  cudaEventCreate(&stop);
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_nside_cth,&(n_side_cth),sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_nside_phi,&(n_side_phi),sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_cth_min,&(cth_min),sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_cth_max,&(cth_max),sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_thmax,&(thmax),sizeof(float));
CUTE/src/correlator_cuda.cu:  //Allocate GPU memory and copy particle positions
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&pos_dev,3*npix*sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(pos_dev,pos,3*npix*sizeof(float),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&npD_dev,npix*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(npD_dev,npD,npix*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&npR_dev,npix*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(npR_dev,npR,npix*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&pix_full_dev,n_boxes2D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(pix_full_dev,pix_full,n_boxes2D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  //Allocate GPU memory for the GPU histogram
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&DD_dev,NB_HISTO_1D*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DD_dev,DD,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&DR_dev,NB_HISTO_1D*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DR_dev,DR,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&RR_dev,NB_HISTO_1D*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(RR_dev,RR,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaEventRecord(start,0); //Time 0
CUTE/src/correlator_cuda.cu:  cudaCrossAngPM<<<n_blocks,NB_HISTO_1D>>>(npix,pix_full_dev,
CUTE/src/correlator_cuda.cu:  cudaEventRecord(stop,0);
CUTE/src/correlator_cuda.cu:  cudaEventSynchronize(stop);
CUTE/src/correlator_cuda.cu:  cudaEventElapsedTime(&elaptime,start,stop);
CUTE/src/correlator_cuda.cu:  printf("  CUDA: Time ellapsed: %3.1f ms\n",elaptime); //Time 1
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DD,DD_dev,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DR,DR_dev,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  cudaMemcpy(RR,RR_dev,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  //Clean up GPU memory
CUTE/src/correlator_cuda.cu:  cudaFree(pos_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(npD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(npR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(pix_full_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(DD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(DR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(RR_dev);
CUTE/src/correlator_cuda.cu:  cudaEventDestroy(start);
CUTE/src/correlator_cuda.cu:  cudaEventDestroy(stop);
CUTE/src/correlator_cuda.cu:void corr_CUDA_Ang(float cth_min,float cth_max,
CUTE/src/correlator_cuda.cu:  cudaEvent_t start, stop;
CUTE/src/correlator_cuda.cu:  cudaEventCreate(&start);
CUTE/src/correlator_cuda.cu:  cudaEventCreate(&stop);
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_nside_cth,&(n_side_cth),sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_nside_phi,&(n_side_phi),sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_cth_min,&(cth_min),sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_cth_max,&(cth_max),sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_thmax,&(thmax),sizeof(float));
CUTE/src/correlator_cuda.cu:  //Allocate GPU memory and copy particle positions
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_posD_dev,3*npD*sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_posD_dev,box_posD,3*npD*sizeof(float),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_posR_dev,3*npR*sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_posR_dev,box_posR,3*npR*sizeof(float),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_npD_dev,n_boxes2D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_npD_dev,box_npD,n_boxes2D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_npR_dev,n_boxes2D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_npR_dev,box_npR,n_boxes2D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_indD_dev,n_boxes2D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_indD_dev,box_indD,n_boxes2D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_indR_dev,n_boxes2D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_indR_dev,box_indR,n_boxes2D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  //Allocate GPU memory for the GPU histogram
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&DD_dev,NB_HISTO_1D*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DD_dev,DD,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&DR_dev,NB_HISTO_1D*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DR_dev,DR,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&RR_dev,NB_HISTO_1D*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(RR_dev,RR,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaEventRecord(start,0); //Time 0
CUTE/src/correlator_cuda.cu:  cudaCrossAng<<<n_blocks,NB_HISTO_1D>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:  cudaEventRecord(stop,0);
CUTE/src/correlator_cuda.cu:  cudaEventSynchronize(stop);
CUTE/src/correlator_cuda.cu:  cudaEventElapsedTime(&elaptime,start,stop);
CUTE/src/correlator_cuda.cu:  printf("  CUDA: Time ellapsed: %3.1f ms\n",elaptime); //Time 1
CUTE/src/correlator_cuda.cu:  cudaEventRecord(start,0); //Time 0
CUTE/src/correlator_cuda.cu:  cudaCrossAng<<<n_blocks,NB_HISTO_1D>>>(npR,box_posR_dev,
CUTE/src/correlator_cuda.cu:  cudaEventRecord(stop,0);
CUTE/src/correlator_cuda.cu:  cudaEventSynchronize(stop);
CUTE/src/correlator_cuda.cu:  cudaEventElapsedTime(&elaptime,start,stop);
CUTE/src/correlator_cuda.cu:  printf("  CUDA: Time ellapsed: %3.1f ms\n",elaptime); //Time 1
CUTE/src/correlator_cuda.cu:  cudaEventRecord(start,0); //Time 0
CUTE/src/correlator_cuda.cu:  cudaCrossAng<<<n_blocks,NB_HISTO_1D>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:  cudaEventRecord(stop,0);
CUTE/src/correlator_cuda.cu:  cudaEventSynchronize(stop);
CUTE/src/correlator_cuda.cu:  cudaEventElapsedTime(&elaptime,start,stop);
CUTE/src/correlator_cuda.cu:  printf("  CUDA: Time ellapsed: %3.1f ms\n",elaptime); //Time 1
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DD,DD_dev,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DR,DR_dev,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  cudaMemcpy(RR,RR_dev,NB_HISTO_1D*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  //Clean up GPU memory
CUTE/src/correlator_cuda.cu:  cudaFree(box_npD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_npR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_indD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_indR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_posD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_posR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(DD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(DR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(RR_dev);
CUTE/src/correlator_cuda.cu:  cudaEventDestroy(start);
CUTE/src/correlator_cuda.cu:  cudaEventDestroy(stop);
CUTE/src/correlator_cuda.cu:void corr_CUDA_3D(float *pos_min,
CUTE/src/correlator_cuda.cu:  cudaEvent_t start, stop;
CUTE/src/correlator_cuda.cu:  cudaEventCreate(&start);
CUTE/src/correlator_cuda.cu:  cudaEventCreate(&stop);
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_nside_x,&(n_side[0]),sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_nside_y,&(n_side[1]),sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_nside_z,&(n_side[2]),sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_irange_x,&irange_x,sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_irange_y,&irange_y,sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_irange_z,&irange_z,sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_l_box_x,&lbx,sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_l_box_y,&lby,sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_l_box_z,&lbz,sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_x_min,&(pos_min[0]),sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_y_min,&(pos_min[1]),sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpyToSymbol(cst_z_min,&(pos_min[2]),sizeof(float));
CUTE/src/correlator_cuda.cu:  //Allocate GPU memory and copy particle positions
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_posD_dev,3*npD*sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_posD_dev,box_posD,3*npD*sizeof(float),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_posR_dev,3*npR*sizeof(float));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_posR_dev,box_posR,3*npR*sizeof(float),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_npD_dev,n_boxes3D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_npD_dev,box_npD,n_boxes3D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_npR_dev,n_boxes3D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_npR_dev,box_npR,n_boxes3D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_indD_dev,n_boxes3D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_indD_dev,box_indD,n_boxes3D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&box_indR_dev,n_boxes3D*sizeof(int));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(box_indR_dev,box_indR,n_boxes3D*sizeof(int),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  //Allocate GPU memory for the GPU histogram
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&DD_dev,nbns*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DD_dev,DD,nbns*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&DR_dev,nbns*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DR_dev,DR,nbns*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaMalloc((void**)&RR_dev,nbns*sizeof(unsigned long long));
CUTE/src/correlator_cuda.cu:  cudaMemcpy(RR_dev,RR,nbns*sizeof(unsigned long long),cudaMemcpyHostToDevice);
CUTE/src/correlator_cuda.cu:  cudaEventRecord(start,0); //Time 0
CUTE/src/correlator_cuda.cu:    cudaCrossMono<<<n_blocks,NB_HISTO_1D>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:	cudaCross3Dps<<<n_blocks,thr>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:	cudaCross3Drm<<<n_blocks,thr>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:  cudaEventRecord(stop,0);
CUTE/src/correlator_cuda.cu:  cudaEventSynchronize(stop);
CUTE/src/correlator_cuda.cu:  cudaEventElapsedTime(&elaptime,start,stop);
CUTE/src/correlator_cuda.cu:  printf("  CUDA: Time ellapsed: %3.1f ms\n",elaptime); //Time 1
CUTE/src/correlator_cuda.cu:  cudaEventRecord(start,0); //Time 0
CUTE/src/correlator_cuda.cu:    cudaCrossMono<<<n_blocks,NB_HISTO_1D>>>(npR,box_posR_dev,
CUTE/src/correlator_cuda.cu:	cudaCross3Dps<<<n_blocks,thr>>>(npR,box_posR_dev,
CUTE/src/correlator_cuda.cu:	cudaCross3Drm<<<n_blocks,thr>>>(npR,box_posR_dev,
CUTE/src/correlator_cuda.cu:  cudaEventRecord(stop,0);
CUTE/src/correlator_cuda.cu:  cudaEventSynchronize(stop);
CUTE/src/correlator_cuda.cu:  cudaEventElapsedTime(&elaptime,start,stop);
CUTE/src/correlator_cuda.cu:  printf("  CUDA: Time ellapsed: %3.1f ms\n",elaptime); //Time 1
CUTE/src/correlator_cuda.cu:  cudaEventRecord(start,0); //Time 0
CUTE/src/correlator_cuda.cu:    cudaCrossMono<<<n_blocks,NB_HISTO_1D>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:	cudaCross3Dps<<<n_blocks,thr>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:	cudaCross3Drm<<<n_blocks,thr>>>(npD,box_posD_dev,
CUTE/src/correlator_cuda.cu:  cudaEventRecord(stop,0);
CUTE/src/correlator_cuda.cu:  cudaEventSynchronize(stop);
CUTE/src/correlator_cuda.cu:  cudaEventElapsedTime(&elaptime,start,stop);
CUTE/src/correlator_cuda.cu:  printf(  "  CUDA: Time ellapsed: %3.1f ms\n",elaptime); //Time 1
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DD,DD_dev,nbns*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  cudaMemcpy(DR,DR_dev,nbns*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  cudaMemcpy(RR,RR_dev,nbns*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
CUTE/src/correlator_cuda.cu:  //Clean up GPU memory
CUTE/src/correlator_cuda.cu:  cudaFree(box_npD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_npR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_indD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_indR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_posD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(box_posR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(DD_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(DR_dev);
CUTE/src/correlator_cuda.cu:  cudaFree(RR_dev);
CUTE/src/correlator_cuda.cu:  cudaEventDestroy(start);
CUTE/src/correlator_cuda.cu:  cudaEventDestroy(stop);
CUTE/src/correlator_cuda.h://                      Correlators with CUDA-C                      //
CUTE/src/correlator_cuda.h:#ifndef _CUTE_CUDACORR_
CUTE/src/correlator_cuda.h:#define _CUTE_CUDACORR_
CUTE/src/correlator_cuda.h:void corr_CUDA_AngPM(float cth_min,float cth_max,
CUTE/src/correlator_cuda.h:void corr_CUDA_Ang(float cth_min,float cth_max,
CUTE/src/correlator_cuda.h:void corr_CUDA_3D(float *pos_min,
CUTE/src/correlator_cuda.h:#endif //_CUTE_CUDACORR_
CUTE/src/boxesCUDA.c:#define FRACTION_AR_CUDA 16.0
CUTE/src/boxesCUDA.c:#define FRACTION_EXTEND_CUDA 0.01
CUTE/src/boxesCUDA.c:  int nside1=(int)(FRACTION_AR_CUDA*lb/rmax);   //nside1 -> 8 boxes per rmax
CUTE/src/boxesCUDA.c:  double ex=FRACTION_EXTEND_CUDA*(x_max_bound-x_min_bound);
CUTE/src/boxesCUDA.c:  double ey=FRACTION_EXTEND_CUDA*(y_max_bound-y_min_bound);
CUTE/src/boxesCUDA.c:  double ez=FRACTION_EXTEND_CUDA*(z_max_bound-z_min_bound);
CUTE/src/io.c:void write_CF_cuda(char *fname,unsigned long long *DD,
CUTE/src/main_cuda.c:#include "correlator_cuda.h"
CUTE/src/main_cuda.c:  corr_CUDA_AngPM(cth_min,cth_max,
CUTE/src/main_cuda.c:  write_CF_cuda(fnameOut,DD,DR,RR,n_dat,n_ran);
CUTE/src/main_cuda.c:  corr_CUDA_Ang(cth_min,cth_max,
CUTE/src/main_cuda.c:  write_CF_cuda(fnameOut,DD,DR,RR,n_dat,n_ran);
CUTE/src/main_cuda.c:  corr_CUDA_3D(pos_min,
CUTE/src/main_cuda.c:  write_CF_cuda(fnameOut,DD,DR,RR,n_dat,n_ran);
CUTE/src/main_cuda.c:  corr_CUDA_3D(pos_min,
CUTE/src/main_cuda.c:  write_CF_cuda(fnameOut,DD,DR,RR,n_dat,n_ran);
CUTE/src/main_cuda.c:  corr_CUDA_3D(pos_min,
CUTE/src/main_cuda.c:  write_CF_cuda(fnameOut,DD,DR,RR,n_dat,n_ran);
CUTE/src/main_cuda.c:  print_info("Using %d CUDA blocks \n",n_blocks);
CUTE/src/main_cuda.c:    fprintf(stderr," in the CUDA implementation \n");
CUTE/src/main_cuda.c:    fprintf(stderr," in the CUDA implementation \n");

```
