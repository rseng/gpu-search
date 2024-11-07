# https://github.com/ahwkuepper/mcluster

```console
gpupot.gpu.cu://#include "gpupot.h"
gpupot.gpu.cu:#ifdef WITH_CUDA5
gpupot.gpu.cu:#  include <helper_cuda.h>
gpupot.gpu.cu:#  define CUDA_SAFE_CALL checkCudaErrors
gpupot.gpu.cu:#include "cuda_pointer.h"
gpupot.gpu.cu:extern "C" void gpupot(int n, double **star, double *pot) {
gpupot.gpu.cu:    int numGPU=0;
gpupot.gpu.cu:    cudaGetDeviceCount(&numGPU);
gpupot.gpu.cu:    assert(numGPU>0);
gpupot.gpu.cu:    cudaSetDevice(0);
gpupot.gpu.cu:	cudaPointer <float2> phi;
gpupot.gpu.cu:	cudaPointer <Particle> ptcl;
gpupot.gpu.cu:	// cudaMemcpy(ptcl_d, ptcl_h, ng * sizeof(Particle), cudaMemcpyHostToDevice);
gpupot.gpu.cu:	// cudaMemcpy(phi_h, phi_d, n * sizeof(float2), cudaMemcpyDeviceToHost);
gpupot.gpu.cu:	fprintf(stderr, "gpupot: %f sec\n", t1 - t0);
Makefile:CUFLAGS= -O3 -D WITH_CUDA5
Makefile:CUDA_PATH = /usr/local/cuda
Makefile:SDK_PATH=$(CUDA_PATH)/samples
Makefile:mcluster_ssegpu: $(OBJECTS) $(LFLAGS) gpupot.gpu.o main.c
Makefile:	$(CC) -c main.c -D SSE -D GPU -lm -I$(CUDA_PATH)/include
Makefile:	$(CC) $(OBJECTS) main.o gpupot.gpu.o  -o mcluster_gpu -L$(CUDA_PATH)/lib64 -lcudart -lstdc++ -lm $(CFLAGS) 
Makefile:mcluster_gpu: gpupot.gpu.o main.c
Makefile:	$(CC) -c main.c -D GPU -I$(CUDA_PATH)/include
Makefile:	$(CC) main.o gpupot.gpu.o -o mcluster_gpu -L$(CUDA_PATH)/lib64 -lcudart -lstdc++ -lm
Makefile:gpupot.gpu.o: gpupot.gpu.cu cuda_pointer.h
Makefile:	nvcc -c $(CUFLAGS) -Xcompiler "-fPIC -O3 -Wall" -I$(SDK_PATH)/common/inc -I. gpupot.gpu.cu 
Makefile:	rm -f *.o mcluster_sse mcluster mcluster_gpu
cuda_pointer.h:struct cudaPointer{
cuda_pointer.h:  cudaPointer(){
cuda_pointer.h:  //  ~cudaPointer(){
cuda_pointer.h:    CUDA_SAFE_CALL(cudaMalloc(&p, size * sizeof(T)));
cuda_pointer.h:    CUDA_SAFE_CALL(cudaMallocHost(&p, size * sizeof(T)));
cuda_pointer.h:    CUDA_SAFE_CALL(cudaFree(dev_pointer));
cuda_pointer.h:    CUDA_SAFE_CALL(cudaFreeHost(host_pointer));
cuda_pointer.h:    CUDA_SAFE_CALL(cudaMemcpy(dev_pointer, host_pointer, count * sizeof(T), cudaMemcpyHostToDevice));
cuda_pointer.h:    CUDA_SAFE_CALL(cudaMemcpy(host_pointer, dev_pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
McLusterManual.tex:      -G & 0/1 & Use GPU with \textsc{Nbody6/7} (no/yes)\\                   
McLusterManual.tex:for the fully mass segregated $N$-body model. The arguments stand for: a total mass of $100.000\msun$ (\texttt{-M}), the EFF density profile (\texttt{-P}) with a 2d core radius of 0.1 pc (\texttt{-r}), a cut-off radius of 20 pc (\texttt{-c}) and a 2d power-law slope of -2 (\texttt{-g}). It is completely mass segregated (\texttt{-S}), the output is for \textsc{Nbody6} (\texttt{-C}), and we use a GPU (\texttt{-G}). The output is named R136 (\texttt{-o}), we use a Kroupa IMF (\texttt{-f}), 20\% binaries (\texttt{-b}) and ordered pairing for massive stars (\texttt{-p}). The random seed of our model is 2 (\texttt{-s}) and the metallicity is 0.01 (\texttt{-Z}). The output is in $N$-body units (\texttt{-u}) since we want to pass it to \textsc{Nbody6}.
main.h:int output0(char *output, int N, int NNBMAX, double RS0, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double dtplot, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star, int sse, int seed, double extmass, double extrad, double extdecay, double extstart);
main.h:int output1(char *output, int N, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star);
main.h:int output2(char *output, int N, int NNBMAX, double RS0, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double dtplot, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star, int sse, int seed, double extmass, double extrad, double extdecay, double extstart);
main.h:int output4(char *output, int N, int NNBMAX, double RS0, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double dtplot, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star, int sse, int seed, double extmass, double extrad, double extdecay, double extstart);
main.h:void info(char *output, int N, double Mcl, int profile, double W0, double S, double D, double Q, double Rh, double gamma[], double a, double Rmax, double tcrit, int tf, double RG[], double VG[], int mfunc, double single_mass, double mlow, double mup, double alpha[], double mlim[], double alpha_L3, double beta_L3, double mu_L3, int weidner, int mloss, int remnant, double epoch, double Z, int prantzos, int nbin, double fbin, int pairing, double msort, int adis, double amin, double amax, int eigen, int BSE, double extmass, double extrad, double extdecay, double extstart, int code, int seed, double dtadj, double dtout, double dtplot, int gpu, int regupdate, int etaupdate, int esc, int units, int match, int symmetry, int OBperiods);
main.c:#ifdef GPU
main.c:#include <cuda.h>
main.c:#include <cuda_runtime.h>
main.c:	int code = 3;					//Nbody version: =0 Nbody6, =1 Nbody4, =2 Nbody6 custom, =3 only create output list of stars, =4 Nbody7 (not yet fully functional), =5 Nbody6++GPU
main.c:	int gpu = 0;					//Use of GPU, 0= off, 1= on
main.c:            gpu = atoi(optarg);
main.c:            if (gpu == 0 || gpu == 1)
main.c:                printf("\nError: Use of GPU (G) needs to "
main.c:                    "be between 0 or 1, %d was given\n", gpu);
main.c:	info(output, N, Mcl, profile, W0, S, D, Q, Rh, gamma, a, Rmax, tcrit, tf, RG, VG, mfunc, single_mass, mlow, mup, alpha, mlim, alpha_L3, beta_L3, mu_L3, weidner, mloss, remnant, epoch, Z, prantzos, nbin, fbin, pairing, msort, adis, amin, amax, eigen, BSE, extmass, extrad, extdecay, extstart, code, seed, dtadj, dtout, dtplot, gpu, regupdate, etaupdate, esc, units, match, symmetry, OBperiods);
main.c:#ifdef GPU
main.c:		gpupot(N,star,&pe);
main.c:        //		printf("PE_GPU %lg ke %lg\n",pe,ke);
main.c:#ifdef GPU
main.c:		gpupot(N,star,&pe);
main.c:        //		printf("PE_GPU %lg ke %lg\n",pe,ke);
main.c:		output0(output, N, NNBMAX, RS0, dtadj, dtout, tcrit, rvir, mmean, tf, regupdate, etaupdate, mloss, bin, esc, M, mlow, mup, MMAX, epoch, dtplot, Z, nbin, Q, RG, VG, rtide, gpu, star, sse, seed, extmass, extrad, extdecay, extstart);
main.c:		output1(output, N, dtadj, dtout, tcrit, rvir, mmean, tf, regupdate, etaupdate, mloss, bin, esc, M, mlow, mup, MMAX, epoch, Z, nbin, Q, RG, VG, rtide, gpu, star);
main.c:		output2(output, N, NNBMAX, RS0, dtadj, dtout, tcrit, rvir, mmean, tf, regupdate, etaupdate, mloss, bin, esc, M, mlow, mup, MMAX, epoch, dtplot, Z, nbin, Q, RG, VG, rtide, gpu, star, sse, seed, extmass, extrad, extdecay, extstart);
main.c:		output4(output, N, NNBMAX, RS0, dtadj, dtout, tcrit, rvir, mmean, tf, regupdate, etaupdate, mloss, bin, esc, M, mlow, mup, MMAX, epoch, dtplot, Z, nbin, Q, RG, VG, rtide, gpu, star, sse, seed, extmass, extrad, extdecay, extstart);
main.c:		output5(output, N, NNBMAX, RS0, dtadj, dtout, tcrit*tscale, tcrit, rvir, mmean, tf, regupdate, etaupdate, mloss, bin, esc, M, mlow, mup, MMAX, epoch, dtplot, Z, nbin, Q, RG, VG, rtide, gpu, star, sse, seed, extmass, extrad, extdecay, extstart);
main.c:int output0(char *output, int N, int NNBMAX, double RS0, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double dtplot, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star, int sse, int seed, double extmass, double extrad, double extdecay, double extstart){
main.c:int output1(char *output, int N, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star){
main.c:	if (gpu) fprintf(PAR,"1.0\n");
main.c:int output2(char *output, int N, int NNBMAX, double RS0, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double dtplot, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star, int sse, int seed, double extmass, double extrad, double extdecay, double extstart){
main.c:int output4(char *output, int N, int NNBMAX, double RS0, double dtadj, double dtout, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double dtplot, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star, int sse, int seed, double extmass, double extrad, double extdecay, double extstart){
main.c:int output5(char *output, int N, int NNBMAX, double RS0, double dtadj, double dtout, double tcritp, double tcrit, double rvir, double mmean, int tf, int regupdate, int etaupdate, int mloss, int bin, int esc, double M, double mlow, double mup, double MMAX, double epoch, double dtplot, double Z, int nbin, double Q, double *RG, double *VG, double rtide, int gpu, double **star, int sse, int seed, double extmass, double extrad, double extdecay, double extstart){
main.c:void info(char *output, int N, double Mcl, int profile, double W0, double S, double D, double Q, double Rh, double gamma[], double a, double Rmax, double tcrit, int tf, double RG[], double VG[], int mfunc, double single_mass, double mlow, double mup, double alpha[], double mlim[], double alpha_L3, double beta_L3, double mu_L3, int weidner, int mloss, int remnant, double epoch, double Z, int prantzos, int nbin, double fbin, int pairing, double msort, int adis, double amin, double amax, int eigen, int BSE, double extmass, double extrad, double extdecay, double extstart, int code, int seed, double dtadj, double dtout, double dtplot, int gpu, int regupdate, int etaupdate, int esc, int units, int match, int symmetry, int OBperiods) {
main.c:	fprintf(INFO, "gpu = %i\n",gpu);
main.c:	printf("       -G <0|1> (GPU usage; 0= no GPU, 1= use GPU)                   \n");
README:       -G <0|1> (GPU usage; 0= no GPU, 1= use GPU)
README:Specify whether you want to use Nbody6 with a GPU.

```
