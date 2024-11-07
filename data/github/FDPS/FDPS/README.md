# https://github.com/FDPS/FDPS

```console
sample/fortran/p3m/test.py:        # use_phantom_grape_x86, use_gpu_cuda in the original Makefile.
sample/fortran/vdw-test/test.py:        # use_phantom_grape_x86, use_gpu_cuda in the original Makefile.
sample/fortran/sph/test.py:        # use_phantom_grape_x86, use_gpu_cuda in the original Makefile.
sample/fortran/nbody+sph/test.py:        # use_phantom_grape_x86, use_gpu_cuda in the original Makefile.
sample/fortran/nbody/test.py:        # use_phantom_grape_x86, use_gpu_cuda in the original Makefile.
sample/c++/water/Makefile:gpu:
sample/c++/water/Makefile:	make -C ./src/cuda/
sample/c++/water/Makefile:	cp ./src/cuda/gpu.out .
sample/c++/water/Makefile:	make reuse -C ./src/cuda/
sample/c++/water/Makefile:	cp ./src/cuda/reuse.out .
sample/c++/water/Makefile:	make clean -C ./src/cuda/
sample/c++/water/Makefile:	make dstclean -C ./src/cuda/
sample/c++/water/README:    cuda		source directory for flexible water simulation using GPU and reuse-list mode
sample/c++/water/README:      cuda_pointer.h:		definition of device and host pointer management class
sample/c++/water/README:      kernel.h:			header file for functions which use cuda APIs
sample/c++/water/README:    for flexible model w/o gpu:
sample/c++/water/README:    for flexible model w/ gpu:
sample/c++/water/README:	make gpu
sample/c++/water/README:	./gpu.out
sample/c++/water/README:    for flexible model w/ gpu and reuse-list mode:
sample/c++/water/src/cuda/Makefile:CUDAPATH?=/usr/local/cuda
sample/c++/water/src/cuda/Makefile:NVCC=$(CUDAPATH)/bin/nvcc
sample/c++/water/src/cuda/Makefile:NVCCFLAGS= -std=c++17 -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -L$(CUDAPATH)/lib64 $(FDPS_INC) -Xcompiler "-std=c++17 -O3"
sample/c++/water/src/cuda/Makefile:LDFLAGS+=-L$(CUDAPATH)/lib64  -lcudart
sample/c++/water/src/cuda/Makefile:all:	gpu
sample/c++/water/src/cuda/Makefile:gpu:	Makefile $(SRC) $(HED) kernel.o
sample/c++/water/src/cuda/Makefile:	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SRC) kernel.o -o gpu.out
sample/c++/water/src/cuda/kernel.cu:#include "cuda_pointer.h"
sample/c++/water/src/cuda/kernel.cu:#if (__CUDA_ARCH__ >= 300)
sample/c++/water/src/cuda/kernel.cu:static cudaPointer<EpiDev>   dev_epi;
sample/c++/water/src/cuda/kernel.cu:static cudaPointer<EpjDev>   dev_epj;
sample/c++/water/src/cuda/kernel.cu:static cudaPointer<ForceDev> dev_force;
sample/c++/water/src/cuda/kernel.cu:static cudaPointer<int>      ij_disp;
sample/c++/water/src/cuda/kernel.cu:      cudaGetDeviceCount(&ndevice);
sample/c++/water/src/cuda/kernel.cu:      cudaSetDevice(PS::Comm::getRank()%ndevice);
sample/c++/water/src/cuda/cuda_pointer.h:#include <cuda.h>
sample/c++/water/src/cuda/cuda_pointer.h:#include <cuda_runtime.h>
sample/c++/water/src/cuda/cuda_pointer.h:#  include <helper_cuda.h>
sample/c++/water/src/cuda/cuda_pointer.h:#  define CUDA_SAFE_CALL checkCudaErrors
sample/c++/water/src/cuda/cuda_pointer.h:struct cudaPointer{
sample/c++/water/src/cuda/cuda_pointer.h:  cudaPointer(){
sample/c++/water/src/cuda/cuda_pointer.h:  //        ~cudaPointer(){
sample/c++/water/src/cuda/cuda_pointer.h:    CUDA_SAFE_CALL(cudaMalloc(&p, size * sizeof(T)));
sample/c++/water/src/cuda/cuda_pointer.h:    CUDA_SAFE_CALL(cudaMallocHost(&p, size * sizeof(T)));
sample/c++/water/src/cuda/cuda_pointer.h:    CUDA_SAFE_CALL(cudaFree(dev_pointer));
sample/c++/water/src/cuda/cuda_pointer.h:    CUDA_SAFE_CALL(cudaFreeHost(host_pointer));
sample/c++/water/src/cuda/cuda_pointer.h:    CUDA_SAFE_CALL(cudaMemcpy(dev_pointer, host_pointer, count * sizeof(T), cudaMemcpyHostToDevice));
sample/c++/water/src/cuda/cuda_pointer.h:    CUDA_SAFE_CALL(cudaMemcpy(host_pointer, dev_pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
sample/c++/nbody/Makefile:#use_gpu_cuda = yes
sample/c++/nbody/Makefile:ifeq ($(use_gpu_cuda),yes)
sample/c++/nbody/Makefile:CUDA_HOME = /usr/local/cuda
sample/c++/nbody/Makefile:#CUDA_HOME = /gwfefs/opt/x86_64/cuda/7.5
sample/c++/nbody/Makefile:NVCC = time $(CUDA_HOME)/bin/nvcc -std=c++11 -Xcompiler="-std=c++11 -O3"
sample/c++/nbody/Makefile:INC  += -I$(CUDA_HOME)/samples/common/inc/
sample/c++/nbody/Makefile:CFLAGS  += -DENABLE_GPU_CUDA
sample/c++/nbody/Makefile:CLIBS = -L$(CUDA_HOME)/lib64 -lcudart -lgomp
sample/c++/nbody/Makefile:force_gpu_cuda.o:force_gpu_cuda.cu
sample/c++/nbody/Makefile:OBJS = force_gpu_cuda.o
sample/c++/nbody/force_gpu_cuda.cu:#include "cuda_pointer.h"
sample/c++/nbody/force_gpu_cuda.cu:#include "force_gpu_cuda.hpp"
sample/c++/nbody/force_gpu_cuda.cu:	N_THREAD_GPU = 32,
sample/c++/nbody/force_gpu_cuda.cu:struct EpiGPU{
sample/c++/nbody/force_gpu_cuda.cu:struct EpjGPU{
sample/c++/nbody/force_gpu_cuda.cu:struct ForceGPU{
sample/c++/nbody/force_gpu_cuda.cu:		const EpiGPU * epi,
sample/c++/nbody/force_gpu_cuda.cu:		const EpjGPU * epj, 
sample/c++/nbody/force_gpu_cuda.cu:		ForceGPU     * force,
sample/c++/nbody/force_gpu_cuda.cu:		const EpjGPU *epj, 
sample/c++/nbody/force_gpu_cuda.cu:	for(int j=j_head; j<j_tail; j+=N_THREAD_GPU){
sample/c++/nbody/force_gpu_cuda.cu:		if(j_tail-j < N_THREAD_GPU){
sample/c++/nbody/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
sample/c++/nbody/force_gpu_cuda.cu:		float4        jpsh[2][N_THREAD_GPU],
sample/c++/nbody/force_gpu_cuda.cu:		const EpjGPU *epj, 
sample/c++/nbody/force_gpu_cuda.cu:	for(int j=0; j<nj_shorter; j+=N_THREAD_GPU){
sample/c++/nbody/force_gpu_cuda.cu:		if(nj_shorter-j < N_THREAD_GPU){
sample/c++/nbody/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
sample/c++/nbody/force_gpu_cuda.cu:	for(int j=nj_shorter; j<nj_longer; j+=N_THREAD_GPU){
sample/c++/nbody/force_gpu_cuda.cu:		if(jrem < N_THREAD_GPU){
sample/c++/nbody/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
sample/c++/nbody/force_gpu_cuda.cu:		const EpjGPU *epj, 
sample/c++/nbody/force_gpu_cuda.cu:		const EpiGPU * epi,
sample/c++/nbody/force_gpu_cuda.cu:		const EpjGPU * epj, 
sample/c++/nbody/force_gpu_cuda.cu:		ForceGPU     * force,
sample/c++/nbody/force_gpu_cuda.cu:	int t_tail = t_head + N_THREAD_GPU - 1;
sample/c++/nbody/force_gpu_cuda.cu:	__shared__ float4 jpsh[2][N_THREAD_GPU];
sample/c++/nbody/force_gpu_cuda.cu:static cudaPointer<EpiGPU>   dev_epi;
sample/c++/nbody/force_gpu_cuda.cu:static cudaPointer<EpjGPU>   dev_epj;
sample/c++/nbody/force_gpu_cuda.cu:static cudaPointer<ForceGPU> dev_force;
sample/c++/nbody/force_gpu_cuda.cu:static cudaPointer<int2>     ij_disp;
sample/c++/nbody/force_gpu_cuda.cu:    if(ni_tot_reg % N_THREAD_GPU){
sample/c++/nbody/force_gpu_cuda.cu:        ni_tot_reg /= N_THREAD_GPU;
sample/c++/nbody/force_gpu_cuda.cu:        ni_tot_reg *= N_THREAD_GPU;
sample/c++/nbody/force_gpu_cuda.cu:    int nblocks  = ni_tot_reg / N_THREAD_GPU;
sample/c++/nbody/force_gpu_cuda.cu:    int nthreads = N_THREAD_GPU;
sample/c++/nbody/cuda_pointer.h:#include <cuda.h>
sample/c++/nbody/cuda_pointer.h:#include <cuda_runtime.h>
sample/c++/nbody/cuda_pointer.h:#  include <helper_cuda.h>
sample/c++/nbody/cuda_pointer.h:#  define CUDA_SAFE_CALL checkCudaErrors
sample/c++/nbody/cuda_pointer.h:struct cudaPointer{
sample/c++/nbody/cuda_pointer.h:	cudaPointer(){
sample/c++/nbody/cuda_pointer.h://        ~cudaPointer(){
sample/c++/nbody/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMalloc(&p, size * sizeof(T)));
sample/c++/nbody/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMallocHost(&p, size * sizeof(T)));
sample/c++/nbody/cuda_pointer.h:		CUDA_SAFE_CALL(cudaFree(dev_pointer));
sample/c++/nbody/cuda_pointer.h:		CUDA_SAFE_CALL(cudaFreeHost(host_pointer));
sample/c++/nbody/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMemcpy(dev_pointer, host_pointer, count * sizeof(T), cudaMemcpyHostToDevice));
sample/c++/nbody/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMemcpy(host_pointer, dev_pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
sample/c++/nbody/nbody.cpp:#ifdef ENABLE_GPU_CUDA
sample/c++/nbody/nbody.cpp:#include"force_gpu_cuda.hpp"
sample/c++/nbody/force_gpu_cuda.hpp:#include <helper_cuda.h>
pikg/sample/c++/Nbody/Makefile:ifeq ($(use_cuda),yes)
pikg/sample/c++/Nbody/Makefile:CONVERSION_TYPE=CUDA
pikg/sample/c++/Nbody/Makefile:CCFLAGS+= -DUSE_CUDA_KERNEL
pikg/sample/c++/Nbody/README.md:# for CUDA mode
pikg/sample/c++/Nbody/README.md:use_cuda=yes make
pikg/sample/c++/Nbody/nbody.cpp:#ifndef USE_CUDA_KERNEL
pikg/README.md:粒子間相互作用をDSLで記述し，パラメータを指定すると，任意のアーキテクチャ(Intel CPU, Fujitsu A64FX, NVIDIA GPU, PEZY-SC2, etc.)向けのカーネルを生成する．
pikg/README.md:- CUDA
pikg/README.md:- GPUカーネル生成機能を追加
pikg/inc/pikg_cuda_pointer.hpp:#ifndef PIKG_CUDA_POINTER
pikg/inc/pikg_cuda_pointer.hpp:#define PIKG_CUDA_POINTER
pikg/inc/pikg_cuda_pointer.hpp:  class CUDAPointer{
pikg/inc/pikg_cuda_pointer.hpp:    ~CUDAPointer(){
pikg/inc/pikg_cuda_pointer.hpp:	cudaFree(dev);
pikg/inc/pikg_cuda_pointer.hpp:      cudaMalloc((void**)&dev,n*sizeof(T));
pikg/inc/pikg_cuda_pointer.hpp:      cudaMemcpy(dev,hst,n*sizeof(T),cudaMemcpyHostToDevice);
pikg/inc/pikg_cuda_pointer.hpp:      cudaMemcpy(hst,dev,n*sizeof(T),cudaMemcpyDeviceToHost);
pikg/inc/pikg_cuda_pointer.hpp:#endif // PIKG_CUDA_POINTER
pikg/src/CUDA.rb:  def generate_optimized_cuda_kernel(conversion_type,h = $varhash)
pikg/src/CUDA.rb:    abort "error: --class-file option must be specified for conversion_type CUDA" if $epi_file == nil
pikg/src/CUDA.rb:    code += "#include \"pikg_cuda_pointer.hpp\"\n"
pikg/src/CUDA.rb:    # GPU class definition
pikg/src/CUDA.rb:    code += "struct EpiGPU{\n"
pikg/src/CUDA.rb:    code += "struct EpjGPU{\n"
pikg/src/CUDA.rb:    code += "struct ForceGPU{\n"
pikg/src/CUDA.rb:    code += "  N_THREAD_GPU = 32,\n"
pikg/src/CUDA.rb:    code += "inline __device__ ForceGPU inner_kernel(\n"
pikg/src/CUDA.rb:    code += "				     EpiGPU epi,\n"
pikg/src/CUDA.rb:    code += "				     EpjGPU epj,\n"
pikg/src/CUDA.rb:    code += "				     ForceGPU force"
pikg/src/CUDA.rb:    code += "__device__ ForceGPU ForceKernel_1walk(\n"
pikg/src/CUDA.rb:    code += "				    EpjGPU *jpsh,\n"
pikg/src/CUDA.rb:    code += "				    const EpiGPU ipos,\n"
pikg/src/CUDA.rb:    code += "				    const EpjGPU *epj, \n"
pikg/src/CUDA.rb:    code += "				    ForceGPU accp"
pikg/src/CUDA.rb:    code += "  for(int j=j_head; j<j_tail; j+=N_THREAD_GPU){\n"
pikg/src/CUDA.rb:    code += "    if(j_tail-j < N_THREAD_GPU){\n"
pikg/src/CUDA.rb:    code += "      for(int jj=0; jj<N_THREAD_GPU; jj++){\n"
pikg/src/CUDA.rb:    code += "__device__ ForceGPU ForceKernel_2walk(\n"
pikg/src/CUDA.rb:    code += "				    EpjGPU jpsh[2][N_THREAD_GPU],\n"
pikg/src/CUDA.rb:    code += "				    const EpiGPU  ipos,\n"
pikg/src/CUDA.rb:    code += "				    const EpjGPU *epj, \n"
pikg/src/CUDA.rb:    code += "				    ForceGPU accp"
pikg/src/CUDA.rb:    code += "  for(int j=0; j<nj_shorter; j+=N_THREAD_GPU){\n"
pikg/src/CUDA.rb:    code += "    if(nj_shorter-j < N_THREAD_GPU){\n"
pikg/src/CUDA.rb:    code += "      for(int jj=0; jj<N_THREAD_GPU; jj++){\n"
pikg/src/CUDA.rb:    code += "  for(int j=nj_shorter; j<nj_longer; j+=N_THREAD_GPU){\n"
pikg/src/CUDA.rb:    code += "    if(jrem < N_THREAD_GPU){\n"
pikg/src/CUDA.rb:    code += "      for(int jj=0; jj<N_THREAD_GPU; jj++){\n"
pikg/src/CUDA.rb:    code += "__device__ ForceGPU ForceKernel_multiwalk(\n"
pikg/src/CUDA.rb:    code += "					const EpiGPU ipos,\n"
pikg/src/CUDA.rb:    code += "					const EpjGPU *epj, \n"
pikg/src/CUDA.rb:    code += "					ForceGPU accp"
pikg/src/CUDA.rb:    code += "    EpjGPU jp = epj[j];\n"
pikg/src/CUDA.rb:    code += "__global__ void #{$kernel_name}_cuda(\n"
pikg/src/CUDA.rb:    code += "                  const EpiGPU * epi,\n"
pikg/src/CUDA.rb:    code += "                  const EpjGPU * epj, \n"
pikg/src/CUDA.rb:    code += "                  ForceGPU     * force"
pikg/src/CUDA.rb:    code += "  const EpiGPU ip = epi[tid];\n"
pikg/src/CUDA.rb:    code += "  ForceGPU accp;\n"
pikg/src/CUDA.rb:    code += "  int t_tail = t_head + N_THREAD_GPU - 1;\n"
pikg/src/CUDA.rb:    code += "  __shared__ EpjGPU jpsh[2][N_THREAD_GPU];\n"
pikg/src/CUDA.rb:    code += "static PIKG::CUDAPointer<EpiGPU>   dev_epi;\n"
pikg/src/CUDA.rb:    code += "static PIKG::CUDAPointer<EpjGPU>   dev_epj;\n"
pikg/src/CUDA.rb:    code += "static PIKG::CUDAPointer<ForceGPU> dev_force;\n"
pikg/src/CUDA.rb:    code += "static PIKG::CUDAPointer<int>  ij_disp;\n"
pikg/src/CUDA.rb:    code += "static PIKG::CUDAPointer<int>   walk;\n"
pikg/src/CUDA.rb:    # epi copy to epi_gpu
pikg/src/CUDA.rb:    # epj copy to epj_gpu
pikg/src/CUDA.rb:      # spj copy to spj_gpu
pikg/src/CUDA.rb:    code += "  if(ni_tot_reg % N_THREAD_GPU){\n"
pikg/src/CUDA.rb:    code += "    ni_tot_reg /= N_THREAD_GPU;\n"
pikg/src/CUDA.rb:    code += "    ni_tot_reg *= N_THREAD_GPU;\n"
pikg/src/CUDA.rb:    code += "  int nblocks  = ni_tot_reg / N_THREAD_GPU;\n"
pikg/src/CUDA.rb:    code += "  int nthreads = N_THREAD_GPU;\n"
pikg/src/CUDA.rb:    code += "  #{$kernel_name}_cuda <<<nblocks, nthreads>>> (ij_disp, walk,  dev_epi, dev_epj, dev_force"
pikg/src/common.rb:  when /CUDA/
pikg/src/parserdriver.rb:require_relative "CUDA.rb"
pikg/src/parserdriver.rb:  when "CUDA"
pikg/src/parserdriver.rb:    program.generate_optimized_cuda_kernel($conversion_type)
src/CHANGELOG:* add functions Comm::getRankMultiDim() and Comm::getNumberOfProcMultiDim().
src/domain_info.hpp:            Comm::setNumberOfProcMultiDim(0, nx);
src/domain_info.hpp:            Comm::setNumberOfProcMultiDim(1, ny);
src/domain_info.hpp:            Comm::setNumberOfProcMultiDim(2, nz);
src/fortran_interface/blueprints/FDPS_ftn_if_blueprint.cpp:   return ((PS::CommInfo*)(ci))->getNumberOfProcMultiDim(id);
src/fortran_interface/blueprints/FDPS_ftn_if_blueprint.cpp:   return ((PS::CommInfo*)(comm_table+ci))->getNumberOfProcMultiDim(id);
src/fortran_interface/blueprints/FDPS_ftn_if_blueprint.cpp:   return PS::Comm::getNumberOfProcMultiDim(id);
src/ps_defs.hpp:        void setNumberOfProcMultiDim(const S32 id, const S32 n) {
src/ps_defs.hpp:        S32 getNumberOfProcMultiDim(const S32 id) {
src/ps_defs.hpp:        static void setNumberOfProcMultiDim(const S32 id, const S32 n) {
src/ps_defs.hpp:        static S32 getNumberOfProcMultiDim(const S32 id) {

```
