# https://github.com/TinkerTools/tinker9

```console
_clang-format:    - SEQ_CUDA
LICENSE.md:      > Tinker9: Next Generation of Tinker with GPU Support.
test/testrt.cpp:   inform::gpucard = 1;
test/reduce.cpp:                    "gpu-package cuda\n";
test/box.cpp:#include "tool/gpucard.h"
test/box.cpp:   gpuData(RcOp::ALLOC | RcOp::INIT);
test/box.cpp:   gpuData(RcOp::DEALLOC);
test/box.cpp:   gpuData(RcOp::ALLOC | RcOp::INIT);
test/box.cpp:   gpuData(RcOp::DEALLOC);
test/box.cpp:   gpuData(RcOp::ALLOC | RcOp::INIT);
test/box.cpp:   gpuData(RcOp::DEALLOC);
test/nblist.cpp:TEST_CASE("NBList-ArBox", "[ff][nblist][arbox][mixcuda]")
test/async.cpp:static int gpu_value;
test/async.cpp:   gpu_value = 0;
test/async.cpp:   int dup_value = gpu_value;
test/async.cpp:   printf("== ASYNC_WRITE_1 %4d | GPU VALUE %4d\n", idx, gpu_value);
test/async.cpp:   // time spent on getting gpu buffer and writing to external files
test/async.cpp:   printf("== ASYNC_WRITE_2 %4d | GPU VALUE %4d | HOST %4d\n", idx, gpu_value, host_value);
test/async.cpp:   // 1. async duplicate gpu value to another gpu buffer;
test/async.cpp:   // 2. async get gpu buffer to host and then write to the external files
test/async.cpp:   // gpu_value may change after exiting this routine even BEFORE
test/async.cpp:      gpu_value -= 1;
test/async.cpp:      printf("STEP %4d | GPU VALUE %4d\n", i, gpu_value);
test/osrw.cpp:TEST_CASE("K-Water", "[ff][osrw][mixcuda]")
test/osrw.cpp:TEST_CASE("K-Water-Analyze", "[ff][osrw][mixcuda]")
test/mathfunc.cpp:#if TINKER_CUDART
ext/ext/catch_2_13_9:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
ext/ext/catch_2_13_9:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
ext/ext/catch_2_13_9:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
ext/ext/catch_2_9_1:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
ext/ext/catch_2_13_8:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
ext/ext/catch_2_13_8:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
ext/ext/catch_2_13_8:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
ext/ext/y3/genall.sh:# Run a python script for all YAML files in this directory to generate their cuda kernals
ext/ext/catch_2_13_4:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__)
ext/ext/catch_2_13_4:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
ext/ext/catch_2_13_4:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
ext/ext/ck3.py:    def cudaReplaceDict(self) -> dict:
ext/ext/ck3.py:        d = self.cudaReplaceDict()
ext/interface/cpp/tinker/detail/inform.hh:extern int& gpucard;
ext/interface/cpp/tinker/detail/inform.hh:extern "C" int TINKER_MOD(inform, gpucard);
ext/interface/cpp/tinker/detail/inform.hh:int& gpucard = TINKER_MOD(inform, gpucard);
doc/Doxyfile:                         TINKER_CUDART \
doc/Doxyfile:                         __CUDACC__
doc/manual/m/key/parallel.rst:**CUDA-DEVICE [integer]**
doc/manual/m/key/parallel.rst:.. index:: CUDA-DEVICE
doc/manual/m/key/parallel.rst:.. index:: CUDA_DEVICE
doc/manual/m/key/parallel.rst:Followed by an integer value starting from 0, sets the CUDA-enabled
doc/manual/m/key/parallel.rst:GPU device for the program. Value will be overwritten by environment variable
doc/manual/m/key/parallel.rst:*CUDA_DEVICE*.
doc/manual/m/key/parallel.rst:For instance, a node has four CUDA devices, and the *CUDA_VISIBLE_DEVICES*
doc/manual/m/key/parallel.rst:environment variable (part of CUDA library) has been set to
doc/manual/m/key/parallel.rst:*CUDA_VISIBLE_DEVICES=1,3*. This means only two CUDA devices are avaiable
doc/manual/m/key/parallel.rst:here, thus the valid values for *CUDA-DEVICE* are 0 and 1.
doc/manual/m/key/parallel.rst:**GPU-PACKAGE [CUDA / OPENACC]** |not8|
doc/manual/m/key/parallel.rst:.. index:: GPU-PACKAGE
doc/manual/m/key/parallel.rst:.. index:: GPU_PACKAGE
doc/manual/m/key/parallel.rst:Selects code paths for some GPU algorithms where both CUDA and
doc/manual/m/key/parallel.rst:OpenACC versions have been implemented.
doc/manual/m/key/parallel.rst:The default value is CUDA. Value will be overwritten by environment variable
doc/manual/m/key/parallel.rst:*GPU_PACKAGE*.
doc/manual/m/install/preq.rst:A relatively recent NVIDIA GPU is mandatory for the GPU code.
doc/manual/m/install/preq.rst:The oldest NVIDIA GPU Tinker9 has been tested on is GeForce GTX 675MX (compute capability 3.0).
doc/manual/m/install/preq.rst:CUDA/nvcc          [b]
doc/manual/m/install/preq.rst:OpenACC/NVHPC/PGI  [c]
doc/manual/m/install/preq.rst:- [b] GPU code only. Version >= 9.0.
doc/manual/m/install/preq.rst:- [c] Optional for the GPU code. A recent `NVIDIA HPC SDK <https://www.developer.nvidia.com/hpc-sdk>`_ is preferred.
doc/manual/m/install/preq.rst:- [d] We have successfully built Tinker9 on Windows WSL2 Ubuntu with CUDA 11.0 and NVHPC 20.9. Please `check this link <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_ for more details.
doc/manual/m/install/preq.rst:**Using NVIDIA HPC SDK on Clusters**
doc/manual/m/install/preq.rst:Prior to rebranding, the current NVIDIA HPC SDK was known as the PGI compiler
doc/manual/m/install/preq.rst:therefore FFTW libraries are no longer mandatory for GPU code.
doc/manual/m/install/buildwithcmake.rst:For a GPU card with compute capability 7.0,
doc/manual/m/install/buildwithcmake.rst:an example to compile the GPU code without OpenACC:
doc/manual/m/install/buildwithcmake.rst:   FC=gfortran compute_capability=70 gpu_lang=cuda cmake ..
doc/manual/m/install/buildwithcmake.rst:Assuming separate CUDA and NVHPC are properly installed,
doc/manual/m/install/buildwithcmake.rst:another example to compile the GPU code with both OpenACC and CUDA:
doc/manual/m/install/buildwithcmake.rst:For the options of other GPU devices and features,
doc/manual/m/install/buildwithcmake.rst:Set *CXX=...*, *CUDACXX=...*, and *FC=...* to specify the non-default C++,
doc/manual/m/install/buildwithcmake.rst:CUDA, and Fortran compilers, respectively. These environmental variables
doc/manual/m/install/buildwithcmake.rst:*only* for the OpenACC GPU code.
doc/manual/m/install/buildwithcmake.rst:If not set, the building script will take a guess at the OpenACC compiler.
doc/manual/m/install/buildwithcmake.rst:computers that had Nvidia support. clang is hardwired in the cmake scripts
doc/manual/m/install/buildwithcmake.rst:Flag to compile to GPU (with value 0 or OFF) or CPU (with value 1 or ON)
doc/manual/m/install/buildwithcmake.rst:**-DGPU_LANG (gpu_lang) = OPENACC**
doc/manual/m/install/buildwithcmake.rst:If set to *CUDA*, the GPU code will only use the cuda source files.
doc/manual/m/install/buildwithcmake.rst:And the program will crash at runtime if it falls into an OpenACC code path.
doc/manual/m/install/buildwithcmake.rst:GPU code only.
doc/manual/m/install/buildwithcmake.rst:CUDA compute capability (multiplied by 10) of GPU.
doc/manual/m/install/buildwithcmake.rst:If left unspecified, the script will attempt to detect the GPU,
doc/manual/m/install/buildwithcmake.rst:`NVIDIA website. <https://developer.nvidia.com/cuda-gpus>`_
doc/manual/m/install/buildwithcmake.rst:**-DCUDA_DIR (cuda_dir) = /usr/local/cuda**
doc/manual/m/install/buildwithcmake.rst:Nvidia GPU code only.
doc/manual/m/install/buildwithcmake.rst:Top-level CUDA installation directory, under which directories *include*,
doc/manual/m/install/buildwithcmake.rst:This option will supersede the CUDA installation identified by the official
doc/manual/m/install/buildwithcmake.rst:*CUDACXX* environmental variable.
doc/manual/m/install/buildwithcmake.rst:instance, although PGI 19.4 supports CUDA 9.2, 10.0, 10.1, but the default
doc/manual/m/install/buildwithcmake.rst:CUDA version configured in PGI 19.4 may be 9.2 and the external NVCC version
doc/manual/m/install/buildwithcmake.rst:is 10.1. One solution is to pass *CUDA_HOME=${cuda_dir}* to the PGI
doc/manual/m/install/buildwithcmake.rst:compiler, in which case, **cuda_dir** should be set to
doc/manual/m/install/buildwithcmake.rst:*/usr/local/cuda-10.1*.
doc/doc.h: *    \defgroup cuda_syntax             CUDA Syntax
doc/doc.h: *    \defgroup acc_syntax              OpenACC Syntax
doc/doc.h: *    \defgroup nvidia                  NVIDIA GPU
doc/doc.h: *      a 32-bit integer and CUDA warp size.
doc/doc.h: *       - `__popc` available in NVCC CUDA.
doc/doc.h: *       - `int __ffs` available in NVCC CUDA.
doc/doc.h: *    5. (CUDA) Given a local variable `var` which may have different values in
doc/doc.h: *    6. (CUDA) *Generally*, calling `__ballot_sync` with `val` will return a
README.md:Tinker9: Next Generation of Tinker with GPU Support
README.md:Tinker9 is a complete rewrite and extension of the canonical Tinker software, currently Tinker8. Tinker9 is implemented as C++ code with OpenACC directives and CUDA kernels providing excellent performance on GPUs. At present, Tinker9 builds against the object library from Tinker8, and provides GPU versions of the Tinker ANALYZE, BAR, DYNAMIC, MINIMIZE and TESTGRAD programs. Existing Tinker file formats and force field parameter files are fully compatible with Tinker9, and nearly all Tinker8 keywords function identically in Tinker9. Over time we plan to port much or all of the remaining portions of Fortran Tinker8 to the C++ Tinker9 code base.
README.md:[C++ and CUDA Code Documentation (Doxygen)](https://tinkertools.github.io/tinker9/)
include/math/libfunc.h:#if TINKER_CUDART
include/math/libfunc.h:#ifdef _OPENACC
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/sinhc.h:SEQ_CUDA
include/math/parallelacc.h:T reduceSum_acc(const T* gpu_a, size_t nelem, int queue);
include/math/parallel.h:T reduceSum(const T* gpu_a, size_t nelem, int queue)
include/math/parallel.h:   return TINKER_FCALL2(acc1, cu1, reduceSum, gpu_a, nelem, queue);
include/math/parallel.h:/// \param queue   OpenACC queue.
include/math/parallel.h:/// \param queue  OpenACC queue.
include/math/switch.h:SEQ_CUDA
include/syntax/acc/seqdef.h:/// Expands to \c _Pragma("acc routine seq") in OpenACC source files.
include/syntax/acc/seqdef.h:/// \def SEQ_CUDA
include/syntax/acc/seqdef.h:/// An empty macro in the OpenACC source code.
include/syntax/acc/seqdef.h:#define SEQ_CUDA
include/syntax/cu/adddef.h:// docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
include/syntax/cu/adddef.h:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
include/syntax/cu/adddef.h:/// \addtogroup cuda_syntax
include/syntax/cu/seqdef.h:/// \ingroup cuda_syntax
include/syntax/cu/seqdef.h:/// Expands to \c __device__ in CUDA source files.
include/syntax/cu/seqdef.h:/// \def SEQ_CUDA
include/syntax/cu/seqdef.h:/// \ingroup cuda_syntax
include/syntax/cu/seqdef.h:/// Expands to \c __device__ in CUDA source files.
include/syntax/cu/seqdef.h:/// Used in CUDA kernel templates.
include/syntax/cu/seqdef.h:#define SEQ_CUDA __device__
include/ff/atom.h:/// molecules back into the periodic box on GPU.
include/ff/image.h:/// This is because of a [bug](https://forums.developer.nvidia.com/t/136445)
include/ff/box.h:/// OpenACC only: Copies the current box data to device asynchronously.
include/ff/box.h:/// The implementations are empty for CPU and CUDA code because it is only in
include/ff/box.h:/// the OpenACC code that a copy of the PBC box is created on device.
include/ff/amoeba/emplar.h:///    - not using GPU;
include/ff/amoeba/emplar.h:///    - not using CUDA as the primary GPU package;
include/tool/fft.h:/// [cuFFT C API](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftplan3d)
include/tool/gpucard.h:#if TINKER_CUDART
include/tool/gpucard.h:/// \addtogroup nvidia
include/tool/gpucard.h:/// \return  CUDA runtime version.
include/tool/gpucard.h:std::string gpuCudaRuntimeVersion();
include/tool/gpucard.h:/// \return  Max CUDA runtime version supported by the driver.
include/tool/gpucard.h:std::string gpuCudaDriverVersion();
include/tool/gpucard.h:std::string gpuThrustVersion();
include/tool/gpucard.h:std::vector<DeviceAttribute>& gpuDeviceAttributes();
include/tool/gpucard.h:/// Sets up the GPU card.
include/tool/gpucard.h:void gpuData(RcOp);
include/tool/gpucard.h:int gpuGridSize(int nthreadsPerBlock);
include/tool/gpucard.h:int gpuMaxNParallel(int idev);
include/tool/gpucard.h:/// \addtogroup nvidia
include/tool/macro.h:/// \def TINKER_CUDART
include/tool/macro.h:/// Macro for the CUDA runtime-enabled GPU code.
include/tool/macro.h:#ifdef TINKER_CUDART
include/tool/macro.h:#undef TINKER_CUDART
include/tool/macro.h:#define TINKER_CUDART 1
include/tool/macro.h:#define TINKER_CUDART 0
include/tool/macro.h:/// \def TINKER_GPULANG_OPENACC
include/tool/macro.h:/// Macro for the OpenACC GPU code.
include/tool/macro.h:///    - only the CUDA code is in use on the GPU platform.
include/tool/macro.h:#ifdef TINKER_GPULANG_OPENACC
include/tool/macro.h:#undef TINKER_GPULANG_OPENACC
include/tool/macro.h:#define TINKER_GPULANG_OPENACC 1
include/tool/macro.h:#define TINKER_GPULANG_OPENACC 0
include/tool/macro.h:/// \def TINKER_GPULANG_CUDA
include/tool/macro.h:/// Macro for the CUDA GPU code.
include/tool/macro.h:///    - OpenACC code is in use on the GPU platform.
include/tool/macro.h:#ifdef TINKER_GPULANG_CUDA
include/tool/macro.h:#undef TINKER_GPULANG_CUDA
include/tool/macro.h:#define TINKER_GPULANG_CUDA 1
include/tool/macro.h:#define TINKER_GPULANG_CUDA 0
include/tool/cudalib.h:/// \brief Sets up the CUDA variables including but not limited to CUDA streams,
include/tool/cudalib.h:/// CUDA library handles, CUDA memory buffers, and integer units for the OpenACC queues.
include/tool/cudalib.h:void cudalibData(RcOp);
include/tool/cudalib.h:#if TINKER_CUDART
include/tool/cudalib.h:#include <cuda_runtime.h>
include/tool/cudalib.h:TINKER_EXTERN cudaStream_t s0;   ///< CUDA stream for the default OpenACC async queue. \ingroup async
include/tool/cudalib.h:TINKER_EXTERN cudaStream_t s1;   ///< CUDA stream for the default OpenACC sync queue. \ingroup async
include/tool/cudalib.h:TINKER_EXTERN cudaStream_t spme; ///< CUDA stream for the OpenACC async %PME queue. \ingroup pme
include/tool/cudalib.h:TINKER_EXTERN cublasHandle_t h0; ///< CUDA BLAS handle using #s0. \ingroup async
include/tool/cudalib.h:TINKER_EXTERN cublasHandle_t h1; ///< CUDA BLAS handle using #s1. \ingroup async
include/tool/cudalib.h:TINKER_EXTERN cudaEvent_t pme_event_start;  ///< \ingroup pme
include/tool/cudalib.h:TINKER_EXTERN cudaEvent_t pme_event_finish; ///< \ingroup pme
include/tool/compilers.h:std::string accCompilerName();  ///< Returns the name of the OpenACC compiler.
include/tool/compilers.h:std::string cudaCompilerName(); ///< Returns the name of the CUDA compiler.
include/tool/platform.h:   ACC = 0x001,     ///< Flag for the OpenACC platform.
include/tool/platform.h:   CUDA = 0x002     ///< Flag for the CUDA platform.
include/tool/accasync.h:/// This is a snippet of the OpenACC code.
include/tool/accasync.h:/// A special integer constant value \c acc_async_sync is defined by the OpenACC
include/tool/accasync.h:/// on the CUDA platform every OpenACC queue is built on top of a CUDA stream.
include/tool/accasync.h:/// Global handles for the GPU runtime libraries. \ingroup async
include/tool/accasync.h:TINKER_EXTERN int q0;   ///< Default OpenACC async queue. \ingroup async
include/tool/accasync.h:TINKER_EXTERN int q1;   ///< Default OpenACC sync queue. \ingroup async
include/tool/accasync.h:TINKER_EXTERN int qpme; ///< OpenACC async queue for %PME. \ingroup async
include/tool/accasync.h:TINKER_EXTERN bool use_pme_stream; ///< Logical flag for use of a separate CUDA stream for %PME. \ingroup async
include/tool/externfunc.h:#if TINKER_GPULANG_OPENACC // mixed source code: openacc and cuda
include/tool/externfunc.h:   (pltfm_config & Platform::CUDA) ? TINKER_FCALL0_NORMAL_(F, cu, __VA_ARGS__) : TINKER_FCALL0_NORMAL_(F, acc, __VA_ARGS__)
include/tool/externfunc.h:#elif TINKER_GPULANG_CUDA // pure cuda
include/tool/darray.h:/// \brief Similar to OpenACC wait and CUDA stream synchronize.
include/tool/darray.h:void waitFor(int queue ///< OpenACC queue.
include/tool/darray.h:/// \brief Similar to OpenACC async copyin, copies data from host to device.
include/tool/darray.h:                                  int queue        ///< OpenACC queue.
include/tool/darray.h:/// \brief Similar to OpenACC async copyout, copies data from device to host.
include/tool/darray.h:                                   int queue        ///< OpenACC queue.
include/tool/darray.h:/// \note Different from OpenACC copy.
include/tool/darray.h:                                int queue        ///< OpenACC queue.
include/tool/darray.h:                                int queue      ///< OpenACC queue.
include/tool/darray.h:                               int q          ///< OpenACC queue.
include/tool/darray.h:                                int q          ///< OpenACC queue.
include/testrt.h:#define COMPARE_ENERGY(gpuptr, ref_eng, eps)       \
include/testrt.h:      double eng = energyReduce(gpuptr);           \
include/testrt.h:#define COMPARE_COUNT(gpuptr, ref_count) \
include/testrt.h:      int count = countReduce(gpuptr);   \
include/testrt.h:#define COMPARE_VIR(gpuptr, ref_v, eps)                    \
include/testrt.h:      virialReduce(vir1, gpuptr);                          \
include/testrt.h:#define COMPARE_VIR2(gpuptr, gpuptr2, ref_v, eps)                    \
include/testrt.h:      virialReduce(vir1, gpuptr);                                    \
include/testrt.h:      virialReduce(vir2, gpuptr2);                                   \
include/seq/improp.h:SEQ_CUDA
include/seq/damp_hippo.h:SEQ_CUDA
include/seq/damp_hippo.h:SEQ_CUDA
include/seq/damp_hippo.h:SEQ_CUDA
include/seq/damp_hippo.h:SEQ_CUDA
include/seq/damp_hippo.h:SEQ_CUDA
include/seq/dampaplus.h:SEQ_CUDA
include/seq/pair_field_chgpen.h:SEQ_CUDA
include/seq/pair_field_chgpen.h:SEQ_CUDA
include/seq/angtor.h:SEQ_CUDA
include/seq/urey.h:SEQ_CUDA
include/seq/pair_lj.h:SEQ_CUDA
include/seq/pair_lj.h:SEQ_CUDA
include/seq/pair_lj.h:SEQ_CUDA
include/seq/pair_lj.h:SEQ_CUDA
include/seq/pair_charge.h:SEQ_CUDA
include/seq/pair_charge.h:SEQ_CUDA
include/seq/pair_charge.h:SEQ_CUDA
include/seq/pair_hal.h:SEQ_CUDA
include/seq/pair_hal.h:SEQ_CUDA
include/seq/pairfieldaplus.h:SEQ_CUDA
include/seq/pairfieldaplus.h:SEQ_CUDA
include/seq/pairfieldaplus.h:SEQ_CUDA
include/seq/pairfieldaplus.h:SEQ_CUDA
include/seq/geom.h:SEQ_CUDA
include/seq/geom.h:SEQ_CUDA
include/seq/geom.h:SEQ_CUDA
include/seq/geom.h:SEQ_CUDA
include/seq/geom.h:SEQ_CUDA
include/seq/pair_polar_chgpen.h:SEQ_CUDA
include/seq/pairpolaraplus.h:SEQ_CUDA
include/seq/pairpolaraplus.h:SEQ_CUDA
include/seq/pairmpoleaplus.h:SEQ_CUDA
include/seq/pair_mpole_chgpen.h:SEQ_CUDA
include/seq/copysign.h:/// \note Standard C and CUDA libraries only have float and double versions.
include/seq/settle.h:SEQ_CUDA
include/seq/settle.h:SEQ_CUDA
include/seq/torsion.h:SEQ_CUDA
include/seq/rotpole.h:#if _OPENACC
include/seq/rotpole.h:#if _OPENACC
include/seq/rotpole.h:#if _OPENACC
include/seq/rotpole.h:#if _OPENACC
include/seq/strbnd.h:SEQ_CUDA
include/seq/pairchgtrnaplus.h:SEQ_CUDA
include/seq/angle.h:SEQ_CUDA
include/seq/imptor.h:SEQ_CUDA
include/seq/opbend.h:SEQ_CUDA
include/seq/pair_field.h:SEQ_CUDA
include/seq/pair_field.h:SEQ_CUDA
include/seq/pair_field.h:SEQ_CUDA
include/seq/pair_field.h:SEQ_CUDA
include/seq/damp_hippodisp.h:SEQ_CUDA
include/seq/bsplgen.h:#ifdef __CUDACC__
include/seq/bsplgen.h: * \param bsbuild_ A CUDA working array of size `MAX_BSORDER*MAX_BSORDER`.
include/seq/bsplgen.h:#ifndef __CUDACC__
include/seq/bsplgen.h:#ifdef __CUDACC__
include/seq/pair_disp.h:SEQ_CUDA
include/seq/pair_disp.h:SEQ_CUDA
include/seq/pair_mpole.h: * \brief OpenACC pairwise multipole electrostatic energy.
include/seq/pair_mpole.h:SEQ_CUDA
include/seq/pair_mpole.h:SEQ_CUDA
include/seq/damp.h:SEQ_CUDA
include/seq/bond.h:SEQ_CUDA
include/seq/tortor.h:SEQ_CUDA
include/seq/pair_polar.h:SEQ_CUDA
include/seq/pair_polar.h:SEQ_CUDA
include/seq/launch.h:#include "tool/cudalib.h"
include/seq/launch.h:#include "tool/gpucard.h"
include/seq/launch.h:/// \addtogroup cuda_syntax
include/seq/launch.h:/// Launch a non-blocking CUDA kernel.
include/seq/launch.h:/// \param st  CUDA stream.
include/seq/launch.h:/// \param k   CUDA \c __global__ kernel.
include/seq/launch.h:void launch_k3s(cudaStream_t st, size_t sh, int bs, int np, K k, Ts&&... a)
include/seq/launch.h:/// Launch a non-blocking CUDA kernel.
include/seq/launch.h:/// \param st  CUDA stream.
include/seq/launch.h:/// \param k   CUDA \c __global__ kernel.
include/seq/launch.h:void launch_k3b(cudaStream_t st, size_t sh, int bs, int np, K k, Ts&&... a)
include/seq/launch.h:/// Launch a non-blocking CUDA kernel with 0 dynamic shared memory.
include/seq/launch.h:void launch_k2s(cudaStream_t st, int bs, int np, K k, Ts&&... a)
include/seq/launch.h:/// Launch a non-blocking CUDA kernel with 0 dynamic shared memory.
include/seq/launch.h:void launch_k2b(cudaStream_t st, int bs, int np, K k, Ts&&... a)
include/seq/launch.h:/// Launch a non-blocking CUDA kernel with 0 dynamic shared memory
include/seq/launch.h:void launch_k1s(cudaStream_t st, int np, K k, Ts&&... a)
include/seq/launch.h:/// Launch a non-blocking CUDA kernel with 0 dynamic shared memory
include/seq/launch.h:void launch_k1b(cudaStream_t st, int np, K k, Ts&&... a)
include/seq/strtor.h:SEQ_CUDA
include/seq/pitors.h:SEQ_CUDA
include/seq/pair_chgtrn.h:SEQ_CUDA
include/seq/pair_alterpol.h:SEQ_CUDA
include/seq/pair_alterpol.h:#if _OPENACC
include/seq/pair_repel.h:SEQ_CUDA
CMakeLists.txt:!!/ For a GPU card with compute capability 7.0,
CMakeLists.txt:!!/ an example to compile the GPU code without OpenACC:
CMakeLists.txt:!!/    FC=gfortran compute_capability=70 gpu_lang=cuda cmake ..
CMakeLists.txt:!!/ Assuming separate CUDA and NVHPC are properly installed,
CMakeLists.txt:!!/ another example to compile the GPU code with both OpenACC and CUDA:
CMakeLists.txt:!!/ For the options of other GPU devices and features,
CMakeLists.txt:!!/ Set *CXX=...*, *CUDACXX=...*, and *FC=...* to specify the non-default C++,
CMakeLists.txt:!!/ CUDA, and Fortran compilers, respectively. These environmental variables
CMakeLists.txt:!!/ *only* for the OpenACC GPU code.
CMakeLists.txt:!!/ If not set, the building script will take a guess at the OpenACC compiler.
CMakeLists.txt:!!/ computers that had Nvidia support. clang is hardwired in the cmake scripts
CMakeLists.txt:!!/ Flag to compile to GPU (with value 0 or OFF) or CPU (with value 1 or ON)
CMakeLists.txt:set (HOST ${hostValue} CACHE BOOL "Build Executables for GPU Platform (OFF) or CPU Platform (ON)")
CMakeLists.txt:!!/ **-DGPU_LANG (gpu_lang) = OPENACC**
CMakeLists.txt:!!/ If set to *CUDA*, the GPU code will only use the cuda source files.
CMakeLists.txt:!!/ And the program will crash at runtime if it falls into an OpenACC code path.
CMakeLists.txt:if (DEFINED ENV{gpu_lang})
CMakeLists.txt:   set (gpuLangValue $ENV{gpu_lang})
CMakeLists.txt:   set (gpuLangValue openacc)
CMakeLists.txt:string (TOUPPER ${gpuLangValue} gpuLangValue)
CMakeLists.txt:set (GPU_LANG ${gpuLangValue} CACHE STRING
CMakeLists.txt:   "GPU Programming Language (experimental): OPENACC, CUDA"
CMakeLists.txt:!!/ GPU code only.
CMakeLists.txt:!!/ CUDA compute capability (multiplied by 10) of GPU.
CMakeLists.txt:!!/ If left unspecified, the script will attempt to detect the GPU,
CMakeLists.txt:!!/ `NVIDIA website. <https://developer.nvidia.com/cuda-gpus>`_
CMakeLists.txt:   "[GPU ONLY] CUDA Compute Capability Multiplied by 10 (Comma-Separated)"
CMakeLists.txt:         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvidiacc.py"
CMakeLists.txt:!!/ **-DCUDA_DIR (cuda_dir) = /usr/local/cuda**
CMakeLists.txt:!!/ Nvidia GPU code only.
CMakeLists.txt:!!/ Top-level CUDA installation directory, under which directories *include*,
CMakeLists.txt:!!/ This option will supersede the CUDA installation identified by the official
CMakeLists.txt:!!/ *CUDACXX* environmental variable.
CMakeLists.txt:!!/ instance, although PGI 19.4 supports CUDA 9.2, 10.0, 10.1, but the default
CMakeLists.txt:!!/ CUDA version configured in PGI 19.4 may be 9.2 and the external NVCC version
CMakeLists.txt:!!/ is 10.1. One solution is to pass *CUDA_HOME=${cuda_dir}* to the PGI
CMakeLists.txt:!!/ compiler, in which case, **cuda_dir** should be set to
CMakeLists.txt:!!/ */usr/local/cuda-10.1*.
CMakeLists.txt:if (DEFINED ENV{cuda_dir})
CMakeLists.txt:   get_filename_component (cudaDirValue "$ENV{cuda_dir}" ABSOLUTE
CMakeLists.txt:elseif (DEFINED ENV{CUDACXX})
CMakeLists.txt:   find_program (T9_CUDA_COMPILER $ENV{CUDACXX})
CMakeLists.txt:   get_filename_component (cudaDirValue "${T9_CUDA_COMPILER}" DIRECTORY) # /usr/local/cuda/bin
CMakeLists.txt:   get_filename_component (cudaDirValue "${cudaDirValue}" DIRECTORY)     # /usr/local/cuda
CMakeLists.txt:   set (cudaDirValue /usr/local/cuda)
CMakeLists.txt:set (CUDA_DIR ${cudaDirValue} CACHE PATH "[GPU ONLY] CUDA Directory")
CMakeLists.txt:## cuda compiler
CMakeLists.txt:## set CMAKE_CUDA_COMPILER before project ()
CMakeLists.txt:   set (CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc")
CMakeLists.txt:## openacc compiler
CMakeLists.txt:elseif (GPU_LANG STREQUAL "OPENACC")
CMakeLists.txt:if (GPU_LANG STREQUAL "CUDA")
CMakeLists.txt:   enable_language (CUDA)
CMakeLists.txt:## macro: TINKER_CUDART
CMakeLists.txt:   list (APPEND T9_DEFS TINKER_CUDART)
CMakeLists.txt:   if (GPU_LANG STREQUAL "CUDA")
CMakeLists.txt:      list (APPEND T9_DEFS TINKER_GPULANG_CUDA)
CMakeLists.txt:   elseif (GPU_LANG STREQUAL "OPENACC")
CMakeLists.txt:      list (APPEND T9_DEFS TINKER_GPULANG_OPENACC)
CMakeLists.txt:   if (GPU_LANG STREQUAL "CUDA")
CMakeLists.txt:   elseif (GPU_LANG STREQUAL "OPENACC")
CMakeLists.txt:      set (__T9_INSTALL_DIR "${__T9_INSTALL_DIR}/gpu")
CMakeLists.txt:      if [ ${GPU_LANG} = CUDA ] \;
CMakeLists.txt:         ./all.tests -a --durations yes --order rand --rng-seed time ~[noassert]~[mixcuda] \;
docker/tinker9.docker:lCudaKeys = ['10.1', '11.2.2']
docker/tinker9.docker:    # cuda10.1
docker/tinker9.docker:    # cuda11.2.2
docker/tinker9.docker:    tinker9.docker [CudaVersion] [Stage]
docker/tinker9.docker:    tinker9.docker [CudaVersion] [Stage] | bash
docker/tinker9.docker:        line += '\nCudaVersion'
docker/tinker9.docker:        for k in lCudaKeys:
docker/tinker9.docker:    def __init__(self, cudaver: str):
docker/tinker9.docker:        self.cudaVersion = cudaver
docker/tinker9.docker:        self.config = ConfigData[cudaver]
docker/tinker9.docker:        return 'Dockerfile-%s' % self.cudaVersion
docker/tinker9.docker:        cuda = self.cudaVersion
docker/tinker9.docker:        return 'cuda%s' % (cuda)
docker/tinker9.docker:    def __init__(self, cudaver: str):
docker/tinker9.docker:        super().__init__(cudaver)
docker/tinker9.docker:        return 'devel-%s.dockerfile' % self.cudaVersion
docker/tinker9.docker:        c = 'FROM nvidia/cuda:%s' % self.config[kDevelUbuntu]
docker/tinker9.docker:        # c += 'ENV PATH="$PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/%s/compilers/bin"' % self.nvhpc
docker/tinker9.docker:        # c += 'ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/%s/compilers/lib"' % self.nvhpc
docker/tinker9.docker:    def __init__(self, cudaver: str):
docker/tinker9.docker:        super().__init__(cudaver)
docker/tinker9.docker:        return 'runtime-%s.dockerfile' % self.cudaVersion
docker/tinker9.docker:        c = '''FROM nvidia/cuda:%s''' % self.config[kRuntimeUbuntu]
docker/tinker9.docker:#         develFile = DevelFile(self.cudaVersion)
docker/tinker9.docker:# FROM nvidia/cuda:%s''' % (develFile.tag(), self.config[kRuntimeUbuntu])
docker/tinker9.docker:#         c += '''ARG LibPath=/opt/nvidia/hpc_sdk/Linux_x86_64/%s/compilers/lib
docker/tinker9.docker:    def __init__(self, cudaver: str):
docker/tinker9.docker:        super().__init__(cudaver)
docker/tinker9.docker:        return 'compile-%s.dockerfile' % self.cudaVersion
docker/tinker9.docker:        return 'temp_build_tinker9-%s' % self.cudaVersion
docker/tinker9.docker:        develFile = DevelFile(self.cudaVersion)
docker/tinker9.docker:ENV CUDA_HOME=/usr/local/cuda
docker/tinker9.docker:RUN cmake $T9Dir -B $T9Dir/build -DCOMPUTE_CAPABILITY=%s -DGPU_LANG=CUDA -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_INSTALL_PREFIX=$T9Dir/bin
docker/tinker9.docker:RUN make install''' % (self.cmakeArgComputeCapability(self.cudaVersion), DefaultNProc())
docker/tinker9.docker:    def __init__(self, cudaver: str):
docker/tinker9.docker:        super().__init__(cudaver)
docker/tinker9.docker:        self._tagInstallFile = 'tinkertools/tinker9:cuda%s-%s-%s' % (self.cudaVersion, t, githash)
docker/tinker9.docker:        return 'install-%s.dockerfile' % self.cudaVersion
docker/tinker9.docker:        runtimeFile = RuntimeFile(self.cudaVersion)
docker/tinker9.docker:        compileFile = CompileFile(self.cudaVersion)
docker/tinker9.docker:ENV PATH="$PATH:/home/tinker9/bin/gpu-m"''' % (compileFile.tag(), runtimeFile.tag())
docker/tinker9.docker:        compileFile = CompileFile(self.cudaVersion)
docker/tinker9.docker:        cmd = 'docker tag %s tinkertools/tinker9:cuda%s-latest' % (self.tag(), self.cudaVersion)
docker/tinker9.docker:def DockerFileFactory(cudaver: str, stage: str) -> DockerFile:
docker/tinker9.docker:    d = DockerFile(cudaver)
docker/tinker9.docker:        d = DevelFile(cudaver)
docker/tinker9.docker:        d = RuntimeFile(cudaver)
docker/tinker9.docker:        d = CompileFile(cudaver)
docker/tinker9.docker:        d = InstallFile(cudaver)
docker/tinker9.docker:            cudaver, stage = sys.argv[1], sys.argv[2]
docker/tinker9.docker:            d = DockerFileFactory(cudaver, stage)
docker/tinker9.docker:    # cuda   : nvhpc
docker/tinker9.docker:            c += '''RUN wget https://developer.download.nvidia.com/hpc-sdk/20.9/nvhpc-20-9_20.9_amd64.deb https://developer.download.nvidia.com/hpc-sdk/20.9/nvhpc-2020_20.9_amd64.deb
docker/tinker9.docker:            c += '''RUN echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | tee /etc/apt/sources.list.d/nvhpc.list
docker/tinker9.docker:RUN apt update -y && apt install -y nvhpc-22-3-cuda-multi'''
docker/README.md:tinker9.docker [CudaVersion] [Stage] | bash
docker/README.md:Use command `tinker9.docker -h` to see the valid values of `CudaVersion` and `Stage`.
docker/README.md:Checking the newest official installation guide for nvidia container toolkit
docker/README.md:### 5. Install the nvidia container toolkit
docker/README.md:curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
docker/README.md:curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
docker/README.md:    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
docker/README.md:sudo apt install -y nvidia-container-toolkit
cmake/device.cmake:if (GPU_LANG STREQUAL "OPENACC")
cmake/device.cmake:      -Mcudalib=cufft,cublas
cmake/device.cmake:# elseif (GPU_LANG STREQUAL "CUDA")
cmake/device.cmake:if (GPU_LANG STREQUAL "OPENACC")
cmake/device.cmake:         LINK_FLAGS " CUDA_HOME=${CUDA_DIR}"
cmake/device.cmake:            ${CMAKE_INSTALL_NAME_TOOL} -add_rpath "${CUDA_DIR}/lib" "${var}"
cmake/nvidiacc.py:This script detects the compute capabilities (X.Y) of the the GPU cards and
cmake/nvidiacc.py:returns them in a comma-separated XY string, if multiple GPUs are detected.
cmake/nvidiacc.py:    libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll', 'cuda.dll')
cmake/nvidiacc.py:            cuda = ctypes.CDLL(libname)
cmake/nvidiacc.py:        CUDA_SUCCESS = 0
cmake/nvidiacc.py:        if result != CUDA_SUCCESS:
cmake/nvidiacc.py:            cuda.cuGetErrorString(result, ctypes.byref(error_str))
cmake/nvidiacc.py:    # from cuda.h
cmake/nvidiacc.py:    checkCall(cuda.cuInit, 0)
cmake/nvidiacc.py:    nGpus = ctypes.c_int()
cmake/nvidiacc.py:    checkCall(cuda.cuDeviceGetCount, ctypes.byref(nGpus))
cmake/nvidiacc.py:    for i in range(nGpus.value):
cmake/nvidiacc.py:        checkCall(cuda.cuDeviceGet, ctypes.byref(device), i)
cmake/nvidiacc.py:        checkCall(cuda.cuDeviceGetAttribute, ctypes.byref(cc_major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
cmake/nvidiacc.py:        checkCall(cuda.cuDeviceGetAttribute, ctypes.byref(cc_minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
src/acc/cudalib.cpp:#if TINKER_CUDART
src/acc/cudalib.cpp:#   include "tool/cudalib.h"
src/acc/cudalib.cpp:#   include <openacc.h>
src/acc/cudalib.cpp:void cudalibDataStreamAndQ_acc(RcOp op)
src/acc/cudalib.cpp:#if TINKER_GPULANG_OPENACC
src/acc/cudalib.cpp:      g::s0 = (cudaStream_t)acc_get_cuda_stream(g::q0);
src/acc/cudalib.cpp:      g::s1 = (cudaStream_t)acc_get_cuda_stream(g::q1);
src/acc/cudalib.cpp:      if (pltfm_config & Platform::CUDA) {
src/acc/cudalib.cpp:         g::spme = (cudaStream_t)acc_get_cuda_stream(g::qpme);
src/acc/pme.cpp:static void pmeConv_acc1(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
src/acc/pme.cpp:               deviceptr(gpu_e,gpu_vir,qgrid,bsmod1,bsmod2,bsmod3)
src/acc/pme.cpp:               atomic_add(eterm, gpu_e, i & (bufsize - 1));
src/acc/pme.cpp:               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_vir, i & (bufsize - 1));
src/acc/pme.cpp:void pmeConv_acc(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
src/acc/pme.cpp:   if (gpu_vir == nullptr) {
src/acc/pme.cpp:      if (gpu_e == nullptr) {
src/acc/pme.cpp:         pmeConv_acc1<true, false>(pme_u, gpu_e, nullptr);
src/acc/pme.cpp:      if (gpu_e == nullptr) {
src/acc/pme.cpp:         pmeConv_acc1<false, true>(pme_u, nullptr, gpu_vir);
src/acc/pme.cpp:         pmeConv_acc1<true, true>(pme_u, gpu_e, gpu_vir);
src/acc/pme.cpp:void fphiMpole_acc(PMEUnit pme_u, real (*gpu_fphi)[20])
src/acc/pme.cpp:   fphiGet_acc<MPOLE>(pme_u, (real*)gpu_fphi, nullptr, nullptr);
src/acc/pme.cpp:void fphiUind_acc(PMEUnit pme_u, real (*gpu_fdip_phi1)[10], real (*gpu_fdip_phi2)[10],
src/acc/pme.cpp:   real (*gpu_fdip_sum_phi)[20])
src/acc/pme.cpp:   fphiGet_acc<UIND>(pme_u, (real*)gpu_fdip_phi1, (real*)gpu_fdip_phi2, (real*)gpu_fdip_sum_phi);
src/acc/pme.cpp:void fphiUind2_acc(PMEUnit pme_u, real (*gpu_fdip_phi1)[10], real (*gpu_fdip_phi2)[10])
src/acc/pme.cpp:   fphiGet_acc<UIND2>(pme_u, (real*)gpu_fdip_phi1, (real*)gpu_fdip_phi2, nullptr);
src/acc/echarge.cpp:#include "tool/gpucard.h"
src/acc/echarge.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/ehal.cpp:#include "tool/gpucard.h"
src/acc/ehal.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/compilers.cpp:#if TINKER_GPULANG_OPENACC
src/acc/CMakeLists.txt:elseif (GPU_LANG STREQUAL "OPENACC")
src/acc/CMakeLists.txt:   string (APPEND CMAKE_CXX_FLAGS " CUDA_HOME=${CUDA_DIR}")
src/acc/elj.cpp:#include "tool/gpucard.h"
src/acc/elj.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/nblist.cpp:#include "tool/gpucard.h"
src/acc/nblist.cpp:#if TINKER_CUDART
src/acc/nblist.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/cmakesrc.txt:cudalib.cpp
src/acc/hippo/edisp.cpp:#include "tool/gpucard.h"
src/acc/hippo/edisp.cpp:static void pmeConvDisp_acc1(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_v)
src/acc/hippo/edisp.cpp:               deviceptr(gpu_e,gpu_v,qgrid,bsmod1,bsmod2,bsmod3)
src/acc/hippo/edisp.cpp:            atomic_add(e, gpu_e, i & (bufsize - 1));
src/acc/hippo/edisp.cpp:            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_v, i & (bufsize - 1));
src/acc/hippo/edisp.cpp:   MAYBE_UNUSED int ngrid = gpuGridSize(BLOCK_DIM);
src/acc/hippo/echgtrn.cpp:#include "tool/gpucard.h"
src/acc/hippo/echgtrn.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/field.cpp:#include "tool/gpucard.h"
src/acc/hippo/field.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/field.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/induce.cpp:#include "tool/gpucard.h"
src/acc/hippo/induce.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/expol.cpp:#include "tool/gpucard.h"
src/acc/hippo/expol.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/expol.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/empole.cpp:#include "tool/gpucard.h"
src/acc/hippo/empole.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/erepel.cpp:#include "tool/gpucard.h"
src/acc/hippo/erepel.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/epolar.cpp:#include "tool/gpucard.h"
src/acc/hippo/epolar.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/hippo/epolar.cpp:static void epolarChgpenEwaldRecipSelf_acc1(const real (*gpu_uind)[3], bool use_cf)
src/acc/hippo/epolar.cpp:   cuindToFuind(pu, gpu_uind, gpu_uind, fuind, fuind);
src/acc/hippo/epolar.cpp:               rpole,cmp,gpu_uind,cphidp,pot)
src/acc/hippo/epolar.cpp:      real uix = gpu_uind[i][0];
src/acc/hippo/epolar.cpp:      real uiy = gpu_uind[i][1];
src/acc/hippo/epolar.cpp:      real uiz = gpu_uind[i][2];
src/acc/hippo/epolar.cpp:         uix = gpu_uind[i][0];
src/acc/hippo/epolar.cpp:         uiy = gpu_uind[i][1];
src/acc/hippo/epolar.cpp:         uiz = gpu_uind[i][2];
src/acc/hippo/epolar.cpp:                  gpu_uind,fphid,cphi,cphidp)\
src/acc/hippo/epolar.cpp:            0.5f * ((gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][1]);
src/acc/hippo/epolar.cpp:               ((gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][1] +
src/acc/hippo/epolar.cpp:                  (gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][2]);
src/acc/hippo/epolar.cpp:               ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][1] +
src/acc/hippo/epolar.cpp:                  (gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][3]);
src/acc/hippo/epolar.cpp:            0.5f * ((gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][2]);
src/acc/hippo/epolar.cpp:               ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][2] +
src/acc/hippo/epolar.cpp:                  (gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][3]);
src/acc/hippo/epolar.cpp:            0.5f * ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][3]);
src/acc/hippo/epolar.cpp:         vxx = vxx - (cphid[1] * gpu_uind[i][0]);
src/acc/hippo/epolar.cpp:         vxy = vxy - 0.25f * (2 * cphid[1] * gpu_uind[i][1] + 2 * cphid[2] * gpu_uind[i][0]);
src/acc/hippo/epolar.cpp:         vxz = vxz - 0.25f * (2 * cphid[1] * gpu_uind[i][2] + 2 * cphid[3] * gpu_uind[i][0]);
src/acc/hippo/epolar.cpp:         vyy = vyy - cphid[2] * gpu_uind[i][1];
src/acc/hippo/epolar.cpp:         vyz = vyz - 0.25f * (2 * cphid[2] * gpu_uind[i][2] + 2 * cphid[3] * gpu_uind[i][1]);
src/acc/hippo/epolar.cpp:         vzz = vzz - cphid[3] * gpu_uind[i][2];
src/acc/hippo/epolar.cpp:      #pragma acc parallel loop independent async deviceptr(cmp,gpu_uind)
src/acc/hippo/epolar.cpp:         cmp[i][1] += gpu_uind[i][0];
src/acc/hippo/epolar.cpp:         cmp[i][2] += gpu_uind[i][1];
src/acc/hippo/epolar.cpp:         cmp[i][3] += gpu_uind[i][2];
src/acc/aplus/echgtrn.cpp:#include "tool/gpucard.h"
src/acc/aplus/echgtrn.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/aplus/field.cpp:#include "tool/gpucard.h"
src/acc/aplus/field.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/aplus/field.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/aplus/induce.cpp:#include "tool/gpucard.h"
src/acc/aplus/induce.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/aplus/empole.cpp:#include "tool/gpucard.h"
src/acc/aplus/empole.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/aplus/epolar.cpp:#include "tool/gpucard.h"
src/acc/aplus/epolar.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/mathparallel.cpp:T reduceSum_acc(const T* gpu_a, size_t cpu_n, int queue)
src/acc/mathparallel.cpp:               deviceptr(gpu_a) copy(val) reduction(+:val)
src/acc/mathparallel.cpp:      val += gpu_a[i];
src/acc/mathparallel.cpp:T dotProd_acc(const T* restrict gpu_a, const T* restrict gpu_b, size_t cpu_n, int queue)
src/acc/mathparallel.cpp:               deviceptr(gpu_a,gpu_b) copy(val) reduction(+:val)
src/acc/mathparallel.cpp:      val += gpu_a[i] * gpu_b[i];
src/acc/mathparallel.cpp:void scaleArray_acc(T* gpu_dst, T scal, size_t nelem, int queue)
src/acc/mathparallel.cpp:   #pragma acc parallel loop independent async(queue) deviceptr(gpu_dst)
src/acc/mathparallel.cpp:      gpu_dst[i] *= scal;
src/acc/mdintg.cpp:#if TINKER_CUDART
src/acc/mdintg.cpp:   if (pltfm_config & Platform::CUDA) {
src/acc/amoeba/empolenonewald.cpp:#include "tool/gpucard.h"
src/acc/amoeba/empolenonewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/epolarewald.cpp:#include "tool/gpucard.h"
src/acc/amoeba/epolarewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/epolarewald.cpp:static void epolarEwaldRecipSelf_acc1(const real (*gpu_uind)[3],
src/acc/amoeba/epolarewald.cpp:   const real (*gpu_uinp)[3])
src/acc/amoeba/epolarewald.cpp:   cuindToFuind(pu, gpu_uind, gpu_uinp, fuind, fuinp);
src/acc/amoeba/epolarewald.cpp:               rpole,cmp,gpu_uind,gpu_uinp,cphidp)
src/acc/amoeba/epolarewald.cpp:      real uix = 0.5f * (gpu_uind[i][0] + gpu_uinp[i][0]);
src/acc/amoeba/epolarewald.cpp:      real uiy = 0.5f * (gpu_uind[i][1] + gpu_uinp[i][1]);
src/acc/amoeba/epolarewald.cpp:      real uiz = 0.5f * (gpu_uind[i][2] + gpu_uinp[i][2]);
src/acc/amoeba/epolarewald.cpp:         uix = gpu_uind[i][0];
src/acc/amoeba/epolarewald.cpp:         uiy = gpu_uind[i][1];
src/acc/amoeba/epolarewald.cpp:         uiz = gpu_uind[i][2];
src/acc/amoeba/epolarewald.cpp:                  gpu_uind,gpu_uinp,fphid,fphip,cphi,cphidp)
src/acc/amoeba/epolarewald.cpp:            - 0.5f * ((gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][1]);
src/acc/amoeba/epolarewald.cpp:               * ((gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][1]
src/acc/amoeba/epolarewald.cpp:                  + (gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][2]);
src/acc/amoeba/epolarewald.cpp:               * ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][1]
src/acc/amoeba/epolarewald.cpp:                  + (gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][3]);
src/acc/amoeba/epolarewald.cpp:            - 0.5f * ((gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][2]);
src/acc/amoeba/epolarewald.cpp:               * ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][2]
src/acc/amoeba/epolarewald.cpp:                  + (gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][3]);
src/acc/amoeba/epolarewald.cpp:            - 0.5f * ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][3]);
src/acc/amoeba/epolarewald.cpp:            - 0.5f * (cphid[1] * gpu_uinp[i][0] + cphip[1] * gpu_uind[i][0]);
src/acc/amoeba/epolarewald.cpp:               * (cphid[1] * gpu_uinp[i][1] + cphip[1] * gpu_uind[i][1]
src/acc/amoeba/epolarewald.cpp:                  + cphid[2] * gpu_uinp[i][0] + cphip[2] * gpu_uind[i][0]);
src/acc/amoeba/epolarewald.cpp:               * (cphid[1] * gpu_uinp[i][2] + cphip[1] * gpu_uind[i][2]
src/acc/amoeba/epolarewald.cpp:                  + cphid[3] * gpu_uinp[i][0] + cphip[3] * gpu_uind[i][0]);
src/acc/amoeba/epolarewald.cpp:            - 0.5f * (cphid[2] * gpu_uinp[i][1] + cphip[2] * gpu_uind[i][1]);
src/acc/amoeba/epolarewald.cpp:               * (cphid[2] * gpu_uinp[i][2] + cphip[2] * gpu_uind[i][2]
src/acc/amoeba/epolarewald.cpp:                  + cphid[3] * gpu_uinp[i][1] + cphip[3] * gpu_uind[i][1]);
src/acc/amoeba/epolarewald.cpp:            - 0.5f * (cphid[3] * gpu_uinp[i][2] + cphip[3] * gpu_uind[i][2]);
src/acc/amoeba/epolarewald.cpp:      #pragma acc parallel loop independent async deviceptr(cmp,gpu_uinp)
src/acc/amoeba/epolarewald.cpp:         cmp[i][1] += gpu_uinp[i][0];
src/acc/amoeba/epolarewald.cpp:         cmp[i][2] += gpu_uinp[i][1];
src/acc/amoeba/epolarewald.cpp:         cmp[i][3] += gpu_uinp[i][2];
src/acc/amoeba/epolarewald.cpp:                  deviceptr(cmp,gpu_uind,gpu_uinp)
src/acc/amoeba/epolarewald.cpp:         cmp[i][1] += (gpu_uind[i][0] - gpu_uinp[i][0]);
src/acc/amoeba/epolarewald.cpp:         cmp[i][2] += (gpu_uind[i][1] - gpu_uinp[i][1]);
src/acc/amoeba/epolarewald.cpp:         cmp[i][3] += (gpu_uind[i][2] - gpu_uinp[i][2]);
src/acc/amoeba/fieldnonewald.cpp:#include "tool/gpucard.h"
src/acc/amoeba/fieldnonewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/fieldnonewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/torque.cpp:// in the OpenACC torque kernel. Running this command will show the segfault
src/acc/amoeba/torque.cpp:// GPU_PACKAGE=OPENACC PGI_ACC_NOTIFY=31 PGI_ACC_DEBUG=1 ./all.tests local-frame-3,nacl-2 -d yes -a
src/acc/amoeba/torque.cpp:static void torque_acc1(VirialBuffer gpu_vir, grad_prec* gx, grad_prec* gy, grad_prec* gz)
src/acc/amoeba/torque.cpp:               deviceptr(x,y,z,gx,gy,gz,zaxis,trqx,trqy,trqz,gpu_vir)
src/acc/amoeba/torque.cpp:         atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_vir, i & (bufsize - 1));
src/acc/amoeba/empoleewald.cpp:#include "tool/gpucard.h"
src/acc/amoeba/empoleewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/induce.cpp:#include "tool/gpucard.h"
src/acc/amoeba/induce.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/induce.cpp:      // OpenACC and CPU save the upper triangle.
src/acc/amoeba/induce.cpp:      // OpenACC and CPU save the upper triangle.
src/acc/amoeba/fieldewald.cpp:#include "tool/gpucard.h"
src/acc/amoeba/fieldewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/fieldewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/acc/amoeba/epolarnonewald.cpp:#include "tool/gpucard.h"
src/acc/amoeba/epolarnonewald.cpp:void epolar0DotProd_acc(const real (*gpu_uind)[3], const real (*gpu_udirp)[3])
src/acc/amoeba/epolarnonewald.cpp:               deviceptr(ep,gpu_uind,gpu_udirp,polarity_inv)
src/acc/amoeba/epolarnonewald.cpp:         * (gpu_uind[i][0] * gpu_udirp[i][0] + gpu_uind[i][1] * gpu_udirp[i][1]
src/acc/amoeba/epolarnonewald.cpp:            + gpu_uind[i][2] * gpu_udirp[i][2]);
src/acc/amoeba/epolarnonewald.cpp:   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
src/cudalib.cpp:#include "tool/cudalib.h"
src/cudalib.cpp:#if TINKER_CUDART
src/cudalib.cpp:#   include "tool/gpucard.h"
src/cudalib.cpp:#   include <cuda_profiler_api.h>
src/cudalib.cpp:TINKER_FVOID2(acc1, cu1, cudalibDataStreamAndQ, RcOp);
src/cudalib.cpp:void cudalibData(RcOp op)
src/cudalib.cpp:#if TINKER_CUDART
src/cudalib.cpp:      check_rt(cudaProfilerStop());
src/cudalib.cpp:      TINKER_FCALL0(cudalibDataStreamAndQ, RcOp::DEALLOC);
src/cudalib.cpp:      check_rt(cudaFreeHost(pinned_buf));
src/cudalib.cpp:      check_rt(cudaFree(dptr_buf));
src/cudalib.cpp:      check_rt(cudaEventDestroy(pme_event_start));
src/cudalib.cpp:      check_rt(cudaEventDestroy(pme_event_finish));
src/cudalib.cpp:      TINKER_FCALL0(cudalibDataStreamAndQ, RcOp::ALLOC);
src/cudalib.cpp:      check_rt(cublasCreate(&g::h0)); // calls cudaMemcpy [sync] here
src/cudalib.cpp:      check_rt(cublasCreate(&g::h1)); // calls cudaMemcpy [sync] here
src/cudalib.cpp:      int nblock = gpuGridSize(BLOCK_DIM);
src/cudalib.cpp:      check_rt(cudaMallocHost(&pinned_buf, nblock * sizeof(double)));
src/cudalib.cpp:      check_rt(cudaMalloc(&dptr_buf, nblock * sizeof(double)));
src/cudalib.cpp:      check_rt(cudaEventCreateWithFlags(&pme_event_start, cudaEventDisableTiming));
src/cudalib.cpp:      check_rt(cudaEventCreateWithFlags(&pme_event_finish, cudaEventDisableTiming));
src/cudalib.cpp:      check_rt(cudaProfilerStart());
src/pme.cpp:   if (pltfm_config & Platform::CUDA)
src/pme.cpp:void pmeConv(PMEUnit pme_u, VirialBuffer gpu_vir)
src/pme.cpp:   TINKER_FCALL2(acc1, cu1, pmeConv, pme_u, nullptr, gpu_vir);
src/pme.cpp:void pmeConv(PMEUnit pme_u, EnergyBuffer gpu_e)
src/pme.cpp:   TINKER_FCALL2(acc1, cu1, pmeConv, pme_u, gpu_e, nullptr);
src/pme.cpp:void pmeConv(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
src/pme.cpp:   TINKER_FCALL2(acc1, cu1, pmeConv, pme_u, gpu_e, gpu_vir);
src/initial.cpp:   gpucard = 0;
src/host/gpucard.cpp:#include "tool/gpucard.h"
src/host/gpucard.cpp:void gpuData(RcOp op)
src/host/gpucard.cpp:int gpuGridSize(int)
src/host/gpucard.cpp:int gpuMaxNParallel(int)
src/energy.cpp:   if (pltfm_config & Platform::CUDA) {
src/mod.cpp:#include "tool/cudalib.h"
src/mod.cpp:#include "tool/gpucard.h"
src/rattle.cpp:            if (TINKER_CUDART and pltfm_config & Platform::CUDA)
src/rattle.cpp:            if (TINKER_CUDART and pltfm_config & Platform::CUDA)
src/rattle.cpp:// cuda: settle, methyl (including CH, CH2, CH3, etc.)
src/rattle.cpp:// openacc / cpu: settle, CH, rattle (including CH2, CH3, etc.)
src/rattle.cpp:   if (pltfm_config & Platform::CUDA)
src/rattle.cpp:   if (pltfm_config & Platform::CUDA)
src/rattle.cpp:   if (pltfm_config & Platform::CUDA)
src/xdynamic.cpp:   inform::gpucard = 1;
src/objc/cmakesrc.txt:gpuutil.m
src/objc/gpuutil.m:int tinkerGpuUtilizationInt32_macos(int iDevice)
src/objc/gpuutil.m:         ssize_t gpuCoreUse = 0;
src/objc/gpuutil.m:            const void* gpuCoreUtilization =
src/objc/gpuutil.m:               CFDictionaryGetValue(perfProperties, CFSTR("GPU Core Utilization"));
src/objc/gpuutil.m:            if (gpuCoreUtilization)
src/objc/gpuutil.m:               CFNumberGetValue((CFNumberRef)gpuCoreUtilization, kCFNumberSInt64Type, &gpuCoreUse);
src/objc/gpuutil.m:            return gpuCoreUse / 10000000;
src/atom.cpp:#include "tool/gpucard.h"
src/atom.cpp:#if TINKER_CUDART
src/atom.cpp:      nelem_buffer = gpuMaxNParallel(idevice);
src/rcman.cpp:#include "tool/gpucard.h"
src/rcman.cpp:#include "tool/cudalib.h"
src/rcman.cpp:   RcMan gpu42{gpuData, op};
src/rcman.cpp:   RcMan cl42{cudalibData, op};
src/mdsave.cpp:#include "tool/cudalib.h"
src/mdsave.cpp:#if TINKER_CUDART
src/mdsave.cpp:#   include "tool/gpucard.h"
src/mdsave.cpp:#   include <cuda_runtime.h>
src/mdsave.cpp:#if TINKER_CUDART
src/mdsave.cpp:static cudaEvent_t mdsave_begin_event, mdsave_end_event;
src/mdsave.cpp:#if TINKER_CUDART
src/mdsave.cpp:   // There is no guarantee that the CUDA runtime will use the same GPU card as
src/mdsave.cpp:   // the main thread, unless cudaSetDevice() is called explicitly.
src/mdsave.cpp:   // Of course this is not a problem if the computer has only one GPU card.
src/mdsave.cpp:   check_rt(cudaSetDevice(idevice));
src/mdsave.cpp:#if TINKER_CUDART
src/mdsave.cpp:   check_rt(cudaEventRecord(mdsave_begin_event, g::s0));
src/mdsave.cpp:   check_rt(cudaStreamWaitEvent(g::s1, mdsave_begin_event, 0));
src/mdsave.cpp:   // get gpu buffer and write to external files
src/mdsave.cpp:#if TINKER_CUDART
src/mdsave.cpp:   check_rt(cudaEventRecord(mdsave_end_event, g::s1));
src/mdsave.cpp:   check_rt(cudaStreamWaitEvent(g::s0, mdsave_end_event, 0));
src/mdsave.cpp:#if TINKER_CUDART
src/mdsave.cpp:      check_rt(cudaEventDestroy(mdsave_begin_event));
src/mdsave.cpp:      check_rt(cudaEventDestroy(mdsave_end_event));
src/mdsave.cpp:#if TINKER_CUDART
src/mdsave.cpp:      check_rt(cudaEventCreateWithFlags(&mdsave_begin_event,
src/mdsave.cpp:         cudaEventDisableTiming));
src/mdsave.cpp:      check_rt(cudaEventCreateWithFlags(&mdsave_end_event,
src/mdsave.cpp:         cudaEventDisableTiming));
src/cu/mdpt.cu:   cudaStream_t st = g::s0;
src/cu/mdpt.cu:   int grid_siz1 = gpuGridSize(BLOCK_DIM);
src/cu/mdpt.cu:   check_rt(cudaMemcpyAsync(hptr, dptr, HN * sizeof(energy_prec), cudaMemcpyDeviceToHost, st));
src/cu/mdpt.cu:   check_rt(cudaStreamSynchronize(st));
src/cu/spatial.cu:   auto policy = thrust::cuda::par(ThrustCache::instance()).on(g::s0);
src/cu/echarge.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/cudalib.cu:#include "tool/cudalib.h"
src/cu/cudalib.cu:void cudalibDataStreamAndQ_cu(RcOp op)
src/cu/cudalib.cu:      check_rt(cudaStreamDestroy(g::s1));
src/cu/cudalib.cu:      check_rt(cudaStreamCreateWithFlags(&g::s1, cudaStreamNonBlocking));
src/cu/elj.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/CMakeLists.txt:enable_language (CUDA)
src/cu/CMakeLists.txt:set (CMAKE_CUDA_STANDARD ${T9_CPPSTD})
src/cu/CMakeLists.txt:set (CMAKE_CUDA_EXTENSIONS OFF)
src/cu/CMakeLists.txt:      CUDA_RESOLVE_DEVICE_SYMBOLS ON
src/cu/CMakeLists.txt:      CUDA_SEPARABLE_COMPILATION  ON
src/cu/CMakeLists.txt:      CUDA_ARCHITECTURES          "${T9_CUCCLIST}"
src/cu/cmakesrc.txt:cudalib.cu
src/cu/evalence.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/compilers.cu:std::string cudaCompilerName()
src/cu/compilers.cu:   return format("nvcc %d.%d.%d", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
src/cu/pme.cu:   check_rt(cudaMemsetAsync(st.qgrid, 0, 2 * nt * sizeof(type), stream));
src/cu/pme.cu:   real aewald, TINKER_IMAGE_PARAMS, real box_volume, EnergyBuffer restrict gpu_e,
src/cu/pme.cu:   VirialBuffer restrict gpu_vir)
src/cu/pme.cu:      atomic_add(ectl, gpu_e, ithread);
src/cu/pme.cu:      atomic_add(vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz, gpu_vir, ithread);
src/cu/pme.cu:static void pmeConv_cu2(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
src/cu/pme.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/pme.cu:      TINKER_IMAGE_ARGS, box_volume, gpu_e, gpu_vir);
src/cu/pme.cu:void pmeConv_cu(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_vir)
src/cu/pme.cu:   if (gpu_vir == nullptr) {
src/cu/pme.cu:      if (gpu_e == nullptr) {
src/cu/pme.cu:         pmeConv_cu2<true, false>(pme_u, gpu_e, nullptr);
src/cu/pme.cu:      if (gpu_e == nullptr) {
src/cu/pme.cu:         pmeConv_cu2<false, true>(pme_u, nullptr, gpu_vir);
src/cu/pme.cu:         pmeConv_cu2<true, true>(pme_u, gpu_e, gpu_vir);
src/cu/ehal.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/precond.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/precond.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/echgtrn.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/edisp.cu:   EnergyBuffer restrict gpu_e, VirialBuffer restrict gpu_vir)
src/cu/hippo/edisp.cu:      atomic_add(ectl, gpu_e, ithread);
src/cu/hippo/edisp.cu:      atomic_add(vctlxx, vctlyx, vctlzx, vctlyy, vctlzy, vctlzz, gpu_vir, ithread);
src/cu/hippo/edisp.cu:static void pmeConvDisp_cu2(PMEUnit pme_u, EnergyBuffer gpu_e, VirialBuffer gpu_v)
src/cu/hippo/edisp.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/edisp.cu:      pme_u->bsmod3, pme_u->aewald, TINKER_IMAGE_ARGS, vbox, gpu_e, gpu_v);
src/cu/hippo/edisp.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/erepel.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/empole.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/expol.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/expol.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/expol.cu:      check_rt(cudaMemcpyAsync((real*)pinned_buf, epsd, sizeof(real), cudaMemcpyDeviceToHost, g::s0));
src/cu/hippo/expol.cu:      check_rt(cudaStreamSynchronize(g::s0));
src/cu/hippo/epolar.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/epolar.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/hippo/pcg.cu:      check_rt(cudaMemcpyAsync((real*)pinned_buf, epsd, sizeof(real), cudaMemcpyDeviceToHost, g::s0));
src/cu/hippo/pcg.cu:      check_rt(cudaStreamSynchronize(g::s0));
src/cu/mdintg.cu:   int grid_siz1 = -4 + gpuGridSize(BLOCK_DIM);
src/cu/echglj.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/epolarrecip.cu:   const real (*restrict gpu_uind)[3], const real (*restrict gpu_uinp)[3],
src/cu/epolarrecip.cu:   if (gpu_uinp) {
src/cu/epolarrecip.cu:         real uix = 0.5f * (gpu_uind[i][0] + gpu_uinp[i][0]);
src/cu/epolarrecip.cu:         real uiy = 0.5f * (gpu_uind[i][1] + gpu_uinp[i][1]);
src/cu/epolarrecip.cu:         real uiz = 0.5f * (gpu_uind[i][2] + gpu_uinp[i][2]);
src/cu/epolarrecip.cu:            uix = gpu_uind[i][0];
src/cu/epolarrecip.cu:            uiy = gpu_uind[i][1];
src/cu/epolarrecip.cu:            uiz = gpu_uind[i][2];
src/cu/epolarrecip.cu:         real uix = gpu_uind[i][0];
src/cu/epolarrecip.cu:         real uiy = gpu_uind[i][1];
src/cu/epolarrecip.cu:         real uiz = gpu_uind[i][2];
src/cu/epolarrecip.cu:            uix = gpu_uind[i][0];
src/cu/epolarrecip.cu:            uiy = gpu_uind[i][1];
src/cu/epolarrecip.cu:            uiz = gpu_uind[i][2];
src/cu/epolarrecip.cu:   const real (*restrict cmp)[10], const real (*restrict gpu_uind)[3],
src/cu/epolarrecip.cu:   const real (*restrict gpu_uinp)[3], const real (*restrict fphid)[10],
src/cu/epolarrecip.cu:   if (gpu_uinp) {
src/cu/epolarrecip.cu:            0.5f * ((gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][1]);
src/cu/epolarrecip.cu:               ((gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][1] +
src/cu/epolarrecip.cu:                  (gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][2]);
src/cu/epolarrecip.cu:               ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][1] +
src/cu/epolarrecip.cu:                  (gpu_uind[i][0] + gpu_uinp[i][0]) * cphi[i][3]);
src/cu/epolarrecip.cu:            0.5f * ((gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][2]);
src/cu/epolarrecip.cu:               ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][2] +
src/cu/epolarrecip.cu:                  (gpu_uind[i][1] + gpu_uinp[i][1]) * cphi[i][3]);
src/cu/epolarrecip.cu:            0.5f * ((gpu_uind[i][2] + gpu_uinp[i][2]) * cphi[i][3]);
src/cu/epolarrecip.cu:         vxx = vxx - 0.5f * (cphid[1] * gpu_uinp[i][0] + cphip[1] * gpu_uind[i][0]);
src/cu/epolarrecip.cu:               (cphid[1] * gpu_uinp[i][1] + cphip[1] * gpu_uind[i][1] + cphid[2] * gpu_uinp[i][0] +
src/cu/epolarrecip.cu:                  cphip[2] * gpu_uind[i][0]);
src/cu/epolarrecip.cu:               (cphid[1] * gpu_uinp[i][2] + cphip[1] * gpu_uind[i][2] + cphid[3] * gpu_uinp[i][0] +
src/cu/epolarrecip.cu:                  cphip[3] * gpu_uind[i][0]);
src/cu/epolarrecip.cu:         vyy = vyy - 0.5f * (cphid[2] * gpu_uinp[i][1] + cphip[2] * gpu_uind[i][1]);
src/cu/epolarrecip.cu:               (cphid[2] * gpu_uinp[i][2] + cphip[2] * gpu_uind[i][2] + cphid[3] * gpu_uinp[i][1] +
src/cu/epolarrecip.cu:                  cphip[3] * gpu_uind[i][1]);
src/cu/epolarrecip.cu:         vzz = vzz - 0.5f * (cphid[3] * gpu_uinp[i][2] + cphip[3] * gpu_uind[i][2]);
src/cu/epolarrecip.cu:            0.5f * ((gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][1]);
src/cu/epolarrecip.cu:               ((gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][1] +
src/cu/epolarrecip.cu:                  (gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][2]);
src/cu/epolarrecip.cu:               ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][1] +
src/cu/epolarrecip.cu:                  (gpu_uind[i][0] + gpu_uind[i][0]) * cphi[i][3]);
src/cu/epolarrecip.cu:            0.5f * ((gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][2]);
src/cu/epolarrecip.cu:               ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][2] +
src/cu/epolarrecip.cu:                  (gpu_uind[i][1] + gpu_uind[i][1]) * cphi[i][3]);
src/cu/epolarrecip.cu:            0.5f * ((gpu_uind[i][2] + gpu_uind[i][2]) * cphi[i][3]);
src/cu/epolarrecip.cu:         vxx = vxx - (cphid[1] * gpu_uind[i][0]);
src/cu/epolarrecip.cu:         vxy = vxy - 0.25f * (2 * cphid[1] * gpu_uind[i][1] + 2 * cphid[2] * gpu_uind[i][0]);
src/cu/epolarrecip.cu:         vxz = vxz - 0.25f * (2 * cphid[1] * gpu_uind[i][2] + 2 * cphid[3] * gpu_uind[i][0]);
src/cu/epolarrecip.cu:         vyy = vyy - cphid[2] * gpu_uind[i][1];
src/cu/epolarrecip.cu:         vyz = vyz - 0.25f * (2 * cphid[2] * gpu_uind[i][2] + 2 * cphid[3] * gpu_uind[i][1]);
src/cu/epolarrecip.cu:         vzz = vzz - cphid[3] * gpu_uind[i][2];
src/cu/epolarrecip.cu:   int n, real (*restrict cmp)[10], const real (*restrict gpu_uinp)[3])
src/cu/epolarrecip.cu:      cmp[i][1] += gpu_uinp[i][0];
src/cu/epolarrecip.cu:      cmp[i][2] += gpu_uinp[i][1];
src/cu/epolarrecip.cu:      cmp[i][3] += gpu_uinp[i][2];
src/cu/epolarrecip.cu:   const real (*restrict gpu_uind)[3], const real (*restrict gpu_uinp)[3])
src/cu/epolarrecip.cu:      cmp[i][1] += (gpu_uind[i][0] - gpu_uinp[i][0]);
src/cu/epolarrecip.cu:      cmp[i][2] += (gpu_uind[i][1] - gpu_uinp[i][1]);
src/cu/epolarrecip.cu:      cmp[i][3] += (gpu_uind[i][2] - gpu_uinp[i][2]);
src/cu/epolarrecip.cu:static void epolarEwaldRecipSelf_cu1(const real (*gpu_uind)[3], const real (*gpu_uinp)[3])
src/cu/epolarrecip.cu:   cuindToFuind(pu, gpu_uind, gpu_uinp, fuind, fuinp);
src/cu/epolarrecip.cu:      rpole, cmp, gpu_uind, gpu_uinp, cphidp);
src/cu/epolarrecip.cu:         cmp, gpu_uind, gpu_uinp, fphid, fphip, cphi, cphidp, //
src/cu/epolarrecip.cu:      launch_k1s(g::s0, n, epolarEwaldRecipSelfVirial_cu3, n, cmp, gpu_uinp);
src/cu/epolarrecip.cu:      launch_k1s(g::s0, n, epolarEwaldRecipSelfVirial_cu4, n, cmp, gpu_uind, gpu_uinp);
src/cu/epolarrecip.cu:static void epolarChgpenEwaldRecipSelf_cu1(const real (*gpu_uind)[3], bool use_cf)
src/cu/epolarrecip.cu:   cuindToFuind(pu, gpu_uind, gpu_uind, fuind, fuind);
src/cu/epolarrecip.cu:      rpole, cmp, gpu_uind, nullptr, cphidp);
src/cu/epolarrecip.cu:         cmp, gpu_uind, nullptr, fphid, nullptr, cphi, cphidp, //
src/cu/epolarrecip.cu:      launch_k1s(g::s0, n, epolarEwaldRecipSelfVirial_cu3, n, cmp, gpu_uind);
src/cu/amoeba/precond.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/torque.cu:void torque_cu1(int n, VirialBuffer restrict gpu_vir, //
src/cu/amoeba/torque.cu:         atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, gpu_vir, ithread);
src/cu/amoeba/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/field.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/empole.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/emplar.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/epolar.cu:void epolar0DotProd_cu1(int n, real f, EnergyBuffer restrict ep, const real (*restrict gpu_uind)[3],
src/cu/amoeba/epolar.cu:   const real (*restrict gpu_udirp)[3], const real* restrict polarity_inv)
src/cu/amoeba/epolar.cu:         * (gpu_uind[i][0] * gpu_udirp[i][0] + gpu_uind[i][1] * gpu_udirp[i][1] + gpu_uind[i][2] * gpu_udirp[i][2]);
src/cu/amoeba/epolar.cu:void epolar0DotProd_cu(const real (*gpu_uind)[3], const real (*gpu_udirp)[3])
src/cu/amoeba/epolar.cu:   launch_k1b(g::s0, n, epolar0DotProd_cu1, n, f, ep, gpu_uind, gpu_udirp, polarity_inv);
src/cu/amoeba/epolar.cu:   int ngrid = gpuGridSize(BLOCK_DIM);
src/cu/amoeba/pcg.cu:      check_rt(cudaMemcpyAsync((real*)pinned_buf, epsd, 2 * sizeof(real), cudaMemcpyDeviceToHost, g::s0));
src/cu/amoeba/pcg.cu:      check_rt(cudaStreamSynchronize(g::s0));
src/cu/amoeba/binding.cu:#include "tool/cudalib.h"
src/cu/amoeba/binding.cu:static cudaMemcpyKind h2d = cudaMemcpyHostToDevice;
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&k1, (const void*)&d::zaxis));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&k2, (const void*)&d::pole));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&k3, (const void*)&d::rpole));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(k1, &zaxis, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(k2, &pole, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(k3, &rpole, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaStreamSynchronize(g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&p1, (const void*)&d::njpolar));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&p2, (const void*)&d::jpolar));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&p3, (const void*)&d::thlval));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(p1, &njpolar, sizeof(int), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(p2, &jpolar, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(p3, &thlval, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&p4, (const void*)&d::polarity));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&p5, (const void*)&d::thole));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&p6, (const void*)&d::pdamp));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(p4, &polarity, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(p5, &thole, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(p6, &pdamp, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaGetSymbolAddress(&p7, (const void*)&d::polarity_inv));
src/cu/amoeba/binding.cu:      check_rt(cudaMemcpyAsync(p7, &polarity_inv, sizeof(void*), h2d, g::s0));
src/cu/amoeba/binding.cu:      check_rt(cudaStreamSynchronize(g::s0));
src/cu/mathparallel.cu:#include "tool/cudalib.h"
src/cu/mathparallel.cu:#include "tool/gpucard.h"
src/cu/mathparallel.cu:#include <cuda_runtime.h>
src/cu/mathparallel.cu:void reduce_to_dptr(const T* a, size_t nelem, cudaStream_t st)
src/cu/mathparallel.cu:   int grid_siz1 = gpuGridSize(BLOCK_DIM);
src/cu/mathparallel.cu:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cu/mathparallel.cu:   check_rt(cudaMemcpyAsync(hptr, dptr, sizeof(T), cudaMemcpyDeviceToHost, st));
src/cu/mathparallel.cu:   check_rt(cudaStreamSynchronize(st));
src/cu/mathparallel.cu:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cu/mathparallel.cu:   int grid_siz1 = gpuGridSize(BLOCK_DIM);
src/cu/mathparallel.cu:   check_rt(cudaMemcpyAsync(hptr, (T*)dptr, HN * sizeof(HT), cudaMemcpyDeviceToHost, st));
src/cu/mathparallel.cu:   check_rt(cudaStreamSynchronize(st));
src/cu/mathparallel.cu:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cu/mathparallel.cu:   int grid_siz1 = gpuGridSize(BLOCK_DIM);
src/cu/mathparallel.cu:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cu/mathparallel.cu:   int grid_siz1 = gpuGridSize(BLOCK_DIM);
src/cu/mathparallel.cu:// cublas gemm does not run as fast here prior to cuda 10.1.
src/cu/mathparallel.cu:// #if CUDART_VERSION >= 10100 // >= 10.1
src/cu/mathparallel.cu:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/xinfo.cc:#include "tool/gpucard.h"
src/xinfo.cc:   gpuData(RcOp::INIT);
src/xinfo.cc:#if TINKER_CUDART
src/xinfo.cc:#if TINKER_GPULANG_OPENACC
src/xinfo.cc:      "OpenACC and CUDA"
src/xinfo.cc:#elif TINKER_GPULANG_CUDA
src/xinfo.cc:      "CUDA"
src/xinfo.cc:   if (pltfm_config & Platform::CUDA)
src/xinfo.cc:      print(out, fmt, "Primary GPU package:", "CUDA");
src/xinfo.cc:      print(out, fmt, "Primary GPU package:", "OpenACC");
src/xinfo.cc:   print(out, fmt, "Latest CUDA supported by driver:", gpuCudaDriverVersion());
src/xinfo.cc:   print(out, fmt, "CUDA runtime version:", gpuCudaRuntimeVersion());
src/xinfo.cc:   print(out, fmt, "Thrust version:", gpuThrustVersion());
src/xinfo.cc:   print(out, fmt, "CUDA compiler:", cudaCompilerName());
src/xinfo.cc:   print(out, fmt, "OpenACC compiler:",
src/xinfo.cc:#if TINKER_GPULANG_OPENACC
src/xinfo.cc:      print(out, fmd, "GPU detected:", ndevice);
src/xinfo.cc:      const auto& attribs = gpuDeviceAttributes();
src/xinfo.cc:         print(out, fm1, format("GPU %d:", a.device));
src/xinfo.cc:         print(out, f2d, "Number of CUDA cores:", a.cores_per_multiprocessor * a.multiprocessor_count);
src/xinfo.cc:         print(out, fm2, "Used/Total GPU memory:",
src/xinfo.cc:   gpuData(RcOp::DEALLOC);
src/mathlu.cpp:#if TINKER_GPULANG_OPENACC
src/mathlu.cpp:#elif TINKER_GPULANG_CUDA
src/CMakeLists.txt:host/gpucard.cpp
src/CMakeLists.txt:cudart/darray.cpp
src/CMakeLists.txt:cudart/error.cpp
src/CMakeLists.txt:cudart/fft.cpp
src/CMakeLists.txt:cudart/gpucard.cpp
src/CMakeLists.txt:cudart/pmestream.cpp
src/CMakeLists.txt:cudart/thrustcache.cpp
src/CMakeLists.txt:   target_include_directories (__t9_intf INTERFACE "${CUDA_DIR}/include")
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:      if (pltfm_config & Platform::CUDA)
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:      if (pltfm_config & Platform::CUDA)
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:      if (pltfm_config & Platform::CUDA)
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:      if (pltfm_config & Platform::CUDA)
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:      if (pltfm_config & Platform::CUDA)
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_GPULANG_CUDA
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:      if (pltfm_config & Platform::CUDA)
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:#if TINKER_CUDART
src/nblist.cpp:#if TINKER_CUDART
src/cmakesrc.txt:cudalib.cpp
src/cudart/fft.cpp:#include "tool/cudalib.h"
src/cudart/pmestream.cpp:#include "tool/cudalib.h"
src/cudart/pmestream.cpp:      check_rt(cudaEventRecord(pme_event_start, g::s0));
src/cudart/pmestream.cpp:      check_rt(cudaStreamWaitEvent(g::spme, pme_event_start, 0));
src/cudart/pmestream.cpp:      check_rt(cudaEventRecord(pme_event_finish, g::spme));
src/cudart/pmestream.cpp:      check_rt(cudaStreamWaitEvent(g::s0, pme_event_finish, 0));
src/cudart/error.cpp:#include <cuda_runtime.h>
src/cudart/error.cpp:std::string translateErrorCode<cudaError_t>(cudaError_t error_num)
src/cudart/error.cpp:   return std::string(cudaGetErrorString(error_num));
src/cudart/gpucard.cpp:#include "tool/gpucard.h"
src/cudart/gpucard.cpp:#include <cuda_runtime.h>
src/cudart/gpucard.cpp:extern "C" int tinkerGpuUtilizationInt32_macos(int);
src/cudart/gpucard.cpp:std::string gpuCudaRuntimeVersion()
src/cudart/gpucard.cpp:   check_rt(cudaRuntimeGetVersion(&ver));
src/cudart/gpucard.cpp:std::string gpuCudaDriverVersion()
src/cudart/gpucard.cpp:   check_rt(cudaDriverGetVersion(&ver));
src/cudart/gpucard.cpp:std::string gpuThrustVersion()
src/cudart/gpucard.cpp:std::vector<DeviceAttribute>& gpuDeviceAttributes()
src/cudart/gpucard.cpp:static std::string getNvidiaSmi()
src/cudart/gpucard.cpp:   std::string smi = "nvidia-smi";
src/cudart/gpucard.cpp:   int val1 = std::system("which nvidia-smi > /dev/null");
src/cudart/gpucard.cpp:      val1 = std::system("which nvidia-smi.exe > /dev/null");
src/cudart/gpucard.cpp:      smi = "nvidia-smi.exe";
src/cudart/gpucard.cpp:         TINKER_THROW("nvidia-smi is not found.");
src/cudart/gpucard.cpp:   cudaDeviceProp prop;
src/cudart/gpucard.cpp:   check_rt(cudaGetDeviceProperties(&prop, device));
src/cudart/gpucard.cpp:   check_rt(cudaDeviceGetPCIBusId(pciBusID, 13, device));
src/cudart/gpucard.cpp:   if (prop.computeMode == cudaComputeModeExclusive)
src/cudart/gpucard.cpp:   else if (prop.computeMode == cudaComputeModeProhibited)
src/cudart/gpucard.cpp:   else if (prop.computeMode == cudaComputeModeExclusiveProcess)
src/cudart/gpucard.cpp:   check_rt(cudaSetDevice(device));
src/cudart/gpucard.cpp:   check_rt(cudaMemGetInfo(&a.free_mem_bytes, &a.total_mem_bytes));
src/cudart/gpucard.cpp:   // nvidia cuda-c-programming-guide compute-capabilities
src/cudart/gpucard.cpp:   // Number of CUDA cores (FP32) per multiprocessor, not tabulated;
src/cudart/gpucard.cpp:   check_rt(cudaDeviceReset());
src/cudart/gpucard.cpp:   int usp = -1; // user-specified cuda device; -1 for not set
src/cudart/gpucard.cpp:      usp_str = "CUDA-DEVICE keyword";
src/cudart/gpucard.cpp:      getKV("CUDA-DEVICE", usp, -1);
src/cudart/gpucard.cpp:   // check environment variable "CUDA_DEVICE"
src/cudart/gpucard.cpp:      usp_str = "CUDA_DEVICE environment variable";
src/cudart/gpucard.cpp:      if (const char* str = std::getenv("CUDA_DEVICE"))
src/cudart/gpucard.cpp:   // check environment variable "cuda_device"
src/cudart/gpucard.cpp:      usp_str = "cuda_device environment variable";
src/cudart/gpucard.cpp:      if (const char* str = std::getenv("cuda_device"))
src/cudart/gpucard.cpp:      const auto& a = gpuDeviceAttributes()[i];
src/cudart/gpucard.cpp:      int macosGpuUtil = tinkerGpuUtilizationInt32_macos(i);
src/cudart/gpucard.cpp:      gpercent.push_back(macosGpuUtil);
src/cudart/gpucard.cpp:      std::string smi = getNvidiaSmi();
src/cudart/gpucard.cpp:      std::string cmd = format("%s --query-gpu=utilization.gpu "
src/cudart/gpucard.cpp:      int igpuutil = gpercent[idev];
src/cudart/gpucard.cpp:      int jgpuutil = gpercent[jdev];
src/cudart/gpucard.cpp:      if (igpuutil + SIGNIFICANT_DIFFERENCE < jgpuutil)
src/cudart/gpucard.cpp:      else if (jgpuutil + SIGNIFICANT_DIFFERENCE < igpuutil)
src/cudart/gpucard.cpp:      usp_str = "GPU utilization";
src/cudart/gpucard.cpp:         " CUDA-DEVICE Warning,"
src/cudart/gpucard.cpp:      " GPU Device :  Setting Device ID to %d from %s\n",
src/cudart/gpucard.cpp:static unsigned int cuda_device_flags = 0;
src/cudart/gpucard.cpp:void gpuData(RcOp op)
src/cudart/gpucard.cpp:      // should not reset these variables for unit tests, if multiple GPUs are available
src/cudart/gpucard.cpp:      gpuDeviceAttributes().clear();
src/cudart/gpucard.cpp:      if (cuda_device_flags)
src/cudart/gpucard.cpp:      // else if (cuda_device_flags == 0)
src/cudart/gpucard.cpp:      cuda_device_flags = cudaDeviceMapHost;
src/cudart/gpucard.cpp:      cuda_device_flags |= cudaDeviceScheduleBlockingSync;
src/cudart/gpucard.cpp:      // Using this flag may reduce the latency for the cudaStreamSynchronize() calls.
src/cudart/gpucard.cpp:      cuda_device_flags |= cudaDeviceScheduleSpin;
src/cudart/gpucard.cpp:      // cudaError_t cudaSetDeviceFlags (unsigned int flags);
src/cudart/gpucard.cpp:      // initialized then this call will fail with the error cudaErrorSetOnActiveProcess.
src/cudart/gpucard.cpp:      // In this case it is necessary to reset device using cudaDeviceReset()
src/cudart/gpucard.cpp:      // Since CUDA 11, cudaSetDeviceFlags should be called after cudaSetDevice.
src/cudart/gpucard.cpp:      // Prior to CUDA 11, cudaSetDeviceFlags must be called before cudaSetDevice.
src/cudart/gpucard.cpp:#if CUDART_VERSION < 11000
src/cudart/gpucard.cpp:      always_check_rt(cudaSetDeviceFlags(cuda_device_flags));
src/cudart/gpucard.cpp:      always_check_rt(cudaGetDeviceCount(&ndevice));
src/cudart/gpucard.cpp:      auto& all = gpuDeviceAttributes();
src/cudart/gpucard.cpp:      check_rt(cudaSetDevice(idevice));
src/cudart/gpucard.cpp:#if CUDART_VERSION >= 11000
src/cudart/gpucard.cpp:      check_rt(cudaSetDeviceFlags(cuda_device_flags));
src/cudart/gpucard.cpp:      check_rt(cudaDeviceSynchronize());
src/cudart/gpucard.cpp:      check_rt(cudaGetDevice(&kdevice));
src/cudart/gpucard.cpp:      check_rt(cudaGetDeviceFlags(&kflags));
src/cudart/gpucard.cpp:      if (kflags != cuda_device_flags)
src/cudart/gpucard.cpp:         TINKER_THROW(format("Cuda device flag %u in use is different than the pre-selected flag %u.", kflags,
src/cudart/gpucard.cpp:            cuda_device_flags));
src/cudart/gpucard.cpp:int gpuGridSize(int nthreads_per_block)
src/cudart/gpucard.cpp:   const auto& a = gpuDeviceAttributes()[idevice];
src/cudart/gpucard.cpp:int gpuMaxNParallel(int idev)
src/cudart/gpucard.cpp:   const auto& a = gpuDeviceAttributes().at(idev);
src/cudart/darray.cpp:#include "tool/cudalib.h"
src/cudart/darray.cpp:#include <cuda_runtime.h>
src/cudart/darray.cpp:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cudart/darray.cpp:   check_rt(cudaStreamSynchronize(st));
src/cudart/darray.cpp:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cudart/darray.cpp:   check_rt(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, st));
src/cudart/darray.cpp:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cudart/darray.cpp:   check_rt(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, st));
src/cudart/darray.cpp:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cudart/darray.cpp:   check_rt(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, st));
src/cudart/darray.cpp:   cudaStream_t st = queue == g::q1 ? g::s1 : g::s0;
src/cudart/darray.cpp:   check_rt(cudaMemsetAsync(dst, 0, nbytes, st));
src/cudart/darray.cpp:   check_rt(cudaFree(ptr));
src/cudart/darray.cpp:   check_rt(cudaMalloc(pptr, nbytes));
src/amoeba/mpole.cpp:      if (pltfm_config & Platform::CUDA) {
src/platform.cpp:#if TINKER_GPULANG_OPENACC
src/platform.cpp:         std::string gpu_package = "";
src/platform.cpp:         if (const char* str = std::getenv("gpu_package")) {
src/platform.cpp:            gpu_package = str;
src/platform.cpp:            Text::upcase(gpu_package);
src/platform.cpp:         if (const char* str = std::getenv("GPU_PACKAGE")) {
src/platform.cpp:            gpu_package = str;
src/platform.cpp:            Text::upcase(gpu_package);
src/platform.cpp:         if (gpu_package == "") {
src/platform.cpp:            getKV("GPU-PACKAGE", gpu_package, "CUDA");
src/platform.cpp:         if (gpu_package == "CUDA") {
src/platform.cpp:            pltfm_config = Platform::CUDA;
src/platform.cpp:            print(stdout, " Primary GPU package :  CUDA\n");
src/platform.cpp:         } else if (gpu_package == "OPENACC") {
src/platform.cpp:            print(stdout, " Primary GPU package :  OpenACC\n");
src/platform.cpp:#elif TINKER_GPULANG_CUDA
src/platform.cpp:      pltfm_config = Platform::CUDA;

```
