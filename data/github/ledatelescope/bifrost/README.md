# https://github.com/ledatelescope/bifrost

```console
python/setup.py:# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
python/bifrost/version/__main__.py:    print(" CUDA support: %s" % _yes_no(BF_CUDA_ENABLED))
python/bifrost/version/__main__.py:    if BF_CUDA_ENABLED:
python/bifrost/version/__main__.py:        print("  CUDA version: %.1f" % BF_CUDA_VERSION)
python/bifrost/version/__main__.py:        print("  CUDA architectures: %s" % BF_GPU_ARCHS)
python/bifrost/version/__main__.py:        print("  CUDA shared memory: %i B" % BF_GPU_SHAREDMEM)
python/bifrost/version/__main__.py:        print("  CUDA managed memory support: %s" % _yes_no(BF_GPU_MANAGEDMEM))
python/bifrost/version/__main__.py:        print("  CUDA map disk cache: %s" % _yes_no(BF_MAP_KERNEL_DISK_CACHE))
python/bifrost/version/__main__.py:        print("  CUDA debugging: %s" % _yes_no(BF_CUDA_DEBUG_ENABLED))
python/bifrost/version/__main__.py:        print("  CUDA tracing enabled: %s" % _yes_no(BF_TRACE_ENABLED))
python/bifrost/pipeline.py:                 gpu: Optional[int]=None,
python/bifrost/pipeline.py:        self._gpu           = gpu
python/bifrost/pipeline.py:                        'cuda':      'limegreen',
python/bifrost/pipeline.py:                        'cuda_host': 'deepskyblue'
python/bifrost/pipeline.py:        if self.gpu is not None:
python/bifrost/pipeline.py:            device.set_device(self.gpu)
python/bifrost/pipeline.py:        default_space = 'cuda_host' if core.cuda_enabled() else 'system'
python/bifrost/romein.py:        # TODO: Work out how to integrate CUDA stream
python/bifrost/device.py:    """Set the CUDA stream to the provided stream handle"""
python/bifrost/device.py:    """Get the current CUDA stream and return its address"""
python/bifrost/device.py:            # pycuda stream?
python/bifrost/device.py:    """Sets a flag on all GPU devices that tells them not to spin the CPU when
python/bifrost/device.py:    synchronizing. This is useful for reducing CPU load in GPU pipelines.
python/bifrost/device.py:    This function must be called _before_ any GPU devices are
python/bifrost/ndarray.py:TODO: Some calls result in segfault with space=cuda (e.g., __getitem__
python/bifrost/ndarray.py:        if (src_bf.bf.space == 'cuda_managed' or
python/bifrost/ndarray.py:            dst_bf.bf.space == 'cuda_managed'):
python/bifrost/ndarray.py:                                            space='cuda',
python/bifrost/ndarray.py:            if 'pycuda' in sys.modules:
python/bifrost/ndarray.py:                from pycuda.gpuarray import GPUArray as pycuda_GPUArray
python/bifrost/ndarray.py:                if isinstance(base, pycuda_GPUArray):
python/bifrost/ndarray.py:                                           space='cuda',
python/bifrost/ndarray.py:                                           buffer=int(base.gpudata),
python/bifrost/ndarray.py:            if self.bf.space == 'cuda_managed':
python/bifrost/ndarray.py:            ## For arrays that can be access from CUDA, use bifrost.map
python/bifrost/ndarray.py:                if space == 'cuda_managed':
python/bifrost/ndarray.py:                ## For arrays that can be access from CUDA, use bifrost.transpose
python/bifrost/ndarray.py:        if space_accessible(self.bf.space, ['cuda']):
python/bifrost/ndarray.py:            umem = cp.cuda.UnownedMemory(self.ctypes.data, self.data.nbytes, self)
python/bifrost/ndarray.py:            mptr = cp.cuda.MemoryPointer(umem, 0)
python/bifrost/ndarray.py:    def as_GPUArray(self, *args, **kwargs):
python/bifrost/ndarray.py:        from pycuda.gpuarray import GPUArray as pycuda_GPUArray
python/bifrost/ndarray.py:        g  = pycuda_GPUArray(shape=self.shape, dtype=self.dtype, *args, **kwargs)
python/bifrost/memory.py:# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
python/bifrost/memory.py:    elif space == 'cuda_host':
python/bifrost/memory.py:    elif space == 'cuda_managed':
python/bifrost/memory.py:        return 'system' in from_spaces or 'cuda' in from_spaces
python/bifrost/memory.py:# Note: These functions operate on numpy or GPU arrays
python/bifrost/affinity.py:# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
python/bifrost/ring.py:# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
python/bifrost/fdmt.py:             exponent: float=-2.0, space: Union[str,_th.BFspace_enum,_bf.BFspace]='cuda'):
python/bifrost/fdmt.py:        # TODO: Work out how to integrate CUDA stream
python/bifrost/block_chainer.py:        bc.blocks.copy('cuda')
python/bifrost/block_chainer.py:        bc.blocks.copy('cuda_host')
python/bifrost/libbifrost.py:# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
python/bifrost/blocks/quantize.py:        Input:  [...], dtype = [c]f32, space = SYSTEM or CUDA
python/bifrost/blocks/quantize.py:        Output: [...], dtype = any (complex) integer type, space = SYSTEM or CUDA
python/bifrost/blocks/reduce.py:        return ('cuda',)
python/bifrost/blocks/reduce.py:        Input:  [..., N, ...], dtype = float, space = CUDA
python/bifrost/blocks/reduce.py:        Output: [..., N / factor, ...], dtype = f32, space = CUDA
python/bifrost/blocks/reduce.py:        Input:  [..., N, ...], dtype = complex, space = CUDA
python/bifrost/blocks/reduce.py:        Output: [..., N / factor, ...], dtype = cf32, space = CUDA
python/bifrost/blocks/reduce.py:        Input:  [..., N, ...], dtype = complex, space = CUDA
python/bifrost/blocks/reduce.py:        Output: [..., N / factor, ...], dtype = f32, space = CUDA
python/bifrost/blocks/fftshift.py:        return ('cuda',)
python/bifrost/blocks/fftshift.py:        Input:  [...], dtype = any, space = CUDA
python/bifrost/blocks/fftshift.py:        Output: [...], dtype = any, space = CUDA
python/bifrost/blocks/scrunch.py:# TODO: This is a bit hacky and inflexible, and has no CUDA backend yet
python/bifrost/blocks/transpose.py:        if space_accessible(self.space, ['cuda']):
python/bifrost/blocks/transpose.py:        Input:  [...], dtype = any , space = SYSTEM or CUDA
python/bifrost/blocks/detect.py:        return ('cuda',)
python/bifrost/blocks/detect.py:        Input:  [..., 'pol', ...], dtype = any complex, space = CUDA
python/bifrost/blocks/detect.py:        Output: [..., 'pol', ...], dtype = real or complex, space = CUDA
python/bifrost/blocks/fdmt.py:        return ('cuda',)
python/bifrost/blocks/fdmt.py:    This uses the GPU. It is used in pulsar and fast radio burst (FRB)
python/bifrost/blocks/fdmt.py:        Input:  ['pol', 'freq',       'time'], dtype = any real, space = CUDA
python/bifrost/blocks/fdmt.py:        Output: ['pol', 'dispersion', 'time'], dtype = f32, space = CUDA
python/bifrost/blocks/reverse.py:        return ('cuda',)
python/bifrost/blocks/reverse.py:        Input:  [...], dtype = any, space = CUDA
python/bifrost/blocks/reverse.py:        Output: [...], dtype = any, space = CUDA
python/bifrost/blocks/correlate.py:        return ('cuda',)
python/bifrost/blocks/correlate.py:        Input:  ['time', 'freq', 'station', 'pol'], dtype = any complex, space = CUDA
python/bifrost/blocks/correlate.py:        Output: ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'] (lower triangle filled), dtype = cf32, space = CUDA
python/bifrost/blocks/correlate.py:        This block is backed by a fast GPU kernel based on the one in the xGPU
python/bifrost/blocks/correlate.py:        https://github.com/GPU-correlators/xGPU
python/bifrost/blocks/unpack.py:        Input:  [...], dtype = one of: i/u2, i/u4, ci2, ci4, space = SYSTEM or CUDA
python/bifrost/blocks/unpack.py:        Output: [...], dtype = i8 or ci8 (matching input), space = SYSTEM or CUDA
python/bifrost/blocks/convert_visibilities.py:        return ('cuda',)
python/bifrost/blocks/convert_visibilities.py:        Input:  ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'], dtype = any complex, space = CUDA
python/bifrost/blocks/convert_visibilities.py:        Output: ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'], dtype = any complex, space = CUDA
python/bifrost/blocks/convert_visibilities.py:        Output: ['time', 'baseline', 'freq', 'stokes'], dtype = any complex, space = CUDA
python/bifrost/blocks/convert_visibilities.py:        Input:  ['time', 'baseline', 'freq', 'stokes'], dtype = any complex, space = CUDA
python/bifrost/blocks/convert_visibilities.py:        Output: ['time', 'freq', 'station_i', 'pol_i', 'station_j', 'pol_j'], dtype = any complex, space = CUDA
python/bifrost/blocks/fft.py:        return ('cuda',)
python/bifrost/blocks/fft.py:    """Apply a GPU FFT to the input ring data.
python/bifrost/blocks/fft.py:        Input:  [...], dtype = any real or complex, space = CUDA
python/bifrost/blocks/fft.py:        Output: [...], dtype = [f32, cf32, f64, or cf64], space = CUDA
python/bifrost/blocks/copy.py:        spaces, such as from system memory to GPU memory.
python/bifrost/blocks/copy.py:        space (str): Output data space (e.g., 'cuda' or 'system').
python/bifrost/blocks/accumulate.py:        return ('cuda',)
python/bifrost/blocks/accumulate.py:    Input:  [..., 'time', ...], dtype = any, space = CUDA
python/bifrost/blocks/accumulate.py:    Output: [..., 'time'/nframe, ...], dtype = any, space = CUDA
python/bifrost/map.py:        Only GPU computation is currently supported.
python/bifrost/__init__.py:# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
python/bifrost/__init__.py:__copyright__  = "Copyright (c) 2016-2020, The Bifrost Authors. All rights reserved.\nCopyright (c) 2016, NVIDIA CORPORATION. All rights reserved."
python/bifrost/fir.py:             space: Union[str,_th.BFspace_enum,_bf.BFspace]='cuda'):
python/bifrost/fir.py:        # TODO: Work out how to integrate CUDA stream
python/bifrost/core.py:# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
python/bifrost/core.py:def cuda_enabled() -> bool:
python/bifrost/core.py:    return bool(_bf.bfGetCudaEnabled())
lgtm.yml:      - export NOCUDA=1
flake.nix:      # Can inspect the cuda version to guess at what architectures would be
flake.nix:      # most useful. Take care not to instatiate the cuda package though, which
flake.nix:      defaultGpuArchs = cudatoolkit:
flake.nix:        if lib.hasPrefix "11." cudatoolkit.version then [
flake.nix:      # cudatoolkit      │       │
flake.nix:      # cudatoolkit_11   │       │  (deprecated)
flake.nix:      # cudatoolkit_11_5 11.5.0  │    (35 37 50)52 53 60 61 62 70 72 75 80 86 87
flake.nix:        , enablePython ? true, python3, enableCuda ? false, cudatoolkit
flake.nix:        , util-linuxMinimal, gpuArchs ? defaultGpuArchs cudatoolkit }:
flake.nix:            + lib.optionalString enableCuda
flake.nix:            "-cuda${lib.versions.majorMinor cudatoolkit.version}"
flake.nix:          ] ++ lib.optionals enableCuda [ cudatoolkit util-linuxMinimal ];
flake.nix:          # wants to quote the args and that fails with spaces in gpuArchs.
flake.nix:              ++ lib.optionals enableCuda [
flake.nix:                "--with-cuda-home=${cudatoolkit}"
flake.nix:                ''--with-gpu-archs="${lib.concatStringsSep " " gpuArchs}"''
flake.nix:                "--with-nvcc-flags='-Wno-deprecated-gpu-targets'"
flake.nix:                "LDFLAGS=-L${cudatoolkit}/lib/stubs"
flake.nix:            lib.optionals enableCuda [ "CUDA_LIBDIR64=$(CUDA_HOME)/lib" ];
flake.nix:          # Which cuda versions should be target by the packages? Let's just do
flake.nix:          isCuda = name: builtins.match "cudaPackages(_1[01])" name != null;
flake.nix:          shortenCuda = lib.replaceStrings [ "Packages" "_" ] [ "" "" ];
flake.nix:          cudaAttrs = lib.filterAttrs (name: pkg:
flake.nix:            isCuda name && lib.elem pkgs.system pkg.cudatoolkit.meta.platforms)
flake.nix:          eachCuda = f: lib.concatMap f ([ null ] ++ lib.attrNames cudaAttrs);
flake.nix:              eachCuda (cuda:
flake.nix:                    + lib.optionalString (cuda != null) "-${shortenCuda cuda}"
flake.nix:                      enableCuda = cuda != null;
flake.nix:                      cudatoolkit = pkgs.${cuda}.cudatoolkit;
flake.nix:                  makeWrapperArgs = lib.optionals config.enableCuda
flake.nix:                    [ "--set LD_PRELOAD /usr/lib/x86_64-linux-gnu/libcuda.so" ];
config/libtool.m4:    nvcc*) # Cuda Compiler Driver 2.2
config/libtool.m4:	nvcc*)	# Cuda Compiler Driver 2.2
config/cuda.m4:AC_DEFUN([AX_CHECK_CUDA],
config/cuda.m4:  AC_PROVIDE([AX_CHECK_CUDA])
config/cuda.m4:  AC_ARG_WITH([cuda_home],
config/cuda.m4:              [AS_HELP_STRING([--with-cuda-home],
config/cuda.m4:                              [CUDA install path (default=/usr/local/cuda)])],
config/cuda.m4:              [with_cuda_home=/usr/local/cuda])
config/cuda.m4:  AC_SUBST(CUDA_HOME, $with_cuda_home)
config/cuda.m4:  AC_ARG_ENABLE([cuda],
config/cuda.m4:                [AS_HELP_STRING([--disable-cuda],
config/cuda.m4:                                [disable cuda support (default=no)])],
config/cuda.m4:                [enable_cuda=no],
config/cuda.m4:                [enable_cuda=yes])
config/cuda.m4:  AC_SUBST([HAVE_CUDA], [0])
config/cuda.m4:  AC_SUBST([CUDA_VERSION], [0])
config/cuda.m4:  AC_SUBST([CUDA_HAVE_CXX20], [0])
config/cuda.m4:  AC_SUBST([CUDA_HAVE_CXX17], [0])
config/cuda.m4:  AC_SUBST([CUDA_HAVE_CXX14], [0])
config/cuda.m4:  AC_SUBST([CUDA_HAVE_CXX11], [0])
config/cuda.m4:  AC_SUBST([GPU_MIN_ARCH], [0])
config/cuda.m4:  AC_SUBST([GPU_MAX_ARCH], [0])
config/cuda.m4:  AC_SUBST([GPU_SHAREDMEM], [0])
config/cuda.m4:  AC_SUBST([GPU_PASCAL_MANAGEDMEM], [0])
config/cuda.m4:  AC_SUBST([GPU_EXP_PINNED_ALLOC], [1])
config/cuda.m4:  if test "$enable_cuda" != "no"; then
config/cuda.m4:    AC_SUBST([HAVE_CUDA], [1])
config/cuda.m4:    AC_PATH_PROG(NVCC, nvcc, no, [$CUDA_HOME/bin:$PATH])
config/cuda.m4:    AC_PATH_PROG(NVPRUNE, nvprune, no, [$CUDA_HOME/bin:$PATH])
config/cuda.m4:    AC_PATH_PROG(CUOBJDUMP, cuobjdump, no, [$CUDA_HOME/bin:$PATH])
config/cuda.m4:  if test "$HAVE_CUDA" = "1"; then
config/cuda.m4:    AC_MSG_CHECKING([for a working CUDA 10+ installation])
config/cuda.m4:    LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
config/cuda.m4:    NVCCLIBS="$LIBS -lcuda -lcudart"
config/cuda.m4:          #include <cuda.h>
config/cuda.m4:          #include <cuda_runtime.h>]],
config/cuda.m4:          [[cudaMalloc(0, 0);]])],
config/cuda.m4:        [AC_SUBST([HAVE_CUDA], [0])])
config/cuda.m4:    if test "$HAVE_CUDA" = "1"; then
config/cuda.m4:      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
config/cuda.m4:      NVCCLIBS="$NVCCLIBS -lcuda -lcudart"
config/cuda.m4:            #include <cuda.h>
config/cuda.m4:            #include <cuda_runtime.h>]],
config/cuda.m4:            [[cudaMalloc(0, 0);]])],
config/cuda.m4:          [CUDA_VERSION=$( ${NVCC} --version | ${GREP} -Po -e "release.*," | cut -d,  -f1 | cut -d\  -f2 )
config/cuda.m4:           AC_MSG_RESULT(yes - v$CUDA_VERSION)],
config/cuda.m4:           AC_SUBST([HAVE_CUDA], [0])])
config/cuda.m4:      AC_SUBST([HAVE_CUDA], [0])
config/cuda.m4:  if test "$HAVE_CUDA" = "1"; then
config/cuda.m4:    AC_MSG_CHECKING([for CUDA CXX standard support])
config/cuda.m4:    CUDA_STDCXX=$( ${NVCC} --help | ${GREP} -Po -e "--std.*}" | ${SED} 's/.*|//;s/}//;' )
config/cuda.m4:    if test "$CUDA_STDCXX" = "c++20"; then
config/cuda.m4:      AC_SUBST([CUDA_HAVE_CXX20], [1])
config/cuda.m4:      if test "$CUDA_STDCXX" = "c++17"; then
config/cuda.m4:        AC_SUBST([CUDA_HAVE_CXX17], [1])
config/cuda.m4:        if test "$CUDA_STDCXX" = "c++14"; then
config/cuda.m4:          AC_SUBST([CUDA_HAVE_CXX14], [1])
config/cuda.m4:          if test "$CUDA_STDCXX" = "c++11"; then
config/cuda.m4:            AC_SUBST([CUDA_HAVE_CXX11], [1])
config/cuda.m4:                              [CUDA default stream model to use: 'legacy' or 'per-thread' (default='per-thread')])],
config/cuda.m4:  if test "$HAVE_CUDA" = "1"; then
config/cuda.m4:    AC_MSG_CHECKING([for different CUDA default stream models])
config/cuda.m4:          AC_MSG_ERROR(Invalid CUDA stream model: '$with_stream_model')
config/cuda.m4:  if test "$HAVE_CUDA" = "1"; then
config/cuda.m4:    CPPFLAGS="$CPPFLAGS -DBF_CUDA_ENABLED=1"
config/cuda.m4:    CXXFLAGS="$CXXFLAGS -DBF_CUDA_ENABLED=1"
config/cuda.m4:    NVCCFLAGS="$NVCCFLAGS -DBF_CUDA_ENABLED=1"
config/cuda.m4:    LDFLAGS="$LDFLAGS -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
config/cuda.m4:    NVCCLIBS="$NVCCLIBS -lcuda -lcudart -lnvrtc -lcublas -lcudadevrt -L. -lcufft_static_pruned -lculibos -lnvToolsExt"
config/cuda.m4:  AC_ARG_WITH([gpu_archs],
config/cuda.m4:              [AS_HELP_STRING([--with-gpu-archs=...],
config/cuda.m4:                              [default GPU architectures (default=detect)])],
config/cuda.m4:              [with_gpu_archs='auto'])
config/cuda.m4:  if test "$HAVE_CUDA" = "1"; then
config/cuda.m4:    AC_MSG_CHECKING([for valid CUDA architectures])
config/cuda.m4:    if test "$with_gpu_archs" = "auto"; then
config/cuda.m4:      AC_MSG_CHECKING([which CUDA architectures to target])
config/cuda.m4:      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
config/cuda.m4:      NVCCLIBS="-lcuda -lcudart"
config/cuda.m4:            #include <cuda.h>
config/cuda.m4:            #include <cuda_runtime.h>
config/cuda.m4:            cudaGetDeviceCount(&deviceCount);
config/cuda.m4:              cudaSetDevice(dev);
config/cuda.m4:              cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
config/cuda.m4:              cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
config/cuda.m4:            [AC_SUBST([GPU_ARCHS], [`cat confarchs.out`])
config/cuda.m4:             ar_valid=$( echo $GPU_ARCHS $ar_supported | xargs -n1 | sort | uniq -d | xargs )
config/cuda.m4:               AC_SUBST([GPU_ARCHS], [$ar_valid])
config/cuda.m4:               AC_MSG_RESULT([$GPU_ARCHS])
config/cuda.m4:      AC_SUBST([GPU_ARCHS], [$with_gpu_archs])
config/cuda.m4:    AC_MSG_CHECKING([for valid requested CUDA architectures])
config/cuda.m4:    ar_requested=$( echo "$GPU_ARCHS" | wc -w )
config/cuda.m4:    ar_valid=$( echo $GPU_ARCHS $ar_supported | xargs -n1 | sort | uniq -d | xargs )
config/cuda.m4:    AC_SUBST([GPU_MIN_ARCH], [$ar_min_valid])
config/cuda.m4:    AC_SUBST([GPU_MAX_ARCH], [$ar_max_valid])
config/cuda.m4:                           [default GPU shared memory per block in bytes (default=detect)])],
config/cuda.m4:      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
config/cuda.m4:      NVCCLIBS="-lcuda -lcudart"
config/cuda.m4:            #include <cuda.h>
config/cuda.m4:            #include <cuda_runtime.h>
config/cuda.m4:            cudaGetDeviceCount(&deviceCount);
config/cuda.m4:              cudaSetDevice(dev);
config/cuda.m4:              cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, dev);
config/cuda.m4:            [AC_SUBST([GPU_SHAREDMEM], [`cat confsmem.out`])
config/cuda.m4:             AC_MSG_RESULT([$GPU_SHAREDMEM B])],
config/cuda.m4:      AC_SUBST([GPU_SHAREDMEM], [$with_shared_mem])
config/cuda.m4:    AC_MSG_CHECKING([for Pascal-style CUDA managed memory])
config/cuda.m4:    cm_invalid=$( echo $GPU_ARCHS | ${SED} -e 's/\b[[1-5]][[0-9]]\b/PRE/g;' )
config/cuda.m4:      AC_SUBST([GPU_PASCAL_MANAGEDMEM], [1])
config/cuda.m4:      AC_SUBST([GPU_PASCAL_MANAGEDMEM], [0])
config/cuda.m4:    LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
config/cuda.m4:    NVCCLIBS="-lcuda -lcudart"
config/cuda.m4:          #include <cuda.h>
config/cuda.m4:          #include <cuda_runtime.h>
config/cuda.m4:          #include <thrust/system/cuda/memory.h>]],
config/cuda.m4:          [AC_SUBST([GPU_EXP_PINNED_ALLOC], [0])
config/cuda.m4:          [AC_SUBST([GPU_EXP_PINNED_ALLOC], [1])
config/cuda.m4:     AC_SUBST([GPU_PASCAL_MANAGEDMEM], [0])
config/cuda.m4:     AC_SUBST([GPU_EXP_PINNED_ALLOC], [1])
share/bifrost.m4:          [[bfGetCudaEnabled();]])],
configure.ac:AC_CONFIG_SRCDIR([src/cuda.cpp])
configure.ac:# CUDA
configure.ac:# include CUDA-specific entries #
configure.ac:AX_CHECK_CUDA
configure.ac:AC_ARG_ENABLE([cuda_debug],
configure.ac:              [AS_HELP_STRING([--enable-cuda-debug],
configure.ac:                              [enable CUDA debugging (nvcc -G; default=no)])],
configure.ac:              [enable_cuda_debug=yes],
configure.ac:              [enable_cuda_debug=no])
configure.ac:AC_SUBST([HAVE_CUDA_DEBUG], [0])
configure.ac:AS_IF([test x$enable_cuda_debug != xno],
configure.ac:      [AC_SUBST([HAVE_CUDA_DEBUG], [1])
configure.ac:      [AC_SUBST([HAVE_MAP_CACHE], [$HAVE_CUDA])])
configure.ac:       AS_IF([test x$CUDA_HAVE_CXX20 = x1],
configure.ac:             [AS_IF([test x$CUDA_HAVE_CXX17 = x1],
configure.ac:                    [AS_IF([test x$CUDA_HAVE_CXX14 = x1],
configure.ac:       AS_IF([test x$CUDA_HAVE_CXX20 = x1],
configure.ac:             [AS_IF([test x$CUDA_HAVE_CXX17 = x1],
configure.ac:                    [AS_IF([test x$CUDA_HAVE_CXX14 = x1],
configure.ac:       AS_IF([test x$CUDA_HAVE_CXX20 = x1],
configure.ac:             [AS_IF([test x$CUDA_HAVE_CXX17 = x1],
configure.ac:                    [AS_IF([test x$CUDA_HAVE_CXX14 = x1],
configure.ac:# Additional CUDA flags
configure.ac:AS_IF([test x$HAVE_CUDA != x0],
configure.ac:      [NVCC_GENCODE=$(echo $GPU_ARCHS | ${SED} -e 's/\([[0-9]]\{2,3\}\)/-gencode arch=compute_\1,\\"code=sm_\1\\"/g;')
configure.ac:       NVCC_GENCODE="$NVCC_GENCODE -gencode arch=compute_${GPU_MAX_ARCH},\\\"code=compute_${GPU_MAX_ARCH}\\\""
configure.ac:       CPPFLAGS="$CPPFLAGS -I$CUDA_HOME/include"
configure.ac:AS_IF([test x$HAVE_CUDA = x1],
configure.ac:      [AC_MSG_NOTICE(cuda: yes - v$CUDA_VERSION - $GPU_ARCHS - $with_stream_model streams)],
configure.ac:      [AC_MSG_NOTICE(cuda: no)])
configure.ac:AS_IF([test x$enable_cuda_debug != xno],
configure.ac:      [AC_SUBST([OPTIONS], ["$OPTIONS cuda_debug"])])
testbench/test_fft.py:takes the FFT of the data (on the GPU no less), and then writes it to a new file. 
testbench/test_fft.py:    b_copy      = CopyBlock(b_read, space='cuda', core=1, gpu=0)
testbench/test_fft.py:    b_fft       = FftBlock(b_copy, axes=1, core=2, gpu=0)
testbench/test_fdmt.py:    d_filterbank = copy(h_filterbank, space='cuda', gpu=0, core=2)
testbench/test_fdmt.py:    with bfp.block_scope(core=2, gpu=0):
testbench/test_fft_detect.py:takes the FFT of the data (on the GPU no less), and then writes it to a new file. 
testbench/test_fft_detect.py:    b_copy      = CopyBlock(b_read, space='cuda', core=1, gpu=0)
testbench/test_fft_detect.py:    b_fft       = FftBlock(b_copy, axes=1, core=2, gpu=0)
testbench/test_guppi.py:This testbench tests a guppi gpuspec reader
testbench/gpuspec_simple.py:    bc.blocks.copy(space='cuda', core=1)
testbench/gpuspec_simple.py:    with bf.block_scope(fuse=True, gpu=0):
testbench/gpuspec_simple.py:    bc.blocks.copy(space='cuda_host', core=2)
testbench/test_guppi_reader.py:This testbench tests a guppi gpuspec reader
docs/source/Create-a-pipeline.rst:#. Channelize it with a GPU FFT.
docs/source/Create-a-pipeline.rst:#. Copy the raw data to the GPU.
docs/source/Create-a-pipeline.rst:also have some CUDA-compatible GPUs to run this example.
docs/source/Create-a-pipeline.rst:Next, we want to put this data onto the GPU. Bifrost makes this simple.
docs/source/Create-a-pipeline.rst:    gpu_raw_data = blocks.copy(raw_data, space='cuda')
docs/source/Create-a-pipeline.rst:for our audio file. Then, by setting ``space='cuda'``, we tell Bifrost
docs/source/Create-a-pipeline.rst:to create a ring in GPU memory, and copy all of the contents of ``raw_data``
docs/source/Create-a-pipeline.rst:into this new ring. With this GPU ring, we can connect more blocks and
docs/source/Create-a-pipeline.rst:do GPU processing.
docs/source/Create-a-pipeline.rst:    chunked_data = views.split_axis(gpu_raw_data, 'time', 256, label='fine_time')
docs/source/Create-a-pipeline.rst:What have we done here? We took ``gpu_raw_data``, which is a block on the GPU,
docs/source/Create-a-pipeline.rst:and which implicitly points to its output ring buffer which sits on the GPU,
docs/source/Create-a-pipeline.rst:256-size axis, and we want to do it on the GPU. Bifrost knows which
docs/source/Create-a-pipeline.rst:ring is on the GPU and CPU, so this is implicitly passed to the FFT block:
docs/source/Create-a-pipeline.rst:The Bifrost FFT block wraps ``cuFFT``, the CUDA FFT package, which is
docs/source/Create-a-pipeline.rst:This block takes in the output of the FFT (we are still on the GPU!),
docs/source/Create-a-pipeline.rst:on the GPU for us:
docs/source/Create-a-pipeline.rst:But first, we have to offload from the GPU:
docs/source/Create-a-pipeline.rst:     host_transposed = blocks.copy(transposed, space='cuda_host')
docs/source/Create-a-pipeline.rst:#. Copied the raw data to the GPU.
docs/source/Create-a-pipeline.rst:    gpu_raw_data = blocks.copy(raw_data, space='cuda')
docs/source/Create-a-pipeline.rst:    chunked_data = views.split_axis(gpu_raw_data, 'time', 256, label='fine_time')
docs/source/Create-a-pipeline.rst:    host_transposed = blocks.copy(transposed, space='cuda_host')
docs/source/intro.rst:high-throughput processing CPU/GPU pipelines. It is specifically
docs/source/intro.rst:may be run on either the CPU or GPU, and the ring buffer will take care
docs/source/intro.rst:of memory copies between the CPU and GPU spaces.
docs/source/Getting-started-guide.rst:You will need a relatively new gcc and CUDA - we have used Bifrost with gcc 4.8 and CUDA 8,
docs/source/Getting-started-guide.rst:`CUDA <https://developer.nvidia.com/cuda-zone>`__
docs/source/Getting-started-guide.rst:CUDA allows you to program your GPU from C and C++. You will need an
docs/source/Getting-started-guide.rst:NVIDIA GPU to do this. If this is your first time trying out Bifrost,
docs/source/Getting-started-guide.rst:and you don't have CUDA yet, we recommend that you skip this step, and
docs/source/Getting-started-guide.rst:If you are ready to work with a GPU, you will want to get the newest
docs/source/Getting-started-guide.rst:`CUDA toolkit <https://developer.nvidia.com/cuda-downloads>`__. Follow 
docs/source/Getting-started-guide.rst:The table below indicates which CUDA toolkit and kernel driver versions Bifrost
docs/source/Getting-started-guide.rst:   :header: "OS","Linux Kernel","Driver Version","GPU","Toolkit","Status"
docs/source/Getting-started-guide.rst:configure: cuda: yes - 50 52
docs/source/Cpp-Development.rst:   #include <cuda_runtime_api.h>
docs/source/tools-intro.rst:NVIDIA Profiler
docs/source/tools-intro.rst:The NVIDIA Profiler and Visual Profiler tools (part of the CUDA Toolkit)
docs/source/tools-intro.rst:     19154  GuppiRawSourceB     0    9.4    0.714    0.000    0.714    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154       FftBlock_0     3    4.4    0.733    0.699    0.034    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154      CopyBlock_0     2    4.4    0.722    0.700    0.021    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154  TransposeBlock_     1    3.5    0.710    0.695    0.015    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154  HdfWriteBlock_0     6    0.4    3.220    3.213    0.007    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154    DetectBlock_0     4    1.0    0.738    0.733    0.005    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154  FftShiftBlock_0     3    4.4    0.738    0.734    0.005    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154      CopyBlock_1     6    0.4    2.816    2.813    0.003    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154  AccumulateBlock     5    4.0    0.005    0.005    0.001    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:     19154  PrintHeaderBloc    -1           3.220    3.220    0.000    0.000  python ./bf_gpuspec_midres.py ../pulsa
docs/source/tools-intro.rst:Note: The CPU fraction will probably be 100% on any GPU block because it's currently set to spin (busy loop) while waiting for the GPU.
docs/source/Common-installation-and-execution-problems.rst:OSError: libcudart.so.x.0: cannot open shared object file: No such file or directory
docs/source/Common-installation-and-execution-problems.rst:Similar to the above error. You need to add the CUDA libraries to the
docs/source/Common-installation-and-execution-problems.rst:``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/cuda/installation/lib64``
docs/source/bfmap.rst:Fast GPU math using bfMap
docs/source/bfmap.rst:provides a simple way to do fast arithmetic operations on the GPU. To
docs/source/bfmap.rst:the GPU. Here is how you would do that with ``map``:
docs/source/bfmap.rst:    # Create two arrays on the GPU, A and B, and an empty output C
docs/source/bfmap.rst:    a = bf.ndarray([1,2,3,4,5], space='cuda')
docs/source/bfmap.rst:    b = bf.ndarray([1,0,1,0,1], space='cuda')
docs/source/bfmap.rst:    c = bf.ndarray(np.zeros(5), space='cuda')
docs/source/bfmap.rst:            Only GPU computation is currently supported.
docs/source/bfmap.rst:    # Create two arrays on the GPU, A and B, and an empty output C
docs/source/bfmap.rst:    a = bf.ndarray([1,2,3,4,5], space='cuda')
docs/source/bfmap.rst:    b = bf.ndarray([1,0,1,0,1], space='cuda')
docs/source/bfmap.rst:    c = bf.ndarray(np.zeros((5, 5)), space='cuda')
Dockerfile_prereq.gpu:FROM nvidia/cuda:10.2-devel-ubuntu18.04
test/test_fft.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED, BF_CUDA_VERSION
test/test_fft.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_fft.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fft.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fft.py:        odata = bf.ndarray(shape=oshape, dtype='cf32', space='cuda')
test/test_fft.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fft.py:        odata = bf.ndarray(shape=oshape, dtype='f32', space='cuda')
test/test_map.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_map.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_map.py:        x = bf.asarray(x, 'cuda')
test/test_map.py:        x = bf.asarray(x, space='cuda')
test/test_map.py:        x = bf.asarray(x, space='cuda')
test/test_map.py:        a = bf.asarray(a, space='cuda')
test/test_map.py:        c = bf.empty((a.shape[0],b.shape[0]), a.dtype, 'cuda') # TODO: Need way to compute broadcast shape
test/test_map.py:        x = bf.asarray(x, space='cuda')
test/test_map.py:        a = bf.asarray(known_data, space='cuda')
test/test_map.py:        a = bf.asarray(a, space='cuda')
test/test_map.py:                a = a_orig.copy(space='cuda')
test/test_map.py:                b = bf.ndarray(shape=(n,), dtype=out_dtype, space='cuda')
test/test_map.py:        a = bf.asarray(a, space='cuda')
test/test_map.py:        a = bf.asarray(a, space='cuda')
test/test_map.py:        b = bf.empty((a.shape[2],a.shape[0], a.shape[1]), a.dtype, 'cuda')
test/test_map.py:        a = bf.asarray(a, space='cuda')
test/test_map.py:        b = bf.empty((a.shape[0],a.shape[2]), a.dtype, 'cuda')
test/test_fir.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_fir.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        odata = bf.empty((idata.shape[0]//2, idata.shape[1]), dtype=idata.dtype, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        odata = bf.empty((idata.shape[0]//2, idata.shape[1], idata.shape[2]), dtype=idata.dtype, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        odata = bf.empty((idata.shape[0]//2, idata.shape[1]), dtype=idata.dtype, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        odata = bf.empty((idata.shape[0]//2, idata.shape[1], idata.shape[2]), dtype=idata.dtype, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        coeffs2 = bf.ndarray(coeffs2, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        coeffs2 = bf.ndarray(coeffs2, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        idata = bf.ndarray(known_data, space='cuda')
test/test_fir.py:        coeffs = bf.ndarray(coeffs, space='cuda')
test/test_fir.py:        coeffs2 = bf.ndarray(coeffs2, space='cuda')
test/test_interop.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_interop.py:    import pycuda.driver as cuda
test/test_interop.py:    import pycuda.autoinit
test/test_interop.py:    import pycuda.gpuarray
test/test_interop.py:    import pycuda.driver
test/test_interop.py:    HAVE_PYCUDA = True
test/test_interop.py:    HAVE_PYCUDA = False
test/test_interop.py:@unittest.skipUnless(BF_CUDA_ENABLED and HAVE_CUPY, "requires GPU support and cupy")
test/test_interop.py:        bf_data = bf.ndarray(data, space='cuda')
test/test_interop.py:        with cp.cuda.Stream() as stream:
test/test_interop.py:                bf_data = bf.ndarray(data, space='cuda')
test/test_interop.py:        with cp.cuda.ExternalStream(stream):
test/test_interop.py:            self.assertEqual(cp.cuda.get_current_stream().ptr, stream)
test/test_interop.py:            bf_data = bf.ndarray(data, space='cuda')
test/test_interop.py:@unittest.skipUnless(BF_CUDA_ENABLED and HAVE_PYCUDA, "requires GPU support and cupy")
test/test_interop.py:class TestPyCUDA(unittest.TestCase):
test/test_interop.py:    def test_as_gpuarray(self):
test/test_interop.py:        bf_data = bf.ndarray(data, space='cuda')
test/test_interop.py:        pc_data = bf_data.as_GPUArray()
test/test_interop.py:    def test_from_gpuarray(self):
test/test_interop.py:        pc_data = pycuda.gpuarray.to_gpu(data)
test/test_interop.py:        stream = pycuda.driver.Stream()
test/test_interop.py:            bf_data = bf.ndarray(data, space='cuda')
test/test_accumulate.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_accumulate.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_accumulate.py:            g_data = blocks.copy(c_data, space='cuda')
test/test_romein.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_romein.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_romein.py:        grid = grid.copy(space='cuda')
test/test_romein.py:        data = data.copy(space='cuda')
test/test_romein.py:        illum = illum.copy(space='cuda')
test/test_romein.py:        locs = locs.copy(space='cuda')
test/test_romein.py:        grid = grid.copy(space='cuda')
test/test_romein.py:        data = data.copy(space='cuda')
test/test_romein.py:        illum = illum.copy(space='cuda')
test/test_romein.py:        locs = locs.copy(space='cuda')
test/test_romein.py:        illum = illum.copy(space='cuda')
test/test_romein.py:        grid = grid.copy(space='cuda')
test/test_romein.py:        data = data.copy(space='cuda')
test/test_romein.py:        illum = illum.copy(space='cuda')
test/test_romein.py:        locs = locs.copy(space='cuda')
test/test_romein.py:        locs = locs.copy(space='cuda')
test/test_linalg.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_linalg.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_linalg.py:        # Note: The xGPU-like correlation kernel does not support input values of -128 (only [-127:127])
test/test_linalg.py:        a = bf.asarray(a, space='cuda')
test/test_linalg.py:        c = bf.zeros_like(c_gold, space='cuda')
test/test_linalg.py:        a = bf.asarray(a, space='cuda')
test/test_linalg.py:        c = bf.zeros_like(c_gold, space='cuda')
test/test_linalg.py:        a = bf.asarray(a, space='cuda')
test/test_linalg.py:        b = bf.asarray(b, space='cuda')
test/test_linalg.py:        c = bf.zeros_like(c_gold, space='cuda')
test/test_linalg.py:        a = bf.asarray(a, space='cuda')
test/test_linalg.py:        b = bf.asarray(b, space='cuda')
test/test_linalg.py:        c = bf.zeros_like(c_gold, space='cuda')
test/test_linalg.py:        x = bf.asarray(x, space='cuda')
test/test_linalg.py:        w = bf.asarray(w, space='cuda')
test/test_linalg.py:        b = bf.zeros_like(b_gold, space='cuda')
test/test_linalg.py:        x = bf.asarray(x, space='cuda')
test/test_linalg.py:        b = bf.zeros_like(b_gold, space='cuda')
test/test_linalg.py:        x = bf.asarray(x, space='cuda')
test/test_linalg.py:        b = bf.zeros_like(b_gold, space='cuda')
test/test_linalg.py:        # Note: The xGPU-like correlation kernel is only invoked when k%4 == 0
test/test_scrunch.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_scrunch.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_gunpack.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_gunpack.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_gunpack.py:        oarray = bf.ndarray(shape=iarray.shape, dtype='ci8', space='cuda')
test/test_gunpack.py:        bf.unpack(iarray.copy(space='cuda'), oarray)
test/test_gunpack.py:        oarray = bf.ndarray(shape=iarray.shape, dtype='cf32', space='cuda')
test/test_gunpack.py:        bf.unpack(iarray.copy(space='cuda'), oarray)
test/test_fdmt.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_fdmt.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_fdmt.py:        fdmt.init(nchan, max_delay, f0, df, exponent, 'cuda')
test/test_fdmt.py:                           .astype(np.float32), space='cuda')
test/test_fdmt.py:                            space='cuda')
test/test_fdmt.py:                            space='cuda')
test/test_fdmt.py:        workspace = bf.asarray(np.empty(workspace_size, np.uint8), space='cuda')
test/test_reduce.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_reduce.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_reduce.py:        a = bf.asarray(a, space='cuda')
test/test_reduce.py:        b = bf.empty_like(b_gold, space='cuda')
test/test_reduce.py:        a = bf.ndarray(a, space='cuda')
test/test_reduce.py:        b = bf.empty_like(b_gold, space='cuda')
test/test_reduce.py:        a = bf.asarray(a, space='cuda')
test/test_reduce.py:        b = bf.empty_like(b_gold, space='cuda')
test/test_reduce.py:        a = bf.ndarray(a, space='cuda')
test/test_reduce.py:        b = bf.empty_like(b_gold, space='cuda')
test/test_managed.py:from bifrost.libbifrost_generated import BF_GPU_MANAGEDMEM
test/test_managed.py:@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
test/test_managed.py:        x = bf.asarray(x, 'cuda_managed')
test/test_managed.py:        x = bf.asarray(x, space='cuda')
test/test_managed.py:@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
test/test_managed.py:        idata = bf.ndarray(known_data, space='cuda_managed')
test/test_managed.py:        odata = bf.ndarray(shape=oshape, dtype='cf32', space='cuda_managed')
test/test_managed.py:@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
test/test_managed.py:        idata = bf.ndarray(known_data, space='cuda_managed')
test/test_managed.py:        coeffs = bf.ndarray(coeffs, space='cuda_managed')
test/test_managed.py:        idata = bf.ndarray(known_data, space='cuda_managed')
test/test_managed.py:        coeffs = bf.ndarray(coeffs, space='cuda_managed')
test/test_managed.py:@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
test/test_managed.py:        a = bf.asarray(a, space='cuda_managed')
test/test_managed.py:        b = bf.empty_like(b_gold, space='cuda_managed')
test/test_managed.py:@unittest.skipUnless(BF_GPU_MANAGEDMEM, "requires GPU managed memory support")
test/test_managed.py:        oarray = bf.ndarray(shape=iarray.shape, dtype='ci8', space='cuda_managed')
test/test_managed.py:        bf.unpack(iarray.copy(space='cuda_managed'), oarray)
test/test_block.py:        Furthermore, this block should automatically move GPU data to CPU,
test/test_block.py:        Furthermore, this block should automatically move GPU data to CPU,
test/benchmarks/development_vs_gpuspec/README.rst:Development-effort vs GPUspec code
test/benchmarks/development_vs_gpuspec/Makefile:WITH_BIFROST := blocks/accumulate.py blocks/hdf.py bf_gpuspec.py
test/benchmarks/development_vs_gpuspec/Makefile:WITHOUT_BIFROST := guppi2spectra.c guppi2spectra_gpu.cu guppi2spectra_gpu.h filterbank.h
test/benchmarks/README.rst:        #. Simple copy to GPU and back
test/benchmarks/README.rst:        #. GUPPI raw to filterbank, using GPU
test/benchmarks/README.rst:    #. CUDA kernel generation
test/benchmarks/README.rst:#. :code:`development_vs_gpuspec/` - Performance comparisons with Serial
test/benchmarks/README.rst:    #. GPU pipeline using new blocks
test/benchmarks/README.rst:    1. Element-wise CUDA kernel
test/benchmarks/README.rst:    #. Non-element-wise CUDA kernel
test/benchmarks/benchmarks.csv:performer, describe your machine, result of 'uname -a', number of CPU cores, CPU model name (see 'lspcu'), CUDA version, nvidia driver version, GPU name(s), number of GPUs, result of 'date', inside docker? (yes/no), is your system shared? (yes/no/maybe), do you have background processes? (yes/no), benchmark, benchmark options, real, user, sys," other times (write with a name and separated with a semicolon like ""name"":""time"";)", notes
test/benchmarks/benchmarks.csv:Miles Cranmer, AWS p2.xlarge instance running amazon linux with nvidia-docker, Linux 2ee4db59fe78 4.9.27-14.31.amzn1.x86_64 #1 SMP Wed May 10 01:58:40 UTC 2017 x86_64 x86_64 x86_64 GNU/Linux,4, Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz,8,375.66, Tesla K80,1, Fri Jun  9 08:32:34 UTC 2017, yes, maybe, no, general/compile_time.sh, , 2m30.859s, 3m52.548s, 0m7.864s, ,
test/benchmarks/performance_vs_serial/benchmarks5.log.txt:start key, # FFT's/2, size multiplier, gulp size, gulp frame, gulp frame fft, bifrost execution time, skcuda execution time, speedup, end key
test/benchmarks/performance_vs_serial/linear_fft_pipeline.py:class GPUFFTBenchmarker(PipelineBenchmarker):
test/benchmarks/performance_vs_serial/linear_fft_pipeline.py:            bc.blocks.copy('cuda', gulp_nframe=GULP_FRAME)
test/benchmarks/performance_vs_serial/linear_fft_pipeline.py:gpufftbenchmarker = GPUFFTBenchmarker()
test/benchmarks/performance_vs_serial/linear_fft_pipeline.py:print gpufftbenchmarker.average_benchmark(1)[0]
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:from skcuda.fft import fft, Plan, ifft
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:import pycuda.gpuarray as gpuarray
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:import pycuda.autoinit
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:def scikit_gpu_fft_pipeline(filename):
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:            g_data = gpuarray.to_gpu(data)
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:            tmp1 = gpuarray.empty(data.shape, dtype=np.complex64)
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:            tmp2 = gpuarray.empty(data.shape, dtype=np.complex64)
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:                tmp1 = gpuarray.empty(data.shape, dtype=np.complex64)
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:                tmp2 = gpuarray.empty(data.shape, dtype=np.complex64)
test/benchmarks/performance_vs_serial/skcuda_fft_pipeline.py:print scikit_gpu_fft_pipeline('numpy_data0.bin')
test/benchmarks/performance_vs_serial/run_benchmark.sh:echo "start key, # FFT's/2, size multiplier, gulp size, gulp frame, gulp frame fft, bifrost execution time, skcuda execution time, speedup, end key"
test/benchmarks/performance_vs_serial/run_benchmark.sh:NUM2="$(python skcuda_fft_pipeline.py)"
test/benchmarks/performance_vs_serial/benchmarks4.log.txt:start key, # FFT's/2, size multiplier, gulp size, gulp frame, gulp frame fft, bifrost execution time, skcuda execution time, speedup, end key
test/benchmarks/performance_vs_serial/benchmarks3.log.txt:start key, # FFT's/2, size multiplier, gulp size, gulp frame, bifrost execution time, skcuda execution time, speedup, end key
test/benchmarks/performance_vs_serial/benchmarks_ben5.log.txt:start key, # FFT's/2, size multiplier, gulp size, gulp frame, gulp frame fft, bifrost execution time, skcuda execution time, speedup, end key
test/test_pipeline.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_pipeline.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_pipeline.py:    def test_cuda_copy(self):
test/test_pipeline.py:                data = copy(data, space='cuda')
test/test_pipeline.py:                data = copy(data, space='cuda_host')
test/test_pipeline.py:            data = copy(data, space='cuda')
test/test_pipeline.py:            data = copy(data, space='cuda_host')
test/test_pipeline.py:            data = copy(data, space='cuda')
test/test_pipeline.py:            data = copy(data, space='cuda')
test/test_pipeline.py:            data = copy(data, space='cuda')
test/test_pipeline.py:            data = copy(data, space='cuda')
test/test_pipeline.py:            data = copy(data, space='cuda')
test/test_ndarray.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        c = c.copy(space='cuda').copy(space='cuda_host').copy(space='system')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        self.run_contiguous_copy(space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        self.run_slice_copy(space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        self.run_contiguous_slice_copy(space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        e = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        f = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        g = bf.ndarray(self.known_vals, dtype='f32', space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        g = bf.ndarray(self.known_vals, space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        g = bf.zeros_like(self.known_vals, space='cuda')
test/test_ndarray.py:    @unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_ndarray.py:        self.run_type_conversion(space='cuda')
test/test_guantize.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED
test/test_guantize.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_guantize.py:        oarray = bf.ndarray(shape=iarray.shape, dtype=out_dtype, space='cuda')
test/test_guantize.py:        bf.quantize(iarray.copy(space='cuda'), oarray)
test/test_transpose.py:from bifrost.libbifrost_generated import BF_CUDA_ENABLED, BF_FLOAT128_ENABLED
test/test_transpose.py:@unittest.skipUnless(BF_CUDA_ENABLED, "requires GPU support")
test/test_transpose.py:        iarray = bf.ndarray(idata, space='cuda')
README.md:| **`CPU/GPU Build`** | **`Coverage`** | 
README.md:(and a GPU is available).
README.md:Fast Dispersion Measure Transform (FDMT) on the GPU, and writes
README.md:data = bf.blocks.copy(data, 'cuda')
README.md:data = bf.blocks.copy(data, 'cuda_host')
README.md:CPU and GPU binding, data views, and dot graph output. This example
README.md:bc.blocks.copy(space='cuda', core=1)
README.md:with bf.block_scope(fuse=True, gpu=0):
README.md:bc.blocks.copy(space='cuda_host', core=2)
README.md: - Python API wraps fast C++/CUDA backend
README.md: - Native support for both system (CPU) and CUDA (GPU) memory spaces and computation
README.md:### CUDA
README.md:CUDA is available at https://developer.nvidia.com/cuda-downloads. You can check the
README.md:in the docs to see which versions of the CUDA toolkit have been confirmed to work with Bifrost. 
README.md: * [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
README.md:    $ nvidia-docker run --rm -it ledatelescope/bifrost
README.md:To run all CPU and GPU tests:
tools/like_top.py:def get_gpu_memory_usage():
tools/like_top.py:    Grab nvidia-smi output and return a dictionary of the memory usage.
tools/like_top.py:    q_flag   = '--query-gpu=memory.used,memory.total,memory.free,power.draw,power.limit,utilization.gpu'
tools/like_top.py:        p = subprocess.Popen(['nvidia-smi', q_flag, fmt_flag], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
tools/like_top.py:    display_gpu = False
tools/like_top.py:                gpu  = get_gpu_memory_usage()
tools/like_top.py:                ## Determine if we have GPU data to display
tools/like_top.py:                if gpu['devCount'] > 0:
tools/like_top.py:                    display_gpu = True
tools/like_top.py:            ### General - GPU, if avaliable
tools/like_top.py:            if display_gpu:
tools/like_top.py:                if gpu['pwrLimit'] != 0.0:
tools/like_top.py:                    if gpu['load'] != 0.0:
tools/like_top.py:                        output = 'GPU(s): %9ik total, %9ik used, %9ik free, %5.1f%%us, %.0f/%.0fW\n' % (gpu['memTotal'], gpu['memUsed'], gpu['memFree'], gpu['load'], gpu['pwrDraw'], gpu['pwrLimit'])
tools/like_top.py:                        output = 'GPU(s): %9ik total, %9ik used, %9ik free, %.0f/%.0fW\n' % (gpu['memTotal'], gpu['memUsed'], gpu['memFree'], gpu['pwrDraw'], gpu['pwrLimit'])
tools/like_top.py:                    output = 'GPU(s): %9ik total, %9ik used, %9ik free, %i device(s)\n' % (gpu['memTotal'], gpu['memUsed'], gpu['memFree'], gpu['devCount'])
NOTICE:Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
NOTICE:This product includes software developed at NVIDIA Corporation.
CHANGELOG: * Improved documentation about CUDA support
CHANGELOG: * Various fixes for CUDA 12.x releases
CHANGELOG: * Added set_stream and get_stream to bifrost.device to help control which CUDA stream is used
CHANGELOG: * Added bifrost.device.ExternalStream as a context manager to help with mixing Bifrost and cupy/pycuda
CHANGELOG: * Added support for the cuda_managed space on Pascal and later GPUs
ROADMAP.md: * CPU backends for existing CUDA-only algorithms
tutorial/README.md:You should be able to run these notebooks in any Jupyter environment that has Bifrost installed — just open the `.ipynb` files in this directory.  To try them without installing Bifrost or configuring Jupyter locally, you can open them in Google Colab (free, cloud) or use Docker (local, requires GPU hardware but dependencies are bundled).
tutorial/README.md:This is the simplest way to try the tutorial, without needing to install or configure anything. It also does not require a local GPU: Google currently provides free access to one GPU-enabled runtime (for foreground computation only).
tutorial/README.md:We provide a Docker image with Bifrost, CUDA, Jupyter, and the tutorial already installed if you want a quicker path to trying it out.  Simply run:
tutorial/README.md: docker run -p 8888:8888 --runtime=nvidia lwaproject/bifrost_tutorial
tutorial/README.md: host.  *Note that this uses Nvidia runtime for Docker to allow access to the host's GPU for the GPU-enabled
tutorial/docker/Dockerfile:ARG BASE_CONTAINER=nvidia/cuda:10.1-devel-ubuntu18.04
tutorial/docker/Dockerfile:      org.label-schema.description="Image with CUDA, Bifrost, LSL, and a useful Jupyter stack" \
tutorial/docker/Dockerfile:    ./configure --with-gpu-archs="35 50 61 75" && \
Makefile.in:#GPU Docker build
Makefile.in:	docker build --pull -t $(IMAGE_NAME):$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) -f Dockerfile.gpu -t $(IMAGE_NAME) .
Makefile.in:#GPU Docker prereq build
Makefile.in:	docker build --pull -t $(IMAGE_NAME)_prereq:$(LIBBIFROST_MAJOR).$(LIBBIFROST_MINOR) -f Dockerfile_prereq.gpu -t $(IMAGE_NAME)_prereq .
LICENSE:Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
configure:ac_unique_file="src/cuda.cpp"
configure:HAVE_CUDA_DEBUG
configure:GPU_ARCHS
configure:GPU_EXP_PINNED_ALLOC
configure:GPU_PASCAL_MANAGEDMEM
configure:GPU_SHAREDMEM
configure:GPU_MAX_ARCH
configure:GPU_MIN_ARCH
configure:CUDA_HAVE_CXX11
configure:CUDA_HAVE_CXX14
configure:CUDA_HAVE_CXX17
configure:CUDA_HAVE_CXX20
configure:CUDA_VERSION
configure:HAVE_CUDA
configure:CUDA_HOME
configure:with_cuda_home
configure:enable_cuda
configure:with_gpu_archs
configure:enable_cuda_debug
configure:  --disable-cuda          disable cuda support (default=no)
configure:  --enable-cuda-debug     enable CUDA debugging (nvcc -G; default=no)
configure:  --with-cuda-home        CUDA install path (default=/usr/local/cuda)
configure:  --with-stream-model     CUDA default stream model to use: 'legacy' or
configure:  --with-gpu-archs=...    default GPU architectures (default=detect)
configure:  --with-shared-mem=N     default GPU shared memory per block in bytes
configure:    nvcc*) # Cuda Compiler Driver 2.2
configure:	nvcc*)	# Cuda Compiler Driver 2.2
configure:# CUDA
configure:# include CUDA-specific entries #
configure:# Check whether --with-cuda_home was given.
configure:if test ${with_cuda_home+y}
configure:  withval=$with_cuda_home;
configure:  with_cuda_home=/usr/local/cuda
configure:  CUDA_HOME=$with_cuda_home
configure:  # Check whether --enable-cuda was given.
configure:if test ${enable_cuda+y}
configure:  enableval=$enable_cuda; enable_cuda=no
configure:  enable_cuda=yes
configure:  HAVE_CUDA=0
configure:  CUDA_VERSION=0
configure:  CUDA_HAVE_CXX20=0
configure:  CUDA_HAVE_CXX17=0
configure:  CUDA_HAVE_CXX14=0
configure:  CUDA_HAVE_CXX11=0
configure:  GPU_MIN_ARCH=0
configure:  GPU_MAX_ARCH=0
configure:  GPU_SHAREDMEM=0
configure:  GPU_PASCAL_MANAGEDMEM=0
configure:  GPU_EXP_PINNED_ALLOC=1
configure:  if test "$enable_cuda" != "no"; then
configure:    HAVE_CUDA=1
configure:as_dummy="$CUDA_HOME/bin:$PATH"
configure:as_dummy="$CUDA_HOME/bin:$PATH"
configure:as_dummy="$CUDA_HOME/bin:$PATH"
configure:  if test "$HAVE_CUDA" = "1"; then
configure:    { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for a working CUDA 10+ installation" >&5
configure:printf %s "checking for a working CUDA 10+ installation... " >&6; }
configure:    LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
configure:    NVCCLIBS="$LIBS -lcuda -lcudart"
configure:          #include <cuda.h>
configure:          #include <cuda_runtime.h>
configure:cudaMalloc(0, 0);
configure:  HAVE_CUDA=0
configure:    if test "$HAVE_CUDA" = "1"; then
configure:      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
configure:      NVCCLIBS="$NVCCLIBS -lcuda -lcudart"
configure:            #include <cuda.h>
configure:            #include <cuda_runtime.h>
configure:cudaMalloc(0, 0);
configure:  CUDA_VERSION=$( ${NVCC} --version | ${GREP} -Po -e "release.*," | cut -d,  -f1 | cut -d\  -f2 )
configure:           { printf "%s\n" "$as_me:${as_lineno-$LINENO}: result: yes - v$CUDA_VERSION" >&5
configure:printf "%s\n" "yes - v$CUDA_VERSION" >&6; }
configure:           HAVE_CUDA=0
configure:      HAVE_CUDA=0
configure:  if test "$HAVE_CUDA" = "1"; then
configure:    { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for CUDA CXX standard support" >&5
configure:printf %s "checking for CUDA CXX standard support... " >&6; }
configure:    CUDA_STDCXX=$( ${NVCC} --help | ${GREP} -Po -e "--std.*}" | ${SED} 's/.*|//;s/}//;' )
configure:    if test "$CUDA_STDCXX" = "c++20"; then
configure:      CUDA_HAVE_CXX20=1
configure:      if test "$CUDA_STDCXX" = "c++17"; then
configure:        CUDA_HAVE_CXX17=1
configure:        if test "$CUDA_STDCXX" = "c++14"; then
configure:          CUDA_HAVE_CXX14=1
configure:          if test "$CUDA_STDCXX" = "c++11"; then
configure:            CUDA_HAVE_CXX11=1
configure:  if test "$HAVE_CUDA" = "1"; then
configure:    { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for different CUDA default stream models" >&5
configure:printf %s "checking for different CUDA default stream models... " >&6; }
configure:          as_fn_error $? "Invalid CUDA stream model: '$with_stream_model'" "$LINENO" 5
configure:  if test "$HAVE_CUDA" = "1"; then
configure:    CPPFLAGS="$CPPFLAGS -DBF_CUDA_ENABLED=1"
configure:    CXXFLAGS="$CXXFLAGS -DBF_CUDA_ENABLED=1"
configure:    NVCCFLAGS="$NVCCFLAGS -DBF_CUDA_ENABLED=1"
configure:    LDFLAGS="$LDFLAGS -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
configure:    NVCCLIBS="$NVCCLIBS -lcuda -lcudart -lnvrtc -lcublas -lcudadevrt -L. -lcufft_static_pruned -lculibos -lnvToolsExt"
configure:# Check whether --with-gpu_archs was given.
configure:if test ${with_gpu_archs+y}
configure:  withval=$with_gpu_archs;
configure:  with_gpu_archs='auto'
configure:  if test "$HAVE_CUDA" = "1"; then
configure:    { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for valid CUDA architectures" >&5
configure:printf %s "checking for valid CUDA architectures... " >&6; }
configure:    if test "$with_gpu_archs" = "auto"; then
configure:      { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking which CUDA architectures to target" >&5
configure:printf %s "checking which CUDA architectures to target... " >&6; }
configure:      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
configure:      NVCCLIBS="-lcuda -lcudart"
configure:            #include <cuda.h>
configure:            #include <cuda_runtime.h>
configure:            cudaGetDeviceCount(&deviceCount);
configure:              cudaSetDevice(dev);
configure:              cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
configure:              cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
configure:  GPU_ARCHS=`cat confarchs.out`
configure:             ar_valid=$( echo $GPU_ARCHS $ar_supported | xargs -n1 | sort | uniq -d | xargs )
configure:               GPU_ARCHS=$ar_valid
configure:               { printf "%s\n" "$as_me:${as_lineno-$LINENO}: result: $GPU_ARCHS" >&5
configure:printf "%s\n" "$GPU_ARCHS" >&6; }
configure:      GPU_ARCHS=$with_gpu_archs
configure:    { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for valid requested CUDA architectures" >&5
configure:printf %s "checking for valid requested CUDA architectures... " >&6; }
configure:    ar_requested=$( echo "$GPU_ARCHS" | wc -w )
configure:    ar_valid=$( echo $GPU_ARCHS $ar_supported | xargs -n1 | sort | uniq -d | xargs )
configure:    GPU_MIN_ARCH=$ar_min_valid
configure:    GPU_MAX_ARCH=$ar_max_valid
configure:      LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
configure:      NVCCLIBS="-lcuda -lcudart"
configure:            #include <cuda.h>
configure:            #include <cuda_runtime.h>
configure:            cudaGetDeviceCount(&deviceCount);
configure:              cudaSetDevice(dev);
configure:              cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, dev);
configure:  GPU_SHAREDMEM=`cat confsmem.out`
configure:             { printf "%s\n" "$as_me:${as_lineno-$LINENO}: result: $GPU_SHAREDMEM B" >&5
configure:printf "%s\n" "$GPU_SHAREDMEM B" >&6; }
configure:      GPU_SHAREDMEM=$with_shared_mem
configure:    { printf "%s\n" "$as_me:${as_lineno-$LINENO}: checking for Pascal-style CUDA managed memory" >&5
configure:printf %s "checking for Pascal-style CUDA managed memory... " >&6; }
configure:    cm_invalid=$( echo $GPU_ARCHS | ${SED} -e 's/\b[1-5][0-9]\b/PRE/g;' )
configure:      GPU_PASCAL_MANAGEDMEM=1
configure:      GPU_PASCAL_MANAGEDMEM=0
configure:    LDFLAGS="-L$CUDA_HOME/lib64 -L$CUDA_HOME/lib"
configure:    NVCCLIBS="-lcuda -lcudart"
configure:          #include <cuda.h>
configure:          #include <cuda_runtime.h>
configure:          #include <thrust/system/cuda/memory.h>
configure:  GPU_EXP_PINNED_ALLOC=0
configure:  GPU_EXP_PINNED_ALLOC=1
configure:     GPU_PASCAL_MANAGEDMEM=0
configure:     GPU_EXP_PINNED_ALLOC=1
configure:# Check whether --enable-cuda_debug was given.
configure:if test ${enable_cuda_debug+y}
configure:  enableval=$enable_cuda_debug; enable_cuda_debug=yes
configure:  enable_cuda_debug=no
configure:HAVE_CUDA_DEBUG=0
configure:if test x$enable_cuda_debug != xno
configure:  HAVE_CUDA_DEBUG=1
configure:  HAVE_MAP_CACHE=$HAVE_CUDA
configure:       if test x$CUDA_HAVE_CXX20 = x1
configure:  if test x$CUDA_HAVE_CXX17 = x1
configure:  if test x$CUDA_HAVE_CXX14 = x1
configure:       if test x$CUDA_HAVE_CXX20 = x1
configure:  if test x$CUDA_HAVE_CXX17 = x1
configure:  if test x$CUDA_HAVE_CXX14 = x1
configure:       if test x$CUDA_HAVE_CXX20 = x1
configure:  if test x$CUDA_HAVE_CXX17 = x1
configure:  if test x$CUDA_HAVE_CXX14 = x1
configure:# Additional CUDA flags
configure:if test x$HAVE_CUDA != x0
configure:  NVCC_GENCODE=$(echo $GPU_ARCHS | ${SED} -e 's/\([0-9]\{2,3\}\)/-gencode arch=compute_\1,\\"code=sm_\1\\"/g;')
configure:       NVCC_GENCODE="$NVCC_GENCODE -gencode arch=compute_${GPU_MAX_ARCH},\\\"code=compute_${GPU_MAX_ARCH}\\\""
configure:       CPPFLAGS="$CPPFLAGS -I$CUDA_HOME/include"
configure:if test x$HAVE_CUDA = x1
configure:  { printf "%s\n" "$as_me:${as_lineno-$LINENO}: cuda: yes - v$CUDA_VERSION - $GPU_ARCHS - $with_stream_model streams" >&5
configure:printf "%s\n" "$as_me: cuda: yes - v$CUDA_VERSION - $GPU_ARCHS - $with_stream_model streams" >&6;}
configure:  { printf "%s\n" "$as_me:${as_lineno-$LINENO}: cuda: no" >&5
configure:printf "%s\n" "$as_me: cuda: no" >&6;}
configure:if test x$enable_cuda_debug != xno
configure:  OPTIONS="$OPTIONS cuda_debug"
Dockerfile.gpu:FROM nvidia/cuda:10.2-devel-ubuntu18.04
.gitignore:test/benchmarks/development_vs_gpuspec/with_bifrost/
.gitignore:test/benchmarks/development_vs_gpuspec/without_bifrost/
Dockerfile.cpu:    make -j NOCUDA=1 && \
.lgtm.yml:      - export NOCUDA=1
src/Complex.hpp:#if BF_CUDA_ENABLED
src/Complex.hpp:#include <cuda_fp16.h>
src/Complex.hpp:#ifndef __CUDACC_RTC__
src/Complex.hpp:#endif // __CUDACC_RTC__
src/Complex.hpp:#ifdef __CUDACC_VER_MAJOR__
src/Complex.hpp:template<typename Real> struct cuda_vector2_type {};
src/Complex.hpp:template<>              struct cuda_vector2_type<float>  { typedef float2  type; };
src/Complex.hpp:template<>              struct cuda_vector2_type<double> { typedef double2 type; };
src/Complex.hpp:#ifdef __CUDACC_VER_MAJOR__
src/Complex.hpp:	inline __host__ __device__ Complex(typename Complex_detail::cuda_vector2_type<T>::type c) : x(c.x), y(c.y) {}
src/Complex.hpp:	inline __host__ __device__ operator typename Complex_detail::cuda_vector2_type<T>::type() const { return make_float2(x,y); }
src/Complex.hpp:#if BF_CUDA_ENABLED
src/Complex.hpp:#ifdef __CUDA_ARCH__
src/Complex.hpp:#endif // __CUDA_ARCH__
src/map.cpp:#include "cuda.hpp"
src/map.cpp:#include <cuda.h>
src/map.cpp:#if CUDA_VERSION >= 7500
src/map.cpp:	cc_ss << "compute_" << BF_GPU_MIN_ARCH;
src/map.cpp:	cc_ss << "compute_" << get_cuda_device_cc();
src/map.cpp:		cudaRuntimeGetVersion(&rt);
src/map.cpp:		cudaDriverGetVersion(&drv);
src/map.cpp:		cudaRuntimeGetVersion(&rt);
src/map.cpp:		cudaDriverGetVersion(&drv);
src/map.cpp:	void load_from_disk(ObjectCache<std::string,std::pair<CUDAKernel,bool> > *kernel_cache) {
src/map.cpp:      CUDAKernel kernel;
src/map.cpp:	bool load(ObjectCache<std::string,std::pair<CUDAKernel,bool> > *kernel_cache) {
src/map.cpp:	thread_local static ObjectCache<std::string,std::pair<CUDAKernel,bool> >
src/map.cpp:		CUDAKernel kernel;
src/map.cpp:	CUDAKernel& kernel = cache_entry.first;
src/map.cpp:			BF_ASSERT(space_accessible_from(args[a]->space, BF_SPACE_CUDA),
src/map.cpp:	                        0, g_cuda_stream,
src/map.cpp:	                        kernel_args) == CUDA_SUCCESS,
src/transpose_gpu_kernel.cuh: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/transpose_gpu_kernel.cuh:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
src/ring_impl.hpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/linalg_kernels.cu:#include "cuda/stream.hpp"
src/linalg_kernels.cu:                                cudaTextureObject_t A_tex,
src/linalg_kernels.cu:// Cherk kernel based on xGPU
src/linalg_kernels.cu:                                   cudaTextureObject_t A_tex,
src/linalg_kernels.cu:                cudaStream_t stream) {
src/linalg_kernels.cu:	//   to the alignment (which should always be safe because cudaMalloc
src/linalg_kernels.cu:	cudaChannelFormatKind channel_format;
src/linalg_kernels.cu:	cudaTextureReadMode   tex_read_mode;
src/linalg_kernels.cu:		channel_format = cudaChannelFormatKindSigned;
src/linalg_kernels.cu:		tex_read_mode  = cudaReadModeNormalizedFloat;
src/linalg_kernels.cu:		channel_format = cudaChannelFormatKindFloat;
src/linalg_kernels.cu:		tex_read_mode  = cudaReadModeElementType;
src/linalg_kernels.cu:	cudaResourceDesc resDesc;
src/linalg_kernels.cu:	resDesc.resType = cudaResourceTypeLinear;
src/linalg_kernels.cu:	cudaTextureDesc texDesc;
src/linalg_kernels.cu:	cudaTextureObject_t A_tex = 0;
src/linalg_kernels.cu:	BF_CHECK_CUDA_EXCEPTION(
src/linalg_kernels.cu:		cudaCreateTextureObject(&A_tex, &resDesc, &texDesc, NULL),
src/linalg_kernels.cu:		// TODO: Replace with cudaLaunchKernel
src/linalg_kernels.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/linalg_kernels.cu:	BF_CHECK_CUDA_EXCEPTION(
src/linalg_kernels.cu:		cudaDestroyTextureObject(A_tex),
src/linalg_kernels.cu:	cudaResourceDesc resDesc2;
src/linalg_kernels.cu:	resDesc2.resType = cudaResourceTypeLinear;
src/linalg_kernels.cu:	cudaTextureDesc texDesc2;
src/linalg_kernels.cu:	cudaTextureObject_t A_tex2 = 0;
src/linalg_kernels.cu:	BF_CHECK_CUDA_EXCEPTION(
src/linalg_kernels.cu:		cudaCreateTextureObject(&A_tex2, &resDesc2, &texDesc2, NULL),
src/linalg_kernels.cu:	BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/linalg_kernels.cu:	BF_CHECK_CUDA_EXCEPTION(
src/linalg_kernels.cu:		cudaDestroyTextureObject(A_tex2),
src/linalg_kernels.cu://   to fill the GPU.
src/linalg_kernels.cu:                                   cudaStream_t stream) {
src/linalg_kernels.cu:	bool B_fits_in_shared_mem = (smem <= BF_GPU_SHAREDMEM);
src/linalg_kernels.cu:	/* // TODO: Use cudaLaunchKernel instead of <<< >>>
src/linalg_kernels.cu:                        cudaStream_t stream) {
src/trace.hpp:#include "cuda.hpp"
src/trace.hpp:#if BF_CUDA_ENABLED
src/trace.hpp:#if BF_CUDA_ENABLED
src/trace.hpp:	cudaStream_t          _stream;
src/trace.hpp:	inline static void range_start_callback(cudaStream_t stream, cudaError_t status, void* userData) {
src/trace.hpp:	inline static void range_end_callback(cudaStream_t stream, cudaError_t status, void* userData) {
src/trace.hpp:	inline AsyncTracer(cudaStream_t stream) : _stream(stream), _id(0), _attrs() {}
src/trace.hpp:		cudaStreamAddCallback(_stream, range_start_callback, (void*)this, 0);
src/trace.hpp:		cudaStreamAddCallback(_stream, range_end_callback, (void*)this, 0);
src/trace.hpp:typedef std::map<cudaStream_t,std::queue<AsyncTracer*> > TracerStreamMap;
src/trace.hpp:#endif // BF_CUDA_ENABLED
src/trace.hpp:#if BF_CUDA_ENABLED
src/trace.hpp:	cudaStream_t _stream;
src/trace.hpp:#if BF_CUDA_ENABLED
src/trace.hpp:#if BF_CUDA_ENABLED
src/trace.hpp:	inline ScopedTracer(std::string name, cudaStream_t stream=0)
src/memory.cpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/memory.cpp:#include "cuda.hpp"
src/memory.cpp:#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
src/memory.cpp:	cudaPointerAttributes ptr_attrs;
src/memory.cpp:	cudaError_t ret = cudaPointerGetAttributes(&ptr_attrs, ptr);
src/memory.cpp:	BF_ASSERT(ret == cudaSuccess || ret == cudaErrorInvalidValue,
src/memory.cpp:	if( ret == cudaErrorInvalidValue ) {
src/memory.cpp:		//           up in cuda-memcheck?
src/memory.cpp:		// Note: cudaPointerGetAttributes only works for memory allocated with
src/memory.cpp:		//         CUDA API functions, so if it fails we just assume sysmem.
src/memory.cpp:		cudaGetLastError();
src/memory.cpp:#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
src/memory.cpp:        		case cudaMemoryTypeUnregistered: *space = BF_SPACE_SYSTEM;       break;
src/memory.cpp:        		case cudaMemoryTypeHost:         *space = BF_SPACE_CUDA_HOST;    break;
src/memory.cpp:        		case cudaMemoryTypeDevice:       *space = BF_SPACE_CUDA;         break;
src/memory.cpp:        		case cudaMemoryTypeManaged:      *space = BF_SPACE_CUDA_MANAGED; break;
src/memory.cpp:		*space = BF_SPACE_CUDA_MANAGED;
src/memory.cpp:		case cudaMemoryTypeHost:   *space = BF_SPACE_SYSTEM; break;
src/memory.cpp:		case cudaMemoryTypeDevice: *space = BF_SPACE_CUDA;   break;
src/memory.cpp:#endif  // defined(CUDA_VERSION) && CUDA_VERSION >= 10000
src/memory.cpp:		case BF_SPACE_CUDA:         return "cuda";
src/memory.cpp:		case BF_SPACE_CUDA_HOST:    return "cuda_host";
src/memory.cpp:		case BF_SPACE_CUDA_MANAGED: return "cuda_managed";
src/memory.cpp:#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
src/memory.cpp:	case BF_SPACE_CUDA: {
src/memory.cpp:		BF_CHECK_CUDA(cudaMalloc((void**)&data, size),
src/memory.cpp:	case BF_SPACE_CUDA_HOST: {
src/memory.cpp:		unsigned flags = cudaHostAllocDefault;
src/memory.cpp:		BF_CHECK_CUDA(cudaHostAlloc((void**)&data, size, flags),
src/memory.cpp:	case BF_SPACE_CUDA_MANAGED: {
src/memory.cpp:		unsigned flags = cudaMemAttachGlobal;
src/memory.cpp:		BF_CHECK_CUDA(cudaMallocManaged((void**)&data, size, flags),
src/memory.cpp:#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
src/memory.cpp:	case BF_SPACE_CUDA:         cudaFree(ptr); break;
src/memory.cpp:	case BF_SPACE_CUDA_HOST:    cudaFreeHost(ptr); break;
src/memory.cpp:	case BF_SPACE_CUDA_MANAGED: cudaFree(ptr); break;
src/memory.cpp:#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
src/memory.cpp:		//         than using cudaMemcpyDefault.
src/memory.cpp:		cudaMemcpyKind kind = cudaMemcpyDefault;
src/memory.cpp:		case BF_SPACE_CUDA_HOST: // fall-through
src/memory.cpp:			case BF_SPACE_CUDA_HOST: // fall-through
src/memory.cpp:			case BF_SPACE_CUDA: kind = cudaMemcpyHostToDevice; break;
src/memory.cpp:			case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
src/memory.cpp:		case BF_SPACE_CUDA: {
src/memory.cpp:			case BF_SPACE_CUDA_HOST: // fall-through
src/memory.cpp:			case BF_SPACE_SYSTEM: kind = cudaMemcpyDeviceToHost; break;
src/memory.cpp:			case BF_SPACE_CUDA: kind = cudaMemcpyDeviceToDevice; break;
src/memory.cpp:			case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
src/memory.cpp:		case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
src/memory.cpp:		BF_TRACE_STREAM(g_cuda_stream);
src/memory.cpp:		BF_CHECK_CUDA(cudaMemcpyAsync(dst, src, count, kind, g_cuda_stream),
src/memory.cpp:#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
src/memory.cpp:		//         than using cudaMemcpyDefault.
src/memory.cpp:		cudaMemcpyKind kind = cudaMemcpyDefault;
src/memory.cpp:		case BF_SPACE_CUDA_HOST: // fall-through
src/memory.cpp:			case BF_SPACE_CUDA_HOST: // fall-through
src/memory.cpp:			case BF_SPACE_CUDA: kind = cudaMemcpyHostToDevice; break;
src/memory.cpp:			case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
src/memory.cpp:		case BF_SPACE_CUDA: {
src/memory.cpp:			case BF_SPACE_CUDA_HOST: // fall-through
src/memory.cpp:			case BF_SPACE_SYSTEM: kind = cudaMemcpyDeviceToHost; break;
src/memory.cpp:			case BF_SPACE_CUDA:   kind = cudaMemcpyDeviceToDevice; break;
src/memory.cpp:			case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
src/memory.cpp:		case BF_SPACE_CUDA_MANAGED: kind = cudaMemcpyDefault; break;
src/memory.cpp:		BF_TRACE_STREAM(g_cuda_stream);
src/memory.cpp:		BF_CHECK_CUDA(cudaMemcpy2DAsync(dst, dst_stride,
src/memory.cpp:		                                kind, g_cuda_stream),
src/memory.cpp:#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
src/memory.cpp:		case BF_SPACE_CUDA_HOST:    ::memset(ptr, value, count); break;
src/memory.cpp:		case BF_SPACE_CUDA: // Fall-through
src/memory.cpp:		case BF_SPACE_CUDA_MANAGED: {
src/memory.cpp:			BF_TRACE_STREAM(g_cuda_stream);
src/memory.cpp:			BF_CHECK_CUDA(cudaMemsetAsync(ptr, value, count, g_cuda_stream),
src/memory.cpp:#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
src/memory.cpp:		case BF_SPACE_CUDA_HOST:    memset2D(ptr, stride, value, width, height); break;
src/memory.cpp:		case BF_SPACE_CUDA: // Fall-through
src/memory.cpp:		case BF_SPACE_CUDA_MANAGED: {
src/memory.cpp:			BF_TRACE_STREAM(g_cuda_stream);
src/memory.cpp:			BF_CHECK_CUDA(cudaMemset2DAsync(ptr, stride, value, width, height, g_cuda_stream),
src/guantize.hu: * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
src/guantize.hu:void launch_foreach_simple_gpu(T const*     in,
src/guantize.hu:                               cudaStream_t stream=0);
src/guantize.hu:void launch_foreach_simple_gpu_4bit(T const*     in,
src/guantize.hu:                                    cudaStream_t stream=0);
src/guantize.hu:void launch_foreach_simple_gpu_2bit(T const*     in,
src/guantize.hu:                                    cudaStream_t stream=0);
src/guantize.hu:void launch_foreach_simple_gpu_1bit(T const*     in,
src/guantize.hu:                                    cudaStream_t stream=0);
src/array_utils.hpp:	//case BF_DTYPE_F128:  return "long double"; // TODO: This doesn't seem to work properly in CUDA
src/cuda.cpp:#include <bifrost/cuda.h>
src/cuda.cpp:#include "cuda.hpp"
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:thread_local cudaStream_t g_cuda_stream = cudaStreamPerThread;
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:	*(cudaStream_t*)stream = g_cuda_stream;
src/cuda.cpp:	BF_FAIL("Built without CUDA support (bfStreamGet)", BF_STATUS_INVALID_STATE);
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:	g_cuda_stream = *(cudaStream_t*)stream;
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:	BF_CHECK_CUDA(cudaGetDevice(device), BF_STATUS_DEVICE_ERROR);
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:	BF_CHECK_CUDA(cudaSetDevice(device), BF_STATUS_DEVICE_ERROR);
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:	BF_CHECK_CUDA(cudaDeviceGetByPCIBusId(&device, pci_bus_id),
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:	BF_CHECK_CUDA(cudaStreamSynchronize(g_cuda_stream),
src/cuda.cpp:#if BF_CUDA_ENABLED
src/cuda.cpp:	BF_CHECK_CUDA(cudaGetDevice(&old_device), BF_STATUS_DEVICE_ERROR);
src/cuda.cpp:	BF_CHECK_CUDA(cudaGetDeviceCount(&ndevices), BF_STATUS_DEVICE_ERROR);
src/cuda.cpp:		BF_CHECK_CUDA(cudaSetDevice(d), BF_STATUS_DEVICE_ERROR);
src/cuda.cpp:		BF_CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),
src/cuda.cpp:	BF_CHECK_CUDA(cudaSetDevice(old_device), BF_STATUS_DEVICE_ERROR);
src/autodep.mk:# This defines build rules for c/c++/cuda sources with automatic dependency generation
src/autodep.mk:	@echo "Building CUDA source file $<"
src/Jones.hpp:#ifdef __CUDACC__
src/Jones.hpp:#ifdef __CUDACC__
src/Jones.hpp:#ifdef __CUDACC__
src/Jones.hpp:#ifdef __CUDACC__
src/fft_kernels.cu:#include "cuda.hpp"
src/fft_kernels.cu:		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
src/fft_kernels.cu:		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
src/fft_kernels.cu:		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
src/fft_kernels.cu:		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
src/fft_kernels.cu:		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
src/fft_kernels.cu:		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
src/fft_kernels.cu:		BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_r_hptr,
src/fft_kernels.cu:			BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_c_hptr,
src/fft_kernels.cu:			BF_CHECK_CUDA( cudaMemcpyFromSymbol(&callback_load_z_hptr,
src/affinity.cpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/romein.cu:Implements the Romein convolutional algorithm onto a GPU using CUDA. 
src/romein.cu:#include "cuda.hpp"
src/romein.cu:#include "cuda/stream.hpp"
src/romein.cu:                                 cudaStream_t stream=0) {
src/romein.cu:    if(loc_size <= BF_GPU_SHAREDMEM) {
src/romein.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)romein_kernel_sloc<InType,OutType>,
src/romein.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)romein_kernel<InType,OutType>,
src/romein.cu:public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
src/romein.cu:    cudaStream_t _stream;
src/romein.cu:                      _maxsupport(1), _stream(g_cuda_stream) {}
src/romein.cu:        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/romein.cu:        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/romein.cu:    void set_stream(cudaStream_t stream) {
src/romein.cu:    BF_ASSERT(space_accessible_from(positions->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/romein.cu:    BF_ASSERT(space_accessible_from(kernels->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/romein.cu:    BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
src/romein.cu:    BF_ASSERT(space_accessible_from(positions->space,   BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/romein.cu:    BF_ASSERT(space_accessible_from(kernels->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/romein.cu:    BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/romein.cu:    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/ring_impl.cpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/ring_impl.cpp:#include <bifrost/cuda.h>
src/ring_impl.cpp:#include "cuda.hpp"
src/ring_impl.cpp:#if defined BF_CUDA_ENABLED && BF_CUDA_ENABLED
src/ring_impl.cpp:	                    space==BF_SPACE_CUDA         ||
src/ring_impl.cpp:	                    space==BF_SPACE_CUDA_HOST    ||
src/ring_impl.cpp:	                    space==BF_SPACE_CUDA_MANAGED,
src/bifrost/affinity.h: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/bifrost/config.h.in:// CUDA support
src/bifrost/config.h.in:#define BF_CUDA_ENABLED @HAVE_CUDA@
src/bifrost/config.h.in:#define BF_CUDA_VERSION @CUDA_VERSION@
src/bifrost/config.h.in:#define BF_GPU_ARCHS "@GPU_ARCHS@"
src/bifrost/config.h.in:#define BF_GPU_MIN_ARCH @GPU_MIN_ARCH@
src/bifrost/config.h.in:#define BF_GPU_MAX_ARCH @GPU_MAX_ARCH@
src/bifrost/config.h.in:#define BF_GPU_SHAREDMEM @GPU_SHAREDMEM@
src/bifrost/config.h.in:#define BF_GPU_MANAGEDMEM @GPU_PASCAL_MANAGEDMEM@
src/bifrost/config.h.in:#define BF_GPU_EXP_PINNED_ALLOC @GPU_EXP_PINNED_ALLOC@
src/bifrost/config.h.in:#define BF_CUDA_DEBUG_ENABLED @HAVE_CUDA_DEBUG@
src/bifrost/common.h: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/bifrost/common.h:BFbool      bfGetCudaEnabled();
src/bifrost/ring.h: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/bifrost/ring.h:// TODO: bfCudaEnabled
src/bifrost/memory.h: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/bifrost/memory.h:	BF_SPACE_CUDA         = 2, // cudaMalloc
src/bifrost/memory.h:	BF_SPACE_CUDA_HOST    = 3, // cudaHostAlloc
src/bifrost/memory.h:	BF_SPACE_CUDA_MANAGED = 4  // cudaMallocManaged
src/bifrost/transpose.h: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/bifrost/cuda.h:#ifndef BF_CUDA_H_INCLUDE_GUARD_
src/bifrost/cuda.h:#define BF_CUDA_H_INCLUDE_GUARD_
src/bifrost/cuda.h:#endif // BF_CUDA_H_INCLUDE_GUARD_
src/bifrost/map.h: *        as CUDA device code. Examples:\n
src/utils.hpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/utils.hpp:#include "cuda.hpp"
src/utils.hpp:#if BF_CUDA_ENABLED
src/utils.hpp:#if !defined BF_CUDA_ENABLED || !BF_CUDA_ENABLED
src/utils.hpp:	                              space == BF_SPACE_CUDA_HOST ||
src/utils.hpp:	                              space == BF_SPACE_CUDA_MANAGED);
src/utils.hpp:	case BF_SPACE_CUDA:   return (space == BF_SPACE_CUDA ||
src/utils.hpp:	                              space == BF_SPACE_CUDA_MANAGED);
src/utils.hpp:#if BF_GPU_MANAGEDMEM
src/utils.hpp:        case BF_SPACE_CUDA_MANAGED: return (space == BF_SPACE_SYSTEM ||
src/utils.hpp:                                            space == BF_SPACE_CUDA_HOST ||
src/utils.hpp:                                            space == BF_SPACE_CUDA_MANAGED ||
src/utils.hpp:                                            space == BF_SPACE_CUDA);
src/utils.hpp:#if BF_CUDA_ENABLED
src/gunpack.cu: * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
src/gunpack.cu:#include "cuda.hpp"
src/gunpack.cu:		byteswap_gpu(oval, &oval);
src/gunpack.cu:		byteswap_gpu(oval, &oval);
src/gunpack.cu:		rshift_subwords_gpu<4,int8_t>(oval);
src/gunpack.cu:		conjugate_subwords_gpu<int8_t>(oval);
src/gunpack.cu:		byteswap_gpu(oval, &oval);
src/gunpack.cu:		rshift_subwords_gpu<6,int8_t>(oval);
src/gunpack.cu:		conjugate_subwords_gpu<int8_t>(oval);
src/gunpack.cu:		byteswap_gpu(oval, &oval);
src/gunpack.cu:		rshift_subwords_gpu<6,int8_t>(oval);
src/gunpack.cu:		conjugate_subwords_gpu<int8_t>(oval);
src/gunpack.cu:__global__ void foreach_simple_gpu(T const* in,
src/gunpack.cu:inline void launch_foreach_simple_gpu(T const*     in,
src/gunpack.cu:                                      cudaStream_t stream) {
src/gunpack.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu<T,U,Func,Size>,
src/gunpack.cu:__global__ void foreach_promote_gpu(T const* in,
src/gunpack.cu:inline void launch_foreach_promote_gpu(T const*     in,
src/gunpack.cu:                                       cudaStream_t stream) {
src/gunpack.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_promote_gpu<T,U,V,Func,Size>,
src/gunpack.cu:// Instantiation - launch_foreach_simple_gpu calls used in unpack.cpp
src/gunpack.cu:template void launch_foreach_simple_gpu<uint8_t,uint16_t,GunpackFunctor<uint8_t,uint16_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                  cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_simple_gpu<uint8_t,uint32_t,GunpackFunctor<uint8_t,uint32_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                  cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_simple_gpu<uint8_t,uint64_t,GunpackFunctor<uint8_t,uint64_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                  cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_simple_gpu<uint8_t,int16_t,GunpackFunctor<uint8_t,int16_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_simple_gpu<uint8_t,int32_t,GunpackFunctor<uint8_t,int32_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_simple_gpu<uint8_t,int64_t,GunpackFunctor<uint8_t,int64_t>,size_t>(uint8_t const *in,
src/gunpack.cu:                                                                                                cudaStream_t   stream);
src/gunpack.cu:// Instantiation - launch_foreach_promote_gpu calls used in unpack.cpp
src/gunpack.cu:template void launch_foreach_promote_gpu<uint8_t,int16_t,float,GunpackFunctor<uint8_t,int16_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                       cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_promote_gpu<uint8_t,int32_t,float,GunpackFunctor<uint8_t,int32_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                       cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_promote_gpu<uint8_t,int64_t,float,GunpackFunctor<uint8_t,int64_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                       cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_promote_gpu<uint8_t,int16_t,double,GunpackFunctor<uint8_t,int16_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                        cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_promote_gpu<uint8_t,int32_t,double,GunpackFunctor<uint8_t,int32_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                        cudaStream_t   stream);
src/gunpack.cu:template void launch_foreach_promote_gpu<uint8_t,int64_t,double,GunpackFunctor<uint8_t,int64_t>,size_t>(uint8_t const* in,
src/gunpack.cu:                                                                                                        cudaStream_t   stream);
src/common.cpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/common.cpp:BFbool bfGetCudaEnabled() {
src/common.cpp:  return BF_CUDA_ENABLED;
src/fir.cu:#include "cuda.hpp"
src/fir.cu:#include <math_constants.h> // For CUDART_NAN_F
src/fir.cu:                              cudaStream_t stream=0) {
src/fir.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)fir_kernel<InType,OutType>,
src/fir.cu:public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
src/fir.cu:	cudaStream_t _stream;
src/fir.cu:	BFfir_impl() : _coeffs(NULL), _decim(1), _stream(g_cuda_stream) {}
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION( cudaMemsetAsync(_state0,
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION( cudaMemsetAsync(_state1,
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_state0,
src/fir.cu:		                                         cudaMemcpyDeviceToDevice,
src/fir.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/fir.cu:	void set_stream(cudaStream_t stream) {
src/fir.cu:	BF_ASSERT(space_accessible_from(coeffs->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/fir.cu:	BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
src/fir.cu:	BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
src/fir.cu:	BF_ASSERT(space_accessible_from(coeffs->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/fir.cu:	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/fir.cu:	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_INVALID_SPACE);
src/Makefile.in:HAVE_CUDA     ?= @HAVE_CUDA@
src/Makefile.in:GPU_ARCHS     ?= @GPU_ARCHS@
src/Makefile.in:CUDA_HOME     ?= @CUDA_HOME@
src/Makefile.in:CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
src/Makefile.in:CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
src/Makefile.in:CUDA_INCDIR   ?= $(CUDA_HOME)/include
src/Makefile.in:  cuda.o \
src/Makefile.in:ifeq ($(HAVE_CUDA),1)
src/Makefile.in:  # These files require the CUDA Toolkit to compile
src/Makefile.in:#NVCCFLAGS += -Xcudafe "--diag_suppress=unrecognized_gcc_pragma"
src/Makefile.in:ifeq ($(HAVE_CUDA),1)
src/Makefile.in:LIBCUFFT_STATIC = $(CUDA_LIBDIR64)/libcufft_static.a
src/Makefile.in:	#   E.g., We may have GPU_ARCHS="35 61" but libcufft_static might only
src/Makefile.in:_cuda_device_link.o: Makefile fft_kernels.o libcufft_static_pruned.a
src/Makefile.in:	@echo "Linking _cuda_device_link.o"
src/Makefile.in:CUDA_DEVICE_LINK_OBJ = _cuda_device_link.o
src/Makefile.in:CUDA_DEVICE_LINK_OBJ =
src/Makefile.in:$(LIBBIFROST_SO): $(LIBBIFROST_OBJS) $(LIBBIFROST_VERSION_FILE) $(CUDA_DEVICE_LINK_OBJ)
src/Makefile.in:	$(LINKER) $(SHARED_FLAG) -Wl,$(WLFLAGS) -o $@ $(LIBBIFROST_OBJS) $(CUDA_DEVICE_LINK_OBJ) $(LIB) $(LDFLAGS)
src/ring.cpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/romein_kernels.cuh://CUDA Includes
src/romein_kernels.cuh:#include "cuda.h"
src/romein_kernels.cuh:#include "cuda_runtime_api.h"
src/romein_kernels.cuh:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:#include "cuda.hpp"
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM) || (space_accessible_from(in->space, BF_SPACE_CUDA) && space_accessible_from(out->space, BF_SPACE_CUDA)),
src/unpack.cpp:	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_SYSTEM) || (space_accessible_from(in->space, BF_SPACE_CUDA) && space_accessible_from(out->space, BF_SPACE_CUDA)),
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:	if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:#define CALL_FOREACH_SIMPLE_GPU_UNPACK(itype,otype) \
src/unpack.cpp:	BF_TRACE_STREAM(g_cuda_stream); \
src/unpack.cpp:	launch_foreach_simple_gpu((itype*)in->data, \
src/unpack.cpp:	                          g_cuda_stream); \
src/unpack.cpp:#define CALL_FOREACH_PROMOTE_GPU_UNPACK(itype,ttype,otype) \
src/unpack.cpp:	BF_TRACE_STREAM(g_cuda_stream); \
src/unpack.cpp:	launch_foreach_promote_gpu((itype*)in->data, \
src/unpack.cpp:	                           g_cuda_stream); \
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_SIMPLE_GPU_UNPACK(uint8_t,int64_t);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_SIMPLE_GPU_UNPACK(uint8_t,int32_t);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_SIMPLE_GPU_UNPACK(uint8_t,int16_t);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_SIMPLE_GPU_UNPACK(uint8_t,uint32_t);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_SIMPLE_GPU_UNPACK(uint8_t,uint16_t);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_PROMOTE_GPU_UNPACK(uint8_t,int64_t,float);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_PROMOTE_GPU_UNPACK(uint8_t,int32_t,float);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_PROMOTE_GPU_UNPACK(uint8_t,int16_t,float);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_PROMOTE_GPU_UNPACK(uint8_t,int64_t,double);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_PROMOTE_GPU_UNPACK(uint8_t,int32_t,double);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/unpack.cpp:				CALL_FOREACH_PROMOTE_GPU_UNPACK(uint8_t,int16_t,double);
src/unpack.cpp:#if BF_CUDA_ENABLED
src/unpack.cpp:#undef CALL_FOREACH_SIMPLE_GPU_UNPACK
src/unpack.cpp:#undef CALL_FOREACH_PROMOTE_GPU_UNPACK
src/assert.hpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/int_fastdiv.h:#ifdef __CUDA_ARCH__
src/linalg_kernels.h:#include "cuda.hpp"
src/linalg_kernels.h:                cudaStream_t stream);
src/linalg_kernels.h:                        cudaStream_t stream);
src/fdmt.cu:#include "cuda.hpp"
src/fdmt.cu:#include <math_constants.h> // For CUDART_NAN_F
src/fdmt.cu:				OutType outval(CUDART_NAN_F);//std::numeric_limits<OutType>::quiet_NaN());
src/fdmt.cu:				//d_out[t + ostride_*r] = CUDART_NAN_F;
src/fdmt.cu:			//	d_out[t + ostride_*r] = CUDART_NAN_F;
src/fdmt.cu:				//	//d_out[t - (ntime-1) + ostride*r] = CUDART_NAN_F;
src/fdmt.cu:                             cudaStream_t   stream=0) {
src/fdmt.cu:	BF_CHECK_CUDA_EXCEPTION(
src/fdmt.cu:		cudaLaunchKernel((void*)fdmt_init_kernel<InType,OutType>,
src/fdmt.cu:                             cudaStream_t stream=0) {
src/fdmt.cu:	BF_CHECK_CUDA_EXCEPTION(
src/fdmt.cu:		cudaLaunchKernel((void*)fdmt_exec_kernel<DType>,
src/fdmt.cu:public: // HACK WAR for what looks like a bug in the CUDA 7.0 compiler
src/fdmt.cu:	cudaStream_t _stream;
src/fdmt.cu:	                _stream(g_cuda_stream) {}
src/fdmt.cu:		BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_d_offsets,
src/fdmt.cu:		                                         cudaMemcpyHostToDevice,
src/fdmt.cu:			BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_d_step_srcrows + step*_plan_stride,
src/fdmt.cu:			                                         cudaMemcpyHostToDevice,
src/fdmt.cu:			BF_CHECK_CUDA_EXCEPTION( cudaMemcpyAsync(_d_step_delays  + step*_plan_stride,
src/fdmt.cu:			                                         cudaMemcpyHostToDevice,
src/fdmt.cu:		BF_CHECK_CUDA_EXCEPTION( cudaStreamSynchronize(_stream),
src/fdmt.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/fdmt.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/fdmt.cu:		BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
src/fdmt.cu:	void set_stream(cudaStream_t stream) {
src/fdmt.cu:	BF_ASSERT(space_accessible_from(space, BF_SPACE_CUDA),
src/fdmt.cu:	BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
src/fdmt.cu:	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
src/fdmt.cu:	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
src/guantize.cu: * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
src/guantize.cu:#include "cuda.hpp"
src/guantize.cu:	return min_gpu(max_gpu(x,F(minval_gpu<I>())),F(maxval_gpu<I>()));
src/guantize.cu:	return min_gpu(max_gpu(x,F(-7)),F(7));
src/guantize.cu:	return min_gpu(max_gpu(x,F(-1)),F(1));
src/guantize.cu:			byteswap_gpu(ival, &ival);
src/guantize.cu:			byteswap_gpu(oval, &oval);
src/guantize.cu:void foreach_simple_gpu(T const* in,
src/guantize.cu:inline void launch_foreach_simple_gpu(T const*     in,
src/guantize.cu:                                      cudaStream_t stream=0) {
src/guantize.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu<T,U,Func,Size>,
src/guantize.cu:void foreach_simple_gpu_4bit(T const* in,
src/guantize.cu:			byteswap_gpu(tempR, &tempR);
src/guantize.cu:			byteswap_gpu(tempI, &tempI);
src/guantize.cu:			byteswap_gpu(tempO, &tempO);
src/guantize.cu:inline void launch_foreach_simple_gpu_4bit(T const*     in,
src/guantize.cu:                                           cudaStream_t stream=0) {
src/guantize.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu_4bit<T,Func,Size>,
src/guantize.cu:void foreach_simple_gpu_2bit(T const* in,
src/guantize.cu:			byteswap_gpu(tempA, &tempA);
src/guantize.cu:			byteswap_gpu(tempB, &tempB);
src/guantize.cu:			byteswap_gpu(tempC, &tempC);
src/guantize.cu:			byteswap_gpu(tempD, &tempD);
src/guantize.cu:			byteswap_gpu(tempO, &tempO);
src/guantize.cu:inline void launch_foreach_simple_gpu_2bit(T const*     in,
src/guantize.cu:                                           cudaStream_t stream=0) {
src/guantize.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu_2bit<T,Func,Size>,
src/guantize.cu:void foreach_simple_gpu_1bit(T const* in,
src/guantize.cu:			byteswap_gpu(tempA, &tempA);
src/guantize.cu:			byteswap_gpu(tempB, &tempB);
src/guantize.cu:			byteswap_gpu(tempC, &tempC);
src/guantize.cu:			byteswap_gpu(tempD, &tempD);
src/guantize.cu:			byteswap_gpu(tempE, &tempE);
src/guantize.cu:			byteswap_gpu(tempF, &tempF);
src/guantize.cu:			byteswap_gpu(tempG, &tempG);
src/guantize.cu:			byteswap_gpu(tempH, &tempH);
src/guantize.cu:			byteswap_gpu(tempO, &tempO);
src/guantize.cu:inline void launch_foreach_simple_gpu_1bit(T const*     in,
src/guantize.cu:                                           cudaStream_t stream=0) {
src/guantize.cu:	BF_CHECK_CUDA_EXCEPTION(cudaLaunchKernel((void*)foreach_simple_gpu_1bit<T,Func,Size>,
src/guantize.cu:// Instantiation - launch_foreach_simple_gpu_1bit calls used in quantize.cpp
src/guantize.cu:template void launch_foreach_simple_gpu_1bit<float,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu_1bit<float,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                 cudaStream_t stream);
src/guantize.cu:// Instantiation - launch_foreach_simple_gpu_2bit calls used in quantize.cpp
src/guantize.cu:template void launch_foreach_simple_gpu_2bit<float,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu_2bit<float,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                 cudaStream_t stream);
src/guantize.cu:// Instantiation - launch_foreach_simple_gpu_4bit calls used in quantize.cpp
src/guantize.cu:template void launch_foreach_simple_gpu_4bit<float,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu_4bit<float,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                 cudaStream_t stream);
src/guantize.cu:// Instantiation - launch_foreach_simple_gpu calls used in quantize.cpp
src/guantize.cu:template void launch_foreach_simple_gpu<float,uint8_t,GuantizeFunctor<float,float,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                   cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,uint8_t,GuantizeFunctor<float,double,uint8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                    cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,uint16_t,GuantizeFunctor<float,float,uint16_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                     cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,uint16_t,GuantizeFunctor<float,double,uint16_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                      cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,uint32_t,GuantizeFunctor<float,float,uint32_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                     cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,uint32_t,GuantizeFunctor<float,double,uint32_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                      cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,int8_t,GuantizeFunctor<float,float,int8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                 cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,int8_t,GuantizeFunctor<float,double,int8_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                  cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,int16_t,GuantizeFunctor<float,float,int16_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                   cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,int16_t,GuantizeFunctor<float,double,int16_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                    cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,int32_t,GuantizeFunctor<float,float,int32_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                   cudaStream_t stream);
src/guantize.cu:template void launch_foreach_simple_gpu<float,int32_t,GuantizeFunctor<float,double,int32_t>,size_t>(float const* in,
src/guantize.cu:                                                                                                    cudaStream_t stream);
src/linalg.cu:                CgemmEx (8bit, cuda >= 8.0, >=sm_50)
src/linalg.cu:                Cgemm3m (fp32, cuda >= 8.0, >=sm_50)
src/linalg.cu:                Cherk3mEx (8bit or fp32, cuda >= 8.0, >=sm_50)
src/linalg.cu:              Eventually it will probably be worth integrating the xGPU kernel,
src/linalg.cu:#include "cuda.hpp"
src/linalg.cu:#include "cuda/stream.hpp"
src/linalg.cu:                                  cudaStream_t stream,
src/linalg.cu:		if( get_cuda_device_cc() >= 50 ) {
src/linalg.cu:			                                CUDA_C_8I,
src/linalg.cu:			                                CUDA_C_32F,
src/linalg.cu:		if( get_cuda_device_cc() >= 50 ) {
src/linalg.cu:			                                CUDA_C_32F,
src/linalg.cu:			                                CUDA_C_32F,
src/linalg.cu:                          cudaStream_t stream,
src/linalg.cu:		cuda::child_stream child_stream(stream);
src/linalg.cu:		cuda::child_stream stream(g_cuda_stream);
src/linalg.cu:                                  cudaStream_t stream,
src/linalg.cu:		if( get_cuda_device_cc() >= 50 ) {
src/linalg.cu:			                              CUDA_C_8I,
src/linalg.cu:			                              CUDA_C_8I,
src/linalg.cu:			                              CUDA_C_32F,
src/linalg.cu:		if( get_cuda_device_cc() >= 50 ) {
src/linalg.cu:                                cudaStream_t stream,
src/linalg.cu:                                                           CUDA_C_8I, 
src/linalg.cu:                                                           CUDA_C_8I, 
src/linalg.cu:                                                           CUDA_C_32F, 
src/linalg.cu:                          cudaStream_t stream,
src/linalg.cu:					cuda::child_stream child_stream(stream);
src/linalg.cu:		cuda::child_stream stream(g_cuda_stream);
src/linalg.cu:	BF_ASSERT(space_accessible_from(c->space, BF_SPACE_CUDA),
src/linalg.cu:		BF_ASSERT(space_accessible_from(a->space, BF_SPACE_CUDA),
src/linalg.cu:		BF_ASSERT(space_accessible_from(b->space, BF_SPACE_CUDA),
src/linalg.cu:		BF_ASSERT(space_accessible_from(input->space, BF_SPACE_CUDA),
src/ShapeIndexer.cuh:#ifdef __CUDA_ARCH__
src/ShapeIndexer.cuh:#ifdef __CUDA_ARCH__
src/ShapeIndexer.cuh:#ifdef __CUDA_ARCH__
src/ArrayIndexer.cuh:#ifndef __CUDACC__
src/ArrayIndexer.cuh:#ifdef __CUDA_ARCH__
src/ArrayIndexer.cuh:#ifdef __CUDA_ARCH__
src/ArrayIndexer.cuh:#ifdef __CUDA_ARCH__
src/cuda.hpp:#if BF_CUDA_ENABLED
src/cuda.hpp://#define CUDA_API_PER_THREAD_DEFAULT_STREAM
src/cuda.hpp:#include <cuda_runtime_api.h>
src/cuda.hpp:#include <cuda.h>
src/cuda.hpp:extern thread_local cudaStream_t g_cuda_stream;
src/cuda.hpp:// TODO: BFstatus bfSetStream(void const* stream) { cuda_stream = *(cudaStream_t*)stream; }
src/cuda.hpp: // TODO: Also need thrust::cuda::par(allocator).on(stream)
src/cuda.hpp: #define thrust_cuda_par_on(stream) thrust::cuda::par.on(stream)
src/cuda.hpp: // TODO: Also need thrust::cuda::par(allocator, stream)
src/cuda.hpp: #define thrust_cuda_par_on(stream) thrust::cuda::par(stream)
src/cuda.hpp:inline int get_cuda_device_cc() {
src/cuda.hpp:	cudaGetDevice(&device);
src/cuda.hpp:	cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
src/cuda.hpp:	cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device);
src/cuda.hpp:  if( cc > BF_GPU_MAX_ARCH ) {
src/cuda.hpp:    cc = BF_GPU_MAX_ARCH;
src/cuda.hpp:#define BF_CHECK_CUDA_EXCEPTION(call, err) \
src/cuda.hpp:		cudaError_t cuda_ret = call; \
src/cuda.hpp:		if( cuda_ret != cudaSuccess ) { \
src/cuda.hpp:			BF_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
src/cuda.hpp:		/*BF_ASSERT(cuda_ret == cudaSuccess, err);*/ \
src/cuda.hpp:		BF_ASSERT_EXCEPTION(cuda_ret == cudaSuccess, err); \
src/cuda.hpp:#define BF_CHECK_CUDA(call, err) \
src/cuda.hpp:		cudaError_t cuda_ret = call; \
src/cuda.hpp:		if( cuda_ret != cudaSuccess ) { \
src/cuda.hpp:			BF_DEBUG_PRINT(cudaGetErrorString(cuda_ret)); \
src/cuda.hpp:		BF_ASSERT(cuda_ret == cudaSuccess, err); \
src/cuda.hpp:class CUDAKernel {
src/cuda.hpp:	inline void cuda_safe_call(CUresult res) {
src/cuda.hpp:		if( res != CUDA_SUCCESS ) {
src/cuda.hpp:		cuda_safe_call(cuModuleLoadDataEx(&_module, _ptx.c_str(),
src/cuda.hpp:		cuda_safe_call(cuModuleGetFunction(&_kernel, _module,
src/cuda.hpp:	inline CUDAKernel() : _module(0), _kernel(0) {}
src/cuda.hpp:	inline CUDAKernel(const CUDAKernel& other) : _module(0), _kernel(0) {
src/cuda.hpp:	inline CUDAKernel(const char*   func_name,
src/cuda.hpp:	inline CUDAKernel& set(const char*   func_name,
src/cuda.hpp:	inline void swap(CUDAKernel& other) {
src/cuda.hpp:	inline CUDAKernel& operator=(const CUDAKernel& other) {
src/cuda.hpp:		CUDAKernel tmp(other);
src/cuda.hpp:	inline ~CUDAKernel() {
src/cuda.hpp:#else // BF_CUDA_ENABLED
src/cuda.hpp:#endif // BF_CUDA_ENABLED
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:#include "cuda.hpp"
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_SYSTEM) || (space_accessible_from(in->space, BF_SPACE_CUDA) && space_accessible_from(out->space, BF_SPACE_CUDA)),
src/quantize.cpp:	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_SYSTEM) || (space_accessible_from(in->space, BF_SPACE_CUDA) && space_accessible_from(out->space, BF_SPACE_CUDA)),
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:	if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:#define CALL_FOREACH_SIMPLE_GPU_QUANTIZE(itype,stype,otype) \
src/quantize.cpp:	BF_TRACE_STREAM(g_cuda_stream); \
src/quantize.cpp:	launch_foreach_simple_gpu((itype*)in->data, \
src/quantize.cpp:	                          g_cuda_stream); \
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				BF_TRACE_STREAM(g_cuda_stream);
src/quantize.cpp:				launch_foreach_simple_gpu_1bit((float*)in->data, \
src/quantize.cpp:				                               g_cuda_stream);
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				BF_TRACE_STREAM(g_cuda_stream);
src/quantize.cpp:				launch_foreach_simple_gpu_2bit((float*)in->data, \
src/quantize.cpp:				                               g_cuda_stream);
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				BF_TRACE_STREAM(g_cuda_stream);
src/quantize.cpp:				launch_foreach_simple_gpu_4bit((float*)in->data, \
src/quantize.cpp:				                               g_cuda_stream);
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,float,int8_t);
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,float,int16_t);
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,double,int32_t);
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,float,uint16_t);
src/quantize.cpp:#if BF_CUDA_ENABLED
src/quantize.cpp:			if( space_accessible_from(in->space, BF_SPACE_CUDA) ) {
src/quantize.cpp:				CALL_FOREACH_SIMPLE_GPU_QUANTIZE(float,double,uint32_t);
src/quantize.cpp:#undef CALL_FOREACH_SIMPLE_GPU_QUANTIZE
src/reduce.cu:#include "cuda/stream.hpp"
src/reduce.cu:                                 cudaStream_t stream) {
src/reduce.cu:	BF_CHECK_CUDA_EXCEPTION(
src/reduce.cu:		cudaLaunchKernel((void*)reduce_vector_kernel<N, IType, OType>,
src/reduce.cu:                               cudaStream_t stream) {
src/reduce.cu:	BF_CHECK_CUDA_EXCEPTION(
src/reduce.cu:		cudaLaunchKernel((void*)reduce_loop_kernel<IType, OType>,
src/reduce.cu:			                               op, g_cuda_stream));
src/reduce.cu:			                               op, g_cuda_stream));
src/reduce.cu:			                               op, g_cuda_stream));
src/reduce.cu:			                                op, g_cuda_stream));
src/reduce.cu:			                          op, g_cuda_stream));
src/reduce.cu:                                                  cudaStream_t stream) {
src/reduce.cu:	BF_CHECK_CUDA_EXCEPTION(
src/reduce.cu:		cudaLaunchKernel((void*)reduce_complex_standard_vector_kernel<N, IType>,
src/reduce.cu:                                                cudaStream_t stream) {
src/reduce.cu:	BF_CHECK_CUDA_EXCEPTION(
src/reduce.cu:		cudaLaunchKernel((void*)reduce_complex_standard_loop_kernel<IType>,
src/reduce.cu:			                                                op, g_cuda_stream));
src/reduce.cu:			                                                op, g_cuda_stream));
src/reduce.cu:                                                             op, g_cuda_stream));
src/reduce.cu:			                                           op, g_cuda_stream));
src/reduce.cu:                                         cudaStream_t stream) {
src/reduce.cu:    BF_CHECK_CUDA_EXCEPTION(
src/reduce.cu:        cudaLaunchKernel((void*)reduce_complex_power_vector_kernel<N, IType>,
src/reduce.cu:                                       cudaStream_t stream) {
src/reduce.cu:    BF_CHECK_CUDA_EXCEPTION(
src/reduce.cu:        cudaLaunchKernel((void*)reduce_complex_power_loop_kernel<IType>,
src/reduce.cu:                                                         op, g_cuda_stream));
src/reduce.cu:                                                         op, g_cuda_stream));
src/reduce.cu:                                                          op, g_cuda_stream));
src/reduce.cu:                                                    op, g_cuda_stream));
src/reduce.cu:	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_CUDA),
src/utils.hu: * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
src/utils.hu:#ifndef BF_UNPACK_GPU_HU_INCLUDE_GUARD_
src/utils.hu:#define BF_UNPACK_GPU_HU_INCLUDE_GUARD_
src/utils.hu:inline __device__ void rshift_subwords_gpu(T& val) {
src/utils.hu:inline __device__ void conjugate_subwords_gpu(T& val) {
src/utils.hu:T min_gpu(T x, U y) {
src/utils.hu:T max_gpu(T x, U y) {
src/utils.hu:T maxval_gpu() {
src/utils.hu:int8_t maxval_gpu() {
src/utils.hu:int16_t maxval_gpu() {
src/utils.hu:int32_t maxval_gpu() {
src/utils.hu:int64_t maxval_gpu() {
src/utils.hu:uint8_t maxval_gpu() {
src/utils.hu:uint16_t maxval_gpu() {
src/utils.hu:uint32_t maxval_gpu() {
src/utils.hu:uint64_t maxval_gpu() {
src/utils.hu:T minval_gpu() {
src/utils.hu:	return -maxval_gpu<T>();
src/utils.hu:uint8_t minval_gpu() {
src/utils.hu:uint16_t minval_gpu() {
src/utils.hu:uint32_t minval_gpu() {
src/utils.hu:uint64_t minval_gpu() {
src/utils.hu:void byteswap_impl_gpu(uint64_t value, uint64_t* result) {
src/utils.hu:void byteswap_impl_gpu(uint32_t value, uint32_t* result) {
src/utils.hu:void byteswap_impl_gpu(uint16_t value, uint16_t* result) {
src/utils.hu:T type_pun_gpu(U x) {
src/utils.hu:void byteswap_gpu(T value, T* result) {
src/utils.hu:void byteswap_gpu(uint64_t value, uint64_t* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint64_t*)result);
src/utils.hu:void byteswap_gpu(int64_t value, int64_t* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint64_t*)result);
src/utils.hu:void byteswap_gpu(double value, double* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint64_t*)result);
src/utils.hu:void byteswap_gpu(uint32_t value, uint32_t* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint32_t*)result);
src/utils.hu:void byteswap_gpu(int32_t value, int32_t* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint32_t*)result);
src/utils.hu:void byteswap_gpu(float value, float* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint32_t*)result);
src/utils.hu:void byteswap_gpu(uint16_t value, uint16_t* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint16_t*)result);
src/utils.hu:void byteswap_gpu(int16_t value, int16_t* result) {
src/utils.hu:	return byteswap_impl_gpu(type_pun_gpu<uint64_t>(value), (uint16_t*)result);
src/utils.hu:#endif // BF_UNPACK_GPU_HU_INCLUDE_GUARD_
src/fft.cu:#include "cuda.hpp"
src/fft.cu:#if defined(BF_GPU_EXP_PINNED_ALLOC) && BF_GPU_EXP_PINNED_ALLOC
src/fft.cu:#include <thrust/system/cuda/experimental/pinned_allocator.h>
src/fft.cu:#include <thrust/system/cuda/memory.h>
src/fft.cu:#if defined(BF_GPU_EXP_PINNED_ALLOC) && BF_GPU_EXP_PINNED_ALLOC
src/fft.cu:	typedef thrust::cuda::experimental::pinned_allocator<CallbackData> pinned_allocator_type;
src/fft.cu:#if CUDA_VERSION >= 7500
src/fft.cu:#if CUDA_VERSION >= 7500
src/fft.cu:	//         We could potentially use a CUDA event as a lighter-weight
src/fft.cu:	cudaStreamSynchronize(g_cuda_stream);
src/fft.cu:	cudaMemcpyAsync(d_callback_data, h_callback_data, sizeof(CallbackData),
src/fft.cu:	                cudaMemcpyHostToDevice, g_cuda_stream);
src/fft.cu:	BF_TRACE_STREAM(g_cuda_stream);
src/fft.cu:	BF_ASSERT(space_accessible_from( in->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
src/fft.cu:	BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA), BF_STATUS_UNSUPPORTED_SPACE);
src/fft.cu:	cudaStream_t stream = g_cuda_stream;
src/gunpack.hu: * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
src/gunpack.hu:void launch_foreach_simple_gpu(T const*     in,
src/gunpack.hu:                               cudaStream_t stream=0);
src/gunpack.hu:void launch_foreach_promote_gpu(T const*     in,
src/gunpack.hu:                                cudaStream_t stream=0);
src/transpose.cu: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/transpose.cu:#if BF_CUDA_ENABLED
src/transpose.cu:  #include "transpose_gpu_kernel.cuh"
src/transpose.cu:  #include "cuda.hpp"
src/transpose.cu:  typedef int cudaStream_t; // WAR
src/transpose.cu:                   cudaStream_t  stream) {
src/transpose.cu:#if BF_CUDA_ENABLED
src/transpose.cu:#if BF_GPU_MIN_ARCH < 40
src/transpose.cu:		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/transpose.cu:		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
src/transpose.cu:	cudaError_t error = cudaGetLastError();
src/transpose.cu:	if( error != cudaSuccess ) {
src/transpose.cu:		std::printf("CUDA ERROR: %s\n", cudaGetErrorString(error));
src/transpose.cu:	BF_ASSERT(error == cudaSuccess, BF_STATUS_INTERNAL_ERROR);
src/transpose.cu:                   cudaStream_t  stream) {
src/transpose.cu:                   cudaStream_t   stream) {
src/transpose.cu:	BF_ASSERT(space_accessible_from(in->space, BF_SPACE_CUDA),
src/transpose.cu:	                                g_cuda_stream);
src/cuda/stream.hpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/cuda/stream.hpp: *  \brief Feature-complete RAII wrapper for CUDA stream objects
src/cuda/stream.hpp:          [1] http://on-demand.gputechconf.com/gtc/2015/presentation/S5530-Stephen-Jones.pdf
src/cuda/stream.hpp:              http://on-demand.gputechconf.com/gtc/2015/video/S5530.html
src/cuda/stream.hpp:  cuda::stream stream;
src/cuda/stream.hpp:  cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
src/cuda/stream.hpp:  CUDAScopedStream stream;
src/cuda/stream.hpp:  cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
src/cuda/stream.hpp:cuda::stream async_memcpy_cxx11(void* dst, const void* src, size_t size) {
src/cuda/stream.hpp:  cuda::stream stream(0, cudaStreamNonBlocking);
src/cuda/stream.hpp:  cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
src/cuda/stream.hpp:  cudaDeviceSynchronize();
src/cuda/stream.hpp:  cuda::stream s1(async_memcpy_cxx11(dst1, src2, size));
src/cuda/stream.hpp:  cuda::stream s2;
src/cuda/stream.hpp:#include <cuda_runtime_api.h>
src/cuda/stream.hpp:namespace cuda {
src/cuda/stream.hpp:inline void check_error(cudaError_t ret) {
src/cuda/stream.hpp:	if( ret != cudaSuccess ) {
src/cuda/stream.hpp:		throw ::std::runtime_error(cudaGetErrorString(ret));
src/cuda/stream.hpp:	cudaStream_t _obj;
src/cuda/stream.hpp:	stream(const cuda::stream& other) = delete;
src/cuda/stream.hpp:	stream& operator=(const cuda::stream& other) = delete;
src/cuda/stream.hpp:	stream(const cuda::stream& other);
src/cuda/stream.hpp:	stream& operator=(const cuda::stream& other);
src/cuda/stream.hpp:	void destroy() { if( _obj ) { cudaStreamDestroy(_obj); _obj = 0; } }
src/cuda/stream.hpp:	inline stream(cuda::stream&& other) : _obj(0) { this->swap(other); }
src/cuda/stream.hpp:	inline cuda::stream& operator=(cuda::stream&& other) {
src/cuda/stream.hpp:	                       unsigned flags=cudaStreamDefault) : _obj(0) {
src/cuda/stream.hpp:			cudaDeviceGetStreamPriorityRange(&least_priority,
src/cuda/stream.hpp:			check_error( cudaStreamCreateWithPriority(&_obj,
src/cuda/stream.hpp:			check_error( cudaStreamCreateWithFlags(&_obj, flags) );
src/cuda/stream.hpp:	inline void swap(cuda::stream& other) { std::swap(_obj, other._obj); }
src/cuda/stream.hpp:		check_error( cudaStreamGetPriority(_obj, &val) );
src/cuda/stream.hpp:		check_error( cudaStreamGetFlags(_obj, &val) );
src/cuda/stream.hpp:		cudaError_t ret = cudaStreamQuery(_obj);
src/cuda/stream.hpp:		if( ret == cudaErrorNotReady ) {
src/cuda/stream.hpp:		cudaStreamSynchronize(_obj);
src/cuda/stream.hpp:		check_error( cudaGetLastError() );
src/cuda/stream.hpp:	inline void wait(cudaEvent_t event, unsigned flags=0) const {
src/cuda/stream.hpp:		check_error( cudaStreamWaitEvent(_obj, event, flags) );
src/cuda/stream.hpp:	inline void addCallback(cudaStreamCallback_t callback,
src/cuda/stream.hpp:		check_error( cudaStreamAddCallback(_obj, callback, userData, flags) );
src/cuda/stream.hpp:		check_error( cudaStreamAttachMemAsync(_obj, devPtr, length, flags) );
src/cuda/stream.hpp:	inline operator const cudaStream_t&() const { return _obj; }
src/cuda/stream.hpp:class scoped_stream : public cuda::stream {
src/cuda/stream.hpp:	typedef cuda::stream super_type;
src/cuda/stream.hpp:	                              unsigned flags=cudaStreamNonBlocking)
src/cuda/stream.hpp:class child_stream : public cuda::stream {
src/cuda/stream.hpp:	typedef cuda::stream super_type;
src/cuda/stream.hpp:	cudaStream_t _parent;
src/cuda/stream.hpp:	void sync_streams(cudaStream_t dependent, cudaStream_t dependee) {
src/cuda/stream.hpp:		cudaEvent_t event;
src/cuda/stream.hpp:		check_error(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
src/cuda/stream.hpp:		check_error(cudaEventRecord(event, dependee));
src/cuda/stream.hpp:		check_error(cudaStreamWaitEvent(dependent, event, 0));
src/cuda/stream.hpp:		check_error(cudaEventDestroy(event));
src/cuda/stream.hpp:	inline explicit child_stream(cudaStream_t parent,
src/cuda/stream.hpp:	                             unsigned     flags=cudaStreamNonBlocking)
src/cuda/stream.hpp:} // namespace cuda
src/stringify.cpp: * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
src/stringify.cpp: * * Neither the name of NVIDIA CORPORATION nor the names of its
src/fft_kernels.h:#if CUDA_VERSION >= 7500
src/fft_kernels.h:#if CUDA_VERSION >= 7500

```
