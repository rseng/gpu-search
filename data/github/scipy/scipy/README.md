# https://github.com/scipy/scipy

```console
doc/source/dev/roadmap.rst:Support for distributed arrays and GPU arrays
doc/source/dev/roadmap.rst:to accept distributed arrays (e.g. ``dask.array.Array``) and GPU arrays (e.g.
doc/source/dev/roadmap.rst:or GPU arrays (see
doc/source/dev/api-dev/array_api.rst:Note that the above example works for PyTorch CPU tensors. For GPU tensors or
doc/source/dev/api-dev/array_api.rst:uses compiled code in its implementation, which won't work on GPU.
doc/source/dev/api-dev/array_api.rst:conversions should be zero-copy, while on GPU and other devices the attempt at
doc/source/dev/api-dev/array_api.rst:``"cpu", "cuda", "mps"``. To run the test suite with the PyTorch MPS
doc/source/release/1.12.0-notes.rst:  (GPU support is limited to functions with pure Python implementations).
doc/source/release/1.12.0-notes.rst:* `#19263 <https://github.com/scipy/scipy/pull/19263>`__: ENH: fft: GPU support for non-standard basic transforms
doc/source/release/1.12.0-notes.rst:* `#19601 <https://github.com/scipy/scipy/pull/19601>`__: ENH: Make special C++ implementations work on CUDA (and beyond!)
scipy/cluster/tests/test_vq.py:# Whole class skipped on GPU for now;
scipy/fft/_backend.py:        copying a NumPy array to the GPU for a CuPy backend. Implies ``only``.
scipy/linalg/_basic.py:       Gaussian Process Inference with GPU Acceleration" with contributions
scipy/special/xsf/config.h:#ifdef __CUDACC__
scipy/special/xsf/config.h:#include <cuda/std/cmath>
scipy/special/xsf/config.h:#include <cuda/std/cstddef>
scipy/special/xsf/config.h:#include <cuda/std/cstdint>
scipy/special/xsf/config.h:#include <cuda/std/limits>
scipy/special/xsf/config.h:#include <cuda/std/tuple>
scipy/special/xsf/config.h:#include <cuda/std/type_traits>
scipy/special/xsf/config.h:#include <cuda/std/utility>
scipy/special/xsf/config.h:#ifdef _LIBCUDACXX_COMPILER_NVRTC
scipy/special/xsf/config.h:#include <cuda_runtime.h>
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double abs(double num) { return cuda::std::abs(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double exp(double num) { return cuda::std::exp(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double log(double num) { return cuda::std::log(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double sqrt(double num) { return cuda::std::sqrt(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline bool isinf(double num) { return cuda::std::isinf(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline bool isnan(double num) { return cuda::std::isnan(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline bool isfinite(double num) { return cuda::std::isfinite(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double pow(double x, double y) { return cuda::std::pow(x, y); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double sin(double x) { return cuda::std::sin(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double cos(double x) { return cuda::std::cos(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double tan(double x) { return cuda::std::tan(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double atan(double x) { return cuda::std::atan(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double acos(double x) { return cuda::std::acos(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double sinh(double x) { return cuda::std::sinh(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double cosh(double x) { return cuda::std::cosh(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double asinh(double x) { return cuda::std::asinh(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline bool signbit(double x) { return cuda::std::signbit(x); }
scipy/special/xsf/config.h:#ifndef _LIBCUDACXX_COMPILER_NVRTC
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double ceil(double x) { return cuda::std::ceil(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double floor(double x) { return cuda::std::floor(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double round(double x) { return cuda::std::round(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double trunc(double x) { return cuda::std::trunc(x); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double fma(double x, double y, double z) { return cuda::std::fma(x, y, z); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double copysign(double x, double y) { return cuda::std::copysign(x, y); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double modf(double value, double *iptr) { return cuda::std::modf(value, iptr); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double fmax(double x, double y) { return cuda::std::fmax(x, y); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double fmin(double x, double y) { return cuda::std::fmin(x, y); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double log10(double num) { return cuda::std::log10(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double log1p(double num) { return cuda::std::log1p(num); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double frexp(double num, int *exp) { return cuda::std::frexp(num, exp); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double ldexp(double num, int exp) { return cuda::std::ldexp(num, exp); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double fmod(double x, double y) { return cuda::std::fmod(x, y); }
scipy/special/xsf/config.h:XSF_HOST_DEVICE inline double nextafter(double from, double to) { return cuda::std::nextafter(from, to); }
scipy/special/xsf/config.h:    cuda::std::swap(a, b);
scipy/special/xsf/config.h:using numeric_limits = cuda::std::numeric_limits<T>;
scipy/special/xsf/config.h:using is_floating_point = cuda::std::is_floating_point<T>;
scipy/special/xsf/config.h:using enable_if = cuda::std::enable_if<Cond, T>;
scipy/special/xsf/config.h:using decay = cuda::std::decay<T>;
scipy/special/xsf/config.h:using invoke_result = cuda::std::invoke_result<T>;
scipy/special/xsf/config.h:using pair = cuda::std::pair<T1, T2>;
scipy/special/xsf/config.h:using tuple = cuda::std::tuple<Types...>;
scipy/special/xsf/config.h:using cuda::std::ptrdiff_t;
scipy/special/xsf/config.h:using cuda::std::size_t;
scipy/special/xsf/config.h:using cuda::std::uint64_t;
scipy/special/xsf/digamma.h:             * and has been observed to vary between C++ stdlib and CUDA stdlib.
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#ifndef _MDSPAN_HAS_CUDA
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#  if defined(__CUDACC__)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#    define _MDSPAN_HAS_CUDA __CUDACC__
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#  if (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10 >= 1170)) && \
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_SYCL)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#  if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:// In CUDA defaulted functions do not need host device markup
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#if defined(_MDSPAN_HAS_CUDA) || defined(_MDSPAN_HAS_HIP)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:  /* Might need this on NVIDIA?
scipy/special/xsf/third_party/kokkos/mdspan.hpp:    #if !defined(_MDSPAN_HAS_HIP) && !defined(_MDSPAN_HAS_CUDA)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:// Depending on the CUDA and GCC version we need both the builtin
scipy/special/xsf/third_party/kokkos/mdspan.hpp:      #ifdef __CUDA_ARCH__
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:  // Even with CUDA_ARCH protection this thing warns about calling host function
scipy/special/xsf/third_party/kokkos/mdspan.hpp:      #ifdef __CUDA_ARCH__
scipy/special/xsf/third_party/kokkos/mdspan.hpp:// Depending on the CUDA and GCC version we need both the builtin
scipy/special/xsf/third_party/kokkos/mdspan.hpp:      #ifdef __CUDA_ARCH__
scipy/special/xsf/third_party/kokkos/mdspan.hpp:    // But Clang-CUDA also doesn't accept the use of deduction guide so disable it for CUDA alltogether
scipy/special/xsf/third_party/kokkos/mdspan.hpp:    #if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:      #ifdef __CUDA_ARCH__
scipy/special/xsf/third_party/kokkos/mdspan.hpp:// Depending on the CUDA and GCC version we need both the builtin
scipy/special/xsf/third_party/kokkos/mdspan.hpp:      #ifdef __CUDA_ARCH__
scipy/special/xsf/third_party/kokkos/mdspan.hpp:    // But Clang-CUDA also doesn't accept the use of deduction guide so disable it for CUDA alltogether
scipy/special/xsf/third_party/kokkos/mdspan.hpp:    #if defined(_MDSPAN_HAS_HIP) || defined(_MDSPAN_HAS_CUDA)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:#if defined(__NVCC__) && !defined(__CUDA_ARCH__) && defined(__GNUC__)
scipy/special/xsf/third_party/kokkos/mdspan.hpp:      #ifdef __CUDA_ARCH__
scipy/special/xsf/third_party/kokkos/mdspan.hpp:    #if defined(_MDSPAN_HAS_HIP) || (defined(__NVCC__) && (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10) < 1120)
scipy/special/tests/meson.build:  'test_xsf_cuda.py',
scipy/conftest.py:    SCIPY_DEVICE = 'cuda'
scipy/_lib/_array_api.py:        num_cuda = torch.cuda.device_count()
scipy/_lib/_array_api.py:        for i in range(0, num_cuda):
scipy/_lib/_array_api.py:            devices += [f'cuda:{i}']
scipy/_lib/_array_api.py:        num_cuda = cupy.cuda.runtime.getDeviceCount()
scipy/_lib/_array_api.py:        for i in range(0, num_cuda):
scipy/_lib/_array_api.py:            devices += [f'cuda:{i}']
scipy/_lib/_array_api.py:        num_gpu = jax.device_count(backend='gpu')
scipy/_lib/_array_api.py:        for i in range(0, num_gpu):
scipy/_lib/_array_api.py:            devices += [f'gpu:{i}']

```
