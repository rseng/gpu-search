# https://github.com/astro-informatics/s2fft

```console
docs/user_guide/install.rst:Installing JAX for NVIDIA GPUs
docs/user_guide/install.rst:however to get things running on GPUs can be a bit more involved. We strongly recommend 
docs/user_guide/install.rst:Google. To summarise you will first need to install NVIDIA drivers for 
docs/user_guide/install.rst:`CUDA <https://developer.nvidia.com/cuda-downloads>`_ and `CuDNN <https://developer.nvidia.com/CUDNN>`_, 
docs/user_guide/install.rst:following which a pre-built CUDA-compatible wheels shoulld be installed by running 
docs/user_guide/install.rst:    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
docs/user_guide/install.rst:    pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
docs/user_guide/install.rst:where the versions of CUDA and CuDNN should match those you have installed on the machine.
docs/index.rst:deployable on modern hardware accelerators (e.g. GPUs and TPUs).
docs/index.rst:    HEALPix long JIT compile time fixed for CPU!  Fix for GPU coming soon.
docs/index.rst:GPUs and TPUs).  In particular, these algorithms are based on new Wigner-d recursions
docs/index.rst:    For algorithmic reasons JIT compilation of HEALPix transforms can become slow at high bandlimits, due to XLA unfolding of loops which currently cannot be avoided. After compiling HEALPix transforms should execute with the efficiency outlined in the associated paper, therefore this additional time overhead need only be incurred once. We are aware of this issue and are working to fix it.  A fix for CPU execution has now been implemented (see example `notebook <https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/JAX_HEALPix_backend.html>`_).  Fix for GPU execution is coming soon.
.coveragerc:    ; Exclude functions requiring CUDA from coverage
.coveragerc:    def \w+_cuda(?:|_\w+)\(
tests/test_spherical_transform.py:multiple_gpus = [False, True]
tests/test_spherical_transform.py:@pytest.mark.parametrize("spmd", multiple_gpus)
tests/test_spherical_transform.py:        pytest.skip("GPU distribution only valid for JAX.")
tests/test_spherical_transform.py:@pytest.mark.parametrize("spmd", multiple_gpus)
tests/test_spherical_transform.py:@pytest.mark.parametrize("spmd", multiple_gpus)
tests/test_spherical_transform.py:        pytest.skip("GPU distribution only valid for JAX.")
tests/test_spherical_transform.py:@pytest.mark.parametrize("spmd", multiple_gpus)
tests/test_healpix_ffts.py:    healpix_fft_cuda,
tests/test_healpix_ffts.py:    healpix_ifft_cuda,
tests/test_healpix_ffts.py:gpu_available = get_backend().platform == "gpu"
tests/test_healpix_ffts.py:@pytest.mark.skipif(not gpu_available, reason="GPU not available")
tests/test_healpix_ffts.py:def test_healpix_fft_cuda(flm_generator, nside):
tests/test_healpix_ffts.py:        healpix_fft_cuda(f, L, nside, False),
tests/test_healpix_ffts.py:@pytest.mark.skipif(not gpu_available, reason="GPU not available")
tests/test_healpix_ffts.py:def test_healpix_ifft_cuda(flm_generator, nside):
tests/test_healpix_ffts.py:        healpix_ifft_cuda(ftm, L, nside, False).flatten(),
README.md:also deployable on hardware accelerators (e.g. GPUs and TPUs).
README.md:> HEALPix long JIT compile time fixed for CPU!  Fix for GPU coming soon.
README.md:of hardware accelerators (i.e. GPUs and TPUs). In particular, these
README.md:> For algorithmic reasons JIT compilation of HEALPix transforms can become slow at high bandlimits, due to XLA unfolding of loops which currently cannot be avoided. After compiling HEALPix transforms should execute with the efficiency outlined in the associated paper, therefore this additional time overhead need only be incurred once. We are aware of this issue and are working to fix it.  A fix for CPU execution has now been implemented (see example [notebook](https://astro-informatics.github.io/s2fft/tutorials/spherical_harmonic/JAX_HEALPix_backend.html)).  Fix for GPU execution is coming soon.
README.md:precompute (right) algorithms, with `S2FFT` running on GPUs (for further
CMakeLists.txt:set(CMAKE_CUDA_STANDARD 17)
CMakeLists.txt:# Check for CUDA
CMakeLists.txt:check_language(CUDA)
CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
CMakeLists.txt:  enable_language(CUDA)
CMakeLists.txt:  find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:  message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
CMakeLists.txt:    find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:    target_link_libraries(_s2fft PRIVATE CUDA::cudart_static CUDA::cufft_static CUDA::culibos)
CMakeLists.txt:                        LINKER_LANGUAGE CUDA
CMakeLists.txt:                        CUDA_SEPARABLE_COMPILATION ON)
CMakeLists.txt:    set(CMAKE_CUDA_ARCHITECTURES "70;80;89" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")
CMakeLists.txt:    message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
CMakeLists.txt:    set_target_properties(_s2fft PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
CMakeLists.txt:    message(WARNING "CUDA compiler not found, building without CUDA support")
CMakeLists.txt:    target_compile_definitions(_s2fft PRIVATE NO_CUDA_COMPILER)
benchmarks/wigner.py:        skip("GPU distribution only valid for JAX.")
benchmarks/wigner.py:        skip("GPU distribution only valid for JAX.")
benchmarks/spherical.py:        skip("GPU distribution only valid for JAX.")
benchmarks/spherical.py:        skip("GPU distribution only valid for JAX.")
.pip_readme.rst:(e.g. GPUs and TPUs).
lib/include/s2fft_kernels.h:#include <cmath>  // has to be included before cuda/std/complex
lib/include/s2fft_kernels.h:#include <cuda/std/complex>
lib/include/s2fft_kernels.h:                                const bool& shift, cudaStream_t stream);
lib/include/s2fft_kernels.h:                                  cudaStream_t stream);
lib/include/s2fft_callbacks.h:#include <cuda_runtime.h>
lib/include/hresult.h:// Macro to check for CUDA errors
lib/include/hresult.h:#define checkCudaErrors(call)                                                                 \
lib/include/hresult.h:        cudaError_t err = call;                                                               \
lib/include/hresult.h:        if (err != cudaSuccess) {                                                             \
lib/include/hresult.h:            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
lib/include/s2fft.h:#include <cmath>  // has to be included before cuda/std/complex
lib/include/s2fft.h:#include <cuda/std/complex>
lib/include/s2fft.h:    HRESULT Forward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data);
lib/include/s2fft.h:    HRESULT Backward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data);
lib/include/kernel_helpers.h:// "opaque" parameter of the GPU custom call. In our example we'll use this
lib/include/plan_cache.h:#include <cuda/std/complex>
lib/src/s2fft_kernels.cu:#include <cmath>  // has to be included before cuda/std/complex
lib/src/s2fft_kernels.cu:#include <cuda/std/complex>
lib/src/s2fft_kernels.cu:                                const bool& shift, cudaStream_t stream) {
lib/src/s2fft_kernels.cu:    checkCudaErrors(cudaGetLastError());
lib/src/s2fft_kernels.cu:                                  cudaStream_t stream) {
lib/src/s2fft_kernels.cu:    checkCudaErrors(cudaGetLastError());
lib/src/s2fft_kernels.cu:                                                       cudaStream_t stream);
lib/src/s2fft_kernels.cu:                                                             cudaStream_t stream);
lib/src/s2fft_kernels.cu:                                                         const int& nside, const int& L, cudaStream_t stream);
lib/src/s2fft_kernels.cu:                                                               const int& L, cudaStream_t stream);
lib/src/s2fft_callbacks.cu:#include <cuda_runtime.h>
lib/src/s2fft.cu:#include <cuda_runtime.h>
lib/src/s2fft.cu:#include <cmath>  // has to be included before cuda/std/complex
lib/src/s2fft.cu:#include <cuda/std/complex>
lib/src/s2fft.cu:        cudaMalloc(&params_dev, 2 * sizeof(int64));
lib/src/s2fft.cu:        cudaMemcpy(params_dev, params, 2 * sizeof(int64), cudaMemcpyHostToDevice);
lib/src/s2fft.cu:    cudaMalloc(&equator_params_dev, sizeof(int64));
lib/src/s2fft.cu:    cudaMemcpy(equator_params_dev, equator_params, sizeof(int64), cudaMemcpyHostToDevice);
lib/src/s2fft.cu:HRESULT s2fftExec<Complex>::Forward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data) {
lib/src/s2fft.cu:HRESULT s2fftExec<Complex>::Backward(const s2fftDescriptor &desc, cudaStream_t stream, Complex *data) {
lib/src/extensions.cc:#ifndef NO_CUDA_COMPILER
lib/src/extensions.cc:#include "cuda_runtime.h"
lib/src/extensions.cc:    throw std::runtime_error("This extension was compiled without CUDA support. Cuda functions are not supported.");
lib/src/extensions.cc:#ifdef NO_CUDA_COMPILER
lib/src/extensions.cc:void healpix_fft_cuda() { print_error(); }
lib/src/extensions.cc:void healpix_forward(cudaStream_t stream, void** buffers, s2fftDescriptor descriptor) {
lib/src/extensions.cc:void healpix_backward(cudaStream_t stream, void** buffers, s2fftDescriptor descriptor) {
lib/src/extensions.cc:void healpix_fft_cuda(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
lib/src/extensions.cc:#endif  // NO_CUDA_COMPILER
lib/src/extensions.cc:    dict["healpix_fft_cuda"] = EncapsulateFunction(healpix_fft_cuda);
lib/src/extensions.cc:#ifndef NO_CUDA_COMPILER
.gitignore:#c++/cuda/cmake
s2fft/transforms/otf_recursions.py:    simultaneously; facilitating GPU/TPU acceleration.
s2fft/transforms/otf_recursions.py:    simultaneously; facilitating GPU/TPU acceleration.
s2fft/transforms/otf_recursions.py:    simultaneously; facilitating GPU/TPU acceleration.
s2fft/transforms/otf_recursions.py:    simultaneously; facilitating GPU/TPU acceleration.
s2fft/transforms/spherical.py:    elif method == "cuda":
s2fft/transforms/spherical.py:        use_healpix_custom_primitive (bool, optional): Whether to use a custom CUDA
s2fft/transforms/spherical.py:            "healpix"` and running on a cuda compatible gpu device. using a custom
s2fft/transforms/spherical.py:            ftm = hp.healpix_fft(f, L, nside, "cuda", reality)
s2fft/utils/healpix_ffts.py:        method (str, optional): Evaluation method in {"numpy", "jax", "torch", "cuda"}.
s2fft/utils/healpix_ffts.py:        ValueError: Deployment method not in {"numpy", "jax", "torch", "cuda"}.
s2fft/utils/healpix_ffts.py:    elif method.lower() == "cuda":
s2fft/utils/healpix_ffts.py:        return healpix_fft_cuda(f, L, nside, reality)
s2fft/utils/healpix_ffts.py:    elif method.lower() == "cuda":
s2fft/utils/healpix_ffts.py:        return healpix_ifft_cuda(ftm, L, nside, reality)
s2fft/utils/healpix_ffts.py:# Custom healpix_fft_cuda primitive
s2fft/utils/healpix_ffts.py:def _healpix_fft_cuda_abstract(f, L, nside, reality, fft_type, norm):
s2fft/utils/healpix_ffts.py:def _healpix_fft_cuda_lowering(ctx, f, *, L, nside, reality, fft_type, norm):
s2fft/utils/healpix_ffts.py:        "healpix_fft_cuda",
s2fft/utils/healpix_ffts.py:# Register healpfix_fft_cuda custom call target
s2fft/utils/healpix_ffts.py:    xla_client.register_custom_call_target(name, fn, platform="gpu")
s2fft/utils/healpix_ffts.py:_healpix_fft_cuda_primitive = register_primitive(
s2fft/utils/healpix_ffts.py:    "healpix_fft_cuda",
s2fft/utils/healpix_ffts.py:    abstract_evaluation=_healpix_fft_cuda_abstract,
s2fft/utils/healpix_ffts.py:    lowering_per_platform={None: _healpix_fft_cuda_lowering},
s2fft/utils/healpix_ffts.py:def healpix_fft_cuda(
s2fft/utils/healpix_ffts.py:    Healpix FFT JAX implementation using custom CUDA primitive.
s2fft/utils/healpix_ffts.py:    return _healpix_fft_cuda_primitive.bind(
s2fft/utils/healpix_ffts.py:def healpix_ifft_cuda(
s2fft/utils/healpix_ffts.py:    Healpix IFFT JAX implementation using custom CUDA primitive.
s2fft/utils/healpix_ffts.py:    return _healpix_fft_cuda_primitive.bind(

```
