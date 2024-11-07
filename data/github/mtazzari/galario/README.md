# https://github.com/mtazzari/galario

```console
python/libcommon.pyx:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/libcommon.pyx:           'ngpus', 'use_gpu', 'threads',
python/libcommon.pyx:#                            GPU HELPER FUNCTIONS                              #
python/libcommon.pyx:def ngpus():
python/libcommon.pyx:    Return how many GPUs are available on the machine.
python/libcommon.pyx:    ngpus : int
python/libcommon.pyx:        Number of GPUs available on the machine.
python/libcommon.pyx:    return cpp.ngpus()
python/libcommon.pyx:def use_gpu(int device_id):
python/libcommon.pyx:    Select the GPU to be used for the computation.
python/libcommon.pyx:        use_gpu(device_id)
python/libcommon.pyx:        ID of the GPU to be used for the computation.
python/libcommon.pyx:    If more than one GPU is present, `device_id` might not coincide with the `ID`
python/libcommon.pyx:    reported by the `nvidia-smi` command, which reflects the PCI order.
python/libcommon.pyx:    we recommend to start from `device_id=0` and simultaneously check which GPU is used with
python/libcommon.pyx:    `watch -n0.1 nvidia-smi`.
python/libcommon.pyx:    cpp.use_gpu(device_id)
python/libcommon.pyx:        On the *GPU*, `num` is the square root of the number of threads per block to be used.
python/libcommon.pyx:    The CUDA documentation suggests starting with `num*num`>=64 and multiples of 32,
python/libcommon.pyx:    e.g. 128, 256. GPU cards with compute capability between 2 and 6.2 have
python/libcommon.pyx:    Check the maximum number of threads per block of your GPU by running
python/wrap_lib.cmake:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/wrap_lib.cmake:# Define cython wrappers for double and cuda libs
python/wrap_lib.cmake:  set(options DOUBLE CUDA)
python/wrap_lib.cmake:  if(WRAP_LIB_CUDA)
python/wrap_lib.cmake:    set(suffix "${suffix}_cuda")
python/wrap_lib.cmake:    set(outdir "${outdir}_cuda")
python/__init__.py.in:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/__init__.py.in:HAVE_CUDA = ("${CUDA_FOUND}" == "TRUE")
python/__init__.py.in:if HAVE_CUDA:
python/__init__.py.in:    from . import single_cuda
python/__init__.py.in:    from . import double_cuda
python/speed_baseline.sh:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/speed_baseline.sh:    $cmd --size=$s --gpu --tpb ${threads_per_block} --ompnthreads ${openmp_threads} >> $output1 2>&1;
python/speed_baseline.sh:    $cmd --size=$s --gpu --tpb ${threads_per_block} --ompnthreads ${openmp_threads} >> $output2 2>&1;
python/galario_defs.pxd:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/galario_defs.pxd:    void use_gpu(int device_id) except +
python/galario_defs.pxd:    int  ngpus() except +
python/__init_module__.py.in:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/CMakeLists.txt:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/CMakeLists.txt:if (CUDA_FOUND)
python/CMakeLists.txt:  wrap_lib(CUDA)
python/CMakeLists.txt:  wrap_lib(DOUBLE CUDA)
python/galario_config.pxi.in:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/utils.py:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/test_galario.py:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/test_galario.py:if galario.HAVE_CUDA and int(environ.get("GALARIO_TEST_GPU", 0)):
python/test_galario.py:    from galario import double_cuda as g_double
python/test_galario.py:    from galario import single_cuda as g_single
python/test_galario.py:# use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
python/test_galario.py:ngpus = g_double.ngpus()
python/test_galario.py:g_double.use_gpu(0) #max(0, ngpus-1))
python/test_galario.py:    # CPU/GPU version (galario)
python/test_galario.py:                         [(1000, 'float32', 1.e-2, g_single), # rtol increased from 1e-4 to pass GPU test
python/conftest.py:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/conftest.py:    parser.addoption("--gpu", action="store", default=0,
python/conftest.py:        help="Run tests on gpu. Default: 0")
python/speed_benchmark.py:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
python/speed_benchmark.py:p.add_argument("--gpu", action="store_true", dest="USE_GPU", default=False,
python/speed_benchmark.py:             help="Use GPU version of galario")
python/speed_benchmark.py:p.add_argument("--gpu_id", action="store", dest="gpu_id", default=0, type=int,
python/speed_benchmark.py:             help="Choose index of GPU if several are available.  Check `watch -n 0.1 nvidia-smi` to see which gpu is used during test execution")
python/speed_benchmark.py:p.add_argument("--tpb", type=int, nargs='+', help="Threads per block on the GPU", default=[16])
python/speed_benchmark.py:if options.USE_GPU:
python/speed_benchmark.py:    if galario.HAVE_CUDA:
python/speed_benchmark.py:        from galario import double_cuda as acc_lib_cuda
python/speed_benchmark.py:        acc_lib_cuda.use_gpu(options.gpu_id)
python/speed_benchmark.py:        print("Option --gpu not valid. galario.HAVE_CUDA is {}.".format(galario.HAVE_CUDA))
python/speed_benchmark.py:        options.USE_GPU = False
python/speed_benchmark.py:def do_timing(options, input_data, gpu=False, tpb=0, omp_num_threads=0):
python/speed_benchmark.py:    if gpu:
python/speed_benchmark.py:        acc_lib_cuda.threads(tpb)
python/speed_benchmark.py:    acc_lib = 'acc_lib_cuda' if gpu else 'acc_lib_cpu'
python/speed_benchmark.py:        if gpu:
python/speed_benchmark.py:            filename += "GPU_{}".format(tpb)
python/speed_benchmark.py:    if options.USE_GPU:
python/speed_benchmark.py:            do_timing(options, input_data, tpb=t, gpu=True)
docs/py-api.rst:GPU related
docs/py-api.rst:.. py:data:: galario.HAVE_CUDA
docs/py-api.rst:    Global variable (`bool`). It is `True` if the GPU libraries
docs/py-api.rst:    (`galario.double_cuda` and `galario.single_cuda`) are available, `False`
docs/py-api.rst:    otherwise. On a machine without a CUDA-enabled GPU it is always `False`.
docs/py-api.rst:.. autofunction:: galario.double.ngpus
docs/py-api.rst:.. autofunction:: galario.double.use_gpu
docs/py-api.rst:Out of memory on GPU                              `std::bad_alloc`         `MemoryError`
docs/py-api.rst:Invalid argument (CPU/GPU)                        `std::invalid_argument`  `ValueError`
docs/py-api.rst:Miscellaneous, including out of memory (CPU/GPU)  `std::runtime_error`     `RuntimeError`
docs/basic_usage.rst:    If you want to compute the visibilities of a model :code:`image` (Jy/px) with pixel size `dxy` (rad) in the same :math:`(u_j, v_j)` locations of the observations, you can easily do it with the GPU accelerated |galario|:
docs/basic_usage.rst:        from galario.double_cuda import sampleImage
docs/basic_usage.rst:        from galario.double_cuda import sampleProfile
docs/basic_usage.rst:        from galario.double_cuda import chi2Image
docs/basic_usage.rst:        from galario.double_cuda import sampleImage
docs/basic_usage.rst:    If you work on a machine **without** a CUDA-enabled GPU, don't worry: you can use the CPU version
docs/basic_usage.rst:    of |galario| by just removing the subscript `"_cuda"` from the imports above and benefit from the openMP parallelization.
docs/basic_usage.rst:    All the function names and interfaces are the same for GPU and CPU version!
docs/cookbook.rst:.. _cookbook_GPU_CPU_version:
docs/cookbook.rst:Using the GPU and CPU version
docs/cookbook.rst:The CPU version of |galario| is always compiled, even on a system without a CUDA-enabled GPU. In this case you can import
docs/cookbook.rst:If built on a machine with a CUDA-enabled GPU, |galario| is compiled also for the GPU. You can still import
docs/cookbook.rst:the CPU version as above, and the GPU version as follows:
docs/cookbook.rst:    from galario import double_cuda
docs/cookbook.rst:    from galario import single_cuda
docs/cookbook.rst:To check programmatically whether the GPU version is available, you can read the global variable :data:`galario.HAVE_CUDA`.
docs/cookbook.rst:The following snippet imports the GPU version of galario if it is available, otherwise it imports the CPU version:
docs/cookbook.rst:    if galario.HAVE_CUDA:
docs/cookbook.rst:        from galario import double_cuda as g_double
docs/cookbook.rst:        from galario import single_cuda as g_single
docs/cookbook.rst:This snippet simplifies the development of portable code. Since the functions in `double`, `double_cuda`, `single` and `single_cuda`
docs/cookbook.rst:without a GPU, then move to a machine with a GPU and run it on the GPU without any change.
docs/cookbook.rst:Selecting the GPU
docs/cookbook.rst:|galario| can be used on machines with one or more CUDA-capable GPUs. The number of GPUs available on the machine can be
docs/cookbook.rst:obtained with the :func:`ngpus() <galario.double.ngpus>` function:
docs/cookbook.rst:    double_cuda.ngpus()   # or single_cuda.ngpus()
docs/cookbook.rst:It is possible to tell |galario| to use a particular GPU for the computation the :func:`use_gpu() <galario.double.use_gpu>` function:
docs/cookbook.rst:    double_cuda.use_gpu(ID)
docs/cookbook.rst:where `ID` is an integer number representing the GPU ID. By default, |galario| uses the GPU with `ID=0`. This means that on machines
docs/cookbook.rst:with only one CUDA-capable GPU it is not necessary to call `double_cuda.use_GPU(0)` as this is the default behaviour.
docs/cookbook.rst:    The `ID` to be used in :func:`use_gpu() <galario.double.use_gpu>` might differ from the device ID reported by the `nvidia-smi` command.
docs/cookbook.rst:    See the documentation of :func:`use_gpu() <galario.double.use_gpu>` for more details.
docs/cookbook.rst:On the GPU
docs/cookbook.rst:It is possible to change the number of threads per block used to launch 1D and 2D kernels on the GPU with:
docs/cookbook.rst:    double_cuda.threads(N)
docs/cookbook.rst:256 threads per block. Due to the physical structure of the current NVIDIA cards, `N` must be equal to 8, 16 or 32.
docs/index.rst:**GPU Accelerated Library for Analysing Radio Interferometer Observations**
docs/index.rst:|galario| is a library that exploits the computing power of modern graphic cards (GPUs) to accelerate the comparison of model
docs/index.rst:Along with the GPU accelerated version based on the
docs/index.rst:`CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_, |galario| offers a CPU counterpart accelerated with
docs/index.rst:    Due to technical limitations, the conda package does not support GPUs at the
docs/index.rst:    moment. If you want to use a GPU, you have to build |galario| by hand.
docs/index.rst:Useful recipes for the CPU/GPU management and the model image creation: see the :doc:`Cookbook <cookbook>` with many code snippets.
docs/index.rst:        title = "{GALARIO: a GPU accelerated library for analysing radio interferometer observations}",
docs/FAQ.rst:    ----> 1 from galario.double_cuda import sampleImage
docs/FAQ.rst:    /Users/tdavis/anaconda/lib/python3.5/site-packages/galario/double_cuda/__init__.py in <module>()
docs/FAQ.rst:    ImportError: No module named 'galario.double_cuda.libcommon'
docs/FAQ.rst:    How can I use more than one GPU?
docs/install.rst:Due to technical limitations, the conda package does not support GPUs at the
docs/install.rst:moment. If you want to use a GPU, read on as you have to build |galario| by hand.
docs/install.rst:* [optional] the `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_ >=8.0 for the GPU version: it can be easily installed from the `NVIDIA website <https://developer.nvidia.com/cuda-toolkit>`_
docs/install.rst:* [optional] Python and numpy for Python bindings to the CPU and GPU
docs/install.rst:On a system with a CUDA-enabled GPU card, also the GPU version will be compiled and installed.
docs/install.rst:To manually turn ON/OFF the GPU CUDA compilation, see :ref:`these instructions <build_details_cuda>` below.
docs/install.rst:    compile also the GPU version, check in the |NVIDIA_docs| which gcc/g++
docs/install.rst:    versions are compatible with the `nvcc` compiler shipped with your CUDA
docs/install.rst:.. _build_details_cuda:
docs/install.rst:CUDA
docs/install.rst:`cmake` tests for compilation on the GPU with cuda by default **except on Mac
docs/install.rst:OS**, where version conflicts between the NVIDIA compiler and the C++ compiler
docs/install.rst:To manually enable or disable checking for cuda, do
docs/install.rst:   cmake -DGALARIO_CHECK_CUDA=0 .. # don't check
docs/install.rst:   cmake -DGALARIO_CHECK_CUDA=1 .. # check
docs/install.rst:If cuda is installed in a non-standard directory or you want to specify the
docs/install.rst:   cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.1 ..
docs/install.rst:By default, tests do not run on the GPU. Activate them by setting an environment variable `GALARIO_TEST_GPU`; e.g. `GALARIO_TEST_GPU=1 py.test.sh ...`.
docs/install.rst:A cuda error such as
docs/install.rst:    [ERROR] Cuda call /home/user/workspace/galario/build/src/cuda_lib.cu: 815
docs/install.rst:can mean that code cannot be executed on the GPU at all rather than that specific call being invalid.
docs/install.rst:Check if `nvidia-smi` fails
docs/install.rst:    $ nvidia-smi
docs/install.rst:.. |NVIDIA_docs| raw:: html
docs/install.rst:   <a href="http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements" target="_blank">NVIDIA Docs</a>
docs/C++-example.rst:The core of |galario| is written in C++/Cuda, all functions are in the header `galario.h`.
docs/C++-example.rst:If |galario| was installed with `cuda` support, you can link in `-lgalario_cuda` or `-lgalario_single_cuda` instead.
docs/C++-example.rst:The actual FFT is done in-place, and the result is stored in `res`. The data layout is described in the `FFTW manual <http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data>`_ and applied to the CPU and GPU versions of |galario|::
docs/C++-api.rst:   For the cuda version, this sets the number of threads per block in cuda kernels.
docs/C++-api.rst:   int ngpus()
docs/C++-api.rst:   Get the number of available GPUs.
docs/C++-api.rst:    void use_gpu(int device_id)
docs/C++-api.rst:    Set the GPU to be used for the computations.
docs/C++-api.rst:    For details, see the python function :py:func:`use_gpu`.
docs/uvtable.txt:# Galario - GPU-accelerated library for the analysis of radio interferometry observations.
docs/quickstart.rst:**7) CPU vs GPU execution**
docs/quickstart.rst:    So far we have run |galario| on the CPU. Running it on a GPU can be done by just changing the import at the beginning:
docs/quickstart.rst:        from galario import double_cuda as g_double
docs/quickstart.rst:    For more details on the GPU vs CPU execution, see the :ref:`Cookbook <cookbook>`.
README.md:**Gpu Accelerated Library for Analysing Radio Interferometer Observations**
README.md:**galario** is a library that exploits the computing power of modern graphic cards (GPUs) to accelerate the comparison of model
README.md:    title = "{GALARIO: a GPU accelerated library for analysing radio interferometer observations}",
CMakeLists.txt:# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
CMakeLists.txt:if(NOT DEFINED GALARIO_CHECK_CUDA)
CMakeLists.txt:    message(STATUS "Ignoring cuda on the mac by default. Force searching for cuda with `cmake -DGALARIO_CHECK_CUDA=1`")
CMakeLists.txt:    set(GALARIO_CHECK_CUDA 0)
CMakeLists.txt:    set(GALARIO_CHECK_CUDA 1)
CMakeLists.txt:if(GALARIO_CHECK_CUDA)
CMakeLists.txt:  find_package(CUDA)
CMakeLists.txt:  #enable_language(CUDA)
AUTHORS.rst:The initial development of |galario| was boosted by the GPU Hackathon at the TU Dresden in February 2016 thanks to two
src/galario_py.h:* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
src/galario.cpp:* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#include <cuda_runtime_api.h>
src/galario.cpp:#include <cuda.h>
src/galario.cpp:// general min function already available in cuda
src/galario.cpp:// `min` is chosen for the kernels that are both on gpu and cpu
src/galario.cpp:// Stuff needed for GPU and CPU but should not be visible any other translation unit so we can use very common names.
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    #define CCheck(err) __cudaSafeCall((err), __FILE__, __LINE__)
src/galario.cpp:    inline void __cudaSafeCall(cudaError err, const char *file, const int line)  {
src/galario.cpp:        if (err == cudaErrorInitializationError) {
src/galario.cpp:            throw_exception(file, line, "cuda", "Could not initialize cuda. Is a CUDA GPU available at all?");
src/galario.cpp:        if (err == cudaErrorMemoryAllocation) {
src/galario.cpp:        if (cudaSuccess != err) {
src/galario.cpp:            throw_exception(file, line, "cuda", cudaGetErrorString(err));
src/galario.cpp:            throw_exception(file, line, "cublas", "Could not initialize cublas. Is a cuda GPU available at all? Or is it ouf memory?");
src/galario.cpp:     * A simple RAII wrapper around cuda memory for exception safety
src/galario.cpp:    struct CudaMemory {
src/galario.cpp:        CudaMemory(size_t n) : nbytes(sizeof(T) * n) {
src/galario.cpp:            const auto error = cudaMalloc(&ptr, nbytes);
src/galario.cpp:            if (error != cudaSuccess) {
src/galario.cpp:                CCheck(cudaFree(ptr));
src/galario.cpp:        CudaMemory(size_t n, const T* source) : CudaMemory(n) {
src/galario.cpp:            CCheck(cudaMemcpy(ptr, source, nbytes, cudaMemcpyHostToDevice));
src/galario.cpp:        CudaMemory(const CudaMemory&) = delete;
src/galario.cpp:        CudaMemory& operator=(const CudaMemory&) = delete;
src/galario.cpp:        CudaMemory(CudaMemory&&) = default;
src/galario.cpp:        CudaMemory& operator=(CudaMemory&&) = default;
src/galario.cpp:        ~CudaMemory() {
src/galario.cpp:            cudaFree(ptr);
src/galario.cpp:            CCheck(cudaMemcpy(destination, ptr, nbytes, cudaMemcpyDeviceToHost));
src/galario.cpp:        struct GPUTimer
src/galario.cpp:            cudaEvent_t start;
src/galario.cpp:            cudaEvent_t stop;
src/galario.cpp:            GPUTimer() {
src/galario.cpp:                CCheck(cudaEventCreate(&start));
src/galario.cpp:                CCheck(cudaEventCreate(&stop));
src/galario.cpp:            ~GPUTimer() {
src/galario.cpp:                CCheck(cudaEventDestroy(start));
src/galario.cpp:                CCheck(cudaEventDestroy(stop));
src/galario.cpp:                CCheck(cudaEventRecord(start, 0));
src/galario.cpp:                CCheck(cudaEventRecord(stop, 0));
src/galario.cpp:                CCheck(cudaEventSynchronize(stop));
src/galario.cpp:                CCheck(cudaEventElapsedTime(&elapsed, start, stop));
src/galario.cpp:                ::out() << "[GPU] " << msg << ": " << elapsed << " ms\n";
src/galario.cpp:        struct GPUTimer
src/galario.cpp:            GPUTimer() {
src/galario.cpp:#endif // __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    // fails if cuda is not available. Let the initialization be done only if
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:CudaMemory<dcomplex> copy_input_d(int nx, int ny, const dreal* realdata) {
src/galario.cpp:    GPUTimer t;
src/galario.cpp:    CudaMemory<dcomplex> data_d(nx * ncol);
src/galario.cpp:    CCheck(cudaMemcpy2D(data_d.ptr, rowsize_complex, realdata, rowsize_real, rowsize_real, nx, cudaMemcpyHostToDevice));
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:     CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CudaMemory<dcomplex> data_d(nx*(ny/2 + 1), data);
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CudaMemory<dcomplex> data_d(nx*(ny/2+1), data);
src/galario.cpp:    CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CudaMemory<dcomplex> matrix_d(nrow * ncol, matrix);
src/galario.cpp:    CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CudaMemory<dcomplex> data_d(nrow * ncol, data);
src/galario.cpp:    CudaMemory<dreal> u_d(nd, u);
src/galario.cpp:    CudaMemory<dreal> v_d(nd, v);
src/galario.cpp:    CudaMemory<dcomplex> vis_int_d(nd);
src/galario.cpp:    CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:     CudaMemory<dreal> u_d(nd, u);
src/galario.cpp:     CudaMemory<dreal> v_d(nd, v);
src/galario.cpp:     CudaMemory<dcomplex> vis_int_d(nd, vis_int);
src/galario.cpp:     CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:     CudaMemory<dreal> u_d(nd, u);
src/galario.cpp:     CudaMemory<dreal> v_d(nd, v);
src/galario.cpp:     CudaMemory<dreal> urot_d(nd);
src/galario.cpp:     CudaMemory<dreal> vrot_d(nd);
src/galario.cpp:        cudaMemcpy(urot_d.ptr, u_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
src/galario.cpp:        cudaMemcpy(vrot_d.ptr, v_d.ptr, v_d.nbytes, cudaMemcpyDeviceToDevice);
src/galario.cpp:     CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:CudaMemory<dcomplex> create_image_d(int nr, const dreal* const intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc) {
src/galario.cpp:    GPUTimer t, t_start;
src/galario.cpp:    CudaMemory<dcomplex> image_d(nxy * (nxy / 2 + 1)); t.Elapsed("create_image_d::malloc_image");
src/galario.cpp:    CCheck(cudaMemset(image_d.ptr, 0, image_d.nbytes)); t.Elapsed("create_image_d::memset");
src/galario.cpp:    CudaMemory<dreal> intensity_d(nr, intensity); t.Elapsed("create_image_d::malloc_copy_intensity_H->D");
src/galario.cpp:    CCheck(cudaDeviceSynchronize());
src/galario.cpp:    CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CudaMemory<dcomplex> image_d = create_image_d(nr, intensity, Rmin, dR, nxy, dxy, inc);
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    GPUTimer t_start;
src/galario.cpp:       use nonzero cudaStream_t
src/galario.cpp:    GPUTimer t;
src/galario.cpp:    CudaMemory<dreal> u_d(nd, u);
src/galario.cpp:    CudaMemory<dreal> v_d(nd, v);
src/galario.cpp:    CudaMemory<dreal> urot_d(nd);
src/galario.cpp:    CudaMemory<dreal> vrot_d(nd);
src/galario.cpp:        cudaMemcpy(urot_d.ptr, u_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
src/galario.cpp:        cudaMemcpy(vrot_d.ptr, v_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    GPUTimer t_total;
src/galario.cpp:    CudaMemory<dcomplex> vis_int_d(nd);
src/galario.cpp:    CCheck(cudaDeviceSynchronize());
src/galario.cpp:    GPUTimer t;
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CudaMemory<dcomplex> image_d = create_image_d(nr, intensity, Rmin, dR, nxy, dxy, inc);
src/galario.cpp:    CudaMemory<dcomplex> vis_int_d(nd);
src/galario.cpp:    CCheck(cudaDeviceSynchronize());
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    GPUTimer t_start, t;
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:     CudaMemory<dreal> vis_obs_re_d(nd, vis_obs_re);
src/galario.cpp:     CudaMemory<dreal> vis_obs_im_d(nd, vis_obs_im);
src/galario.cpp:     CudaMemory<dcomplex> vis_int_d(nd, vis_int);
src/galario.cpp:     CudaMemory<dreal> weights_d(nd, weights);
src/galario.cpp:int ngpus()
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CCheck(cudaGetDeviceCount(&num_devices));
src/galario.cpp:void use_gpu(int device_id)
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    CCheck(cudaSetDevice(device_id));
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    GPUTimer t;
src/galario.cpp:      use nonzero cudaStream_t
src/galario.cpp:    CudaMemory<dcomplex> vis_int_d(nd);
src/galario.cpp:    CudaMemory<dreal> vis_obs_re_d(nd, vis_obs_re);
src/galario.cpp:    CudaMemory<dreal> vis_obs_im_d(nd, vis_obs_im);
src/galario.cpp:    CudaMemory<dreal> weights_d(nd, weights);
src/galario.cpp:#ifdef __CUDACC__
src/galario.cpp:    GPUTimer t, t_start2;
src/galario.cpp:    CudaMemory<dcomplex> vis_int_d(nd);
src/galario.cpp:    CudaMemory<dreal> vis_obs_re_d(nd, vis_obs_re);
src/galario.cpp:    CudaMemory<dreal> vis_obs_im_d(nd, vis_obs_im);
src/galario.cpp:    CudaMemory<dreal> weights_d(nd, weights);
src/galario.cpp:    t_start2.Elapsed("chi2_profile_tot_gputimer");
src/galario_defs.h:* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
src/galario_defs.h:#ifdef __CUDACC__
src/galario_defs.h:    #ifdef __CUDACC__
src/galario_defs.h:    #ifdef __CUDACC__
src/galario.h:* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
src/galario.h:/* GPU related functions */
src/galario.h:int ngpus();
src/galario.h:void use_gpu(int device_id);
src/CMakeLists.txt:# GALARIO - Gpu Accelerated Library for Analysing Radio Interferometer Observations #
src/CMakeLists.txt:if (CUDA_FOUND)
src/CMakeLists.txt:  # We require at least 30 because anything before is deprecated in cuda 8. For
src/CMakeLists.txt:  # maximum compatibility with future GPUs, we don't specify an exact GPU code
src/CMakeLists.txt:  # http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#further-mechanisms
src/CMakeLists.txt:  list(APPEND CUDA_NVCC_FLAGS --gpu-architecture=compute_30 --gpu-code=compute_30)
src/CMakeLists.txt:  # Issue with cuda 7.5 on Ubuntu 16.04
src/CMakeLists.txt:  list(APPEND CUDA_NVCC_FLAGS -D_FORCE_INLINES)
src/CMakeLists.txt:  # Allow multiple processes to use the GPU concurrently
src/CMakeLists.txt:  list(APPEND CUDA_NVCC_FLAGS --default-stream per-thread)
src/CMakeLists.txt:    list(APPEND CUDA_NVCC_FLAGS -DGALARIO_TIMING)
src/CMakeLists.txt:  SET(CUDA_PROPAGATE_HOST_FLAGS ON)
src/CMakeLists.txt:  set(common_cu "${CMAKE_CURRENT_BINARY_DIR}/cuda_lib.cu")
src/CMakeLists.txt:  cuda_add_library(galario_single_cuda ${common_cu})
src/CMakeLists.txt:  list(APPEND CUDA_NVCC_FLAGS -DDOUBLE_PRECISION)
src/CMakeLists.txt:  cuda_add_library(galario_cuda ${common_cu})
src/CMakeLists.txt:  foreach(t IN ITEMS galario_single_cuda galario_cuda)
src/CMakeLists.txt:    cuda_add_cublas_to_target(${t})
src/CMakeLists.txt:    cuda_add_cufft_to_target(${t})
src/CMakeLists.txt:endif() # cuda
src/galario_test.cpp:* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
CHANGELOG.rst:- [interface] Python and C++ code now throw detailed exceptions allowing fine-grained control, e.g. for executions on GPU.
CHANGELOG.rst:- [core] Memory handling on GPU: memory is now automatically freed in case of an error (allows catching errors with Exceptions).
CHANGELOG.rst:- [core/bugfix] Fix memory leak in GPU version.
CHANGELOG.rst:- [core] Allow multiple processes to use the GPU concurrently by default.
CHANGELOG.rst:- [interface] Allow enabling/disabling check for CUDA on Mac OS with `cmake -DGALARIO_CHECK_CUDA=1`.

```
