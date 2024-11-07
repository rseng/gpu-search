# https://github.com/EigenDev/simbi

```console
dev.py:default["gpu_compilation"] = "disabled"
dev.py:flag_overrides["gpu_compilation"] = ["--gpu-compilation", "--cpu-compilation"]
dev.py:    args: argparse.Namespace, reconfigure: str, hdf5_include: str, gpu_include: str
dev.py:    command = f"""meson setup {args.build_dir} -Dgpu_compilation={args.gpu_compilation}  
dev.py:    -Dhdf5_include_dir={hdf5_include} -Dgpu_include_dir={gpu_include} \
dev.py:    -Dprofile={args.install_mode} -Dgpu_arch={args.dev_arch} -Dfour_velocity={args.four_velocity} \
dev.py:        help="SM architecture specification for gpu compilation",
dev.py:        help="flag to enable / disable shared memory for gpu builds",
dev.py:        "--gpu-compilation",
dev.py:        dest="gpu_compilation",
dev.py:        dest="gpu_compilation",
dev.py:        gpu_compilation="disabled",
dev.py:    gpu_runtime_dir = ""
dev.py:        which_cuda = Path(get_tool("nvcc"))
dev.py:        gpu_runtime_dir = " ".join(
dev.py:                for path in which_cuda.parents
dev.py:                if "cuda" in str(path.parent)
dev.py:        gpu_runtime_dir = get_output(["hipconfig", "--rocmpath"])
dev.py:        gpu_include = f"{gpu_runtime_dir.split()[0]}/include"
dev.py:        gpu_include = ""
dev.py:    config_command = configure(args, reconfigure_flag, hdf5_include, gpu_include)
build_options.hpp.in: * @brief      file to configure build mode for various cpu / gpu architectures
build_options.hpp.in:        GPU = 1
build_options.hpp.in:        CUDA = 0,
build_options.hpp.in:        ROCM = 1,
build_options.hpp.in:#if defined(GPU_PLATFORM_AMD)
build_options.hpp.in:#if defined(GPU_PLATFORM_AMD)
build_options.hpp.in:#if GPU_CODE
build_options.hpp.in:#define GPU_SHARED __device__
build_options.hpp.in:    // shorthand flag for using gpu shared memory
build_options.hpp.in:#define GPU_SHARED __device__ const
build_options.hpp.in:    // shorthand flag for using gpu shared memory
build_options.hpp.in:    constexpr Platform BuildPlatform = Platform::GPU;
build_options.hpp.in:#define GPU_DEV                    __device__
build_options.hpp.in:#define GPU_DEV_INLINE             __device__ inline
build_options.hpp.in:#define GPU_LAUNCHABLE             __global__
build_options.hpp.in:#define GPU_LAMBDA                 __device__
build_options.hpp.in:#define GPU_CALLABLE               __host__ __device__
build_options.hpp.in:#define GPU_CALLABLE_INLINE        __host__ __device__ inline
build_options.hpp.in:#define GPU_CALLABLE_MEMBER        __host__ __device__
build_options.hpp.in:#define GPU_CALLABLE_INLINE_MEMBER __host__ __device__ inline
build_options.hpp.in:#define GPU_EXTERN_SHARED          extern __shared__
build_options.hpp.in:#if GPU_PLATFORM_NVIDIA
build_options.hpp.in:#include <cuda_runtime.h>
build_options.hpp.in:#define CUDA_CODE 1
build_options.hpp.in:    return cudaMalloc(devPtr, size);
build_options.hpp.in:    return cudaMallocManaged(devPtr, size);
build_options.hpp.in:inline auto devEventCreate(cudaEvent_t* stamp)
build_options.hpp.in:    return cudaEventCreate(stamp);
build_options.hpp.in:inline auto devEventRecord(cudaEvent_t stamp)
build_options.hpp.in:    return cudaEventRecord(stamp);
build_options.hpp.in:constexpr auto devMemcpy               = cudaMemcpy;
build_options.hpp.in:constexpr auto devFree                 = cudaFree;
build_options.hpp.in:constexpr auto devMemset               = cudaMemset;
build_options.hpp.in:constexpr auto devDeviceSynchronize    = cudaDeviceSynchronize;
build_options.hpp.in:constexpr auto devMemcpyHostToDevice   = cudaMemcpyHostToDevice;
build_options.hpp.in:constexpr auto devMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
build_options.hpp.in:constexpr auto devMemcpyDeviceToHost   = cudaMemcpyDeviceToHost;
build_options.hpp.in:constexpr auto devGetErrorString       = cudaGetErrorString;
build_options.hpp.in:constexpr auto devEventDestroy         = cudaEventDestroy;
build_options.hpp.in:constexpr auto devEventSynchronize     = cudaEventSynchronize;
build_options.hpp.in:constexpr auto devEventElapsedTime     = cudaEventElapsedTime;
build_options.hpp.in:constexpr auto devGetDeviceProperties  = cudaGetDeviceProperties;
build_options.hpp.in:constexpr auto devGetDeviceCount       = cudaGetDeviceCount;
build_options.hpp.in:using devProp_t                        = cudaDeviceProp;
build_options.hpp.in:using devError_t                       = cudaError_t;
build_options.hpp.in:using devEvent_t                       = cudaEvent_t;
build_options.hpp.in:using simbiStream_t                    = cudaStream_t;
build_options.hpp.in:#elif GPU_PLATFORM_AMD
build_options.hpp.in:    // shorthand flag for using gpu shared memory
build_options.hpp.in:#define CUDA_CODE 0
build_options.hpp.in:#define GPU_DEV
build_options.hpp.in:#define GPU_LAUNCHABLE
build_options.hpp.in:#define GPU_LAMBDA
build_options.hpp.in:#define GPU_CALLABLE
build_options.hpp.in:#define GPU_CALLABLE_INLINE inline
build_options.hpp.in:#define GPU_CALLABLE_MEMBER
build_options.hpp.in:#define GPU_CALLABLE_INLINE_MEMBER inline
build_options.hpp.in:#define GPU_DEV_INLINE             inline
build_options.hpp.in:#define GPU_SHARED                 const
build_options.hpp.in:#define GPU_EXTERN_SHARED
build_options.hpp.in:    // shorthand flag for gpu compilation check
build_options.hpp.in:    constexpr bool on_gpu = BuildPlatform == Platform::GPU;
simbi/__main__.py:            'gpu'],
simbi/__main__.py:        '--gpu-block-dims',
simbi/__main__.py:        help='gpu dim3 thread block dimensions',
simbi/__main__.py:        '--gpu', 
simbi/__main__.py:        const='gpu'
simbi/__main__.py:    for coord, block in zip(['X','Y','Z'], args.gpu_block_dims):
simbi/__main__.py:        os.environ[f'GPU{coord}BLOCK_SIZE'] = str(block)
simbi/simulator.py:            compute_mode (string):       The compute mode for simulation execution (cpu or gpu)
simbi/simulator.py:                if user_set := f"GPU{coord}BLOCK_SIZE" in os.environ:
simbi/simulator.py:                        dim3[idx] = int(os.environ[f"GPU{coord}BLOCK_SIZE"])
simbi/simulator.py:            logger.debug(f"In GPU mode, GPU block dims are: {tuple(dim3)}")
simbi/simulator.py:        if compute_mode == "gpu":
simbi/simulator.py:                if "GPUXBLOCK_SIZE" not in os.environ:
simbi/simulator.py:                    os.environ["GPUXBLOCK_SIZE"] = "128"
simbi/simulator.py:                if "GPUXBLOCK_SIZE" not in os.environ:
simbi/simulator.py:                    os.environ["GPUXBLOCK_SIZE"] = "16"
simbi/simulator.py:                if "GPUYBLOCK_SIZE" not in os.environ:
simbi/simulator.py:                    os.environ["GPUYBLOCK_SIZE"] = "16"
simbi/simulator.py:                if "GPUXBLOCK_SIZE" not in os.environ:
simbi/simulator.py:                    os.environ["GPUXBLOCK_SIZE"] = "4"
simbi/simulator.py:                if "GPUYBLOCK_SIZE" not in os.environ:
simbi/simulator.py:                    os.environ["GPUYBLOCK_SIZE"] = "4"
simbi/simulator.py:                if "GPUZBLOCK_SIZE" not in os.environ:
simbi/simulator.py:                    os.environ["GPUZBLOCK_SIZE"] = "4"
simbi/simulator.py:        lib_mode = "cpu" if compute_mode in ["cpu", "omp"] else "gpu"
README.md:## For GPU capability
README.md:8)  HIP/ROCm if wanting to run on NVIDIA or AMD GPUs, or just CUDA if
README.md:    running purely NVIDIA
README.md:3)  If `meson` detected `hip` or `cuda`, the install script will install
README.md:    both the cpu and gpu extensions into the `simbi/libs` directory.
README.md:When compiling on a GPU, you must provide your GPU's respective architecture identifier.
README.md:That is to say, if I am compiling on an NVIDIA V100 device with compute capability 7.0, I would
README.md:$ CC=<your_c_compiler> CXX=<your_cpp_compiler> python dev.py install --gpu-compilation --dev-arch 70 [options]
README.md:$ CC=<your_c_compiler> CXX=<your_cpp_compiler> meson setup <build_dir> -Dgpu_arch=70 -Dgpu_compilation=enabled [options]
README.md:    $ simbi run simbi_configs/examples/marti_muller.py --mode gpu --nzones 100 --ad-gamma 1.4 
README.md:    $ simbi run marti_muller --mode gpu --nzones 100 --ad-gamma 1.4
README.md:    $ simbi run marti-muller --mode gpu --nzones 100 --ad-gamma 1.4
README.md:  - [ ] multi-gpu support
pyproject.toml:description = "Python module to solve hydrodynamic equations using a hip/cuda/c++ backend"
pyproject.toml:    "Programming Language :: CUDA",
meson_options.txt:    'gpu_compilation',
meson_options.txt:    'gpu_arch',
meson_options.txt:    'gpu_include_dir',
meson_options.txt:    value: '/opt/cuda/include',
meson.build:gpu_depends   = []
meson.build:gpu_linkers   = []
meson.build:gpu_comp_args = []
meson.build:# this is useful because cuda lags behind on tested gcc versions
meson.build:# GPU CHECK
meson.build:if get_option('gpu_compilation').enabled()
meson.build:    hip  = dependency('HIP', cmake_module_path : '/opt/rocm', modules: ['hip::device', 'hip::host'], required: false)
meson.build:    cuda = dependency('CUDA', required: false)
meson.build:            warning(f'The detected gcc version: gcc-@host_compiler_version@ might not be compatible with cuda version')
meson.build:        gpu_arch = get_option('gpu_arch')
meson.build:        if hip_platform == 'nvidia'
meson.build:            hip_runtime  = 'cuda'
meson.build:            gpu_depends += [cuda]
meson.build:            gpu_linkers += ['-lcudart']
meson.build:            gpu_comp_args = [
meson.build:                '-DGPU_PLATFORM_NVIDIA=1',
meson.build:                f'-arch=sm_@gpu_arch@',
meson.build:            hip_runtime  = 'rocm'
meson.build:            gpu_linkers += ['-lamdhip64']
meson.build:            gpu_comp_args = [
meson.build:                '-DGPU_PLATFORM_AMD=1',
meson.build:                f'--offload-arch=gfx@gpu_arch@',
meson.build:            rocm_path = run_command('hipconfig', '--rocmpath', check: true).stdout().strip()
meson.build:                link_args: [f'-L@rocm_path@/lib', '-lamdhip64', '-O3', '-lgcc_s', '-lpthread', '-lm', '-lrt'],
meson.build:                include_directories: [f'@rocm_path@/include']
meson.build:            gpu_depends += [hip_dep]
meson.build:        message(f'GPU Platform -- @hip_platform@')
meson.build:        message(f'GPU Runtime  -- @hip_runtime@')
meson.build:        message(f'GPU Compiler -- @hip_compiler@')
meson.build:    elif cuda.found()
meson.build:        gpu_depends += [cuda]
meson.build:        gpu_linkers += ['-lcudart']
meson.build:        gpu_arch = get_option('gpu_arch')
meson.build:        gpu_comp_args = [
meson.build:                '-DGPU_PLATFORM_NVIDIA=1',
meson.build:                f'-arch=sm_@gpu_arch@', 
meson.build:        message(f'GPU Platform -- nvidia')
meson.build:        message(f'GPU Runtime  -- cuda')
meson.build:        message(f'GPU Compiler -- nvcc')
meson.build:    #### GPU / CPU IMPLEMENTATIONS
meson.build:    cpp_args: ['-DGPU_CODE=0'],
meson.build:if get_option('gpu_compilation').enabled()
meson.build:    if hip.found() or cuda.found()
meson.build:        gpu_cc = hip.found() ? 'hipcc' : 'nvcc'
meson.build:        gpu_compiler = find_program(f'@gpu_cc@')
meson.build:        gpu_objs = []
meson.build:        gpu_includes = ['-I'+meson.current_source_dir()+'/src', '-I.', '-I'+meson.current_build_dir()]
meson.build:        gpu_includes += ['-I' + get_option('gpu_include_dir'), '-I' + get_option('hdf5_include_dir')]
meson.build:        gpu_link_trgs = []
meson.build:            gpu_trg = custom_target(
meson.build:                    gpu_compiler,
meson.build:                    gpu_includes,
meson.build:                    gpu_comp_args,
meson.build:                    '-DGPU_CODE=1',
meson.build:            gpu_objs += [gpu_trg]
meson.build:        gpu_lib = custom_target(
meson.build:            'gpu_library',
meson.build:            input: gpu_objs,
meson.build:            output: 'libsimbi_gpu.a',
meson.build:        gpu_ext = py3.extension_module(
meson.build:            'gpu_ext',
meson.build:            ['src/gpu_ext.pyx', 'src/call_obj.pyx'],
meson.build:            link_with: [gpu_lib],
meson.build:            dependencies: depends + gpu_depends,
src/gpu_ext.pyx:# in Cython the extension name and file name need to match, but the gpu
src/gpu_ext.pyx:# implementation is identical for the cpu / gpu extensions, so instead of 
src/util/device_api.cpp:    namespace gpu {
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:            void gpuMalloc(void* obj, size_t elements)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:            void gpuMallocManaged(void* obj, size_t elements)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:            void gpuFree(void* obj)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(devFree(obj));
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:            void gpuMemset(void* obj, int val, size_t bytes)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                    simbi::gpu::error::status_t(devMemset(obj, val, bytes));
src/util/device_api.cpp:                simbi::gpu::error::check_err(status, "Failed to memset");
src/util/device_api.cpp:            void gpuEventSynchronize(devEvent_t a)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                    simbi::gpu::error::status_t(devEventSynchronize(a));
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:            void gpuEventCreate(devEvent_t* a)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(devEventCreate(a));
src/util/device_api.cpp:                simbi::gpu::error::check_err(status, "Failed to create event");
src/util/device_api.cpp:            void gpuEventDestroy(devEvent_t a)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(devEventDestroy(a));
src/util/device_api.cpp:                simbi::gpu::error::check_err(status, "Failed to destroy event");
src/util/device_api.cpp:            void gpuEventRecord(devEvent_t a)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                auto status = simbi::gpu::error::status_t(devEventRecord(a));
src/util/device_api.cpp:                simbi::gpu::error::check_err(status, "Failed to record event");
src/util/device_api.cpp:            void gpuEventElapsedTime(float* time, devEvent_t a, devEvent_t b)
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                    simbi::gpu::error::status_t(devEventElapsedTime(time, a, b)
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                    simbi::gpu::error::status_t(devGetDeviceCount(devCount));
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:                    simbi::gpu::error::status_t(devGetDeviceProperties(props, i)
src/util/device_api.cpp:                simbi::gpu::error::check_err(
src/util/device_api.cpp:#if GPU_CODE
src/util/device_api.cpp:    }   // namespace gpu
src/util/launch.tpp:#if GPU_CODE
src/util/parallel_for.hpp: * @brief      implementation of custom parallel-for exec on cpu-gpu lambdas
src/util/parallel_for.hpp:#include "build_options.hpp"   // for global::BuildPlatform, GPU_LAMBDA, Platform ...
src/util/parallel_for.hpp:        simbi::launch(p, [first, last, function] GPU_LAMBDA() {
src/util/parallel_for.hpp:#if GPU_CODE
src/util/parallel_for.hpp:        simbi::launch(p, [=] GPU_LAMBDA() {
src/util/parallel_for.hpp:#if GPU_CODE
src/util/ndarray.tpp:    // Copy from GPU if data exists there
src/util/ndarray.tpp:    copyBetweenGpu(rhs);
src/util/ndarray.tpp:    // Copy GPU data from rhs to lhs
src/util/ndarray.tpp:    copyToGpu();
src/util/ndarray.tpp:void simbi::ndarray<DT, build_mode>::copyToGpu()
src/util/ndarray.tpp:            dev_arr.reset((DT*) myGpuMalloc(nd_capacity));
src/util/ndarray.tpp:        gpu::api::copyHostToDevice(dev_arr.get(), arr.get(), nd_capacity);
src/util/ndarray.tpp:void simbi::ndarray<DT, build_mode>::copyFromGpu()
src/util/ndarray.tpp:        gpu::api::copyDevToHost(arr.get(), dev_arr.get(), nd_capacity);
src/util/ndarray.tpp:void simbi::ndarray<DT, build_mode>::copyBetweenGpu(const ndarray& rhs)
src/util/ndarray.tpp:        gpu::api::copyDevToDev(
src/util/ndarray.tpp:    if constexpr (global::on_gpu) {
src/util/device_api.hpp: * @brief      houses the gpu device-specific api calls
src/util/device_api.hpp:#include "build_options.hpp"   // for blockDim, threadIdx, GPU_CALLABLE_INLINE
src/util/device_api.hpp:    namespace gpu {
src/util/device_api.hpp:                gpuError
src/util/device_api.hpp:#if GPU_CODE
src/util/device_api.hpp:                 * Obtain the GPU status code which resulted in this error being
src/util/device_api.hpp:            void gpuMalloc(void* obj, size_t bytes);
src/util/device_api.hpp:            void gpuMallocManaged(void* obj, size_t bytes);
src/util/device_api.hpp:            void gpuFree(void* obj);
src/util/device_api.hpp:            void gpuEventSynchronize(devEvent_t a);
src/util/device_api.hpp:            void gpuEventCreate(devEvent_t* a);
src/util/device_api.hpp:            void gpuEventDestroy(devEvent_t a);
src/util/device_api.hpp:            void gpuEventRecord(devEvent_t a);
src/util/device_api.hpp:            void gpuEventElapsedTime(float* time, devEvent_t a, devEvent_t b);
src/util/device_api.hpp:            void gpuMemset(void* obj, int val, size_t bytes);
src/util/device_api.hpp:            GPU_DEV_INLINE void synchronize()
src/util/device_api.hpp:#if GPU_CODE
src/util/device_api.hpp:    }   // namespace gpu
src/util/device_api.hpp:    GPU_CALLABLE_INLINE
src/util/device_api.hpp:        if constexpr (global::on_gpu) {
src/util/device_api.hpp:    GPU_CALLABLE_INLINE
src/util/device_api.hpp:    GPU_CALLABLE_INLINE
src/util/device_api.hpp:    GPU_CALLABLE_INLINE
src/util/device_api.hpp:    GPU_CALLABLE_INLINE unsigned int get_tx()
src/util/device_api.hpp:        if constexpr (P == global::Platform::GPU) {
src/util/device_api.hpp:    GPU_CALLABLE_INLINE unsigned int get_ty()
src/util/device_api.hpp:        if constexpr (P == global::Platform::GPU) {
src/util/device_api.hpp:    GPU_CALLABLE_INLINE unsigned int get_threadId()
src/util/device_api.hpp:#if GPU_CODE
src/util/ndarray.hpp: * @brief      implementation of custom cpu-gpu translatable array class
src/util/ndarray.hpp:#include "device_api.hpp"      // for gpuFree, gpuMalloc, gpuMallocManaged
src/util/ndarray.hpp:        void* myGpuMalloc(size_type size)
src/util/ndarray.hpp:            if constexpr (build_mode == global::Platform::GPU) {
src/util/ndarray.hpp:                gpu::api::gpuMalloc(&ptr, size);
src/util/ndarray.hpp:        void* myGpuMallocManaged(size_type size)
src/util/ndarray.hpp:            if constexpr (build_mode == global::Platform::GPU) {
src/util/ndarray.hpp:                gpu::api::gpuMallocManaged(&ptr, size);
src/util/ndarray.hpp:        struct gpuDeleter {
src/util/ndarray.hpp:                if constexpr (build_mode == global::Platform::GPU) {
src/util/ndarray.hpp:                    gpu::api::gpuFree(ptr);
src/util/ndarray.hpp:        unique_p<gpuDeleter> dev_arr;
src/util/ndarray.hpp:        // get pointers to underlying data ambiguously, on host, or on gpu
src/util/ndarray.hpp:        // GPU memory copy helpers
src/util/ndarray.hpp:        void copyToGpu();
src/util/ndarray.hpp:        void copyFromGpu();
src/util/ndarray.hpp:        void copyBetweenGpu(const ndarray& rhs);
src/util/kernel.hpp: * @brief      the generic gpu kernel that runs a generic functor
src/util/kernel.hpp:    GPU_LAUNCHABLE void Kernel(Function f, Arguments... args)
src/util/exec_policy.hpp: * @brief      houses the execution policy object for gpu-specific runs
src/util/exec_policy.hpp:            if constexpr (global::on_gpu) {
src/util/range.hpp:#include "build_options.hpp"   // for GPU_CALLABLE, GPU_CALLABLE_INLINE
src/util/range.hpp:        GPU_CALLABLE
src/util/range.hpp:        GPU_CALLABLE
src/util/range.hpp:        GPU_CALLABLE
src/util/range.hpp:        GPU_CALLABLE
src/util/range.hpp:        GPU_CALLABLE
src/util/range.hpp:        GPU_CALLABLE
src/util/range.hpp:        GPU_CALLABLE
src/util/range.hpp:    GPU_CALLABLE
src/util/range.hpp:    GPU_CALLABLE
src/util/range.hpp:    GPU_CALLABLE
src/util/range.hpp:    GPU_CALLABLE
src/util/range.hpp:GPU_CALLABLE_INLINE range_t<T> range(T begin, T end, U step = 1)
src/util/managed.hpp: * @brief    houses the gpu-Managed object for modified new and delete operators
src/util/managed.hpp:#include "device_api.hpp"      // for deviceSynch, gpuFree, gpuMallocManaged
src/util/managed.hpp:    template <bool gpu_managed = global::managed_memory> class Managed
src/util/managed.hpp:            if constexpr (gpu_managed) {
src/util/managed.hpp:                gpu::api::gpuMallocManaged(&ptr, len);
src/util/managed.hpp:                gpu::api::deviceSynch();
src/util/managed.hpp:            if constexpr (gpu_managed) {
src/util/managed.hpp:                gpu::api::deviceSynch();
src/util/managed.hpp:                gpu::api::gpuFree(ptr);
src/util/logger.hpp:#include "device_api.hpp"       // for gpuEventCreate, gpuEventDestroy
src/util/logger.hpp:                global::on_gpu,
src/util/logger.hpp:                std::conditional_t<global::on_gpu, float, double>;
src/util/logger.hpp:                if constexpr (P == global::Platform::GPU) {
src/util/logger.hpp:                    gpu::api::gpuEventCreate(&stamp);
src/util/logger.hpp:                if constexpr (P == global::Platform::GPU) {
src/util/logger.hpp:                    gpu::api::gpuEventDestroy(stamp);
src/util/logger.hpp:                    gpu::api::gpuEventRecord(stamp);
src/util/logger.hpp:                    gpu::api::gpuEventSynchronize(t2);
src/util/logger.hpp:                    gpu::api::gpuEventElapsedTime(&dt, t1, t2);
src/util/logger.hpp:                    // time output from GPU automatically in ms so convert to
src/util/logger.hpp:                                sim_state.outer_zones.copyToGpu();
src/util/logger.hpp:                                sim_state.outer_zones.copyToGpu();
src/util/logger.hpp:                                    sim_state.outer_zones.copyToGpu();
src/util/logger.hpp:                            if constexpr (global::on_gpu) {
src/util/logger.hpp:                                const real gpu_emperical_bw =
src/util/logger.hpp:                                    100.0 * gpu_emperical_bw /
src/util/logger.hpp:                                        gpu_theoretical_bw
src/common/helpers.cpp://              GPU HELPERS
src/common/helpers.cpp:real gpu_theoretical_bw = 1.0;
src/common/helpers.cpp:// https://stackoverflow.com/questions/5689028/how-to-get-card-specs-programmatically-in-cuda
src/common/helpers.cpp:#if GPU_CODE
src/common/helpers.cpp:            gpu::api::getDeviceCount(&devCount);
src/common/helpers.cpp:            std::cout << "GPU Device(s): " << std::endl << std::endl;
src/common/helpers.cpp:                gpu::api::getDeviceProperties(&props, i);
src/common/helpers.cpp:                gpu_theoretical_bw = 2.0 * props.memoryClockRate *
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real p)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real p, real chi)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Primitive(const Primitive& prim)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator+(const Derived& prim) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator-(const Derived& prim) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator/(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator*(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator+(const Derived& prims) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator-(const Derived& prims) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator*(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Derived operator/(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Conserved(real den, real m1, real nrg)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Conserved(real den, real m1, real nrg, real chi)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Conserved(const Conserved& prim)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v() const { return v1; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real vcomponent(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real& momentum() { return m1; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real momentum(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real momentum(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real& momentum(const luint nhat)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v1() const { return v1; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v2() const { return v2; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real momentum(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real& momentum(const luint nhat)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v1() const { return v1; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v2() const { return v2; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v3() const { return v3; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real csL, real csR)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real momentum(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real& momentum(const luint nhat)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real csL, real csR)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved(real den, real m1, real nrg, real b1)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved(
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved(
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved(
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved(const AnyConserved& u)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved operator+(const AnyConserved& p) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved operator-(const AnyConserved& p) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved operator*(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved operator/(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved& operator+=(const AnyConserved& cons)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved& operator-=(const AnyConserved& cons)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyConserved& operator*=(const real c)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER real total_energy() { return den + nrg; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real momentum(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real& momentum(const luint nhat)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real& momentum() { return m1; }
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real bcomponent(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real& bcomponent(const luint nhat)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive(real rho, real v1, real p, real b1)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive(
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive(
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive(
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive(const AnyPrimitive& c)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive operator+(const AnyPrimitive& e) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive operator-(const AnyPrimitive& e) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive operator*(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive operator/(const real c) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive& operator+=(const AnyPrimitive& prims)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive& operator-=(const AnyPrimitive& prims)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER AnyPrimitive& operator*=(const real c)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v1() const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v2() const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real get_v3() const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real lorentz_factor() const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real lorentz_factor_squared() const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER mag_four_vec(const AnyPrimitive<dim>& prim)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER mag_four_vec(const mag_four_vec& c)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER real inner_product() const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER constexpr real normal(const luint nhat) const
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR) : afL(afL), afR(afR)
src/common/hydro_structs.hpp:        GPU_CALLABLE_MEMBER
src/common/hydro_structs.hpp:        // GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR,
src/common/helpers.tpp:            sim_state.prims.copyFromGpu();
src/common/helpers.tpp:            sim_state.cons.copyFromGpu();
src/common/helpers.tpp:            simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA(const int gid) {
src/common/helpers.tpp:            simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA(const int gid) {
src/common/helpers.tpp:            simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA(const int gid) {
src/common/helpers.tpp:        GPU_LAUNCHABLE typename std::enable_if<is_1D_primitive<T>::value>::type
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_LAUNCHABLE typename std::enable_if<is_2D_primitive<T>::value>::type
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_LAUNCHABLE typename std::enable_if<is_3D_primitive<T>::value>::type
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_LAUNCHABLE
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_LAUNCHABLE
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_LAUNCHABLE
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_LAUNCHABLE void deviceReduceKernel(T* self, real* dt_min, lint nmax)
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_LAUNCHABLE void
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:        GPU_CALLABLE T cubic(T b, T c, T d)
src/common/helpers.tpp:        GPU_CALLABLE int quartic(T b, T c, T d, T e, T res[4])
src/common/helpers.tpp:            if constexpr (global::BuildPlatform == global::Platform::GPU) {
src/common/helpers.tpp:        GPU_CALLABLE void myswap(T& a, T& b)
src/common/helpers.tpp:        GPU_CALLABLE index_type
src/common/helpers.tpp:        GPU_CALLABLE void
src/common/helpers.tpp:        GPU_CALLABLE void
src/common/helpers.tpp:        GPU_SHARED T* sm_proxy(const U object)
src/common/helpers.tpp:#if GPU_CODE
src/common/helpers.tpp:                GPU_EXTERN_SHARED unsigned char memory[];
src/common/helpers.tpp:        GPU_CALLABLE index_type flattened_index(
src/common/helpers.tpp:        GPU_CALLABLE T axid(T idx, T ni, T nj, T kk)
src/common/helpers.tpp:                    if constexpr (global::on_gpu) {
src/common/helpers.tpp:                    if constexpr (global::on_gpu) {
src/common/helpers.tpp:                    if constexpr (global::on_gpu) {
src/common/helpers.tpp:                    if constexpr (global::on_gpu) {
src/common/helpers.tpp:                    if constexpr (global::on_gpu) {
src/common/helpers.tpp:        GPU_DEV void load_shared_buffer(
src/common/helpers.tpp:                gpu::api::synchronize();
src/common/helpers.tpp:                gpu::api::synchronize();
src/common/helpers.tpp:                gpu::api::synchronize();
src/common/helpers.tpp:        GPU_CALLABLE void
src/common/helpers.tpp:        GPU_CALLABLE bool ib_check(
src/common/helpers.hpp:#include "build_options.hpp"   // for real, GPU_CALLABLE_INLINE, luint, lint
src/common/helpers.hpp:// https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
src/common/helpers.hpp:extern real gpu_theoretical_bw;   //  = 1875e6 * (192.0 / 8.0) * 2 / 1e9;
src/common/helpers.hpp:        GPU_CALLABLE_INLINE index_type
src/common/helpers.hpp:        GPU_CALLABLE_INLINE index_type
src/common/helpers.hpp:        GPU_CALLABLE_INLINE index_type
src/common/helpers.hpp:        GPU_CALLABLE_INLINE constexpr T my_max(const T a, const T b)
src/common/helpers.hpp:        GPU_CALLABLE_INLINE constexpr T my_min(const T a, const T b)
src/common/helpers.hpp:        GPU_CALLABLE_INLINE constexpr T my_max3(const T a, const T b, const T c)
src/common/helpers.hpp:        GPU_CALLABLE_INLINE constexpr T my_min3(const T a, const T b, const T c)
src/common/helpers.hpp:        GPU_CALLABLE_INLINE constexpr int sgn(T val)
src/common/helpers.hpp:        GPU_CALLABLE_INLINE real
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE real calc_rmhd_lorentz(
src/common/helpers.hpp:        GPU_CALLABLE_INLINE real
src/common/helpers.hpp:        GPU_CALLABLE_INLINE real
src/common/helpers.hpp:        GPU_CALLABLE_INLINE real
src/common/helpers.hpp:        GPU_CALLABLE_INLINE real newton_f_mhd(
src/common/helpers.hpp:        GPU_CALLABLE_INLINE real
src/common/helpers.hpp:        //          GPU TEMPLATES
src/common/helpers.hpp:        GPU_LAUNCHABLE typename std::enable_if<is_1D_primitive<T>::value>::type
src/common/helpers.hpp:        GPU_LAUNCHABLE typename std::enable_if<is_2D_primitive<T>::value>::type
src/common/helpers.hpp:        GPU_LAUNCHABLE typename std::enable_if<is_3D_primitive<T>::value>::type
src/common/helpers.hpp:        GPU_LAUNCHABLE
src/common/helpers.hpp:        GPU_LAUNCHABLE
src/common/helpers.hpp:        GPU_LAUNCHABLE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        GPU_CALLABLE_INLINE
src/common/helpers.hpp:        inline GPU_DEV real warpReduceMin(real val)
src/common/helpers.hpp:#if CUDA_CODE
src/common/helpers.hpp:            // to work with older cuda versions
src/common/helpers.hpp:#if __CUDA_ARCH__ >= 700
src/common/helpers.hpp:         * @brief perform the reduction in the GPU block
src/common/helpers.hpp:        inline GPU_DEV real blockReduceMin(real val)
src/common/helpers.hpp:#if GPU_CODE
src/common/helpers.hpp:                shared[global::WARP_SIZE];   // Shared mem for 32 (Nvidia) / 64
src/common/helpers.hpp:        GPU_LAUNCHABLE void deviceReduceKernel(T* self, lint nmax);
src/common/helpers.hpp:        GPU_LAUNCHABLE void deviceReduceWarpAtomicKernel(T* self, lint nmax);
src/common/helpers.hpp:        // display the CPU / GPU device properties
src/common/helpers.hpp:#if GPU_CODE
src/common/helpers.hpp:        GPU_CALLABLE T cubic(T b, T c, T d);
src/common/helpers.hpp:        GPU_CALLABLE int quartic(T b, T c, T d, T e, T res[4]);
src/common/helpers.hpp:        GPU_CALLABLE int cubicPluto(T b, T c, T d, T z[]);
src/common/helpers.hpp:        GPU_CALLABLE int quarticPluto(T b, T c, T d, T e, T res[4]);
src/common/helpers.hpp:        GPU_CALLABLE void swap(T& a, T& b);
src/common/helpers.hpp:        GPU_CALLABLE index_type
src/common/helpers.hpp:        GPU_CALLABLE void
src/common/helpers.hpp:        GPU_CALLABLE void
src/common/helpers.hpp:        GPU_SHARED T* sm_proxy(const U object);
src/common/helpers.hpp:        GPU_CALLABLE void
src/common/helpers.hpp:        GPU_CALLABLE bool ib_check(
src/common/helpers.hpp:        GPU_CALLABLE index_type flattened_index(
src/common/helpers.hpp:        GPU_CALLABLE T axid(T idx, T ni, T nj, T kk = T(0));
src/common/helpers.hpp:        GPU_DEV void load_shared_buffer(
src/cpu_ext.pyx:# in Cython the extension name and file name need to match, but the gpu
src/cpu_ext.pyx:# implementation is identical for the cpu / gpu extensions, so instead of 
src/hydro/srhd.hpp:#include "build_options.hpp"    // for real, GPU_CALLABLE_MEMBER, lint, luint
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:#include "build_options.hpp"    // for real, GPU_CALLABLE_MEMBER, lint, luint
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER conserved_t prims2cons(const primitive_t& prims
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/newt.hpp:        GPU_CALLABLE_MEMBER
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER constexpr real SRHD<dim>::get_x1_differential(const lint ii
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER constexpr real SRHD<dim>::get_x2_differential(const lint ii
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER constexpr real SRHD<dim>::get_x3_differential(const lint ii
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER real
src/hydro/srhd.tpp:        [prim_data, cons_data, press_data, troubled_data, this] GPU_LAMBDA(
src/hydro/srhd.tpp:            simbi::gpu::api::synchronize();
src/hydro/srhd.tpp:                simbi::gpu::api::synchronize();
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER SRHD<dim>::eigenvals_t SRHD<dim>::calc_eigenvals(
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER SRHD<dim>::conserved_t
src/hydro/srhd.tpp:#if GPU_CODE
src/hydro/srhd.tpp:    gpu::api::deviceSynch();
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER SRHD<dim>::conserved_t
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER SRHD<dim>::conserved_t SRHD<dim>::calc_hll_flux(
src/hydro/srhd.tpp:GPU_CALLABLE_MEMBER SRHD<dim>::conserved_t SRHD<dim>::calc_hllc_flux(
src/hydro/srhd.tpp:         this] GPU_LAMBDA(const luint idx) {
src/hydro/srhd.tpp:            if constexpr (global::on_gpu) {
src/hydro/srhd.tpp:            outer_zones.copyToGpu();
src/hydro/srhd.tpp:            outer_zones.copyToGpu();
src/hydro/srhd.tpp:            outer_zones.copyToGpu();
src/hydro/srhd.tpp:    cons.copyToGpu();
src/hydro/srhd.tpp:    prims.copyToGpu();
src/hydro/srhd.tpp:    pressure_guess.copyToGpu();
src/hydro/srhd.tpp:    dt_min.copyToGpu();
src/hydro/srhd.tpp:    density_source.copyToGpu();
src/hydro/srhd.tpp:    m1_source.copyToGpu();
src/hydro/srhd.tpp:        m2_source.copyToGpu();
src/hydro/srhd.tpp:        m3_source.copyToGpu();
src/hydro/srhd.tpp:        object_pos.copyToGpu();
src/hydro/srhd.tpp:    energy_source.copyToGpu();
src/hydro/srhd.tpp:    inflow_zones.copyToGpu();
src/hydro/srhd.tpp:    bcs.copyToGpu();
src/hydro/srhd.tpp:    troubled_cells.copyToGpu();
src/hydro/srhd.tpp:    sourceG1.copyToGpu();
src/hydro/srhd.tpp:        sourceG2.copyToGpu();
src/hydro/srhd.tpp:        sourceG3.copyToGpu();
src/hydro/srhd.tpp:        xactive_grid > gpu_block_dimx ? gpu_block_dimx : xactive_grid;
src/hydro/srhd.tpp:        yactive_grid > gpu_block_dimy ? gpu_block_dimy : yactive_grid;
src/hydro/srhd.tpp:        zactive_grid > gpu_block_dimz ? gpu_block_dimz : zactive_grid;
src/hydro/srhd.tpp:    if constexpr (global::on_gpu) {
src/hydro/srhd.tpp:            if constexpr (global::on_gpu) {
src/hydro/srhd.tpp:        troubled_cells.copyFromGpu();
src/hydro/srhd.tpp:        cons.copyFromGpu();
src/hydro/srhd.tpp:        prims.copyFromGpu();
src/hydro/base.hpp:        luint gpu_block_dimx, gpu_block_dimy, gpu_block_dimz;
src/hydro/base.hpp:        //=========================== GPU Threads Per Dimension
src/hydro/base.hpp:        std::string readGpuEnvVar(std::string const& key) const
src/hydro/base.hpp:            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUXBLOCK_SIZE"))
src/hydro/base.hpp:            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUYBLOCK_SIZE"))
src/hydro/base.hpp:            return static_cast<luint>(std::stoi(readGpuEnvVar("GPUZBLOCK_SIZE"))
src/hydro/base.hpp:              gpu_block_dimx(get_xblock_dims()),
src/hydro/base.hpp:              gpu_block_dimy(get_yblock_dims()),
src/hydro/base.hpp:              gpu_block_dimz(get_zblock_dims()),
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER constexpr real RMHD<dim>::get_x1_differential(const lint ii
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER constexpr real RMHD<dim>::get_x2_differential(const lint ii
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER constexpr real RMHD<dim>::get_x3_differential(const lint ii
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER real
src/hydro/rmhd.tpp:        [prim_data, cons_data, edens_data, troubled_data, gr, this] GPU_LAMBDA(
src/hydro/rmhd.tpp:            simbi::gpu::api::synchronize();
src/hydro/rmhd.tpp:                simbi::gpu::api::synchronize();
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER void RMHD<dim>::calc_max_wave_speeds(
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER RMHD<dim>::eigenvals_t RMHD<dim>::calc_eigenvals(
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t
src/hydro/rmhd.tpp:#if GPU_CODE
src/hydro/rmhd.tpp:    gpu::api::deviceSynch();
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t RMHD<dim>::calc_hll_flux(
src/hydro/rmhd.tpp:            // #if !GPU_CODE
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t RMHD<dim>::calc_hllc_flux(
src/hydro/rmhd.tpp:GPU_CALLABLE_MEMBER RMHD<dim>::conserved_t RMHD<dim>::calc_hlld_flux(
src/hydro/rmhd.tpp:         this] GPU_LAMBDA(const luint idx) {
src/hydro/rmhd.tpp:            if constexpr (global::on_gpu) {
src/hydro/rmhd.tpp:            outer_zones.copyToGpu();
src/hydro/rmhd.tpp:            outer_zones.copyToGpu();
src/hydro/rmhd.tpp:            outer_zones.copyToGpu();
src/hydro/rmhd.tpp:    cons.copyToGpu();
src/hydro/rmhd.tpp:    prims.copyToGpu();
src/hydro/rmhd.tpp:    edens_guess.copyToGpu();
src/hydro/rmhd.tpp:    dt_min.copyToGpu();
src/hydro/rmhd.tpp:    density_source.copyToGpu();
src/hydro/rmhd.tpp:    m1_source.copyToGpu();
src/hydro/rmhd.tpp:        m2_source.copyToGpu();
src/hydro/rmhd.tpp:        m3_source.copyToGpu();
src/hydro/rmhd.tpp:        object_pos.copyToGpu();
src/hydro/rmhd.tpp:    energy_source.copyToGpu();
src/hydro/rmhd.tpp:    inflow_zones.copyToGpu();
src/hydro/rmhd.tpp:    bcs.copyToGpu();
src/hydro/rmhd.tpp:    troubled_cells.copyToGpu();
src/hydro/rmhd.tpp:    sourceG1.copyToGpu();
src/hydro/rmhd.tpp:        sourceG2.copyToGpu();
src/hydro/rmhd.tpp:        sourceG3.copyToGpu();
src/hydro/rmhd.tpp:    sourceB1.copyToGpu();
src/hydro/rmhd.tpp:        sourceB2.copyToGpu();
src/hydro/rmhd.tpp:        sourceB3.copyToGpu();
src/hydro/rmhd.tpp:        bstag1.copyToGpu();
src/hydro/rmhd.tpp:        bstag2.copyToGpu();
src/hydro/rmhd.tpp:            bstag3.copyToGpu();
src/hydro/rmhd.tpp:        xactive_grid > gpu_block_dimx ? gpu_block_dimx : xactive_grid;
src/hydro/rmhd.tpp:        yactive_grid > gpu_block_dimy ? gpu_block_dimy : yactive_grid;
src/hydro/rmhd.tpp:        zactive_grid > gpu_block_dimz ? gpu_block_dimz : zactive_grid;
src/hydro/rmhd.tpp:    if constexpr (global::on_gpu) {
src/hydro/rmhd.tpp:            if constexpr (global::on_gpu) {
src/hydro/rmhd.tpp:        troubled_cells.copyFromGpu();
src/hydro/rmhd.tpp:        cons.copyFromGpu();
src/hydro/rmhd.tpp:        prims.copyFromGpu();
src/hydro/state.cpp: * they are templated and cython is unaware of the gpu-specific
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER constexpr real
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER real Newtonian<dim>::get_cell_volume(
src/hydro/newt.tpp:        [cons_data, prim_data, troubled_data, this] GPU_LAMBDA(const luint gid
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER Newtonian<dim>::eigenvals_t Newtonian<dim>::calc_eigenvals(
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER Newtonian<dim>::conserved_t
src/hydro/newt.tpp:#if GPU_CODE
src/hydro/newt.tpp:    gpu::api::deviceSynch();
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER Newtonian<dim>::conserved_t Newtonian<dim>::prims2flux(
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER Newtonian<dim>::conserved_t Newtonian<dim>::calc_hll_flux(
src/hydro/newt.tpp:GPU_CALLABLE_MEMBER Newtonian<dim>::conserved_t Newtonian<dim>::calc_hllc_flux(
src/hydro/newt.tpp:         this] GPU_LAMBDA(const luint idx) {
src/hydro/newt.tpp:            if constexpr (global::on_gpu) {
src/hydro/newt.tpp:            outer_zones.copyToGpu();
src/hydro/newt.tpp:            outer_zones.copyToGpu();
src/hydro/newt.tpp:            outer_zones.copyToGpu();
src/hydro/newt.tpp:    cons.copyToGpu();
src/hydro/newt.tpp:    prims.copyToGpu();
src/hydro/newt.tpp:    dt_min.copyToGpu();
src/hydro/newt.tpp:    density_source.copyToGpu();
src/hydro/newt.tpp:    m1_source.copyToGpu();
src/hydro/newt.tpp:        m2_source.copyToGpu();
src/hydro/newt.tpp:        m3_source.copyToGpu();
src/hydro/newt.tpp:        object_pos.copyToGpu();
src/hydro/newt.tpp:    energy_source.copyToGpu();
src/hydro/newt.tpp:    inflow_zones.copyToGpu();
src/hydro/newt.tpp:    bcs.copyToGpu();
src/hydro/newt.tpp:    troubled_cells.copyToGpu();
src/hydro/newt.tpp:    sourceG1.copyToGpu();
src/hydro/newt.tpp:        sourceG2.copyToGpu();
src/hydro/newt.tpp:        sourceG3.copyToGpu();
src/hydro/newt.tpp:        xactive_grid > gpu_block_dimx ? gpu_block_dimx : xactive_grid;
src/hydro/newt.tpp:        yactive_grid > gpu_block_dimy ? gpu_block_dimy : yactive_grid;
src/hydro/newt.tpp:        zactive_grid > gpu_block_dimz ? gpu_block_dimz : zactive_grid;
src/hydro/newt.tpp:    if constexpr (global::on_gpu) {
src/hydro/newt.tpp:            if constexpr (global::on_gpu) {
src/hydro/newt.tpp:        troubled_cells.copyFromGpu();
src/hydro/newt.tpp:        cons.copyFromGpu();
src/hydro/newt.tpp:        prims.copyFromGpu();
src/hydro/rmhd.hpp:#include "build_options.hpp"    // for real, GPU_CALLABLE_MEMBER, lint, luint
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER
src/hydro/rmhd.hpp:        GPU_CALLABLE_MEMBER

```
