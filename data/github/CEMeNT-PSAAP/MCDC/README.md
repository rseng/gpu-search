# https://github.com/CEMeNT-PSAAP/MCDC

```console
docs/paper.bib:    title = {Divergence Reduction in {M}onte {C}arlo Neutron Transport with On-{GPU} Asynchronous Scheduling},
docs/paper.bib:    keywords = {asynchronous, divergence, scheduling, GPGPU, GPU}
docs/paper.bib:	title = {Continuous-energy {Monte} {Carlo} neutron transport on {GPUs} in the {Shift} code},
docs/paper.bib:	abstract = {A continuous-energy Monte Carlo neutron transport solver executing on GPUs has been developed within the Shift code. Several algorithmic approaches are considered, including both history-based and event-based implementations. Unlike in previous work involving multigroup Monte Carlo transport, it is demonstrated that event-based algorithms signiﬁcantly outperform a historybased approach for continuous-energy transport as a result of increased device occupancy and reduced thread divergence. Numerical results are presented for detailed full-core models of a small modular reactor (SMR), including a model containing depleted fuel materials. These results demonstrate the substantial gains in performance that are possible with the latest-generation of GPUs. On the depleted SMR core conﬁguration, an NVIDIA P100 GPU with 56 streaming multiprocessors provides performance equivalent to 90 CPU cores, and the latest V100 GPU with 80 multiprocessors oﬀers the performance of more than 150 CPU cores.},
docs/source/user/first_mcdc.rst:and GPUs (AMD and Nvidia) and supports threading with MPI (Python or compiled modes).
docs/source/user/first_mcdc.rst:For more performance see how to execute MC/DC on CPUs and GPUs
docs/source/user/index.rst:We include a simple "first simulation guide" as well as more in-depth descriptions on how to execute MC/DC in compiled modes to CPUs and GPUs with or without MPI.
docs/source/user/index.rst:    gpu
docs/source/user/cpu.rst:.. _gpu:
docs/source/user/gpu.rst:.. _gpu:
docs/source/user/gpu.rst:Running MC/DC on GPUs
docs/source/user/gpu.rst:MC/DC supports most of its Numba enabled features for GPU compilation.
docs/source/user/gpu.rst:When targeting GPUs execution MC/DC uses the Harmonize library to schedule events.
docs/source/user/gpu.rst:Harmonize acts as the GPU runtime for MC/DC and has two major scheduling schemes including a novel asynchronous event scheduler.
docs/source/user/gpu.rst:Single GPU Launches
docs/source/user/gpu.rst:To run problems on the GPU evoke input decks with a ``--mode=numba --target=gpu`` option appended on the python command.
docs/source/user/gpu.rst:    python input.py --mode=numba --target=gpu
docs/source/user/gpu.rst:At runtime the user can interface with the Harmonize scheduler that MC/DC uses as its GPU runtime.
docs/source/user/gpu.rst:#. Specifying scheduling modes with ``--gpu_strat=`` either ``event`` (default) or ``async`` (only enabled for Nvidia GPUs) 
docs/source/user/gpu.rst:#. Declaring the GPU arena size (size of memory allocated on the GPU measured in particles) ``--gpu_arena_size= [int_value]`` 
docs/source/user/gpu.rst:MPI+GPU Operability
docs/source/user/gpu.rst:Multi-GPU runs are enabled and require only to be dispatched with appropriate MPI calls.
docs/source/user/gpu.rst:The workflow for MPI+GPU calls is the same as with normal MPI calls and looks something like (assuming you are on an HPC): 
docs/source/user/gpu.rst:    flux run -N 2 -n 8 -g 1 --queue=mi300a python3 input.py --mode=numba --target=gpu --gpu_arena_size=100000000 --gpu_strat=event
docs/source/user/gpu.rst:which launches event scheduled MC/DC on GPUs with a GPU arena 1e9 2 nodes with 8 GPUs total (4/node) on the MI300A partition.
docs/source/user/gpu.rst:    jsrun -n 4 -r 4 -a 1 -g 1 python input.py --mode=numba --target=gpu --gpu_strat=async
docs/source/user/gpu.rst:which launches async scheduled MC/DC on Nvidia GPUs with a GPU arena of 1e9 on 1 node with 4 GPUs total (4/node).
docs/source/user/gpu.rst:GPU Profiling
docs/source/pubs.rst:Neutron Transport with On-GPU Asynchronous Scheduling. ACM Trans. 
docs/source/index.rst:a rapid methods development platform for for modern HPCs and is targeting CPUs and GPUs.
docs/source/index.rst:* linux-nvidia-cuda
docs/source/theory/index.rst:    gpu
docs/source/theory/gpu.rst:.. _gpu:
docs/source/theory/gpu.rst:GPU Functionality
docs/source/install.rst:On Lassen, ``module load gcc/8 cuda/11.8``. Then, 
docs/source/install.rst:GPU Operability (MC/DC+Harmonize)
docs/source/install.rst:MC/DC supports most of its Numba enabled features for GPU compilation and execution.
docs/source/install.rst:When targeting GPUs, MC/DC uses the `Harmonize <https://github.com/CEMeNT-PSAAP/harmonize>`_ library as its GPU runtime, a.k.a. the thing that actually executes MC/DC functions.
docs/source/install.rst:Harmonize acts as MC/DC's GPU runtime by using two major scheduling schemes: an event schedular similar to those implemented in OpenMC and Shift, plus a novel scheduler.
docs/source/install.rst:Nvidia GPUs
docs/source/install.rst:To compile and execute MC/DC on Nvidia GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (CUDA=11.8, Numba>=0.58.0) and a working MC/DC version >=0.10.0. Then,
docs/source/install.rst:AMD GPUs
docs/source/install.rst:To compile and execute MC/DC on AMD GPUs first ensure you have the `Harmonize prerecs <https://github.com/CEMeNT-PSAAP/harmonize/blob/main/install.sh>`_ (ROCm=6.0.0, Numba>=0.58.0) and a working MC/DC version >=0.11.0. Then,
docs/source/install.rst:#. Patch Numba to enable HIP (`instructions here <https://github.com/ROCm/numba-hip>`_)
docs/paper.md:  - GPU
docs/paper.md:It uses the Numba compiler for Python to compile compute kernels to a desired hardware target, including support for graphics processing units (GPUs) [@lam_numba_2015].
docs/paper.md:The main `MC/DC` branch currently only supports CPU architectures enabled by Numba (`x86-64`, `arm64`, and `ppc64`) but we are rapidly extending support to GPUs.
docs/paper.md:We currently have operability on Nvidia GPUs (supported via Numba), and work is ongoing to enable compilation for AMD GPUs.
docs/paper.md:On GPUs, `MC/DC` will use the `harmonize` asynchronous GPU scheduler to increase performance [@brax2023].
docs/paper.md:`harmonize` works by batching jobs during execution such that similar operations get executed simultaneously, reducing the divergence between parallel threads running on the GPU.
test/regression/run.py:parser.add_argument("--target", type=str, choices=["cpu", "gpu"], default="cpu")
test/regression/run.py:# Skip iqmc if GPU run
test/regression/run.py:if target == "gpu":
test/regression/run.py:                Fore.YELLOW + "Note: Skipping %s (GPU target)" % name + Style.RESET_ALL
test/regression/run.py:        gpus_per_task = ""
test/regression/run.py:        if target == "gpu":
test/regression/run.py:            gpus_per_task = f"--gpus-per-task=1 "
test/regression/run.py:            % (srun, gpus_per_task, mode, target)
test/regression/run.py:                        args.target == "gpu"
CITATION.cff:  - gpu
pyproject.toml:keywords = ["Monte Carlo", "nuclear engineering", "neutron transport", "HPC", "GPU", "numba", "mpi4py"]
mcdc/geometry.py:from mcdc.adapt import for_cpu, for_gpu
mcdc/geometry.py:@for_gpu()
mcdc/kernel.py:from mcdc.adapt import toggle, for_cpu, for_gpu
mcdc/kernel.py:@for_gpu()
mcdc/kernel.py:@for_gpu()
mcdc/kernel.py:@for_gpu()
mcdc/kernel.py:@for_gpu()
mcdc/loop.py:# Functions for GPU Interop
mcdc/loop.py:# manages GPU execution (if GPU execution is supported and selected)
mcdc/loop.py:# If GPU execution is supported and selected, the functions shown below will
mcdc/loop.py:# finalization of GPU state
mcdc/loop.py:def setup_gpu(mcdc):
mcdc/loop.py:def teardown_gpu(mcdc):
mcdc/loop.py:def gpu_sources_spec():
mcdc/loop.py:    def step(prog: nb.uintp, P_input: adapt.particle_gpu):
mcdc/loop.py:BLOCK_COUNT = config.args.gpu_block_count
mcdc/loop.py:ASYNC_EXECUTION = config.args.gpu_strat == "async"
mcdc/loop.py:def gpu_loop_source(seed, data, mcdc):
mcdc/loop.py:    # GPU Interop
mcdc/loop.py:        # Store the global state to the GPU
mcdc/loop.py:        src_store_constant(mcdc["gpu_state_pointer"], mcdc)
mcdc/loop.py:        src_store_data(mcdc["gpu_state_pointer"], data)
mcdc/loop.py:        src_load_constant(mcdc, mcdc["gpu_state_pointer"])
mcdc/loop.py:        src_load_data(data, mcdc["gpu_state_pointer"])
mcdc/loop.py:def gpu_precursor_spec():
mcdc/loop.py:    def step(prog: nb.uintp, P_input: adapt.particle_gpu):
mcdc/loop.py:def gpu_loop_source_precursor(seed, data, mcdc):
mcdc/loop.py:    # GPU Interop
mcdc/loop.py:    # Store the global state to the GPU
mcdc/loop.py:    pre_store_constant(mcdc["gpu_state_pointer"], mcdc)
mcdc/loop.py:    pre_store_data(mcdc["gpu_state_pointer"], data)
mcdc/loop.py:    pre_load_constant(mcdc, mcdc["gpu_state_pointer"])
mcdc/loop.py:    pre_load_data(data, mcdc["gpu_state_pointer"])
mcdc/loop.py:def build_gpu_progs(input_deck, args):
mcdc/loop.py:    STRAT = args.gpu_strat
mcdc/loop.py:    src_spec = gpu_sources_spec()
mcdc/loop.py:    pre_spec = gpu_precursor_spec()
mcdc/loop.py:    device_id = rank % args.gpu_share_stride
mcdc/loop.py:        args.gpu_arena_size = args.gpu_arena_size // 32
mcdc/loop.py:    ARENA_SIZE = args.gpu_arena_size
mcdc/loop.py:    BLOCK_COUNT = args.gpu_block_count
mcdc/loop.py:    def real_setup_gpu(mcdc):
mcdc/loop.py:        mcdc["gpu_state_pointer"] = adapt.cast_voidptr_to_uintp(alloc_state())
mcdc/loop.py:            src_alloc_program(mcdc["gpu_state_pointer"], ARENA_SIZE)
mcdc/loop.py:            pre_alloc_program(mcdc["gpu_state_pointer"], ARENA_SIZE)
mcdc/loop.py:    def real_teardown_gpu(mcdc):
mcdc/loop.py:        free_state(adapt.cast_uintp_to_voidptr(mcdc["gpu_state_pointer"]))
mcdc/loop.py:    global setup_gpu, teardown_gpu
mcdc/loop.py:    setup_gpu = real_setup_gpu
mcdc/loop.py:    teardown_gpu = real_teardown_gpu
mcdc/loop.py:    loop_source = gpu_loop_source
mcdc/loop.py:    loop_source_precursor = gpu_loop_source_precursor
mcdc/adapt.py:# Generic GPU/CPU Local Array Variable Constructors
mcdc/adapt.py:    elif isinstance(context, numba.cuda.target.CUDATypingContext):
mcdc/adapt.py:        # Function repurposed from Numba's Cuda_array_decl.
mcdc/adapt.py:    elif isinstance(context, numba.cuda.target.CUDATargetContext):
mcdc/adapt.py:        return numba.cuda.cudaimpl._generic_array(
mcdc/adapt.py:            symbol_name="_cudapy_harm_lmem",
mcdc/adapt.py:            addrspace=numba.cuda.cudadrv.nvvm.ADDRSPACE_LOCAL,
mcdc/adapt.py:        if target == "gpu":
mcdc/adapt.py:def for_gpu(on_target=[]):
mcdc/adapt.py:    return for_("gpu", on_target=on_target)
mcdc/adapt.py:# GPU Type / Extern Functions Forward Declarations
mcdc/adapt.py:mcdc_global_gpu = None
mcdc/adapt.py:mcdc_data_gpu = None
mcdc/adapt.py:group_gpu = None
mcdc/adapt.py:thread_gpu = None
mcdc/adapt.py:particle_gpu = None
mcdc/adapt.py:prep_gpu = None
mcdc/adapt.py:def gpu_forward_declare(args):
mcdc/adapt.py:    if args.gpu_rocm_path != None:
mcdc/adapt.py:        harm.config.set_rocm_path(args.gpu_rocm_path)
mcdc/adapt.py:    if args.gpu_cuda_path != None:
mcdc/adapt.py:        harm.config.set_cuda_path(args.gpu_cuda_path)
mcdc/adapt.py:    global mcdc_global_gpu, mcdc_data_gpu
mcdc/adapt.py:    global group_gpu, thread_gpu
mcdc/adapt.py:    global particle_gpu, particle_record_gpu
mcdc/adapt.py:    mcdc_global_gpu = access_fns["device"]["global"]
mcdc/adapt.py:    mcdc_data_gpu = access_fns["device"]["data"]
mcdc/adapt.py:    group_gpu = access_fns["group"]
mcdc/adapt.py:    thread_gpu = access_fns["thread"]
mcdc/adapt.py:    particle_gpu = numba.from_dtype(type_.particle)
mcdc/adapt.py:    particle_record_gpu = numba.from_dtype(type_.particle_record)
mcdc/adapt.py:    def step(prog: numba.uintp, P: particle_gpu):
mcdc/adapt.py:    def find_cell(prog: numba.uintp, P: particle_gpu):
mcdc/adapt.py:# Seperate GPU/CPU Functions to Target Different Platforms
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:    return mcdc_global_gpu(prog)
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:    return mcdc_data_gpu(prog)
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:    return group_gpu(prog)
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:    return thread_gpu(prog)
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:@for_gpu()
mcdc/adapt.py:device_gpu, group_gpu, thread_gpu = None, None, None
mcdc/adapt.py:    global device_gpu, group_gpu, thread_gpu
mcdc/adapt.py:    if target == "gpu":
mcdc/adapt.py:        device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
mcdc/adapt.py:def make_gpu_loop(
mcdc/adapt.py:    device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
mcdc/adapt.py:    elif target == "gpu_device":
mcdc/adapt.py:        return cuda.jit(func, device=True)
mcdc/adapt.py:    elif target == "gpu":
mcdc/adapt.py:        return cuda.jit(func)
mcdc/config.py:    "--target", type=str, help="Target", choices=["cpu", "gpu"], default="cpu"
mcdc/config.py:    "--gpu_strat",
mcdc/config.py:    help="Strategy used in GPU execution (event or async).",
mcdc/config.py:    "--gpu_block_count",
mcdc/config.py:    help="Number of blocks used in GPU execution.",
mcdc/config.py:    "--gpu_arena_size",
mcdc/config.py:    "--gpu_rocm_path",
mcdc/config.py:    help="Path to ROCm installation for use in GPU execution.",
mcdc/config.py:    "--gpu_cuda_path",
mcdc/config.py:    help="Path to CUDA installation for use in GPU execution.",
mcdc/config.py:    "--gpu_share_stride",
mcdc/config.py:    help="Number of gpus that are shared across adjacent ranks.",
mcdc/type_.py:# While CPU execution can robustly handle all sorts of Numba types, GPU
mcdc/type_.py:# If these rules are violated, memory accesses made in GPUs may encounter
mcdc/type_.py:            ("gpu_state_pointer", uintp),
mcdc/main.py:    build_gpu_progs,
mcdc/main.py:        if config.args.target == "gpu":
mcdc/main.py:            padding = config.args.gpu_block_count * 64 * 16
mcdc/main.py:    if config.target == "gpu":
mcdc/main.py:                "No module named 'harmonize' - GPU functionality not available. "
mcdc/main.py:        adapt.gpu_forward_declare(config.args)
mcdc/main.py:    if config.target == "gpu":
mcdc/main.py:        build_gpu_progs(input_deck, config.args)
mcdc/main.py:    loop.setup_gpu(mcdc)
mcdc/main.py:    loop.teardown_gpu(mcdc)
examples/fixed_source/kobayashi3-TD/scraper.py:# - 'event'  : Event-based GPU execution
examples/fixed_source/kobayashi3-TD/scraper.py:# - 'async'  : Async GPU execution
examples/fixed_source/kobayashi3-TD/scraper.py:    machine_arena_size_opt = "--gpu_arena_size=100000000"
examples/fixed_source/kobayashi3-TD/scraper.py:    machine_arena_size_opt = "--gpu_arena_size=20000000"
examples/fixed_source/kobayashi3-TD/scraper.py:        target_opt = "--target=gpu"
examples/fixed_source/kobayashi3-TD/scraper.py:        strat_opt = "--gpu_strat=event"
examples/fixed_source/kobayashi3-TD/scraper.py:        target_opt = "--target=gpu"
examples/fixed_source/kobayashi3-TD/scraper.py:        strat_opt = "--gpu_strat=async"
.gitignore:# GPU cache

```
