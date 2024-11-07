# https://github.com/phirling/pyc2ray

```console
meson.options:# This option is used to target the CUDA architecture of the GPU that will be used. Default is sm_60 for Tesla P100 (CSCS Piz-Daint)
meson.options:option('gpu-architecture', type : 'string', value : 'sm_60', description : 'Target Architecture for CUDA extension module')
install_scripts/install_daint.sh:export PATH=/apps/daint/UES/hackaton/software/CUDAcore/11.8.0/bin:$PATH
install_scripts/install_daint.sh:# Makefile we set the flag --gpu-architecture=sm_60 = Nvidia P100, A100 = sm_80
install_scripts/install_daint.sh:grep gpu-architecture= Makefile
install_scripts/example_install.sh:module load daint-gpu
install_scripts/example_install.sh:module load nvidia
install_scripts/example_install.sh:# compile CUDA extension module
test/paper_tests/raytracing_benchmark/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/paper_tests/raytracing_benchmark/run_test.py:parser.add_argument("--gpu",action='store_true')
test/paper_tests/raytracing_benchmark/run_test.py:use_gpu = args.gpu
test/paper_tests/raytracing_benchmark/run_test.py:sim = pc2r.C2Ray_Test(paramfile, N, use_gpu)
test/paper_tests/raytracing_benchmark/run_test.py:    if use_gpu:
test/paper_tests/raytracing_benchmark/run_test.py:        # Copy positions & fluxes of sources to the GPU in advance
test/paper_tests/raytracing_benchmark/run_test.py:        if use_gpu:
test/paper_tests/raytracing_benchmark/run_test.py:if use_gpu:
test/paper_tests/test4_shadow/shadow.py:parser.add_argument("--gpu",action='store_true')
test/paper_tests/test4_shadow/shadow.py:use_octa = args.gpu
test/paper_tests/test4_shadow/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/paper_tests/test3_multisource/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/paper_tests/test3_multisource/run_test.py:parser.add_argument("--gpu",action='store_true')
test/paper_tests/test3_multisource/run_test.py:use_octa = args.gpu
test/paper_tests/test2_Ifront_cosmo/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/paper_tests/test2_Ifront_cosmo/run_test.py:parser.add_argument("--gpu",action='store_true')
test/paper_tests/test2_Ifront_cosmo/run_test.py:use_octa = args.gpu                   # Determines which raytracing algorithm to use
test/paper_tests/test1_Ifront/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/paper_tests/test1_Ifront/run_test.py:parser.add_argument("--gpu",action='store_true')
test/paper_tests/test1_Ifront/run_test.py:use_octa = args.gpu                   # Determines which raytracing algorithm to use
test/unit_tests_hackathon/2_multiple_sources/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/unit_tests_hackathon/2_multiple_sources/run_test.py:parser.add_argument("--gpu",action='store_true')
test/unit_tests_hackathon/2_multiple_sources/run_test.py:use_octa = args.gpu
test/unit_tests_hackathon/1_single_black_body/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/unit_tests_hackathon/1_single_black_body/run_test.py:parser.add_argument("--gpu",action='store_true')
test/unit_tests_hackathon/1_single_black_body/run_test.py:use_octa = args.gpu
test/unit_tests_hackathon/4_multiple_sources_mpi/plots.py:d2 = np.load(path+'phi_ion_gpu.npy')
test/unit_tests_hackathon/4_multiple_sources_mpi/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/unit_tests_hackathon/4_multiple_sources_mpi/run_test.py:parser.add_argument("--gpu", action='store_true')
test/unit_tests_hackathon/4_multiple_sources_mpi/run_test.py:use_octa = args.gpu
test/unit_tests_hackathon/3_multiple_sources_quick/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/unit_tests_hackathon/3_multiple_sources_quick/run_test.py:parser.add_argument("--gpu",action='store_true')
test/unit_tests_hackathon/3_multiple_sources_quick/run_test.py:use_octa = args.gpu
test/paper_eor_simulation/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/paper_eor_simulation/run_test.py:sim = pc2r.C2Ray_244Test(paramfile=paramfile, Nmesh=N, use_gpu=use_asora)
test/archive/cosmo_sim/run_mpi.py:sim = pc2r.C2Ray_CubeP3M(paramfile=paramfile, Nmesh=N, use_gpu=use_octa)
test/archive/cosmo_sim/run_cosmo.py:sim = pc2r.C2Ray_CubeP3M(paramfile=paramfile, Nmesh=N, use_gpu=use_octa)
test/archive/c2raypaper_Test_3/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/archive/fake_cosmo_for_report/run.slurm:#SBATCH --constraint=gpu
test/archive/fake_cosmo_for_report/run.slurm:#SBATCH --gres=gpu:1
test/archive/fake_cosmo_for_report/run.slurm:#export CRAY_CUDA_MPS=1
test/archive/single_source/run_example.py:# By default, the C2Ray raytracing is used. To use the gpu-accelerated
test/archive/single_source/run_example.py:# version, set use_octa = True. A CUDA-compatible GPU must be present
test/archive/single_source/run_example.py:parser.add_argument("--gpu",action='store_true',help="Use GPU raytracing")
test/archive/single_source/run_example.py:use_gpu = args.gpu                # Determines which raytracing algorithm to use
test/archive/single_source/run_example.py:sim = pc2r.C2Ray_Test(paramfile, N, use_gpu)
test/archive/rt_benchmark_old/t_vs_r/without_octa.py:    # Init GPU
test/archive/rt_benchmark_old/t_vs_r/with_octa.py:# Init GPU
test/archive/rt_benchmark_old/t_vs_r/with_octa.slurm:#SBATCH --constraint=gpu
test/archive/rt_benchmark_old/t_vs_r/with_octa.slurm:#SBATCH --gres=gpu:1
test/archive/rt_benchmark_old/t_vs_r/with_octa.slurm:#export CRAY_CUDA_MPS=1
test/archive/rt_benchmark_old/t_vs_r/without_octa.slurm:#SBATCH --constraint=gpu
test/archive/rt_benchmark_old/t_vs_r/without_octa.slurm:#SBATCH --gres=gpu:1
test/archive/rt_benchmark_old/t_vs_r/without_octa.slurm:#export CRAY_CUDA_MPS=1
test/archive/rt_benchmark_old/100k/with_octa.py:# Init GPU
test/archive/rt_benchmark_old/100k/with_octa.slurm:#SBATCH --constraint=gpu
test/archive/rt_benchmark_old/100k/with_octa.slurm:#SBATCH --gres=gpu:1
test/archive/rt_benchmark_old/100k/with_octa.slurm:#export CRAY_CUDA_MPS=1
test/archive/rt_benchmark_old/t_vs_numsrc/with_octa.py:# Init GPU
test/archive/rt_benchmark_old/t_vs_numsrc/with_octa.slurm:#SBATCH --constraint=gpu
test/archive/rt_benchmark_old/t_vs_numsrc/with_octa.slurm:#SBATCH --gres=gpu:1
test/archive/rt_benchmark_old/t_vs_numsrc/with_octa.slurm:#export CRAY_CUDA_MPS=1
test/archive/rt_benchmark_old/t_vs_numsrc/without_octa.slurm:#SBATCH --constraint=gpu
test/archive/rt_benchmark_old/t_vs_numsrc/without_octa.slurm:#SBATCH --gres=gpu:1
test/archive/rt_benchmark_old/t_vs_numsrc/without_octa.slurm:#export CRAY_CUDA_MPS=1
test/archive/rt_benchmark_old/check_rates.py:# Init GPU
test/archive/c2raypaper_Test_2/run_coarse.slurm:#SBATCH --constraint=gpu
test/archive/c2raypaper_Test_2/run_coarse.slurm:#SBATCH --gres=gpu:1
test/archive/c2raypaper_Test_2/run_coarse.slurm:#export CRAY_CUDA_MPS=1
test/archive/c2raypaper_Test_2/run_fine.slurm:#SBATCH --constraint=gpu
test/archive/c2raypaper_Test_2/run_fine.slurm:#SBATCH --gres=gpu:1
test/archive/c2raypaper_Test_2/run_fine.slurm:#export CRAY_CUDA_MPS=1
test/archive/c2raypaper_Test_2/parameters.yml:  # Source batch size, i.e. number of sources handled in parallel on the GPU.
test/archive/paper_singlesource_old/singlesource.py:parser.add_argument("--gpu",action='store_true')
test/archive/paper_singlesource_old/singlesource.py:use_octa = args.gpu
test/archive/244Mpc_test/run_cosmotest.py:#sim = pc2r.C2Ray_CubeP3M(paramfile=paramfile, Nmesh=N, use_gpu=use_octa)
test/archive/244Mpc_test/run_cosmotest.py:sim = pc2r.C2Ray_244Test(paramfile=paramfile, Nmesh=N, use_gpu=use_octa)
README.md:# pyc2ray: A flexible and GPU-accelerated radiative transfer framework
README.md:`pyc2ray` is the updated version of [C2Ray](https://github.com/garrelt/C2-Ray3Dm/tree/factorization) [(G. Mellema, I.T. Illiev, A. Alvarez and P.R. Shapiro)](https://ui.adsabs.harvard.edu/link_gateway/2006NewA...11..374M/doi:10.48550/arXiv.astro-ph/0508416), an astrophysical radiative transfer code widely used to simulate the Epoch of Reionization (EoR). `pyc2ray` features a new raytracing method developed for GPUs, named <b>A</b>ccelerated <b>S</b>hort-characteristics <b>O</b>cthaedral <b>RA</b>ytracing (<b>ASORA</b>). `pyc2ray` has a modern python interface that allows easy and customizable use of the code without compromising computational efficiency. A full description of the update and new ray-tracing method can be found online: [arXiv:2311.01492](https://arxiv.org/abs/2311.01492).
README.md:The core features of `C2Ray`, written in Fortran90, are wrapped using `f2py` as a python extension module, while the new raytracing library, _ASORA_, is implemented in C++ using CUDA. Both are native python C-extensions and can be directly accessed from any python script.
README.md:- `nvcc` CUDA compiler
README.md:### 2. Build CUDA extension module (Asora)
README.md:python run_example.py --gpu
README.md:The relevant script is located at `paper_tests/raytracing_benchmark/run_test.py`. This script is quite general, and allows you to measure the runtime of the GPU raytracing function for a varying number of sources, batch sizes and raytracing radii. The steps to reproduce exactly the test shown in the paper are outlined in the Jupyter Notebooks in `paper_tests/raytracing_benchmark/`. 
README.md:- GPU implementation of the chemistry solver
INSTALL.md:- `nvcc` CUDA compiler
INSTALL.md:On daint, load the modules `daint-gpu` and `PrgEnv-cray` (not sure if the latter is actually needed). The tool to build the modules is `f2py`, provided by the `numpy` package. The build requires version 1.24.4 or higher, to check run `f2py` without any options. If the version is too old or the command doesn't exist, install the latest numpy version in your current virtual environment. To build the extension module, run
INSTALL.md:### 2. Build CUDA extension module (Asora)
INSTALL.md:On daint, load the modules `daint-gpu` and `nvidia` (to access the nvcc compiler). Go to `/src/asora/`. Here, you need to edit the Makefile and add the correct include paths at lines 3 and 4. To find the correct python include path (line 3), run
INSTALL.md:module load daint-gpu PrgEnv-cray
INSTALL.md:module load daint-gpu nvidia
INSTALL.md:python run_example.py --gpu
meson.build:        'cuda_std=c++14'])
meson.build:# CUDA compiler is optional, required to build ASORA
meson.build:has_cuda = add_languages('cuda',required:false)
meson.build:if not has_cuda
meson.build:    warning('CUDA compiler not found, will not build ASORA library')
meson.build:    add_global_arguments(['-O2','-Xcompiler','-fPIC','-rdc','true'], language : 'cuda')
meson.build:    add_global_arguments(['-D PERIODIC','-D LOCALRATES'], language : 'cuda')
meson.build:# CUDA sources for ASORA
meson.build:if has_cuda
meson.build:    # Get CUDA architecture to be used for compilation
meson.build:    gpu_arch = get_option('gpu-architecture')
meson.build:        cuda_args : '--gpu-architecture='+gpu_arch)
pyc2ray/c2ray_base.py:# and other things such as memory allocation when using the GPU.
pyc2ray/c2ray_base.py:    def __init__(self, paramfile, Nmesh, use_gpu, use_mpi):
pyc2ray/c2ray_base.py:        use_gpu : bool
pyc2ray/c2ray_base.py:            Whether to use the GPU-accelerated ASORA library for raytracing
pyc2ray/c2ray_base.py:        if use_gpu:
pyc2ray/c2ray_base.py:            self.gpu = True
pyc2ray/c2ray_base.py:            # Allocate GPU memory
pyc2ray/c2ray_base.py:            atexit.register(self._gpu_close)
pyc2ray/c2ray_base.py:            self.gpu = False
pyc2ray/c2ray_base.py:            if self.gpu:
pyc2ray/c2ray_base.py:            else: self.printlog(f"Running in non-MPI (single-GPU/CPU) mode")
pyc2ray/c2ray_base.py:                    self.gpu, self.max_subbox,self.subboxsize,self.loss_fraction,
pyc2ray/c2ray_base.py:                    self.gpu, self.max_subbox,self.subboxsize,self.loss_fraction,
pyc2ray/c2ray_base.py:                self.gpu, self.max_subbox,self.subboxsize,self.loss_fraction,
pyc2ray/c2ray_base.py:            self.gpu,self.max_subbox,self.subboxsize,
pyc2ray/c2ray_base.py:            # 1. Add heating rate computation to ASORA (GPU raytracing)
pyc2ray/c2ray_base.py:        # Copy radiation table to GPU
pyc2ray/c2ray_base.py:        if self.gpu:
pyc2ray/c2ray_base.py:            if(self.rank == 0): self.printlog("Successfully copied radiation tables to GPU memory.")
pyc2ray/c2ray_base.py:    def _gpu_close(self):
pyc2ray/c2ray_base.py:        """ Deallocate GPU memory
pyc2ray/visualization/tomography.py:        self.ax2.set_title(f"Ionization Rate, OCTA GPU",fontsize=12)
pyc2ray/raytracing.py:from .asora_core import cuda_is_init
pyc2ray/raytracing.py:# the ASORA library on the GPU.
pyc2ray/raytracing.py:# data is moved between the CPU and the GPU (this is a big bottleneck).
pyc2ray/raytracing.py:# over the run of a simulation, need to be copied separately to the GPU
pyc2ray/raytracing.py:        use_gpu,max_subbox,subboxsize,loss_fraction,
pyc2ray/raytracing.py:    # Allow a call with GPU only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
pyc2ray/raytracing.py:    if (use_gpu and not cuda_is_init()):
pyc2ray/raytracing.py:        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")
pyc2ray/raytracing.py:    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
pyc2ray/raytracing.py:    if use_gpu:
pyc2ray/raytracing.py:        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
pyc2ray/raytracing.py:        # Copy positions & fluxes of sources to the GPU in advance
pyc2ray/raytracing.py:        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
pyc2ray/raytracing.py:    # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
pyc2ray/raytracing.py:    if not use_gpu:
pyc2ray/raytracing.py:    if use_gpu:
pyc2ray/raytracing.py:        # Use GPU raytracing
pyc2ray/raytracing.py:    # Since chemistry (ODE solving) is done on the CPU in Fortran, flattened CUDA arrays need to be reshaped
pyc2ray/raytracing.py:    if use_gpu:
pyc2ray/raytracing.py:    if (stats and not use_gpu):
pyc2ray/raytracing.py:        r_RT,use_gpu,max_subbox,loss_fraction,
pyc2ray/raytracing.py:    Warning: Calling this function with use_gpu = True assumes that the radiation
pyc2ray/raytracing.py:    tables have previously been copied to the GPU using photo_table_to_device()
pyc2ray/raytracing.py:        * When using GPU (octahedral) RT with ASORA, this sets the size of the octahedron such that a sphere of
pyc2ray/raytracing.py:    use_gpu : bool
pyc2ray/raytracing.py:        Whether or not to use the GPU-accelerated ASORA library for raytracing.
pyc2ray/raytracing.py:        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true
pyc2ray/raytracing.py:        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true
pyc2ray/raytracing.py:        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
pyc2ray/raytracing.py:        when using GPU RT. Default is False
pyc2ray/raytracing.py:    nsubbox : int, only when stats=True and use_gpu=False
pyc2ray/raytracing.py:    photonloss : float, only when stats=True and use_gpu=False
pyc2ray/raytracing.py:     # Allow a call with GPU only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
pyc2ray/raytracing.py:    if (use_gpu and not cuda_is_init()):
pyc2ray/raytracing.py:        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")
pyc2ray/raytracing.py:    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
pyc2ray/raytracing.py:    if use_gpu:
pyc2ray/raytracing.py:        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
pyc2ray/raytracing.py:        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
pyc2ray/raytracing.py:    # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
pyc2ray/raytracing.py:    if use_gpu:
pyc2ray/raytracing.py:        # Use GPU raytracing
pyc2ray/raytracing.py:    if (stats and not use_gpu):
pyc2ray/c2ray_test.py:    def __init__(self, paramfile, Nmesh, use_gpu, use_mpi=None):
pyc2ray/c2ray_test.py:        use_gpu : bool
pyc2ray/c2ray_test.py:            Whether to use the GPU-accelerated ASORA library for raytracing
pyc2ray/c2ray_test.py:        super().__init__(paramfile, Nmesh, use_gpu, use_mpi)
pyc2ray/c2ray_cubep3m.py:    def __init__(self, paramfile, Nmesh, use_gpu):
pyc2ray/c2ray_cubep3m.py:        use_gpu : bool
pyc2ray/c2ray_cubep3m.py:            Whether to use the GPU-accelerated ASORA library for raytracing
pyc2ray/c2ray_cubep3m.py:        super().__init__(paramfile, Nmesh, use_gpu)
pyc2ray/asora_core.py:# GPU memory has been allocated when GPU-accelerated functions are called.
pyc2ray/asora_core.py:__all__ = ['cuda_is_init','device_init','device_close','photo_table_to_device']
pyc2ray/asora_core.py:# This flag indicates whether GPU memory has been correctly allocated before calling any methods.
pyc2ray/asora_core.py:cuda_init = False
pyc2ray/asora_core.py:def cuda_is_init():
pyc2ray/asora_core.py:    global cuda_init
pyc2ray/asora_core.py:    return cuda_init
pyc2ray/asora_core.py:    """Initialize GPU and allocate memory for grid data
pyc2ray/asora_core.py:        Number of sources the GPU handles in parallel. Increasing this parameter
pyc2ray/asora_core.py:    global cuda_init
pyc2ray/asora_core.py:        cuda_init = True
pyc2ray/asora_core.py:        raise RuntimeError("Could not initialize GPU: ASORA library not loaded")
pyc2ray/asora_core.py:    """Deallocate GPU memory
pyc2ray/asora_core.py:    global cuda_init
pyc2ray/asora_core.py:    if cuda_init:
pyc2ray/asora_core.py:        cuda_init = False
pyc2ray/asora_core.py:        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")
pyc2ray/asora_core.py:    """Copy radiation tables to GPU (optically thin & thick tables)
pyc2ray/asora_core.py:    global cuda_init
pyc2ray/asora_core.py:    if cuda_init:
pyc2ray/asora_core.py:        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)") 
pyc2ray/load_extensions.py:# Centralized place to load Fortran and C++/CUDA extensions
pyc2ray/load_extensions.py:            # If asora is not found, the package can still be used but we inform the user that GPU raytracing is not available
pyc2ray/evolve.py:from .asora_core import cuda_is_init
pyc2ray/evolve.py:# which runs using the ASORA library on the GPU.
pyc2ray/evolve.py:# data is moved between the CPU and the GPU (this is a big bottleneck).
pyc2ray/evolve.py:# over the run of a simulation, need to be copied separately to the GPU
pyc2ray/evolve.py:# This file defines two variants of evolve3D: The reference, single-gpu
pyc2ray/evolve.py:# version, and a MPI version which enables usage on multiple GPU nodes.
pyc2ray/evolve.py:        use_gpu,max_subbox,subboxsize,loss_fraction,
pyc2ray/evolve.py:    Warning: Calling this function with use_gpu = True assumes that the radiation
pyc2ray/evolve.py:    tables have previously been copied to the GPU using photo_table_to_device()
pyc2ray/evolve.py:    use_gpu : bool
pyc2ray/evolve.py:        Whether or not to use the GPU-accelerated ASORA library for raytracing.
pyc2ray/evolve.py:        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true
pyc2ray/evolve.py:        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true
pyc2ray/evolve.py:        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
pyc2ray/evolve.py:    # Allow a call with GPU only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
pyc2ray/evolve.py:    if (use_gpu and not cuda_is_init()):
pyc2ray/evolve.py:        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")
pyc2ray/evolve.py:    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
pyc2ray/evolve.py:    if use_gpu:
pyc2ray/evolve.py:        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
pyc2ray/evolve.py:        # Copy positions & fluxes of sources to the GPU in advance
pyc2ray/evolve.py:        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
pyc2ray/evolve.py:        # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
pyc2ray/evolve.py:        if not use_gpu:
pyc2ray/evolve.py:        if use_gpu:
pyc2ray/evolve.py:            # Use GPU raytracing
pyc2ray/evolve.py:        # Since chemistry (ODE solving) is done on the CPU in Fortran, flattened CUDA arrays need to be reshaped
pyc2ray/evolve.py:        if use_gpu:
pyc2ray/evolve.py:        # Finally, when using GPU, need to reshape x back for the next ASORA call
pyc2ray/evolve.py:        if (use_gpu and not converged):
pyc2ray/evolve.py:                use_gpu,max_subbox,subboxsize,loss_fraction,
pyc2ray/evolve.py:    Warning: Calling this function with use_gpu = True assumes that the radiation
pyc2ray/evolve.py:    tables have previously been copied to the GPU using photo_table_to_device()
pyc2ray/evolve.py:        * When using GPU (octahedral) RT with ASORA, this sets the size of the octahedron such that a sphere of
pyc2ray/evolve.py:    use_gpu : bool
pyc2ray/evolve.py:        Whether or not to use the GPU-accelerated ASORA library for raytracing.
pyc2ray/evolve.py:        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true
pyc2ray/evolve.py:        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true
pyc2ray/evolve.py:        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
pyc2ray/evolve.py:    # Allow a call with GPU only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
pyc2ray/evolve.py:    if (use_gpu and not cuda_is_init()):
pyc2ray/evolve.py:        raise RuntimeError("GPU not initialized. Please initialize it by calling device_init(N)")
pyc2ray/evolve.py:    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
pyc2ray/evolve.py:    if use_gpu:
pyc2ray/evolve.py:        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
pyc2ray/evolve.py:        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
pyc2ray/evolve.py:        # Copy positions & fluxes of sources to the GPU in advance
pyc2ray/evolve.py:        # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
pyc2ray/evolve.py:        if not use_gpu:
pyc2ray/evolve.py:        if use_gpu:
pyc2ray/evolve.py:            # Use GPU raytracing
pyc2ray/evolve.py:        # Since chemistry (ODE solving) is done on the CPU in Fortran, flattened CUDA arrays need to be reshaped
pyc2ray/evolve.py:        if use_gpu:
pyc2ray/evolve.py:            # Finally, when using GPU, need to reshape x back for the next ASORA call
pyc2ray/evolve.py:            if (use_gpu and not converged):
pyc2ray/utils/sourceutils.py:    """Convert source data to correct shape & data type for GPU extension module
pyc2ray/README:- libocta: CUDA C++ implementation of the OCTA raytracing method, compiled with nvcc. ../src/octa/
pyc2ray/c2ray_244paper.py:    def __init__(self,paramfile,Nmesh,use_gpu):
pyc2ray/c2ray_244paper.py:        use_gpu : bool
pyc2ray/c2ray_244paper.py:            Whether to use the GPU-accelerated ASORA library for raytracing
pyc2ray/c2ray_244paper.py:        super().__init__(paramfile, Nmesh, use_gpu)
src/asora/python_module.cu:        do_all_sources_gpu(R,coldensh_out_data,sig,dr,ndens_data,xh_av_data,phi_ion_data,NumSrc,m1,minlogtau,dlogtau,NumTau);
src/asora/python_module.cu:    // Allocate GPU memory for grid data
src/asora/python_module.cu:    // Deallocate GPU memory
src/asora/python_module.cu:    // Copy density grid to GPU
src/asora/python_module.cu:    // Copy radiation table to GPU
src/asora/python_module.cu:    // Copy source data to GPU
src/asora/python_module.cu:        {"do_all_sources",  asora_do_all_sources, METH_VARARGS,"Do OCTA raytracing (GPU)"},
src/asora/python_module.cu:        {"device_init",  asora_device_init, METH_VARARGS,"Free GPU memory"},
src/asora/python_module.cu:        {"device_close",  asora_device_close, METH_VARARGS,"Free GPU memory"},
src/asora/python_module.cu:        {"density_to_device",  asora_density_to_device, METH_VARARGS,"Copy density field to GPU"},
src/asora/python_module.cu:        {"photo_table_to_device",  asora_photo_table_to_device, METH_VARARGS,"Copy radiation table to GPU"},
src/asora/python_module.cu:        {"source_data_to_device",  asora_source_data_to_device, METH_VARARGS,"Copy radiation table to GPU"},
src/asora/python_module.cu:        "CUDA C++ implementation of the short-characteristics RT", /* module documentation, may be NULL */
src/asora/raytracing.cu:#define CUDA_BLOCK_SIZE 256                         // Size of blocks used to treat sources
src/asora/raytracing.cu:inline __device__ int modulo_gpu(const int & a,const int & b) { return (a%b+b)%b; }
src/asora/raytracing.cu:inline __device__ int sign_gpu(const double & x) { if (x>=0) return 1; else return -1;}
src/asora/raytracing.cu:inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N) { return N*N*i + N*j + k;}
src/asora/raytracing.cu:// Weight function for C2Ray interpolation function (see cinterp_gpu below)
src/asora/raytracing.cu:__device__ inline double weightf_gpu(const double & cd, const double & sig) { return 1.0/max(0.6,cd*sig);}
src/asora/raytracing.cu:// When using a GPU with compute capability < 6.0, we must manually define the atomicAdd function for doubles
src/asora/raytracing.cu:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
src/asora/raytracing.cu:void do_all_sources_gpu(
src/asora/raytracing.cu:        // CUDA Grid size: since 1 block = 1 source, this sets the number of sources treated in parallel
src/asora/raytracing.cu:        // CUDA Block size: more of a tuning parameter (see above), in practice anything ~128 is fine
src/asora/raytracing.cu:        dim3 bs(CUDA_BLOCK_SIZE);
src/asora/raytracing.cu:        cudaMemset(phi_dev,0,meshsize);
src/asora/raytracing.cu:        // cudaMemcpy(n_dev,ndens,meshsize,cudaMemcpyHostToDevice);  < --- !! density array is not modified, asora assumes that it has been copied to the device before
src/asora/raytracing.cu:        cudaMemcpy(x_dev,xh_av,meshsize,cudaMemcpyHostToDevice);
src/asora/raytracing.cu:            evolve0D_gpu<<<gs,bs>>>(R,max_q,ns,NumSrc,NUM_SRC_PAR,src_pos_dev,src_flux_dev,cdh_dev,
src/asora/raytracing.cu:            auto error = cudaGetLastError();
src/asora/raytracing.cu:            if(error != cudaSuccess) {
src/asora/raytracing.cu:                                        + std::string(cudaGetErrorName(error)) + " - "
src/asora/raytracing.cu:                                        + std::string(cudaGetErrorString(error)));
src/asora/raytracing.cu:            cudaDeviceSynchronize();
src/asora/raytracing.cu:        auto error = cudaMemcpy(phi_ion,phi_dev,meshsize,cudaMemcpyDeviceToHost);
src/asora/raytracing.cu:__global__ void evolve0D_gpu(
src/asora/raytracing.cu:                        if (in_box_gpu(i,j,k,m1))
src/asora/raytracing.cu:                            pos[0] = modulo_gpu(i,m1);
src/asora/raytracing.cu:                            pos[1] = modulo_gpu(j,m1);
src/asora/raytracing.cu:                            pos[2] = modulo_gpu(k,m1);
src/asora/raytracing.cu:                            xh_av_p = xh_av[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
src/asora/raytracing.cu:                            nHI_p = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);
src/asora/raytracing.cu:                                cinterp_gpu(i,j,k,i0,j0,k0,coldensh_in,path,coldensh_out + cdh_offset,sig,m1);
src/asora/raytracing.cu:                            coldensh_out[cdh_offset + mem_offst_gpu(pos[0],pos[1],pos[2],m1)] = cdho;
src/asora/raytracing.cu:                                double phi = photoion_rates_test_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig);
src/asora/raytracing.cu:                                double phi = photoion_rates_gpu(strength,coldensh_in,cdho,vol_ph,sig,photo_thin_table,photo_thick_table,minlogtau,dlogtau,NumTau);
src/asora/raytracing.cu:                                atomicAdd(phi_ion + mem_offst_gpu(pos[0],pos[1],pos[2],m1),phi);
src/asora/raytracing.cu:__device__ void cinterp_gpu(
src/asora/raytracing.cu:    sgni=sign_gpu(idel);
src/asora/raytracing.cu:    sgnj=sign_gpu(jdel);
src/asora/raytracing.cu:    sgnk=sign_gpu(kdel);
src/asora/raytracing.cu:        ip  = modulo_gpu(i  ,m1);
src/asora/raytracing.cu:        imp = modulo_gpu(im ,m1);
src/asora/raytracing.cu:        jp  = modulo_gpu(j  ,m1);
src/asora/raytracing.cu:        jmp = modulo_gpu(jm ,m1);
src/asora/raytracing.cu:        kmp = modulo_gpu(km ,m1);
src/asora/raytracing.cu:        c1=     coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
src/asora/raytracing.cu:        c2=     coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
src/asora/raytracing.cu:        c3=     coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
src/asora/raytracing.cu:        c4=     coldensh_out[mem_offst_gpu(ip,jp,kmp,m1)];
src/asora/raytracing.cu:        w1=   s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w2=   s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w3=   s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w4=   s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        ip  = modulo_gpu(i,m1);
src/asora/raytracing.cu:        imp = modulo_gpu(im,m1);
src/asora/raytracing.cu:        jmp = modulo_gpu(jm,m1);
src/asora/raytracing.cu:        kp  = modulo_gpu(k,m1);
src/asora/raytracing.cu:        kmp = modulo_gpu(km,m1);
src/asora/raytracing.cu:        c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
src/asora/raytracing.cu:        c2=  coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
src/asora/raytracing.cu:        c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
src/asora/raytracing.cu:        c4=  coldensh_out[mem_offst_gpu(ip,jmp,kp,m1)];
src/asora/raytracing.cu:        w1=s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w2=s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w3=s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w4=s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        imp=modulo_gpu(im ,m1);
src/asora/raytracing.cu:        jp= modulo_gpu(j  ,m1);
src/asora/raytracing.cu:        jmp=modulo_gpu(jm ,m1);
src/asora/raytracing.cu:        kp= modulo_gpu(k  ,m1);
src/asora/raytracing.cu:        kmp=modulo_gpu(km ,m1);
src/asora/raytracing.cu:        c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
src/asora/raytracing.cu:        c2=  coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
src/asora/raytracing.cu:        c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
src/asora/raytracing.cu:        c4=  coldensh_out[mem_offst_gpu(imp,jp,kp,m1)];
src/asora/raytracing.cu:        w1   =s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w2   =s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w3   =s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w4   =s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:__device__ void cinterp_gpu2(
src/asora/raytracing.cu:    sgni=sign_gpu(is);
src/asora/raytracing.cu:    sgnj=sign_gpu(js);
src/asora/raytracing.cu:    sgnk=sign_gpu(ks);
src/asora/raytracing.cu:        // c1=     coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
src/asora/raytracing.cu:        // c2=     coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
src/asora/raytracing.cu:        // c3=     coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
src/asora/raytracing.cu:        // c4=     coldensh_out[mem_offst_gpu(ip,jp,kmp,m1)];
src/asora/raytracing.cu:        w1 = s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w2 = s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w3 = s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w4 = s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        // c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
src/asora/raytracing.cu:        // c2=  coldensh_out[mem_offst_gpu(ip,jmp,kmp,m1)];
src/asora/raytracing.cu:        // c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
src/asora/raytracing.cu:        // c4=  coldensh_out[mem_offst_gpu(ip,jmp,kp,m1)];
src/asora/raytracing.cu:        w1=s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w2=s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w3=s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w4=s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        // c1=  coldensh_out[mem_offst_gpu(imp,jmp,kmp,m1)];
src/asora/raytracing.cu:        // c2=  coldensh_out[mem_offst_gpu(imp,jp,kmp,m1)];
src/asora/raytracing.cu:        // c3=  coldensh_out[mem_offst_gpu(imp,jmp,kp,m1)];
src/asora/raytracing.cu:        // c4=  coldensh_out[mem_offst_gpu(imp,jp,kp,m1)];
src/asora/raytracing.cu:        w1   =s1*weightf_gpu(c1,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w2   =s2*weightf_gpu(c2,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w3   =s3*weightf_gpu(c3,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        w4   =s4*weightf_gpu(c4,sigma_HI_at_ion_freq);
src/asora/raytracing.cu:        pos[0] = modulo_gpu(i,m1);
src/asora/raytracing.cu:        pos[1] = modulo_gpu(j,m1);
src/asora/raytracing.cu:        pos[2] = modulo_gpu(k,m1);
src/asora/raytracing.cu:        double cdh_in = coldensh_in[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
src/asora/raytracing.cu:        double nHI = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
src/asora/raytracing.cu:            vol_ph = dist2 * path[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
src/asora/raytracing.cu:        cdh_out = cdh_in + path[mem_offst_gpu(pos[0],pos[1],pos[2],m1)]*nHI;
src/asora/raytracing.cu:        phi = photoion_rates_test_gpu(strength,cdh_in,cdh_out,vol_ph,sig);
src/asora/raytracing.cu:        phi_ion[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi / nHI;
src/asora/raytracing.cu:__global__ void evolve0D_gpu_old(
src/asora/raytracing.cu:        if (in_box_gpu(i,j,k,m1))
src/asora/raytracing.cu:            pos[0] = modulo_gpu(i,m1);
src/asora/raytracing.cu:            pos[1] = modulo_gpu(j,m1);
src/asora/raytracing.cu:            pos[2] = modulo_gpu(k,m1);
src/asora/raytracing.cu:            xh_av_p = xh_av[mem_offst_gpu(pos[0],pos[1],pos[2],m1)];
src/asora/raytracing.cu:            nHI_p = ndens[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] * (1.0 - xh_av_p);
src/asora/raytracing.cu:            if (coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] == 0.0)
src/asora/raytracing.cu:                    cinterp_gpu(i,j,k,i0,j0,k0,coldensh_in,path,coldensh_out,sig,m1);
src/asora/raytracing.cu:                coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] = coldensh_in + nHI_p * path;
src/asora/raytracing.cu:                    double phi = photoion_rates_test_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig);
src/asora/raytracing.cu:                    double phi = photoion_rates_gpu(strength,coldensh_in,coldensh_out[mem_offst_gpu(pos[0],pos[1],pos[2],m1)],vol_ph,sig,photo_table,photo_table,minlogtau,dlogtau,NumTau);
src/asora/raytracing.cu:                    phi_ion[mem_offst_gpu(pos[0],pos[1],pos[2],m1)] += phi;
src/asora/make_patrick:CUDAFLAGS+= -std=c++14 -O2  -Xcompiler -fPIC -D PERIODIC -D LOCALRATES -rdc true #--gpu-architecture=sm_60 #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
src/asora/make_patrick:	$(NVCC) $(CUDAFLAGS) $(INC) -shared -o libasora.so memory.o rates.o raytracing.o python_module.o
src/asora/make_patrick:	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing.cu
src/asora/make_patrick:	$(NVCC) $(CUDAFLAGS) $(INC) -dc rates.cu
src/asora/make_patrick:	$(NVCC) $(CUDAFLAGS) $(INC) -dc memory.cu
src/asora/make_patrick:	$(NVCC) $(CUDAFLAGS) $(INC) -c python_module.cu
src/asora/make_michele:CUDAFLAGS+=-std=c++14 -O2  -Xcompiler -fPIC -D PERIODIC -D LOCALRATES --gpu-architecture=sm_60 #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
src/asora/make_michele:	$(NVCC) $(CUDAFLAGS) $(INC) -shared -o libasora.so memory.o rates.o raytracing.o python_module.o
src/asora/make_michele:	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing.cu
src/asora/make_michele:	$(NVCC) $(CUDAFLAGS) $(INC) -dc rates.cu
src/asora/make_michele:	$(NVCC) $(CUDAFLAGS) $(INC) -dc memory.cu
src/asora/make_michele:	$(NVCC) $(CUDAFLAGS) $(INC) -c python_module.cu
src/asora/rates.cu:__device__ double photoion_rates_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig,
src/asora/rates.cu:__device__ double photoion_rates_test_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig)
src/asora/Makefile:CUDAFLAGS+=-std=c++14 -O2  -Xcompiler -fPIC -D PERIODIC -D LOCALRATES --gpu-architecture=sm_60 #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
src/asora/Makefile:	$(NVCC) $(CUDAFLAGS) $(INC) -shared -o libasora.so memory.o rates.o raytracing.o python_module.o
src/asora/Makefile:	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing.cu
src/asora/Makefile:	$(NVCC) $(CUDAFLAGS) $(INC) -dc rates.cu
src/asora/Makefile:	$(NVCC) $(CUDAFLAGS) $(INC) -dc memory.cu
src/asora/Makefile:	$(NVCC) $(CUDAFLAGS) $(INC) -c python_module.cu
src/asora/raytracing.cuh:#include <cuda_runtime.h>
src/asora/raytracing.cuh:// Functions defined and documented in raytracing_gpu.cu
src/asora/raytracing.cuh:inline __device__ int modulo_gpu(const int & a,const int & b);
src/asora/raytracing.cuh:inline __device__ int sign_gpu(const double & x);
src/asora/raytracing.cuh:inline __device__ int mem_offst_gpu(const int & i,const int & j,const int & k,const int & N);
src/asora/raytracing.cuh:void do_all_sources_gpu(
src/asora/raytracing.cuh:__global__ void evolve0D_gpu(
src/asora/raytracing.cuh:__device__ void cinterp_gpu(
src/asora/raytracing.cuh:inline __device__ bool in_box_gpu(const int & i,const int & j,const int & k,const int & N)
src/asora/memory.cu:// Global variables. Pointers to GPU memory to store grid data
src/asora/memory.cu:    cudaDeviceProp device_prop;
src/asora/memory.cu:    cudaGetDevice(&dev_id);
src/asora/memory.cu:    cudaGetDeviceProperties(&device_prop, dev_id);
src/asora/memory.cu:    if (device_prop.computeMode == cudaComputeModeProhibited) {
src/asora/memory.cu:                    "threads can use ::cudaSetDevice()"
src/asora/memory.cu:    cudaError_t error = cudaGetLastError();
src/asora/memory.cu:    if (error != cudaSuccess) {
src/asora/memory.cu:        std::cout << "cudaGetDeviceProperties returned error code " << error
src/asora/memory.cu:        std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
src/asora/memory.cu:    // Set the source batch size, i.e. the number of sources done in parallel (on the same GPU)
src/asora/memory.cu:    cudaMalloc(&cdh_dev,NUM_SRC_PAR * bytesize);
src/asora/memory.cu:    cudaMalloc(&n_dev,bytesize);
src/asora/memory.cu:    cudaMalloc(&x_dev,bytesize);
src/asora/memory.cu:    cudaMalloc(&phi_dev,bytesize);
src/asora/memory.cu:    error = cudaGetLastError();
src/asora/memory.cu:    if (error != cudaSuccess) {
src/asora/memory.cu:            + std::string(cudaGetErrorName(error)) + " - "
src/asora/memory.cu:            + std::string(cudaGetErrorString(error)));
src/asora/memory.cu:    cudaMemcpy(n_dev,ndens,N*N*N*sizeof(double),cudaMemcpyHostToDevice);
src/asora/memory.cu:    cudaMalloc(&photo_thin_table_dev,NumTau*sizeof(double));
src/asora/memory.cu:    cudaMemcpy(photo_thin_table_dev,thin_table,NumTau*sizeof(double),cudaMemcpyHostToDevice);
src/asora/memory.cu:    cudaMalloc(&photo_thick_table_dev,NumTau*sizeof(double));
src/asora/memory.cu:    cudaMemcpy(photo_thick_table_dev,thick_table,NumTau*sizeof(double),cudaMemcpyHostToDevice);
src/asora/memory.cu:    cudaFree(src_pos_dev);
src/asora/memory.cu:    cudaFree(src_flux_dev);
src/asora/memory.cu:    cudaMalloc(&src_pos_dev,3*NumSrc*sizeof(int));
src/asora/memory.cu:    cudaMalloc(&src_flux_dev,NumSrc*sizeof(double));
src/asora/memory.cu:    cudaMemcpy(src_pos_dev,pos,3*NumSrc*sizeof(int),cudaMemcpyHostToDevice);
src/asora/memory.cu:    cudaMemcpy(src_flux_dev,flux,NumSrc*sizeof(double),cudaMemcpyHostToDevice);
src/asora/memory.cu:    cudaFree(cdh_dev);
src/asora/memory.cu:    cudaFree(n_dev);
src/asora/memory.cu:    cudaFree(x_dev);
src/asora/memory.cu:    cudaFree(phi_dev);
src/asora/memory.cu:    cudaFree(photo_thin_table_dev);
src/asora/memory.cu:    cudaFree(src_pos_dev);
src/asora/memory.cu:    cudaFree(src_flux_dev);
src/asora/rates.cuh:#include <cuda_runtime.h>
src/asora/rates.cuh:__device__ double photoion_rates_gpu(const double & strength,const double & coldens_in,const double & coldens_out,
src/asora/rates.cuh:__device__ double photoion_rates_test_gpu(const double & strength,const double & coldens_in,const double & coldens_out,const double & Vfact,const double & sig);

```
