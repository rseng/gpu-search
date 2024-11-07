# https://github.com/lorenzo-rovigatti/oxDNA

```console
setup.py:        ('cuda', None, 'enable CUDA support')
setup.py:        self.cuda = None
setup.py:        if self.cuda is not None:
setup.py:            cmake_args += ['-DCUDA=On']
breaking_changes.md:* On the CUDA side, `number` has become `c_number` and `number4` has become `c_number4`. Both are set at compile time to `float` and `float4` or `double` and `LR_double4`. If the CUDA backend is compiled in `float` precision (*i.e.* if `-DCUDA_DOUBLE=OFF`, default behaviour), the `mixed` precision is available and can be enabled by putting in the input file `backend_precision=mixed`. The `backend_precision` option is otherwise not used any more.
setup.cfg:#cuda = True
docs/source/relaxation.md:With oxDNA, a standard relaxation procedure starts with running short ($10^2$ to $10^4$ steps) Monte Carlo simulations on a CPU, followed by order of $10^6$ steps of an MD relaxation with a maximum-value of the cutoff for the backbone potential (ideally on CUDA, as simulations of large structures are prohibitively slow on CPU).
docs/source/relaxation.md:but this will depend on the extent of the displacements required. To deal with the potentially large forces, it is also advantageous to increase the damping compared to a standard oxDNA MD simulation (by reducing the `diff_coeff` parameter). For large DNA structures, this more computationally demanding step is ideally performed on GPU-based architectures (CUDA).
docs/source/input.md:* `[reload_from = <string>]`: checkpoint to reload from. This option is incompatible with the keys `conf_file` and `seed`, and requires `restart_step_counter = false` as well as `binary_initial_conf = true`. Note that this option is incompatible with `backend = CUDA`.
docs/source/input.md:* `backend = CPU|CUDA`: MD simulations can be run either on single CPU cores or on single CUDA-enabled GPUs.
docs/source/input.md:* `backend_precision = <any>`: by default CPU simulations are run with `double` precision, CUDA with `mixed` precision (see [here](https://doi.org/10.1002/jcc.23763) for details). The CUDA backend also supports single precision (`backend_precision = float`), but we do not recommend to use it. Optionally, [by using CMake switches](install.md#cmake-options) it is possible to run CPU simulations in single precision or CUDA simulations in double precision.
docs/source/input.md:* `[thermostat = no|refresh|brownian|langevin|DPD]`:  Select the simulation thermostat for MD simulations. `no` means constant-energy simulations.  `refresh` is the Anderson thermostat. `brownian` is an Anderson-like thermostat that refreshes momenta of randomly chosen particles. `langevin` implements a regular Langevin thermostat. `bussi` is the Bussi-Donadio-Parrinello thermostat. `DPD` is a Dissipative Particle Dynamics thermostat. The `no`, `brownian`, `langevin` and `bussi` thermostats are also available on CUDA. Defaults to `no`.
docs/source/input.md:## CUDA options
docs/source/input.md:The following options require `backend = CUDA`.
docs/source/input.md:* `[CUDA_list = no|verlet]`: neighbour lists for CUDA simulations. Defaults to `verlet`.
docs/source/input.md:* `[cells_auto_optimisation = <bool>`: increase the size of the cells used to build Verlet lists if the total number of cells exceeds two times the number of nucleotides. Sometimes disabling this option increases performance. Used only if `CUDA_list = verlet`, defaults to `true`.
docs/source/input.md:* `[CUDA_device = <int>]`: CUDA-enabled device to run the simulation on. If it is not specified or it is given a negative number, a suitable device will be automatically chosen.
docs/source/input.md:* `[CUDA_sort_every = <int>]`: sort particles according to a 3D Hilbert curve after the lists have been updated `CUDA_sort_every` times. This will greatly enhnance performances for some types of interaction. Defaults to `0`, which disables sorting.
docs/source/input.md:* `[threads_per_block = <int>]`: Number of threads per block on the CUDA grid. defaults to 2 * the size of a warp.
docs/source/input.md:* `[CUDA_avoid_cpu_calculations = <bool>]`: Do not run any computations on the CPU. If set to `true`, the energy will not be printed. It may speed up the simulation of very large systems. Defaults to `false`.
docs/source/input.md:* `[CUDA_barostat_always_refresh = <bool>]`: Refresh the momenta of all particles after a successful volume move. Used only if `use_barostat = true`, defaults to `false`.
docs/source/input.md:* `[CUDA_print_energy = <bool>]`: print the potential energy as computed on the GPU to the standard output. Useful for debugging purposes, since the "regular" potential energy is computed on the CPU.
docs/source/input.md:* `backend = CPU/CUDA`: FFS simulations can be run either on CPU or GPU. Note that, unlike the CPU implementation, the CUDA implementation does not print extra columns with the current order parameter values whenever the energy is printed.
docs/source/input.md:* `backend_precision = <any>/mixed`: CPU FFS may use any precision allowed for a normal CPU MD simulation, while CUDA FFS is currently only implemented for mixed precision.
docs/source/input.md:* `ffs_file = <string>`: path to the file with the simulation stopping conditions. Optionally, one may use `master conditions` (CUDA FFS only), which allow one to more easily handle very high dimensional order parameters. See the `EXAMPLES/CUDA_FFS/README` file for more information.
docs/source/input.md:* `[ffs_generate_flux = <bool>]`: **CUDA FFS only**. If `false`, the simulation will run until a stopping condition is reached; if `true`, a flux generation simulation will be run, in which case reaching a condition will cause a configuration to be saved but will not terminate the simulation. In the stopping condition file, the conditions must be labelled forward1, forward2, ... (for the forward conditions); and backward1, backward2, ... (for the backward conditions), ... instead of condition1, condition2, ... . To get standard flux generation, set the forward and backward conditions to correspond to crossing the same interface. As with the single shooting run mode, the name of the condition crossed will be printed to stderr each time. Defaults to `false`.
docs/source/input.md:* `[gen_flux_save_every = <integer>]`: **CUDA FFS only**. Save a configuration each time this number of forward crossings is achieved. Mandatory if `ffs_generate_flux = `true`.
docs/source/input.md:* `[gen_flux_total_crossings = <integer>]`: **CUDA FFS only**. Stop the simulation after this number of crossings is achieved. Mandatory if `ffs_generate_flux = `true`.
docs/source/input.md:* `[gen_flux_conf_prefix = <string>]`: **CUDA FFS only**. the prefix used for the file names of configurations corresponding to the saved forward crossings. Counting starts at zero so the 3rd crossing configuration will be saved as `MY_PREFIX_N2.dat`. Mandatory if `ffs_generate_flux = `true`.
docs/source/input.md:* `[gen_flux_debug = <bool>]`: **CUDA FFS only**. In a flux generation simulation, set to `true` to save backward-crossing configurations for debugging. Defaults to `false`.
docs/source/input.md:* `[check_initial_state = <bool>]`: **CUDA FFS only**. In a flux generation simulation, set to `true` to turn on initial state checking. In this mode an initial configuration that crosses the forward conditions after only 1 step will cause the code to complain and exit. Useful for checking that a flux generation simulation does not start out of the A-state. Defaults to `false`
docs/source/input.md:* `[die_on_unexpected_master = <bool>]`: **CUDA FFS only**. In a flux generation simulation that uses master conditions, set to true to cause the simulation to die if any master conditions except master_forward1 or master_backward1 are reached. Useful for checking that a flux generation simulation does not enter any unwanted free energy basins (i.e. other than the initial state and the desired final state). Defaults to `false`.
docs/source/input.md:* `[unexpected_master_prefix = <string>]`: **CUDA FFS only**. The prefix used for the file names of configurations corresponding to reaching any unexpected master conditions (see
docs/source/scaling.md:# Efficient GPU usage
docs/source/scaling.md:## Simulation performance on GPU classes
docs/source/scaling.md:As NVIDIA releases new GPU architectures, the performance of oxDNA generally increases.  On current hardware (June 2023), it is not uncommon to be able to run well-sampled equilibrium simulations of DNA origami structures within 1-2 days. The following graph compares the performace of NVIDIA v100 vs a100 scientific GPUs on oxDNA simulations of varying sizes.
docs/source/scaling.md:Compared with the AI applications that modern GPUs are optimized for, oxDNA simulations use relatively little memory.  This means that you can get significance increases in simulation throughput by activating [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html), an alternative CUDA API which allows multiple processes to share the same GPU.  If your cluster is MPS compatible (GPUs set to EXCLUSIVE_PROCESS, preferably Volta architecture or newer), using oxDNA is as simple as:
docs/source/scaling.md:nvidia-cuda-mps-control -d
docs/source/scaling.md:echo quit | nvidia-cuda-mps-control
docs/source/scaling.md:It is generally advisable to include the `sleep` between starting each oxDNA process as if they start too close together, the memory allocation on the GPU will occasionally fail.
docs/source/scaling.md:The following graph compares the performance of various numbers of MPS processes on 80GB v100 and a100 GPUs with various number of processes
docs/source/scaling.md:**NOTE** Some variation in the data was caused by a few jobs per-batch failing due to too many members in the neighbor list.  This can be avoided by adding the [max_density_multiplier](https://lorenzo-rovigatti.github.io/oxDNA/input.html#cuda-options) argument to the input files; however this increases memory usage and may cause the MPS runs with higher numbers of replicates to fail.
docs/source/scaling.md:**NOTE** The clusters used to test here limited the number of CPUs associated with each GPU to 20 (v100) and 16 (a100), this likely impacted the simulation efficiency for large numbers of processes.
docs/source/install.md:### CUDA
docs/source/install.md:Compiling with CUDA support requires CMake >= 3.5 and a CUDA toolkit >= 10. If your current setup cannot meet these requirements we advise you to use [older versions of oxDNA](https://sourceforge.net/projects/oxdna/files/).
docs/source/install.md:* `-DCUDA=ON` Enables CUDA support
docs/source/install.md:* `-DCUDA_COMMON_ARCH=ON` Choose the target CUDA compute architecture based on the nvcc version. Set it to off to autodetect the CUDA compute arch GPU installed.
docs/source/install.md:* `-DCUDA_DOUBLE=ON` Set the numerical precision of the CUDA backends to `double`, which is not compatible with the `mixed` precision.
docs/source/oxpy/index.md:my_input["backend"] = "CUDA"
docs/source/ffs.md:The provided Python scripts in `EXAMPLE/FFS_example` directory show how to setup FFS simulation to estimate the rate of melting of a duplex. If you want to use them to study a different system, you need to adapt the defintion of interfaces in the script and `input` files for oxDNA simulation accordingly. The Python scripts support multiple core execution, but requires all CPU cores to be present on a single machine. Hence, if you are submitting these scripts as part of a CPU cluster, all CPUs have to be physically on a single computer. Similarly, if you are using the GPU version of FFS, all used GPU cards have to be on the same node. We next provide description of use of FFS to estimate the rate of association of a DNA 8-mer.
docs/source/configurations.md:The CUDA backend supports base types whose absolute values do not exceed $2^{9} - 1 = 511$. In other words, base types larger than $511$ or smaller than $-511$ are not allowed.
docs/source/performance.md:|Hilbert sorting|MD on CUDA|Sort the particles to optimise data access on GPUs.Used only if [`CUDA_sort` is set](input.md#cuda-options)|
docs/source/performance.md:### GPU simulations
docs/source/performance.md:When running CUDA-powered simulations, the box size has a non-trivial effect on performance, and its interaction with other parameters such as `salt_concentration`, `verlet_skin`, and possibly others, make it hard to come up with a way of automatically set the best options for a given case.
docs/source/performance.md:Since there is no dynamic memory on GPUs, in order to avoid crashing simulations oxDNA sets the size of the cells used to build neighbouring lists so that their memory footprint is not too high. If you want to optimise performance is sometimes worth to set `cells_auto_optimisation = false` so that oxDNA uses the smallest possible cells (at the cost of memory consumption). If the resulting memory footprint can be handled by your GPU you'll probably see some (possibly large) performance gains.
docs/source/performance.md:There are some heuristics that attempt to limit the memory consumption of CUDA simulations. First of all, the given combination of parameters is used to evaluate the minimum size of the cells required to build neighbouring lists, $r_m$. In turn, $r_m$ is used to compute the number of cells along each coordinate $i$ (where $i = x, y, z$) as
docs/source/performance.md:On newer versions of oxDNA (> 3.6.1), setting `debug = true` will report in the log file (or on screen if `log_file` is not set) the amount of memory that is requested by each allocation on the GPU.
docs/source/index.md:oxDNA can perform both molecular dynamics (MD) and Monte Carlo (MC) simulations of the oxDNA and oxRNA models. MD simulations can be run on single CPUs or single CUDA-enabled GPUs, while MC simulations, which can only be run serially, can exploit the Virtual Move Monte Carlo algorithm to greatly speed-up equilibration and sampling, and [Umbrella Sampling biasing](umbrella_sampling.md) to efficiently obtain free-energy profiles. The package also features a [Forward-Flux Sampling interface](ffs.md) to study the kinetics of rare events, and makes it possible to alter the behaviour of the systems by adding [*external forces*](forces.md) that can be used, for instance, to pull on or apply torques to strands or confine nucleotides within semi-planes or spheres.
docs/source/index.md:- for the CUDA-powered code:
docs/outdated/coding_guidelines.txt:CUDA and MPI versions. 
docs/outdated/input_options.txt:        on CUDA. Defaults to 'no'.
docs/outdated/input_options.txt:CUDA options:
docs/outdated/input_options.txt:    [CUDA_list = no|verlet]
docs/outdated/input_options.txt:        Neighbour lists for CUDA simulations. Defaults to 'no'.
docs/outdated/input_options.txt:    backend = CUDA
docs/outdated/input_options.txt:        For CUDA FFS -- NB unlike the CPU implementation, the CUDA
docs/outdated/input_options.txt:        CUDA FFS is currently only implemented for mixed precision
docs/outdated/input_options.txt:        one may use 'master conditions' (CUDA FFS only), which allow one to
docs/outdated/input_options.txt:        EXAMPLES/CUDA_FFS/README file for more information
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; if False, the simulation will run until
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if ffs_generate_flux is True; save a
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if ffs_generate_flux is True; stop the
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if ffs_generate_flux is True; the prefix used
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; In a flux generation simulation, set to
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; in a flux generation simulation, set to
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; in a flux generation simulation that
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if die_on_unexpected_master is True; the
docs/outdated/input_options.txt:    [CUDA_device = <int>]
docs/outdated/input_options.txt:        CUDA-enabled device to run the simulation on. If it is not specified
docs/outdated/input_options.txt:    [CUDA_sort_every = <int>]
docs/outdated/input_options.txt:        sort particles according to a 3D Hilbert curve every CUDA_sort_every
docs/outdated/input_options.txt:        Number of threads per block on the CUDA grid. defaults to 2 * the size
docs/outdated/input_options.txt:    backend = CPU/CUDA
docs/outdated/input_options.txt:        For CPU FFS/For CUDA FFS -- NB unlike the CPU implementation, the CUDA
docs/outdated/input_options.txt:        simulation/CUDA FFS is currently only implemented for mixed precision
docs/outdated/input_options.txt:        one may use 'master conditions' (CUDA FFS only), which allow one to
docs/outdated/input_options.txt:        EXAMPLES/CUDA_FFS/README file for more information
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; if False, the simulation will run until
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if ffs_generate_flux is True; save a
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if ffs_generate_flux is True; stop the
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if ffs_generate_flux is True; the prefix used
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; In a flux generation simulation, set to
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; in a flux generation simulation, set to
docs/outdated/input_options.txt:        CUDA FFS only. Default: False; in a flux generation simulation that
docs/outdated/input_options.txt:        CUDA FFS only. Mandatory if die_on_unexpected_master is True; the
oxpy/OxpyManager.cpp:		Update the CPU data structures. Useful only when running CUDA simulations.
oxpy/OxpyManager.cpp:		This method copies simulation data from the GPU to the CPU, and as such calling it too often may severely decrease performance.
oxpy/pybind11/README.rst:6. NVCC (CUDA 11.0 tested in CI)
oxpy/pybind11/README.rst:7. NVIDIA PGI (20.9 tested in CI)
oxpy/pybind11/.pre-commit-config.yaml:    types_or: [c++, c, cuda]
oxpy/pybind11/tools/pybind11Common.cmake:      # instance, projects that include other types of source files like CUDA
oxpy/pybind11/tools/pybind11Tools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
oxpy/pybind11/tools/pybind11Tools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
oxpy/pybind11/tools/pybind11NewTools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
oxpy/pybind11/tools/pybind11NewTools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
oxpy/pybind11/include/pybind11/detail/common.h:// For CUDA, GCC7, GCC8:
oxpy/pybind11/include/pybind11/detail/common.h:// 1.7% for CUDA, -0.2% for GCC7, and 0.0% for GCC8 (using -DCMAKE_BUILD_TYPE=MinSizeRel,
oxpy/pybind11/include/pybind11/detail/common.h:    && (defined(__CUDACC__) || (defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)))
oxpy/pybind11/include/pybind11/cast.h:    // static_cast works around compiler error with MSVC 17 and CUDA 10.2
oxpy/pybind11/include/pybind11/numpy.h:#ifdef __CUDACC__
README.md:oxDNA is a simulation code that was initially conceived as an implementation of the coarse-grained DNA model introduced by [T. E. Ouldridge, J. P. K. Doye and A. A. Louis](http://dx.doi.org/10.1063/1.3552946). It has been since reworked and it is now an extensible simulation+analysis framework. It natively supports DNA, RNA, Lennard-Jones and patchy particle simulations of different kinds on both single CPU cores and NVIDIA GPUs.
README.md:**Q: Can oxDNA be run on multiple CPU cores or GPUs?**
README.md:**A:** No, oxDNA can run simulations on single cores or single GPUs only.
README.md:- for the CUDA-powered code:
README.md:oxDNA depends on a minimum number of external libraries (a c++-14-compliant standard library and NVIDIA's CUDA if the user wishes to enable it).
analysis/README.md:cmake -DPython=1 ..      #Also include -DCUDA=1 if compiling for GPU and `-DOxpySystemInstall=1` if on a personal machine or if using Python managed by Conda.
analysis/example_input_files/input_relax_MD:backend = CUDA
analysis/example_input_files/input_relax_MD:CUDA_list = verlet
analysis/example_input_files/input_relax_MD:CUDA_sort_every = 0
analysis/example_input_files/input_run:backend = CUDA
analysis/example_input_files/input_run:CUDA_list = verlet
analysis/example_input_files/input_run:CUDA_sort_every = 0
analysis/paper_examples/angles/input_23:backend = CUDA
analysis/paper_examples/angles/input_23:CUDA_list = verlet
analysis/paper_examples/angles/input_23:CUDA_sort_every = 0
analysis/paper_examples/angles/input_20:backend = CUDA
analysis/paper_examples/angles/input_20:CUDA_list = verlet
analysis/paper_examples/angles/input_20:CUDA_sort_every = 0
analysis/paper_examples/angles/README.md:   **A couple of notes on these simulations:** Both of these are large structures, and as such use the CUDA implementation of oxDNA.  In order to run these simulations you must be on a computer that has a GPU and the proper NVIDIA drivers. The input files here will run for 1e8 steps. For production runs we usually recommend 1e9 steps, however 1e8 will be fine for an example.
analysis/paper_examples/mds_mean/README.md:     This is a large structure, and as such use the CUDA implementation of oxDNA.  In order to run this simulation you must be on a computer that has a GPU and the proper NVIDIA drivers. The input file here will run for 1e8 steps.  For production runs we usually recommend 1e9 steps, however 1e8 will be fine for an example.
analysis/paper_examples/mds_mean/input_rna:backend = CUDA
analysis/paper_examples/mds_mean/input_rna:CUDA_list = verlet
analysis/paper_examples/mds_mean/input_rna:CUDA_sort_every = 0
analysis/paper_examples/README.md:The following examples run on the CUDA version of oxDNA which requires a UNIX computer with a GPU:
analysis/paper_examples/distances/README.md:     This is a large structure, and as such use the CUDA implementation of oxDNA.  In order to run this simulation you must be on a computer that has a GPU and the proper NVIDIA drivers.
analysis/paper_examples/distances/input_open:backend = CUDA
analysis/paper_examples/distances/input_open:CUDA_list = verlet
analysis/paper_examples/distances/input_open:CUDA_sort_every = 0
analysis/paper_examples/distances/input_open:CUDA_list = verlet
analysis/paper_examples/distances/input_open:CUDA_sort_every = 0
analysis/paper_examples/distances/input_closed:backend = CUDA
analysis/paper_examples/distances/input_closed:CUDA_list = verlet
analysis/paper_examples/distances/input_closed:CUDA_sort_every = 0
analysis/paper_examples/distances/input_closed:CUDA_list = verlet
analysis/paper_examples/distances/input_closed:CUDA_sort_every = 0
analysis/paper_examples/svd_mean/input_19:backend = CUDA
analysis/paper_examples/svd_mean/input_19:CUDA_list = verlet
analysis/paper_examples/svd_mean/input_19:CUDA_sort_every = 0
analysis/paper_examples/svd_mean/README.md:     Both of these are large structures, and as such use the CUDA implementation of oxDNA. In order to run these simulations you must be on a computer that has a GPU and the proper NVIDIA drivers. The input files here will run for 1e8 steps.  For production runs we usually recommend 1e9 steps, however 1e8 will be fine for an example. Both input files will write to the same output file names.  If you want to run both simulations, we recommend either copying them to their own directories or changing the ouput file names in the input file.
analysis/paper_examples/svd_mean/input_rna:backend = CUDA
analysis/paper_examples/svd_mean/input_rna:CUDA_list = verlet
analysis/paper_examples/svd_mean/input_rna:CUDA_sort_every = 0
analysis/src/oxDNA_analysis_tools/UTILS/boilerplate.py:    "backend" :"CUDA",
analysis/src/oxDNA_analysis_tools/UTILS/boilerplate.py:    "CUDA_list" :"verlet",
analysis/src/oxDNA_analysis_tools/UTILS/boilerplate.py:    "CUDA_sort_every" :"0",
CHANGELOG:	- Fix how the trap, twist and string forces behave on CUDA (see #71) [e92695c]
CHANGELOG:	- Remove a few warnings when compiling with CUDA
CHANGELOG:	- Fix the computation of the stress tensor on the GPU
CHANGELOG:	- Add the stress tensor calculation on GPUs for DNA
CHANGELOG:	- Improve the numerical stability of the CUDA RNA and DNA codes
CHANGELOG:	- Fix a bug whereby a DNA force was sometimes incorrectly evaluated on GPU
CHANGELOG:	- Fix two minor bugs in the CUDARNA interaction [0112a3e]
CHANGELOG:	- Add methods to sync data from and to CUDA interactions (see 7e437f63a8c551333b677d56edc20121551741da)
CHANGELOG:	- Fix a bug whereby the code wouldn't sometimes compile with CUDA 10.x
CHANGELOG:	- Update a CUDA error message
CHANGELOG:	- Improve error messages on CUDA
CHANGELOG:	- Fix a bug whereby RNA simulations with CUDA11 would crash
CHANGELOG:	- Fix a bug in the CUDA barostat that sometimes made the code crash
CHANGELOG:		- Fix a bug whereby oxpy simulations on CUDA couldn't be run consequently
CHANGELOG:	- Fix a bug by avoiding free'ing uninitialised CUDA arrays
CHANGELOG:	- Substitute a call to the deprecated cudaThreadSetCacheConfig function
CHANGELOG:	- Fix compilation with double precision on CUDA
CHANGELOG:	- Port the COMForce force to CUDA
CHANGELOG:	- Reduce the host memory consumption when using forces on CUDA
CHANGELOG:	- Make it possible to install a GPU-powered oxpy with pip
CHANGELOG:		- Fix a bug whereby a failing list update on CUDA would not throw an exception as it should
CHANGELOG:		- Fix compilation of oxpy and with CUDA
CHANGELOG:	- Make the code compatible with CUDA 11
CHANGELOG:		- Fix a bug whereby the Hilbert sorting done on GPU would lead to wrong results due to a wrong updating of the particle data structures
CHANGELOG:		- Fix a bug a CPU simulation run with a CUDA-enabled oxDNA would take up a GPU
CHANGELOG:		- Fix a small bug in VolumeMove and CUDA barostat
CHANGELOG:		* on the CUDA side `number` and `number4` have become `c_number` and `c_number4` and default to `float` and `float4`;
CHANGELOG:		* on CUDA the mixed precision can be used only if `-DCUDA_DOUBLE=OFF`, which is the default behaviour.
CHANGELOG:	- Made oxDNA compatible with CUDA 9.1 and remove a few subsequent warnings
CHANGELOG:	- Added a new option to cmake (CUDA_COMMON_ARCH) that is set to ON by default. If ON, this optional will make oxDNA be compiled for a list of architectures that depend on the installed CUDA version. This might result in a very slow compilation. Use -DCUDA_COMMON_ARCH=OFF to have cmake detect the installed GPU and compile only for its arch
CHANGELOG:	- Mutual traps now work in the same way on the CPU and GPU backends
CHANGELOG:	- oxDNA now supports non-cubic boxes. Both CPU and CUDA backends have this.
CHANGELOG:	use them. This cannot work on CUDA due to the operation order not being predictable.
CHANGELOG:	CPU and GPU) will not exceed the number of particles, thus avoiding running out of memory when using 
CHANGELOG:	- Added GPU support for the salt-dependent RNA model.
CHANGELOG:	- The Langevin thermostat may now be used on GPUs.
CHANGELOG:	GPUs. It seems working but we did not have thoroughly tested it. Use it at your own risk.
CMakeLists.txt:OPTION(CUDA "Set to ON to compile with CUDA support" OFF)
CMakeLists.txt:OPTION(CUDA_DOUBLE "Set the numerical precision for the CUDA backend to double" OFF)
CMakeLists.txt:OPTION(CUDA_COMMON_ARCH "Set to OFF to autodetect the GPU and compile for its architecture, set to ON (default) to compile for the most common architectures" ON)
CMakeLists.txt:	if(CUDA)
CMakeLists.txt:		MESSAGE(FATAL_ERROR "oxDNA with CUDA support cannot be compiled with the Intel compiler")
CMakeLists.txt:	ENDIF(CUDA)
contrib/rovigatti/CMakeLists.txt:IF(CUDA)
contrib/rovigatti/CMakeLists.txt:	find_package("CUDA")
contrib/rovigatti/CMakeLists.txt:	# same thing but for CUDA libs
contrib/rovigatti/CMakeLists.txt:	function(cuda_add_library_no_prefix target source)
contrib/rovigatti/CMakeLists.txt:		cuda_add_library(${target} MODULE EXCLUDE_FROM_ALL ${source} ${ARGN})
contrib/rovigatti/CMakeLists.txt:		target_link_libraries(${target} ${CUDA_LIBRARIES})
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDADetailedPatchySwapInteraction src/Interactions/CUDADetailedPatchySwapInteraction.cu src/Interactions/DetailedPatchySwapInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDAPatchySwapInteraction src/Interactions/CUDAPatchySwapInteraction.cu src/Interactions/PatchySwapInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDAFSInteraction src/Interactions/CUDAFSInteraction.cu src/Interactions/FSInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDAmWInteraction src/Interactions/CUDAmWInteraction.cu src/Interactions/mWInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDAStarrInteraction src/Interactions/CUDAStarrInteraction.cu src/Interactions/StarrInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDACPMixtureInteraction src/Interactions/CUDACPMixtureInteraction.cu src/Interactions/CPMixtureInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDAMGInteraction src/Interactions/CUDAMGInteraction.cu src/Interactions/MGInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDAAOInteraction src/Interactions/CUDAAOInteraction.cu src/Interactions/AOInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDAPolymerSwapInteraction src/Interactions/CUDAPolymerSwapInteraction.cu src/Interactions/PolymerSwapInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	cuda_add_library_no_prefix(CUDADetailedPolymerSwapInteraction src/Interactions/CUDADetailedPolymerSwapInteraction.cu src/Interactions/DetailedPolymerSwapInteraction.cpp)
contrib/rovigatti/CMakeLists.txt:	ADD_DEPENDENCIES(rovigatti CUDADetailedPolymerSwapInteraction CUDADetailedPatchySwapInteraction CUDAPatchySwapInteraction CUDAAOInteraction CUDAMGInteraction CUDACPMixtureInteraction CUDAFSInteraction CUDAmWInteraction CUDAStarrInteraction CUDAPolymerSwapInteraction)
contrib/rovigatti/CMakeLists.txt:ENDIF(CUDA)
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu: * CUDACPMixtureInteraction.cu
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:#include "CUDACPMixtureInteraction.h"
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:__global__ void cp_forces(c_number4 *poss, c_number4 *forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:__global__ void cp_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *c_number_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:CUDACPMixtureInteraction::CUDACPMixtureInteraction() {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:CUDACPMixtureInteraction::~CUDACPMixtureInteraction() {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:void CUDACPMixtureInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:void CUDACPMixtureInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n, &this->_n, 3 * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_int_type, &this->_CP_int_type, 3 * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:void CUDACPMixtureInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.cu:	CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu: * Cudafsinteraction.cu
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:#include "CUDAFSInteraction.h"
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:#define CUDA_MAX_FS_PATCHES 5
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:#define CUDA_MAX_FS_NEIGHS 20
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:/* BEGIN CUDA */
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__constant__ float4 MD_base_patches[2][CUDA_MAX_FS_PATCHES];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:struct __align__(16) CUDA_FS_bond {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:struct __align__(16) CUDA_FS_bond_list {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_FS_bond bonds[CUDA_MAX_FS_NEIGHS];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_FS_bond_list() :
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_FS_bond &new_bond() {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		if(n_bonds > CUDA_MAX_FS_NEIGHS) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__device__ void _patchy_two_body_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &a1, c_number4 &a2, c_number4 &a3, c_number4 &b1, c_number4 &b2, c_number4 &b3, c_number4 &F, c_number4 &torque, CUDA_FS_bond_list *bonds, int q_idx, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:			c_number dist = CUDA_DOT(patch_dist, patch_dist);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:					CUDA_FS_bond_list &bond_list = bonds[pi];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:					CUDA_FS_bond &my_bond = bond_list.new_bond();
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__device__ void _polymer_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, c_number4 &torque, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r) / MD_polymer_length_scale_sqr[0];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__device__ void _three_body(CUDA_FS_bond_list *bonds, c_number4 &F, c_number4 &T, c_number4 *forces, c_number4 *torques) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	for(int pi = 0; pi < CUDA_MAX_FS_PATCHES; pi++) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		CUDA_FS_bond_list &bond_list = bonds[pi];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:			CUDA_FS_bond &b1 = bond_list.bonds[bi];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:				CUDA_FS_bond &b2 = bond_list.bonds[bj];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__device__ void _fene(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r) / MD_polymer_length_scale_sqr[0];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__global__ void FS_bonded_forces(c_number4 *poss, c_number4 *forces, int *bonded_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__global__ void FS_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *three_body_forces, c_number4 *torques, c_number4 *three_body_torques, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	GPU_quat po = orientations[IND];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_FS_bond_list bonds[CUDA_MAX_FS_PATCHES];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:				GPU_quat qo = orientations[j];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:__global__ void FS_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *three_body_forces, c_number4 *torques, c_number4 *three_body_torques, int *matrix_neighs, int *c_number_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	GPU_quat po = orientations[IND];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_FS_bond_list bonds[CUDA_MAX_FS_PATCHES];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:			GPU_quat qo = orientations[k_index];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:/* END CUDA PART */
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:CUDAFSInteraction::CUDAFSInteraction() :
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:				CUDABaseInteraction(),
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:CUDAFSInteraction::~CUDAFSInteraction() {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	if(_d_three_body_forces != NULL) CUDA_SAFE_CALL(cudaFree(_d_three_body_forces));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	if(_d_three_body_torques != NULL) CUDA_SAFE_CALL(cudaFree(_d_three_body_torques));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_bonded_neighs));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:void CUDAFSInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	getInputInt(&inp, "CUDA_sort_every", &sort_every, 0);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:			throw oxDNAException("CUDAFSInteraction: Defective A-particles and CUDA_sort_every > 0 are incompatible");
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:void CUDAFSInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_forces, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_torques, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:			throw oxDNAException("CUDAFSInteraction does not support FS_polymer_alpha > 0");
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_bonded_neighs, n_elems * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:					throw oxDNAException("CUDAFSInteraction: particle %d has more than %d bonded neighbours", p->index, max_n_neighs);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpy(_d_bonded_neighs, h_bonded_neighs, n_elems * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_in_polymers, &_N_in_polymers, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_def_A, &_N_def_A, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_one_component, &_one_component, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_same_patches, &_same_patches, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_B_attraction, &_B_attraction, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patches, &_N_patches, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	if(!_one_component) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patches, &_N_patches_B, sizeof(int), sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	float4 base_patches[CUDA_MAX_FS_PATCHES];
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:			c_number factor = 0.5 / sqrt(CUDA_DOT(base_patches[j], base_patches[j]));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_base_patches, base_patches, sizeof(float4)*n_patches, i*sizeof(float4)*CUDA_MAX_FS_PATCHES));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	if(_N_patches > CUDA_MAX_FS_PATCHES) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		throw oxDNAException("The CUDAFSInteraction supports only particles with up to %d patches", CUDA_MAX_FS_PATCHES);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &_n_forces, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:void CUDAFSInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	int N = CUDABaseInteraction::_N;
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		if(_v_lists->use_edge()) throw oxDNAException("CUDAFSInteraction: use_edge is unsupported");
contrib/rovigatti/src/Interactions/CUDAFSInteraction.cu:		CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu: * CudaDetailedPatchySwapInteraction.cu
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:#include "CUDADetailedPatchySwapInteraction.h"
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:/* BEGIN CUDA */
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:__constant__ int MD_N_patches[CUDADetailedPatchySwapInteraction::MAX_SPECIES];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:__constant__ int MD_patch_types[CUDADetailedPatchySwapInteraction::MAX_SPECIES][CUDADetailedPatchySwapInteraction::MAX_PATCHES];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:struct __align__(16) CUDA_FS_bond {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:struct __align__(16) CUDA_FS_bond_list {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_FS_bond bonds[CUDADetailedPatchySwapInteraction::MAX_NEIGHS];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_FS_bond_list() :
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_FS_bond &new_bond() {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		if(n_bonds > CUDADetailedPatchySwapInteraction::MAX_NEIGHS) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		c_number4 &b2, c_number4 &b3, c_number4 &F, c_number4 &torque, CUDA_FS_bond_list *bonds, int q_idx, cudaTextureObject_t tex_patchy_eps,
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		cudaTextureObject_t tex_base_patches, CUDAStressTensor &p_st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		c_number4 p_base_patch = tex1Dfetch<c_number4>(tex_base_patches, p_patch + ptype * CUDADetailedPatchySwapInteraction::MAX_PATCHES);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:			c_number4 q_base_patch = tex1Dfetch<c_number4>(tex_base_patches, q_patch + qtype * CUDADetailedPatchySwapInteraction::MAX_PATCHES);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:			c_number dist_sqr = CUDA_DOT(patch_dist, patch_dist);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:						CUDA_FS_bond &my_bond = bonds[p_patch].new_bond();
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		c_number4 &b2, c_number4 &b3, c_number4 &F, c_number4 &torque, CUDA_FS_bond_list *bonds, int q_idx, cudaTextureObject_t tex_patchy_eps,
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		cudaTextureObject_t tex_base_patches, CUDAStressTensor &p_st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		c_number4 p_base_patch = tex1Dfetch<c_number4>(tex_base_patches, p_patch + ptype * CUDADetailedPatchySwapInteraction::MAX_PATCHES);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		c_number cospr = CUDA_DOT(p_patch_pos, r_versor);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:				c_number4 q_base_patch = tex1Dfetch<c_number4>(tex_base_patches, q_patch + qtype * CUDADetailedPatchySwapInteraction::MAX_PATCHES);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:				c_number cosqr = -CUDA_DOT(q_patch_pos, r_versor);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:							CUDA_FS_bond &my_bond = bonds[p_patch].new_bond();
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:__device__ void _three_body(CUDA_FS_bond_list *bonds, c_number4 &F, c_number4 &T, CUDAStressTensor &p_st, c_number4 *forces, c_number4 *torques) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	for(int pi = 0; pi < CUDADetailedPatchySwapInteraction::MAX_PATCHES; pi++) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_FS_bond_list &bond_list = bonds[pi];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:			CUDA_FS_bond &b1 = bond_list.bonds[bi];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:				CUDA_FS_bond &b2 = bond_list.bonds[bj];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:__global__ void DPS_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *three_body_forces,
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		c_number4 *torques, c_number4 *three_body_torques, int *matrix_neighs, int *number_neighs, cudaTextureObject_t tex_patchy_eps,
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		cudaTextureObject_t tex_base_patches, bool update_st, CUDAStressTensor *st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	GPU_quat po = orientations[IND];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_FS_bond_list bonds[CUDADetailedPatchySwapInteraction::MAX_PATCHES];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDAStressTensor p_st;
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:			GPU_quat qo = orientations[k_index];
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:/* END CUDA PART */
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:CUDADetailedPatchySwapInteraction::CUDADetailedPatchySwapInteraction() :
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:				CUDABaseInteraction(),
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:CUDADetailedPatchySwapInteraction::~CUDADetailedPatchySwapInteraction() {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_forces));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_torques));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_patchy_eps));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		cudaDestroyTextureObject(_tex_patchy_eps);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_base_patches));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		cudaDestroyTextureObject(_tex_base_patches);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:void CUDADetailedPatchySwapInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	getInputInt(&inp, "CUDA_sort_every", &sort_every, 0);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:void CUDADetailedPatchySwapInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_forces, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_torques, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_is_KF, &_is_KF, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_power, &_patch_power, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	// the following quantities are initialised by read_topology and hence have to be copied over to the GPU after its call
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patch_types, &_N_patch_types, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patches, _N_patches.data(), sizeof(int) * N_species));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_patchy_eps, _patchy_eps.size() * sizeof(float)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_patchy_eps, h_patchy_eps.data(), _patchy_eps.size() * sizeof(float), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	GpuUtils::init_texture_object(&_tex_patchy_eps, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat), _d_patchy_eps, _patchy_eps.size());
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:			throw oxDNAException("CUDADetailedPatchySwapInteraction: cannot simulate particles with more than %d patches. You can increase this number in the DetailedPatchySwapInteraction.h file", MAX_PATCHES);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_types, patch_types, sizeof(int) * n_patches, i * sizeof(int) * MAX_PATCHES));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_base_patches, N_base_patches * sizeof(float4)));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_base_patches, h_base_patches.data(), N_base_patches * sizeof(float4), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	GpuUtils::init_texture_object(&_tex_base_patches, cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat), _d_base_patches, h_base_patches.size());
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:void CUDADetailedPatchySwapInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:	int N = CUDABaseInteraction::_N;
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaMemset(_d_st, 0, N * sizeof(CUDAStressTensor)));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu: * CudaPatchySwapInteraction.cu
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:#include "CUDAPatchySwapInteraction.h"
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:/* BEGIN CUDA */
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:__constant__ int MD_N_patches[CUDAPatchySwapInteraction::MAX_SPECIES];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:__constant__ float4 MD_base_patches[CUDAPatchySwapInteraction::MAX_SPECIES][CUDAPatchySwapInteraction::MAX_PATCHES];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:__constant__ float MD_patchy_eps[CUDAPatchySwapInteraction::MAX_SPECIES * CUDAPatchySwapInteraction::MAX_SPECIES];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:struct __align__(16) CUDA_FS_bond {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:struct __align__(16) CUDA_FS_bond_list {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_FS_bond bonds[CUDAPatchySwapInteraction::MAX_NEIGHS];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_FS_bond_list() :
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_FS_bond &new_bond() {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:		if(n_bonds > CUDAPatchySwapInteraction::MAX_NEIGHS) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:__device__ void _patchy_two_body_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &a1, c_number4 &a2, c_number4 &a3, c_number4 &b1, c_number4 &b2, c_number4 &b3, c_number4 &F, c_number4 &torque, CUDA_FS_bond_list *bonds, int q_idx, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:			c_number dist = CUDA_DOT(patch_dist, patch_dist);
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:					CUDA_FS_bond_list &bond_list = bonds[pi];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:					CUDA_FS_bond &my_bond = bond_list.new_bond();
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:__device__ void _three_body(CUDA_FS_bond_list *bonds, c_number4 &F, c_number4 &T, c_number4 *forces, c_number4 *torques) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	for(int pi = 0; pi < CUDAPatchySwapInteraction::MAX_PATCHES; pi++) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:		CUDA_FS_bond_list &bond_list = bonds[pi];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:			CUDA_FS_bond &b1 = bond_list.bonds[bi];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:				CUDA_FS_bond &b2 = bond_list.bonds[bj];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:__global__ void PS_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *three_body_forces, c_number4 *torques, c_number4 *three_body_torques, int *matrix_neighs, int *number_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	GPU_quat po = orientations[IND];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_FS_bond_list bonds[CUDAPatchySwapInteraction::MAX_PATCHES];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:			GPU_quat qo = orientations[k_index];
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:/* END CUDA PART */
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:CUDAPatchySwapInteraction::CUDAPatchySwapInteraction() :
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:				CUDABaseInteraction(),
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:CUDAPatchySwapInteraction::~CUDAPatchySwapInteraction() {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_forces));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_torques));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:void CUDAPatchySwapInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	getInputInt(&inp, "CUDA_sort_every", &sort_every, 0);
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:void CUDAPatchySwapInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_forces, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_torques, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	// the following quantities are initialised by read_topology and hence have to be copied over to the GPU after its call
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_species, &_N_species, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patches, _N_patches.data(), sizeof(int) * _N_patches.size()));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_base_patches, base_patches, sizeof(float4) * n_patches, i * sizeof(float4) * MAX_PATCHES));
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:void CUDAPatchySwapInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.cu:	int N = CUDABaseInteraction::_N;
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu: * CUDADetailedPolymerSwapInteraction.cu
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:#include "CUDADetailedPolymerSwapInteraction.h"
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:#define CUDA_MAX_SWAP_NEIGHS 20
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:struct __align__(16) CUDA_FS_bond {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:struct __align__(16) CUDA_FS_bond_list {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_FS_bond bonds[CUDA_MAX_SWAP_NEIGHS];
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_FS_bond_list() :
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:		if(n_bonds > CUDA_MAX_SWAP_NEIGHS) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:__device__ void _repulsion(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box, bool with_yukawa) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:__device__ void _sticky(c_number4 &ppos, c_number4 &qpos, int eps_idx, int q_idx, c_number4 &F, CUDA_FS_bond_list &bond_list, cudaTextureObject_t tex_eps, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:__device__ void _FENE(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:__device__ void _sticky_three_body(CUDA_FS_bond_list &bond_list, c_number4 &F, c_number4 *forces) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:		CUDA_FS_bond &b1 = bond_list.bonds[bi];
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:			CUDA_FS_bond &b2 = bond_list.bonds[bj];
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:__device__ void _flexibility_three_body(c_number4 &ppos, c_number4 &n1_pos, c_number4 &n2_pos, int n1_idx, int n2_idx, c_number4 &F, c_number4 *poss, c_number4 *three_body_forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	c_number sqr_dist_pn1 = CUDA_DOT(dist_pn1, dist_pn1);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	c_number sqr_dist_pn2 = CUDA_DOT(dist_pn2, dist_pn2);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	c_number cost = CUDA_DOT(dist_pn1, dist_pn2) * i_pn1_pn2;
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:__global__ void ps_FENE_flexibility_forces(c_number4 *poss, c_number4 *forces, c_number4 *three_body_forces, int *bonded_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:__global__ void ps_forces(c_number4 *poss, c_number4 *forces, c_number4 *three_body_forces, int *matrix_neighs, int *number_neighs, int *bonded_neighs, cudaTextureObject_t tex_eps, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_FS_bond_list bonds;
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:CUDADetailedPolymerSwapInteraction::CUDADetailedPolymerSwapInteraction() :
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:CUDADetailedPolymerSwapInteraction::~CUDADetailedPolymerSwapInteraction() {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_bonded_neighs));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_forces));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_3b_epsilon));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:		cudaDestroyTextureObject(_tex_eps);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:void CUDADetailedPolymerSwapInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:void CUDADetailedPolymerSwapInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_forces, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_bonded_neighs, n_elems * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:				throw oxDNAException("CUDADetailedPolymerSwapInteraction: particle %d has more than %d bonded neighbours", p->index, max_n_neighs);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_bonded_neighs, h_bonded_neighs.data(), n_elems * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n, &_PS_n, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_interaction_matrix_size, &interaction_matrix_size, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_3b_epsilon, _3b_epsilon.size() * sizeof(float)));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_3b_epsilon, h_3b_epsilon.data(), _3b_epsilon.size() * sizeof(float), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	GpuUtils::init_texture_object(&_tex_eps, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat), _d_3b_epsilon, _3b_epsilon.size());
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_enable_semiflexibility, &_enable_semiflexibility, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:void CUDADetailedPolymerSwapInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.cu:	/*number energy = GpuUtils::sum_c_number4_to_double_on_GPU(d_forces, _N);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h: * CUDANathanStarInteraction.h
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:#ifndef CUDANATHANSTARINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:#define CUDANATHANSTARINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h: * @brief CUDA implementation of the {@link NathanStarInteraction}.
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:class CUDANathanStarInteraction: public CUDABaseInteraction, public NathanStarInteraction {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:	void _setup_cuda_interp();
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:	CUDANathanStarInteraction();
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:	virtual ~CUDANathanStarInteraction();
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:extern "C" BaseInteraction *make_CUDANathanStarInteraction() {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:	return new CUDANathanStarInteraction();
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.h:#endif /* CUDANATHANSTARINTERACTION_H_ */
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu: * CUDATSPInteraction.cu
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:#include "CUDATSPInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:__device__ void _TSP_particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:__global__ void tsp_forces(c_number4 *poss, c_number4 *forces, LR_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:__global__ void tsp_forces_edge_nonbonded(c_number4 *poss, c_number4 *forces, edge_bond *edge_list, int n_edges, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:__global__ void tsp_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *c_number_neighs, LR_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:CUDATSPInteraction::CUDATSPInteraction() :
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:CUDATSPInteraction::~CUDATSPInteraction() {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_anchors));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_anchor_neighs));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:void CUDATSPInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	if(getInputInt(&inp, "CUDA_sort_every", &sort_every, 0) == KEY_FOUND) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:void CUDATSPInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_per_star, &N_per_star, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_stars, &this->_N_stars, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_TSP_n, &this->_TSP_n, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_TSP_only_chains, &this->_only_chains, sizeof(bool)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_yukawa_repulsion, &this->_yukawa_repulsion, sizeof(bool)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	if(this->_use_edge) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &this->_n_forces, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:void CUDATSPInteraction::_setup_anchors() {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_anchors, this->_N_stars * sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<TSP_anchor_bonds>(&_d_anchor_neighs, this->_N_stars*TSP_MAX_ARMS*sizeof(TSP_anchor_bonds)));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_anchors, _h_anchors, this->_N_stars * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_anchor_neighs, _h_anchor_neighs, this->_N_stars*TSP_MAX_ARMS*sizeof(TSP_anchor_bonds), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:void CUDATSPInteraction::compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:	CUDASimpleVerletList*_v_lists = dynamic_cast<CUDASimpleVerletList*>(lists);
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:			cudaThreadSynchronize();
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.cu:		CUDANoList*_no_lists = dynamic_cast<CUDANoList*>(lists);
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h: * CUDATSPInteraction.h
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:#ifndef CUDATSPINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:#define CUDATSPINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h: * @brief Handles interactions between TSPs on CUDA. See TSPInteraction for a list of options.
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:class CUDATSPInteraction: public CUDABaseInteraction, public TSPInteraction {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:	CUDATSPInteraction();
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:	virtual ~CUDATSPInteraction();
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:	void compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box);
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:extern "C" BaseInteraction *make_CUDATSPInteraction() {
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:	return new CUDATSPInteraction();
contrib/rovigatti/src/Interactions/old/CUDATSPInteraction.h:#endif /* CUDATSPINTERACTION_H_ */
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h: * CUDAPolymerInteraction.h
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:#ifndef CUDAPOLYMERINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:#define CUDAPOLYMERINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:class CUDAPolymerInteraction: public CUDABaseInteraction, public PolymerInteraction {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:	CUDAPolymerInteraction();
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:	virtual ~CUDAPolymerInteraction();
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:	void cuda_init(int N) override;
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:extern "C" BaseInteraction *make_CUDAPolymerInteraction() {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:	return new CUDAPolymerInteraction();
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.h:#endif /* CUDAPOLYMERINTERACTION_H_ */
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h: * CUDALevyInteraction.h
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:#ifndef CUDALEVYINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:#define CUDALEVYINTERACTION_H_
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:	 * @brief Handles interactions between Levy tetramers on CUDA.
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:	class CUDALevyInteraction: public CUDABaseInteraction, public LevyInteraction {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:		CUDALevyInteraction();
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:		virtual ~CUDALevyInteraction();
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:		c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:		void cuda_init(int N);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:		void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:	extern "C" BaseInteraction *make_CUDALevyInteraction();
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.h:#endif /* CUDALEVYINTERACTION_H_ */
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu: * CUDANathanStarInteraction.cu
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:#include "CUDANathanStarInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:/* BEGIN CUDA */
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:texture<float, 1, cudaReadModeElementType> tex_patchy_star;
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:__device__ void _patchy_patchy_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &p_axis, c_number4 &q_axis, c_number4 &F, c_number4 &torque, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	c_number cospr = -CUDA_DOT(p_axis, r_versor);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	c_number cosqr = CUDA_DOT(q_axis, r_versor);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu://	printf("%d %lf %lf %lf %lf %lf\n", IND, p_mod, q_mod, cospr_part, cosqr_part, sqrt(CUDA_DOT(tmp_force, tmp_force)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:__device__ void _patchy_star_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:__device__ void _star_star_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:__global__ void NS_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	GPU_quat po = orientations[IND];
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:				GPU_quat qo = orientations[j];
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:__global__ void NS_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, int *matrix_neighs, int *c_number_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	GPU_quat po = orientations[IND];
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:			GPU_quat qo = orientations[k_index];
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:/* END CUDA PART */
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:CUDANathanStarInteraction::CUDANathanStarInteraction() :
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:				CUDABaseInteraction(),
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:CUDANathanStarInteraction::~CUDANathanStarInteraction() {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	if(_d_patchy_star != NULL) CUDA_SAFE_CALL(cudaFree(_d_patchy_star));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:void CUDANathanStarInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:void CUDANathanStarInteraction::_setup_cuda_interp() {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<float>(&_d_patchy_star, v_size));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_patchy_star, fx, size * sizeof(float), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaBindTexture(NULL, tex_patchy_star, _d_patchy_star, v_size));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_interp_size, &size, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_xmin, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_xmax, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_bin, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:void CUDANathanStarInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_rep_power, &this->_rep_power, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_half_power, &patch_half_power, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_patchy_rcut, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_patchy_star_rcut, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_star_rcut, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_rep_E_cut, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_angular_cutoff, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_pow_alpha, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_pow_sigma, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_T, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_star_f1_2, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_star_f3_2, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_star_sigma_s, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_star_sigma_s, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_star_factor, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	_setup_cuda_interp();
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:void CUDANathanStarInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/old/CUDANathanStarInteraction.cu:	CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu: * CUDALevyInteraction.cu
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:#include "CUDALevyInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:/* CUDA constants */
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:__device__ void _particle_particle_bonded_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box, bool only_fene = false) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &ppatch, c_number4 &qpatch, c_number4 &F, c_number4 &T, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:		c_number sqr_dist = CUDA_DOT(patch_dist, patch_dist);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:__global__ void Levy_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, LR_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:__global__ void Levy_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, int *matrix_neighs, int *c_number_neighs, LR_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:__device__ void _three_body(c_number4 &ppos, LR_bonds &bs, c_number4 &F, c_number4 *poss, c_number4 *n3_forces, c_number4 *n5_forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	c_number sqr_dist_pn3 = CUDA_DOT(dist_pn3, dist_pn3);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	c_number sqr_dist_pn5 = CUDA_DOT(dist_pn5, dist_pn5);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	c_number cost = CUDA_DOT(dist_pn3, dist_pn5) * i_pn3_pn5;
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:__global__ void three_body_forces(c_number4 *poss, c_number4 *forces, c_number4 *n3_forces, c_number4 *n5_forces, LR_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:__global__ void centre_forces(c_number4 *poss, c_number4 *forces, c_number4 *n3_forces, c_number4 *n5_forces, int *centres, centre_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:CUDALevyInteraction::CUDALevyInteraction() {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:CUDALevyInteraction::~CUDALevyInteraction() {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_centres));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_centre_neighs));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_n3_forces));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_n5_forces));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:void CUDALevyInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	if(getInputInt(&inp, "CUDA_sort_every", &sort_every, 0) == KEY_FOUND) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:void CUDALevyInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc < c_number4 > (&_d_n3_forces, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc < c_number4 > (&_d_n5_forces, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n3_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n5_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_rigid_model, &this->_rigid_model, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patchy_power, &this->_patchy_power, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:void CUDALevyInteraction::_setup_centres() {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_centres, N_centres * sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc < centre_bonds > (&_d_centre_neighs, N_centres * sizeof(centre_bonds)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_centres, h_centres, N_centres * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_centre_neighs, h_centre_neighs, N_centres * sizeof(centre_bonds), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_centres, &N_centres, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:void CUDALevyInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n3_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n5_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:		CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:extern "C" BaseInteraction *make_CUDALevyInteraction() {
contrib/rovigatti/src/Interactions/old/CUDALevyInteraction.cu:	return new CUDALevyInteraction();
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu: * CUDAPolymerInteraction.cu
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:#include "CUDAPolymerInteraction.h"
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:/* BEGIN CUDA */
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:__global__ void polymer_forces_edge_nonbonded(c_number4 *poss, c_number4 *forces, edge_bond *edge_list, int n_edges, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:__global__ void polymer_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *number_neighs, LR_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:/* END CUDA */
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:CUDAPolymerInteraction::CUDAPolymerInteraction() {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:CUDAPolymerInteraction::~CUDAPolymerInteraction() {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:void CUDAPolymerInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:	if(getInputInt(&inp, "CUDA_sort_every", &sort_every, 0) == KEY_FOUND) {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:void CUDAPolymerInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:	if(this->_use_edge) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &this->_n_forces, sizeof(int)));
contrib/rovigatti/src/Interactions/old/CUDAPolymerInteraction.cu:void CUDAPolymerInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu: * CUDACPMixtureInteraction.cu
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:#include "CUDAMGInteraction.h"
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:__device__ void _nonbonded(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:__device__ void _bonded(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:__global__ void cp_bonded_forces(c_number4 *poss, c_number4 *forces, int *bonded_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:__global__ void cp_forces(c_number4 *poss, c_number4 *forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:__global__ void cp_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *c_number_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:CUDAMGInteraction::CUDAMGInteraction() :
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:CUDAMGInteraction::~CUDAMGInteraction() {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_bonded_neighs));
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:void CUDAMGInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:void CUDAMGInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_bonded_neighs, n_elems * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:			if(nb > max_n_neighs) throw oxDNAException("CUDAMGInteraction: particle %d has more than %d bonded neighbours", p->index, max_n_neighs);
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_bonded_neighs, h_bonded_neighs, n_elems * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n, &this->_MG_n, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:void CUDAMGInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/CUDAMGInteraction.cu:	CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h: * CUDAAOInteraction.h
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:#ifndef CUDAAOINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:#define CUDAAOINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h: * @brief CUDA implementation of the {@link AOInteraction Asakura-Oosawa interaction}.
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:class CUDAAOInteraction: public CUDABaseInteraction, public AOInteraction {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:	CUDAAOInteraction();
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:	virtual ~CUDAAOInteraction();
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:extern "C" BaseInteraction *make_CUDAAOInteraction() {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:	return new CUDAAOInteraction();
contrib/rovigatti/src/Interactions/CUDAAOInteraction.h:#endif /* CUDAAOINTERACTION_H_ */
contrib/rovigatti/src/Interactions/FSInteraction.cpp:		// CUDA backend's get_particle_btype will return btype % 4
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h: * CUDAFSInteraction.h
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:#ifndef CUDAFSINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:#define CUDAFSINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:struct CUDA_FS_bonding_pattern;
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h: * @brief CUDA implementation of the {@link FSInteraction}.
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:class CUDAFSInteraction: public CUDABaseInteraction, public FSInteraction {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:	CUDAFSInteraction();
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:	virtual ~CUDAFSInteraction();
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:extern "C" BaseInteraction *make_CUDAFSInteraction() {
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:	return new CUDAFSInteraction();
contrib/rovigatti/src/Interactions/CUDAFSInteraction.h:#endif /* CUDAFSINTERACTION_H_ */
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h: * CUDADetailedPatchySwapInteraction.h
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:#ifndef CUDADETAILEDPATCHYSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:#define CUDADETAILEDPATCHYSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:struct CUDA_FS_bonding_pattern;
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h: * @brief CUDA implementation of the {@link PatchySwapInteraction}.
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:class CUDADetailedPatchySwapInteraction: public CUDABaseInteraction, public DetailedPatchySwapInteraction {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	cudaTextureObject_t _tex_patchy_eps = 0;
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	cudaTextureObject_t _tex_base_patches = 0;
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	CUDADetailedPatchySwapInteraction();
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	virtual ~CUDADetailedPatchySwapInteraction();
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:extern "C" BaseInteraction *make_CUDADetailedPatchySwapInteraction() {
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:	return new CUDADetailedPatchySwapInteraction();
contrib/rovigatti/src/Interactions/CUDADetailedPatchySwapInteraction.h:#endif /* CUDAPATCHYSWAPINTERACTION_H_ */
contrib/rovigatti/src/Interactions/DetailedPatchySwapInteraction.cpp:		// we need to save the base patches so that the CUDA backend has access to them
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu: * CUDAStarrInteraction.cu
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:#include "CUDAStarrInteraction.h"
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:/* CUDA constants */
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__device__ void _particle_particle_bonded_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box, bool only_fene = false) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, int *strand_ids, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__device__ void _particle_all_bonded_interactions(c_number4 &ppos, LR_bonds &bs, c_number4 &F, c_number4 *poss, c_number4 *forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__global__ void Starr_forces(c_number4 *poss, c_number4 *forces, LR_bonds *bonds, int *strand_ids, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__global__ void Starr_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *c_number_neighs, LR_bonds *bonds, int *strand_ids, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__device__ void _three_body(c_number4 &ppos, LR_bonds &bs, c_number4 &F, c_number4 *poss, c_number4 *n3_forces, c_number4 *n5_forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	c_number sqr_dist_pn3 = CUDA_DOT(dist_pn3, dist_pn3);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	c_number sqr_dist_pn5 = CUDA_DOT(dist_pn5, dist_pn5);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	c_number cost = CUDA_DOT(dist_pn3, dist_pn5) * i_pn3_pn5;
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__global__ void three_body_forces(c_number4 *poss, c_number4 *forces, c_number4 *n3_forces, c_number4 *n5_forces, LR_bonds *bonds, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:__global__ void hub_forces(c_number4 *poss, c_number4 *forces, c_number4 *n3_forces, c_number4 *n5_forces, int *hubs, hub_bonds *bonds, LR_bonds *n3n5, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:CUDAStarrInteraction::CUDAStarrInteraction() {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:CUDAStarrInteraction::~CUDAStarrInteraction() {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_strand_ids));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_hubs));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_hub_neighs));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_n3_forces));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_n5_forces));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:void CUDAStarrInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	if(getInputInt(&inp, "CUDA_sort_every", &sort_every, 0) == KEY_FOUND) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:void CUDAStarrInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc < c_number4 > (&_d_n3_forces, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc < c_number4 > (&_d_n5_forces, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n3_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n5_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_starr_model, &this->_starr_model, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_mode, &this->_mode, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_per_strand, &this->_N_per_strand, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_lin_k, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_fene_K, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_fene_sqr_r0, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_rcut, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:void CUDAStarrInteraction::_setup_strand_ids() {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_strand_ids, this->_N * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_strand_ids, h_strand_ids, this->_N * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:void CUDAStarrInteraction::_setup_hubs() {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_hubs, _N_hubs * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc < hub_bonds > (&_d_hub_neighs, _N_hubs * sizeof(hub_bonds)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_hubs, h_hubs, _N_hubs * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_hub_neighs, h_hub_neighs, _N_hubs * sizeof(hub_bonds), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_hubs, &_N_hubs, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:void CUDAStarrInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n3_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_n5_forces, 0, this->_N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:		CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:extern "C" BaseInteraction *make_CUDAStarrInteraction() {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.cu:	return new CUDAStarrInteraction();
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h: * CUDACPMixtureInteraction.h
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:#ifndef CUDAMGINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:#define CUDAMGINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h: * @brief CUDA implementation of the {@link MGInteraction MicroGel interaction}.
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:class CUDAMGInteraction: public CUDABaseInteraction, public MGInteraction {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:	CUDAMGInteraction();
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:	virtual ~CUDAMGInteraction();
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:extern "C" BaseInteraction *make_CUDAMGInteraction() {
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:	return new CUDAMGInteraction();
contrib/rovigatti/src/Interactions/CUDAMGInteraction.h:#endif /* CUDAMGINTERACTION_H_ */
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h: * CUDAPolymerSwapInteraction.h
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:#ifndef CUDAPOLYMERSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:#define CUDAPOLYMERSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h: * @brief CUDA implementation of the {@link PolymerSwapInteraction interaction}.
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:class CUDAPolymerSwapInteraction: public CUDABaseInteraction, public PolymerSwapInteraction {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:	CUDAPolymerSwapInteraction();
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:	virtual ~CUDAPolymerSwapInteraction();
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:extern "C" BaseInteraction *make_CUDAPolymerSwapInteraction() {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:	return new CUDAPolymerSwapInteraction();
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.h:#endif /* CUDAPOLYMERSWAPINTERACTION_H_ */
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu: * CUDAmWInteraction.cu
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:#include "CUDAmWInteraction.h"
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:#define CUDA_MAX_MW_NEIGHS 50
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:/* BEGIN CUDA */
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:struct __align__(16) cuda_mW_bond {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:struct __align__(16) cuda_mW_bond_list {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	cuda_mW_bond bonds[CUDA_MAX_MW_NEIGHS];
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	cuda_mW_bond_list() :
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	cuda_mW_bond &new_bond() {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:		if(n_bonds > CUDA_MAX_MW_NEIGHS) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, cuda_mW_bond_list &bond_list, int q_idx, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	cuda_mW_bond &my_bond = bond_list.new_bond();
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:__device__ void _three_body(cuda_mW_bond_list &bond_list, c_number4 &F, c_number4 *forces, c_number4 *forces_3body) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:		cuda_mW_bond b1 = bond_list.bonds[bi];
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:			cuda_mW_bond b2 = bond_list.bonds[bj];
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:			c_number cos_theta = CUDA_DOT(b1.r, b2.r) / irpq;
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:__global__ void mW_forces(c_number4 *poss, c_number4 *forces, c_number4 *forces_3body, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	cuda_mW_bond_list bond_list;
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:__global__ void mW_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *c_number_neighs, c_number4 *forces_3body, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	cuda_mW_bond_list bond_list;
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:/* END CUDA PART */
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:CUDAmWInteraction::CUDAmWInteraction() :
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:				CUDABaseInteraction(),
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:CUDAmWInteraction::~CUDAmWInteraction() {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	if(_d_forces_3body != NULL) CUDA_SAFE_CALL(cudaFree(_d_forces_3body));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:void CUDAmWInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:void CUDAmWInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_rcut, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_lambda, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_gamma, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_a, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_A, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_B, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_cos_theta0, &f_copy, sizeof(float)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &n_forces, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc < c_number4 > (&_d_forces_3body, size));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDA_SAFE_CALL(cudaMemset(_d_forces_3body, 0, size));
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:void CUDAmWInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/CUDAmWInteraction.cu:	CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu: * CUDAAOInteraction.cu
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:#include "CUDAAOInteraction.h"
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:	c_number r_norm = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:__global__ void cp_forces(c_number4 *poss, c_number4 *forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:__global__ void cp_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *c_number_neighs, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:CUDAAOInteraction::CUDAAOInteraction() {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:CUDAAOInteraction::~CUDAAOInteraction() {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:void CUDAAOInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:void CUDAAOInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:void CUDAAOInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:	CUDASimpleVerletList *_v_lists = dynamic_cast<CUDASimpleVerletList *>(lists);
contrib/rovigatti/src/Interactions/CUDAAOInteraction.cu:	CUDANoList *_no_lists = dynamic_cast<CUDANoList *>(lists);
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h: * CUDACPMixtureInteraction.h
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:#ifndef CUDACPMIXTUREINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:#define CUDACPMIXTUREINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h: * @brief CUDA implementation of the {@link CPMixtureInteraction Colloid-polymer mixture interaction}.
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:class CUDACPMixtureInteraction: public CUDABaseInteraction, public CPMixtureInteraction {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:	CUDACPMixtureInteraction();
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:	virtual ~CUDACPMixtureInteraction();
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:extern "C" BaseInteraction *make_CUDACPMixtureInteraction() {
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:	return new CUDACPMixtureInteraction();
contrib/rovigatti/src/Interactions/CUDACPMixtureInteraction.h:#endif /* CUDACPMIXTUREINTERACTION_H_ */
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h: * CUDAmWInteraction.h
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:#ifndef CUDAMWINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:#define CUDAMWINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h: * @brief CUDA implementation of the {@link mWInteraction}.
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:class CUDAmWInteraction: public CUDABaseInteraction, public mWInteraction {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:	CUDAmWInteraction();
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:	virtual ~CUDAmWInteraction();
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:extern "C" BaseInteraction *make_CUDAmWInteraction() {
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:	return new CUDAmWInteraction();
contrib/rovigatti/src/Interactions/CUDAmWInteraction.h:#endif /* CUDAMWINTERACTION_H_ */
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h: * CUDAPatchySwapInteraction.h
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:#ifndef CUDAPATCHYSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:#define CUDAPATCHYSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:struct CUDA_FS_bonding_pattern;
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h: * @brief CUDA implementation of the {@link PatchySwapInteraction}.
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:class CUDAPatchySwapInteraction: public CUDABaseInteraction, public PatchySwapInteraction {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:	CUDAPatchySwapInteraction();
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:	virtual ~CUDAPatchySwapInteraction();
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:extern "C" BaseInteraction *make_CUDAPatchySwapInteraction() {
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:	return new CUDAPatchySwapInteraction();
contrib/rovigatti/src/Interactions/CUDAPatchySwapInteraction.h:#endif /* CUDAPATCHYSWAPINTERACTION_H_ */
contrib/rovigatti/src/Interactions/PatchySwapInteraction.cpp:			// we need to save the base patches so that the CUDA backend has access to them
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h: * CUDAStarrInteraction.h
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:#ifndef CUDASTARRINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:#define CUDASTARRINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:	 * @brief Handles interactions between Starr tetramers on CUDA.
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:	class CUDAStarrInteraction: public CUDABaseInteraction, public StarrInteraction {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:		CUDAStarrInteraction();
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:		virtual ~CUDAStarrInteraction();
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:		c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:		void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:		void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:	extern "C" BaseInteraction *make_CUDAStarrInteraction();
contrib/rovigatti/src/Interactions/CUDAStarrInteraction.h:#endif /* CUDASTARRINTERACTION_H_ */
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h: * CUDADetailedPolymerSwapInteraction.h
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:#ifndef CUDADETAILEDPOLYMERSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:#define CUDADETAILEDPOLYMERSWAPINTERACTION_H_
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h: * @brief CUDA implementation of the {@link DetailedPolymerSwapInteraction interaction}.
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:class CUDADetailedPolymerSwapInteraction: public CUDABaseInteraction, public DetailedPolymerSwapInteraction {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:	cudaTextureObject_t _tex_eps = 0;
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:	CUDADetailedPolymerSwapInteraction();
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:	virtual ~CUDADetailedPolymerSwapInteraction();
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:	void cuda_init(int N);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:	c_number get_cuda_rcut() {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:extern "C" BaseInteraction *make_CUDADetailedPolymerSwapInteraction() {
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:	return new CUDADetailedPolymerSwapInteraction();
contrib/rovigatti/src/Interactions/CUDADetailedPolymerSwapInteraction.h:#endif /* CUDADETAILEDPOLYMERSWAPINTERACTION_H_ */
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu: * CUDAPolymerSwapInteraction.cu
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:#include "CUDAPolymerSwapInteraction.h"
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:#define CUDA_MAX_SWAP_NEIGHS 20
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:struct __align__(16) CUDA_FS_bond {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:struct __align__(16) CUDA_FS_bond_list {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_FS_bond bonds[CUDA_MAX_SWAP_NEIGHS];
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_FS_bond_list() :
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:		if(n_bonds > CUDA_MAX_SWAP_NEIGHS) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:__device__ void _WCA(c_number4 &ppos, c_number4 &qpos, int int_type, c_number4 &F, CUDAStressTensor &p_st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:__device__ void _sticky(c_number4 &ppos, c_number4 &qpos, int q_idx, c_number4 &F, CUDA_FS_bond_list &bond_list, CUDAStressTensor &p_st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:__device__ void _FENE(c_number4 &ppos, c_number4 &qpos, int int_type, c_number4 &F, CUDAStressTensor &p_st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:__device__ void _patchy_three_body(CUDA_FS_bond_list &bond_list, c_number4 &F, CUDAStressTensor &p_st, c_number4 *forces) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:		CUDA_FS_bond &b1 = bond_list.bonds[bi];
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:			CUDA_FS_bond &b2 = bond_list.bonds[bj];
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:__device__ void _flexibility_three_body(c_number4 &ppos, c_number4 &n1_pos, c_number4 &n2_pos, int n1_idx, int n2_idx, c_number4 &F, c_number4 *poss, c_number4 *three_body_forces, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	c_number sqr_dist_pn1 = CUDA_DOT(dist_pn1, dist_pn1);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	c_number sqr_dist_pn2 = CUDA_DOT(dist_pn2, dist_pn2);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	c_number cost = CUDA_DOT(dist_pn1, dist_pn2) * i_pn1_pn2;
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:__global__ void ps_FENE_flexibility_forces(c_number4 *poss, c_number4 *forces, c_number4 *three_body_forces, int *bonded_neighs, bool update_st, CUDAStressTensor *st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDAStressTensor p_st;
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:__global__ void ps_forces(c_number4 *poss, c_number4 *forces, c_number4 *three_body_forces, int *matrix_neighs, int *number_neighs, bool update_st, CUDAStressTensor *st, CUDABox *box) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_FS_bond_list bonds;
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDAStressTensor p_st;
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:CUDAPolymerSwapInteraction::CUDAPolymerSwapInteraction() :
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:CUDAPolymerSwapInteraction::~CUDAPolymerSwapInteraction() {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_bonded_neighs));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_forces));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:void CUDAPolymerSwapInteraction::get_settings(input_file &inp) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:void CUDAPolymerSwapInteraction::cuda_init(int N) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_forces, N * sizeof(c_number4)));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_bonded_neighs, n_elems * sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:				throw oxDNAException("CUDAPolymerSwapInteraction: particle %d has more than %d bonded neighbours", p->index, max_n_neighs);
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_bonded_neighs, h_bonded_neighs.data(), n_elems * sizeof(int), cudaMemcpyHostToDevice));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n, &_PS_n, sizeof(int)));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_same_sticky_only_interaction, &_same_sticky_only_interaction, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_enable_semiflexibility, &_enable_semiflexibility, sizeof(bool)));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:void CUDAPolymerSwapInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:		CUDA_SAFE_CALL(cudaMemset(_d_st, 0, _N * sizeof(CUDAStressTensor)));
contrib/rovigatti/src/Interactions/CUDAPolymerSwapInteraction.cu:	/*number energy = GpuUtils::sum_c_number4_to_double_on_GPU(d_forces, _N);
contrib/randisi/CMakeLists.txt:IF(CUDA)
contrib/randisi/CMakeLists.txt:	CUDA_ADD_LIBRARY(CUDADNA2ModInteraction SHARED EXCLUDE_FROM_ALL src/Interactions/CUDADNA2ModInteraction.cu src/Interactions/DNA2ModInteraction.cpp)
contrib/randisi/CMakeLists.txt:	ADD_DEPENDENCIES(randisi CUDADNA2ModInteraction)
contrib/randisi/CMakeLists.txt:ENDIF(CUDA)
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h: * CUDADNA2ModInteraction.h
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h: *      re-Author: Ferdinando, after CUDADNAInteraction.h by Lorenzo
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:#ifndef CUDADNA2MODINTERACTION_H_
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:#define CUDADNA2MODINTERACTION_H_
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h: * @brief CUDA implementation of the oxDNA model, as provided by DNAInteraction.
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:class CUDADNA2ModInteraction: public CUDABaseInteraction<number, number4>, public DNA2ModInteraction<number> {
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:	CUDADNA2ModInteraction();
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:	virtual ~CUDADNA2ModInteraction();
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:	void cuda_init(number box_side, int N);
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:	number get_cuda_rcut() { return this->get_rcut(); }
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:	void compute_forces(CUDABaseList<number, number4> *lists, number4 *d_poss, GPU_quat<number> *d_qorientations, number4 *d_forces, number4 *d_torques, LR_bonds *d_bonds, CUDABox<number, number4> *d_box);
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:extern "C" BaseInteraction<float> *make_CUDADNA2ModInteraction_float() { return new CUDADNA2ModInteraction<float, float4>(); } 
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:extern "C" BaseInteraction<double> *make_CUDADNA2ModInteraction_double() { return new CUDADNA2ModInteraction<double, LR_double4>(); }
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.h:#endif /* CUDADNA2MODINTERACTION_H_ */
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number rsqr = CUDA_DOT(r, r);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:  number4 n3y_1 = CUDA_rotateVectorAroundVersor(n3y,n3x,stacking_roll);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:  number4 n3z_1 = CUDA_rotateVectorAroundVersor(n3z,n3x,stacking_roll);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:  n3x_1 = CUDA_rotateVectorAroundVersor(n3x_1,n3y,stacking_tilt);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:  n3z_1 = CUDA_rotateVectorAroundVersor(n3z_1,n3y,stacking_tilt);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number bcos_1 = CUDA_DOT(n3x, n3x_1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number angle_1 = CUDA_LRACOS(bcos_1) * 180./M_PI;
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number bcos_2 = CUDA_DOT(n3y, n3y_1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number angle_2 = CUDA_LRACOS(bcos_2) * 180./M_PI;
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number bcos_3 = CUDA_DOT(n3z, n3z_1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number angle_3 = CUDA_LRACOS(bcos_3) * 180./M_PI;
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number t4 = CUDA_LRACOS(CUDA_DOT(n3z_1, n5z));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number t5 = CUDA_LRACOS(CUDA_DOT(n5z, rstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number t6 = CUDA_LRACOS(-CUDA_DOT(n3z_1, rstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number cosphi1 = CUDA_DOT(n5y, rbackref) / rbackrefmod;
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number cosphi2 = CUDA_DOT(n3y_1, rbackref) / rbackrefmod;
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number ra2 = CUDA_DOT(rstackdir, n5y);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number ra1 = CUDA_DOT(rstackdir, n5x);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number rb1 = CUDA_DOT(rstackdir, n3x_1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number a2b1 = CUDA_DOT(n5y, n3x_1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		ra2 = CUDA_DOT(rstackdir, n3y_1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		rb1 = CUDA_DOT(rstackdir, n5x);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		a2b1 = CUDA_DOT(n3y_1, n5x);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:__device__ void _particle_particle_interaction(number4 ppos, number4 a1, number4 a2, number4 a3, number4 qpos, number4 b1, number4 b2, number4 b3, number4 &F, number4 &T, bool grooving, bool use_debye_huckel, bool use_oxDNA2_coaxial_stacking, LR_bonds pbonds, LR_bonds qbonds, int pind, int qind, CUDABox<number, number4> *box, number hb_multi_extra) {
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t2 = CUDA_LRACOS(-CUDA_DOT(b1, rhydrodir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t3 = CUDA_LRACOS(CUDA_DOT(a1, rhydrodir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t7 = CUDA_LRACOS(-CUDA_DOT(rhydrodir, b3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t8 = CUDA_LRACOS(CUDA_DOT(rhydrodir, a3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t1 = CUDA_LRACOS (-CUDA_DOT(a1, b1));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t2 = CUDA_LRACOS (-CUDA_DOT(b1, rcstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t3 = CUDA_LRACOS ( CUDA_DOT(a1, rcstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t4 = CUDA_LRACOS ( CUDA_DOT(a3, b3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t7 = CUDA_LRACOS (-CUDA_DOT(rcstackdir, b3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number t8 = CUDA_LRACOS ( CUDA_DOT(rcstackdir, a3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number rstackmodsqr = CUDA_DOT(rstack, rstack);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t1 = CUDA_LRACOS (-CUDA_DOT(a1, b1));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t4 = CUDA_LRACOS ( CUDA_DOT(a3, b3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t5 = CUDA_LRACOS ( CUDA_DOT(a3, rstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t6 = CUDA_LRACOS (-CUDA_DOT(b3, rstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:		number rstackmodsqr = CUDA_DOT(rstack, rstack);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t1 = CUDA_LRACOS (-CUDA_DOT(a1, b1));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t4 = CUDA_LRACOS ( CUDA_DOT(a3, b3));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t5 = CUDA_LRACOS ( CUDA_DOT(a3, rstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number t6 = CUDA_LRACOS (-CUDA_DOT(b3, rstackdir));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:			number cosphi3 = CUDA_DOT(rstackdir, (_cross<number, number4>(rbackbonerefdir, a1)));
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:				number a2b1 = CUDA_DOT(a2, b1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:				number a3b1 = CUDA_DOT(a3, b1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:				number ra1 = CUDA_DOT(rstackdir, a1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:				number ra2 = CUDA_DOT(rstackdir, a2);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:				number ra3 = CUDA_DOT(rstackdir, a3);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:				number rb1 = CUDA_DOT(rstackdir, b1);
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:__global__ void dna_forces(number4 *poss, GPU_quat<number> *orientations, number4 *forces, number4 *torques, LR_bonds *bonds, bool grooving, bool use_debye_huckel, bool use_oxDNA2_coaxial_stacking, bool use_oxDNA2_FENE, bool use_mbf, number mbf_xmax, number mbf_finf, CUDABox<number, number4> *box, number *_d_stacking_roll, number *_d_stacking_r_roll, number *_d_stacking_tilt, number *_d_stacking_multiplier, number *_d_hb_multiplier) {
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	get_vectors_from_quat<number,number4>(orientations[IND], a1, a2, a3); //Returns vectors a1,a2 and a3 as they would be in the GPU matrix. These are necessary even in pure quaternion dynamics
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:__global__ void dna_forces_edge_nonbonded(number4 *poss, GPU_quat<number> *orientations, number4 *forces, number4 *torques, edge_bond *edge_list, int n_edges, LR_bonds *bonds, bool grooving, bool use_debye_huckel, bool use_oxDNA2_coaxial_stacking, CUDABox<number, number4> *box, number * _d_hb_multiplier) {
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:__global__ void dna_forces_edge_bonded(number4 *poss, GPU_quat<number> *orientations,  number4 *forces, number4 *torques, LR_bonds *bonds, bool grooving, bool use_oxDNA2_FENE, bool use_mbf, number mbf_xmax, number mbf_finf, number *_d_stacking_roll, number *_d_stacking_r_roll, number *_d_stacking_tilt, number *_d_stacking_multiplier) {
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:__global__ void dna_forces(number4 *poss, GPU_quat<number> *orientations,  number4 *forces, number4 *torques, int *matrix_neighs, int *number_neighs, LR_bonds *bonds, bool grooving, bool use_debye_huckel, bool use_oxDNA2_coaxial_stacking, bool use_oxDNA2_FENE, bool use_mbf, number mbf_xmax, number mbf_finf, CUDABox<number, number4> *box, number *_d_stacking_roll,  number *_d_stacking_r_roll,  number *_d_stacking_tilt,  number *_d_stacking_multiplier,  number *_d_hb_multiplier) {
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:__forceinline__ __device__ number4 CUDA_rotateVectorAroundVersor(number4& vector, number4& versor, number& angle){
contrib/randisi/src/Interactions/CUDA_DNA2ModInteraction.cuh:	number scalar = CUDA_DOT(vector,versor);
contrib/randisi/src/Interactions/DNA2ModInteraction.cpp:		if( getInputNumber(&inp, "mod_stacking_r_roll_1", &_mod_stacking_r_roll_1,0) == KEY_FOUND and backend == "CUDA")
contrib/randisi/src/Interactions/DNA2ModInteraction.cpp:			throw oxDNAException("mod_stacking_r_roll_1 has been disabled on CUDA to try and prevent Bus Error.");
contrib/randisi/src/Interactions/DNA2ModInteraction.cpp:		if( getInputNumber(&inp, "mod_stacking_r_roll_2", &_mod_stacking_r_roll_2,0) == KEY_FOUND and backend == "CUDA")
contrib/randisi/src/Interactions/DNA2ModInteraction.cpp:			throw oxDNAException("mod_stacking_r_roll_2 has been disabled on CUDA to try and prevent Bus Error.");
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu: * CUDADNA2ModInteraction.cu
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu: *      Author: Ferdinando, after CUDADNAInteraction.cu by Lorenzo
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:#include "CUDADNA2ModInteraction.h"
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:#include "CUDA_DNA2ModInteraction.cuh"
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:CUDADNA2ModInteraction<number, number4>::CUDADNA2ModInteraction() {
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:CUDADNA2ModInteraction<number, number4>::~CUDADNA2ModInteraction() {
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	if(_d_stacking_roll != NULL) CUDA_SAFE_CALL( cudaFree(_d_stacking_roll) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	if(_d_stacking_r_roll != NULL) CUDA_SAFE_CALL( cudaFree(_d_stacking_r_roll) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	if(_d_stacking_tilt != NULL) CUDA_SAFE_CALL( cudaFree(_d_stacking_tilt) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	if(_d_hb_multiplier != NULL) CUDA_SAFE_CALL( cudaFree(_d_hb_multiplier) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	if(_d_stacking_multiplier != NULL) CUDA_SAFE_CALL( cudaFree(_d_stacking_multiplier) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:void CUDADNA2ModInteraction<number, number4>::get_settings(input_file &inp) {
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:			// we don't need the F4_... terms as the macros are used in the CUDA_DNA.cuh file; this doesn't apply for the F2_K term
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:void CUDADNA2ModInteraction<number, number4>::cuda_init(number box_side, int N) {
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDABaseInteraction<number, number4>::cuda_init(box_side, N);
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_hb_multi, &f_copy, sizeof(float)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_N, &N, sizeof(int)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	if(this->_use_edge) CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_n_forces, &this->_n_forces, sizeof(int)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:		CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dh_RC, &_debye_huckel_RC, sizeof(float)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:		CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dh_RHIGH, &_debye_huckel_RHIGH, sizeof(float)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:		CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dh_prefactor, &_debye_huckel_prefactor, sizeof(float)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:		CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dh_B, &_debye_huckel_B, sizeof(float)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:		CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dh_minus_kappa, &_minus_kappa, sizeof(float)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:		CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dh_half_charged_ends, &_debye_huckel_half_charged_ends, sizeof(bool)) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<number>(&_d_stacking_roll, k_size) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<number>(&_d_stacking_r_roll, k_size) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<number>(&_d_stacking_tilt, k_size) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<number>(&_d_stacking_multiplier, k_size) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<number>(&_d_hb_multiplier, k_size) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( cudaMemcpy(_d_stacking_roll, this->_a_stacking_roll, k_size, cudaMemcpyHostToDevice) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	//CUDA_SAFE_CALL( cudaMemcpy(_d_stacking_r_roll, this->_a_stacking_r_roll, k_size, cudaMemcpyHostToDevice) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( cudaMemcpy(_d_stacking_tilt, this->_a_stacking_tilt, k_size, cudaMemcpyHostToDevice) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( cudaMemcpy(_d_stacking_multiplier, this->_a_stacking_multiplier, k_size, cudaMemcpyHostToDevice) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDA_SAFE_CALL( cudaMemcpy(_d_hb_multiplier, this->_a_hb_multiplier, k_size, cudaMemcpyHostToDevice) );
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:void CUDADNA2ModInteraction<number, number4>::compute_forces(CUDABaseList<number, number4> *lists, number4 *d_poss, GPU_quat<number> *d_orientations, number4 *d_forces, number4 *d_torques, LR_bonds *d_bonds, CUDABox<number, number4> *d_box) {
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDASimpleVerletList<number, number4> *_v_lists = dynamic_cast<CUDASimpleVerletList<number, number4> *>(lists);
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:				cudaThreadSynchronize();
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:	CUDANoList<number, number4> *_no_lists = dynamic_cast<CUDANoList<number, number4> *>(lists);
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:template class CUDADNA2ModInteraction<float, float4>;
contrib/randisi/src/Interactions/CUDADNA2ModInteraction.cu:template class CUDADNA2ModInteraction<double, LR_double4>;
contrib/tostiguerra/CMakeLists.txt:if(CUDA)
contrib/tostiguerra/CMakeLists.txt:	find_package("CUDA")
contrib/tostiguerra/CMakeLists.txt:	# same thing but for CUDA libs
contrib/tostiguerra/CMakeLists.txt:	function(cuda_add_library_no_prefix target source)
contrib/tostiguerra/CMakeLists.txt:		cuda_add_library(${target} MODULE EXCLUDE_FROM_ALL ${source} ${ARGN})
contrib/tostiguerra/CMakeLists.txt:		target_link_libraries(${target} ${CUDA_LIBRARIES})
contrib/tostiguerra/CMakeLists.txt:	cuda_add_library_no_prefix(CUDACGNucleicAcidsInteraction src/CUDACGNucleicAcidsInteraction.cu src/CGNucleicAcidsInteraction.cpp)
contrib/tostiguerra/CMakeLists.txt:	ADD_DEPENDENCIES(tostiguerra CUDACGNucleicAcidsInteraction)
contrib/tostiguerra/CMakeLists.txt:endif(CUDA)
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu: * CUDACGNucleicAcidsInteraction.cu
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:#include "CUDACGNucleicAcidsInteraction.h"
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:#include "CUDA/Lists/CUDASimpleVerletList.h"
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:#include "CUDA/Lists/CUDANoList.h"
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:#define CUDA_MAX_SWAP_NEIGHS 20
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:#include "CUDA/cuda_utils/CUDA_lr_common.cuh"
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:struct __align__(16) CUDA_FS_bond {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:struct __align__(16) CUDA_FS_bond_list {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_FS_bond bonds[CUDA_MAX_SWAP_NEIGHS];
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_FS_bond_list() :
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_FS_bond &add_bond() {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:		if(n_bonds > CUDA_MAX_SWAP_NEIGHS) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:__device__ void _WCA(c_number4 &ppos, c_number4 &qpos, c_number &sigma, c_number &sqr_rep_rcut, c_number4 &F, CUDAStressTensor &p_st, CUDABox *box) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:__device__ void _sticky(c_number4 &ppos, c_number4 &p_a1, c_number4 &p_a3, c_number4 &qpos, c_number4 &q_a1, c_number4 &q_a2, c_number4 &q_a3, int eps_idx, int q_idx, c_number4 &F, c_number4 &T, CUDA_FS_bond_list &bond_list,
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:		cudaTextureObject_t tex_eps, CUDAStressTensor &p_st, CUDABox *box) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	c_number sqr_r = CUDA_DOT(patch_dist, patch_dist);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:			c_number cost_a3 = CUDA_DOT(p_a3, q_a3);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:			CUDA_FS_bond &new_bond = bond_list.add_bond();
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:__device__ void _FENE(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDAStressTensor &p_st, CUDABox *box) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	c_number sqr_r = CUDA_DOT(r, r);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:__device__ void _patchy_three_body(CUDA_FS_bond_list &bond_list, c_number4 &F, c_number4 &T, CUDAStressTensor &p_st, c_number4 *forces, c_number4 *torques) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:		CUDA_FS_bond &b1 = bond_list.bonds[bi];
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:			CUDA_FS_bond &b2 = bond_list.bonds[bj];
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:__device__ void _flexibility_three_body(c_number4 &ppos, c_number4 &n1_pos, c_number4 &n2_pos, int n1_idx, int n2_idx, c_number4 &F, c_number4 *poss, c_number4 *three_body_forces, CUDABox *box) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	c_number sqr_dist_pn1 = CUDA_DOT(dist_pn1, dist_pn1);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	c_number sqr_dist_pn2 = CUDA_DOT(dist_pn2, dist_pn2);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	c_number cost = CUDA_DOT(dist_pn1, dist_pn2) * i_pn1_pn2;
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	c_number cost_a1 = CUDA_DOT(p_a1, q_a1);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:__global__ void ps_bonded_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, c_number4 *three_body_forces, LR_bonds *bonded_neighs, bool update_st, CUDAStressTensor *st, CUDABox *box) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDAStressTensor p_st;
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:__global__ void ps_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, c_number4 *three_body_forces, 
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:						cudaTextureObject_t tex_eps, CUDAStressTensor *st, CUDABox *box) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_FS_bond_list bonds;
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDAStressTensor p_st;
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:CUDACGNucleicAcidsInteraction::CUDACGNucleicAcidsInteraction() :
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:CUDACGNucleicAcidsInteraction::~CUDACGNucleicAcidsInteraction() {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_forces));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_three_body_torques));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:void CUDACGNucleicAcidsInteraction::get_settings(input_file &inp) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:void CUDACGNucleicAcidsInteraction::cuda_init(int N) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDABaseInteraction::cuda_init(N);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:		throw oxDNAException("DPS_semiflexibility is not available on CUDA");
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_crossovers, N * sizeof(int)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_crossovers, crossovers.data(), N * sizeof(int), cudaMemcpyHostToDevice));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_forces, N * sizeof(c_number4)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_three_body_torques, N * sizeof(c_number4)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n, &_PS_n, sizeof(int)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_interaction_matrix_size, &interaction_matrix_size, sizeof(int)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_enable_semiflexibility_3b, &_enable_semiflexibility_3b, sizeof(bool)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_enable_patch_stacking, &_enable_patch_stacking, sizeof(bool)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_3b_epsilon, _3b_epsilon.size() * sizeof(float)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_3b_epsilon, h_3b_epsilon.data(), _3b_epsilon.size() * sizeof(float), cudaMemcpyHostToDevice));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	GpuUtils::init_texture_object(&_tex_eps, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat), _d_3b_epsilon, _3b_epsilon.size());
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:void CUDACGNucleicAcidsInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:		CUDA_SAFE_CALL(cudaMemset(_d_st, 0, _N * sizeof(CUDAStressTensor)));
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.cu:	/*number energy = GpuUtils::sum_c_number4_to_double_on_GPU(d_forces, _N);
contrib/tostiguerra/src/CGNucleicAcidsInteraction.cpp:			/* the n3 and n5 members are only used on CUDA */
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h: * CUDACGNucleicAcidsInteraction.h
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:#ifndef CUDACGNUCLEICACIDSINTERACTION_H_
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:#define CUDACGNUCLEICACIDSINTERACTION_H_
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:#include "CUDA/Interactions/CUDABaseInteraction.h"
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h: * @brief CUDA implementation of the {@link CGNucleicAcidsInteraction interaction}.
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:class CUDACGNucleicAcidsInteraction: public CUDABaseInteraction, public CGNucleicAcidsInteraction {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:	cudaTextureObject_t _tex_eps = 0;
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:	CUDACGNucleicAcidsInteraction();
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:	virtual ~CUDACGNucleicAcidsInteraction();
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:	void cuda_init(int N);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:	c_number get_cuda_rcut() {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:	void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box);
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:extern "C" BaseInteraction *make_CUDACGNucleicAcidsInteraction() {
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:	return new CUDACGNucleicAcidsInteraction();
contrib/tostiguerra/src/CUDACGNucleicAcidsInteraction.h:#endif /* CUDACGNUCLEICACIDSINTERACTION_H_ */
select_compute_arch.cmake:#   CUDA_SELECT_NVCC_ARCH_FLAGS(out_variable [target_CUDA_architectures])
select_compute_arch.cmake:#   -- Selects GPU arch flags for nvcc based on target_CUDA_architectures
select_compute_arch.cmake:#      target_CUDA_architectures : Auto | Common | All | LIST(ARCH_AND_PTX ...)
select_compute_arch.cmake:#       - "Auto" detects local machine GPU compute arch at runtime.
select_compute_arch.cmake:#      Returns LIST of flags to be added to CUDA_NVCC_FLAGS in ${out_variable}
select_compute_arch.cmake:#       CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.2(5.0) Maxwell)
select_compute_arch.cmake:#        LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
select_compute_arch.cmake:#      More info on CUDA architectures: https://en.wikipedia.org/wiki/CUDA
select_compute_arch.cmake:if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
select_compute_arch.cmake:  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
select_compute_arch.cmake:      AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
select_compute_arch.cmake:    set(CUDA_VERSION "${CMAKE_MATCH_1}")
select_compute_arch.cmake:# See: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
select_compute_arch.cmake:# https://developer.nvidia.com/cuda-toolkit-archive
select_compute_arch.cmake:# The initial status here is for CUDA 7.0
select_compute_arch.cmake:set(CUDA_KNOWN_GPU_ARCHITECTURES  "Fermi" "Kepler" "Maxwell" "Kepler+Tegra" "Kepler+Tesla" "Maxwell+Tegra")
select_compute_arch.cmake:set(CUDA_COMMON_GPU_ARCHITECTURES "2.0" "2.1" "3.0" "3.5" "5.0" "5.3")
select_compute_arch.cmake:set(CUDA_LIMIT_GPU_ARCHITECTURE "6.0")
select_compute_arch.cmake:set(CUDA_ALL_GPU_ARCHITECTURES "2.0" "2.1" "3.0" "3.2" "3.5" "3.7" "5.0" "5.2" "5.3")
select_compute_arch.cmake:set(_CUDA_MAX_COMMON_ARCHITECTURE "5.2+PTX")
select_compute_arch.cmake:if(CUDA_VERSION VERSION_GREATER_EQUAL "8.0")
select_compute_arch.cmake:  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Pascal")
select_compute_arch.cmake:  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "6.0" "6.1")
select_compute_arch.cmake:  list(APPEND CUDA_ALL_GPU_ARCHITECTURES "6.0" "6.1" "6.2")
select_compute_arch.cmake:  set(_CUDA_MAX_COMMON_ARCHITECTURE "6.2+PTX")
select_compute_arch.cmake:  set(CUDA_LIMIT_GPU_ARCHITECTURE "7.0")
select_compute_arch.cmake:  list(REMOVE_ITEM CUDA_COMMON_GPU_ARCHITECTURES "2.0" "2.1")
select_compute_arch.cmake:if(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
select_compute_arch.cmake:  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Volta")
select_compute_arch.cmake:  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "7.0")
select_compute_arch.cmake:  list(APPEND CUDA_ALL_GPU_ARCHITECTURES "7.0" "7.2")
select_compute_arch.cmake:  set(_CUDA_MAX_COMMON_ARCHITECTURE "7.2+PTX")
select_compute_arch.cmake:  set(CUDA_LIMIT_GPU_ARCHITECTURE "8.0")
select_compute_arch.cmake:  list(REMOVE_ITEM CUDA_KNOWN_GPU_ARCHITECTURES "Fermi")
select_compute_arch.cmake:  list(REMOVE_ITEM CUDA_ALL_GPU_ARCHITECTURES "2.0" "2.1")
select_compute_arch.cmake:if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
select_compute_arch.cmake:  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Turing")
select_compute_arch.cmake:  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "7.5")
select_compute_arch.cmake:  list(APPEND CUDA_ALL_GPU_ARCHITECTURES "7.5")
select_compute_arch.cmake:  set(_CUDA_MAX_COMMON_ARCHITECTURE "7.5+PTX")
select_compute_arch.cmake:  set(CUDA_LIMIT_GPU_ARCHITECTURE "8.0")
select_compute_arch.cmake:  list(REMOVE_ITEM CUDA_COMMON_GPU_ARCHITECTURES "3.0")
select_compute_arch.cmake:# https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html#cuda-general-new-features
select_compute_arch.cmake:# https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html#deprecated-features
select_compute_arch.cmake:if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
select_compute_arch.cmake:  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Ampere")
select_compute_arch.cmake:  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "8.0")
select_compute_arch.cmake:  list(APPEND CUDA_ALL_GPU_ARCHITECTURES "8.0")
select_compute_arch.cmake:  set(_CUDA_MAX_COMMON_ARCHITECTURE "8.0+PTX")
select_compute_arch.cmake:  set(CUDA_LIMIT_GPU_ARCHITECTURE "8.6")
select_compute_arch.cmake:  list(REMOVE_ITEM CUDA_COMMON_GPU_ARCHITECTURES "3.5" "5.0")
select_compute_arch.cmake:  list(REMOVE_ITEM CUDA_ALL_GPU_ARCHITECTURES "3.0" "3.2")
select_compute_arch.cmake:if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
select_compute_arch.cmake:  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "8.6")
select_compute_arch.cmake:  list(APPEND CUDA_ALL_GPU_ARCHITECTURES "8.6")
select_compute_arch.cmake:  set(_CUDA_MAX_COMMON_ARCHITECTURE "8.6+PTX")
select_compute_arch.cmake:  set(CUDA_LIMIT_GPU_ARCHITECTURE "9.0")
select_compute_arch.cmake:list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "${_CUDA_MAX_COMMON_ARCHITECTURE}")
select_compute_arch.cmake:# Check with: cmake -DCUDA_VERSION=7.0 -P select_compute_arch.cmake
select_compute_arch.cmake:  cmake_print_variables(CUDA_KNOWN_GPU_ARCHITECTURES)
select_compute_arch.cmake:  cmake_print_variables(CUDA_COMMON_GPU_ARCHITECTURES)
select_compute_arch.cmake:  cmake_print_variables(CUDA_LIMIT_GPU_ARCHITECTURE)
select_compute_arch.cmake:  cmake_print_variables(CUDA_ALL_GPU_ARCHITECTURES)
select_compute_arch.cmake:# A function for automatic detection of GPUs installed  (if autodetection is enabled)
select_compute_arch.cmake:#   CUDA_DETECT_INSTALLED_GPUS(OUT_VARIABLE)
select_compute_arch.cmake:function(CUDA_DETECT_INSTALLED_GPUS OUT_VARIABLE)
select_compute_arch.cmake:  if(NOT CUDA_GPU_DETECT_OUTPUT)
select_compute_arch.cmake:    if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
select_compute_arch.cmake:      set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cu")
select_compute_arch.cmake:      set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cpp")
select_compute_arch.cmake:      "#include <cuda_runtime.h>\n"
select_compute_arch.cmake:      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
select_compute_arch.cmake:      "    cudaDeviceProp prop;\n"
select_compute_arch.cmake:      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
select_compute_arch.cmake:    if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
select_compute_arch.cmake:              CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
select_compute_arch.cmake:              LINK_LIBRARIES ${CUDA_LIBRARIES}
select_compute_arch.cmake:      set(CUDA_GPU_DETECT_OUTPUT ${compute_capabilities}
select_compute_arch.cmake:        CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
select_compute_arch.cmake:  if(NOT CUDA_GPU_DETECT_OUTPUT)
select_compute_arch.cmake:    message(STATUS "Automatic GPU detection failed. Building for common architectures.")
select_compute_arch.cmake:    set(${OUT_VARIABLE} ${CUDA_COMMON_GPU_ARCHITECTURES} PARENT_SCOPE)
select_compute_arch.cmake:    # Filter based on CUDA version supported archs
select_compute_arch.cmake:    set(CUDA_GPU_DETECT_OUTPUT_FILTERED "")
select_compute_arch.cmake:    separate_arguments(CUDA_GPU_DETECT_OUTPUT)
select_compute_arch.cmake:    foreach(ITEM IN ITEMS ${CUDA_GPU_DETECT_OUTPUT})
select_compute_arch.cmake:        if(CUDA_LIMIT_GPU_ARCHITECTURE AND ITEM VERSION_GREATER_EQUAL CUDA_LIMIT_GPU_ARCHITECTURE)
select_compute_arch.cmake:        list(GET CUDA_COMMON_GPU_ARCHITECTURES -1 NEWITEM)
select_compute_arch.cmake:        string(APPEND CUDA_GPU_DETECT_OUTPUT_FILTERED " ${NEWITEM}")
select_compute_arch.cmake:        string(APPEND CUDA_GPU_DETECT_OUTPUT_FILTERED " ${ITEM}")
select_compute_arch.cmake:    set(${OUT_VARIABLE} ${CUDA_GPU_DETECT_OUTPUT_FILTERED} PARENT_SCOPE)
select_compute_arch.cmake:# Function for selecting GPU arch flags for nvcc based on CUDA architectures from parameter list
select_compute_arch.cmake:#   SELECT_NVCC_ARCH_FLAGS(out_variable [list of CUDA compute archs])
select_compute_arch.cmake:function(CUDA_SELECT_NVCC_ARCH_FLAGS out_variable)
select_compute_arch.cmake:  set(CUDA_ARCH_LIST "${ARGN}")
select_compute_arch.cmake:  if("X${CUDA_ARCH_LIST}" STREQUAL "X" )
select_compute_arch.cmake:    set(CUDA_ARCH_LIST "Auto")
select_compute_arch.cmake:  set(cuda_arch_bin)
select_compute_arch.cmake:  set(cuda_arch_ptx)
select_compute_arch.cmake:  if("${CUDA_ARCH_LIST}" STREQUAL "All")
select_compute_arch.cmake:    set(CUDA_ARCH_LIST ${CUDA_KNOWN_GPU_ARCHITECTURES})
select_compute_arch.cmake:  elseif("${CUDA_ARCH_LIST}" STREQUAL "Common")
select_compute_arch.cmake:    set(CUDA_ARCH_LIST ${CUDA_COMMON_GPU_ARCHITECTURES})
select_compute_arch.cmake:  elseif("${CUDA_ARCH_LIST}" STREQUAL "Auto")
select_compute_arch.cmake:    CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
select_compute_arch.cmake:    message(STATUS "Autodetected CUDA architecture(s): ${CUDA_ARCH_LIST}")
select_compute_arch.cmake:  string(REGEX REPLACE "[ \t]+" ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
select_compute_arch.cmake:  list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
select_compute_arch.cmake:  foreach(arch_name ${CUDA_ARCH_LIST})
select_compute_arch.cmake:        message(SEND_ERROR "Unknown CUDA Architecture Name ${arch_name} in CUDA_SELECT_NVCC_ARCH_FLAGS")
select_compute_arch.cmake:    list(APPEND cuda_arch_bin ${arch_bin})
select_compute_arch.cmake:      list(APPEND cuda_arch_ptx ${arch_ptx})
select_compute_arch.cmake:  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
select_compute_arch.cmake:  string(REGEX REPLACE "\\." "" cuda_arch_ptx "${cuda_arch_ptx}")
select_compute_arch.cmake:  string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
select_compute_arch.cmake:  string(REGEX MATCHALL "[0-9]+"   cuda_arch_ptx "${cuda_arch_ptx}")
select_compute_arch.cmake:  if(cuda_arch_bin)
select_compute_arch.cmake:    list(REMOVE_DUPLICATES cuda_arch_bin)
select_compute_arch.cmake:  if(cuda_arch_ptx)
select_compute_arch.cmake:    list(REMOVE_DUPLICATES cuda_arch_ptx)
select_compute_arch.cmake:  # Tell NVCC to add binaries for the specified GPUs
select_compute_arch.cmake:  foreach(arch ${cuda_arch_bin})
select_compute_arch.cmake:  foreach(arch ${cuda_arch_ptx})
input_options.md:        on CUDA. Defaults to 'no'.
input_options.md:CUDA options:
input_options.md:    [CUDA_list = no|verlet]
input_options.md:        Neighbour lists for CUDA simulations. Defaults to 'no'.
input_options.md:    backend = CUDA
input_options.md:        For CUDA FFS -- NB unlike the CPU implementation, the CUDA
input_options.md:        CUDA FFS is currently only implemented for mixed precision
input_options.md:        one may use 'master conditions' (CUDA FFS only), which allow one to
input_options.md:        EXAMPLES/CUDA_FFS/README file for more information
input_options.md:        CUDA FFS only. Default: False; if False, the simulation will run until
input_options.md:        CUDA FFS only. Mandatory if ffs_generate_flux is True; save a
input_options.md:        CUDA FFS only. Mandatory if ffs_generate_flux is True; stop the
input_options.md:        CUDA FFS only. Mandatory if ffs_generate_flux is True; the prefix used
input_options.md:        CUDA FFS only. Default: False; In a flux generation simulation, set to
input_options.md:        CUDA FFS only. Default: False; in a flux generation simulation, set to
input_options.md:        CUDA FFS only. Default: False; in a flux generation simulation that
input_options.md:        CUDA FFS only. Mandatory if die_on_unexpected_master is True; the
input_options.md:    [CUDA_device = <int>]
input_options.md:        CUDA-enabled device to run the simulation on. If it is not specified
input_options.md:    [CUDA_sort_every = <int>]
input_options.md:        sort particles according to a 3D Hilbert curve every CUDA_sort_every
input_options.md:        Number of threads per block on the CUDA grid. defaults to 2 * the size
input_options.md:    backend = CPU/CUDA
input_options.md:        For CPU FFS/For CUDA FFS -- NB unlike the CPU implementation, the CUDA
input_options.md:        simulation/CUDA FFS is currently only implemented for mixed precision
input_options.md:        one may use 'master conditions' (CUDA FFS only), which allow one to
input_options.md:        EXAMPLES/CUDA_FFS/README file for more information
input_options.md:        CUDA FFS only. Default: False; if False, the simulation will run until
input_options.md:        CUDA FFS only. Mandatory if ffs_generate_flux is True; save a
input_options.md:        CUDA FFS only. Mandatory if ffs_generate_flux is True; stop the
input_options.md:        CUDA FFS only. Mandatory if ffs_generate_flux is True; the prefix used
input_options.md:        CUDA FFS only. Default: False; In a flux generation simulation, set to
input_options.md:        CUDA FFS only. Default: False; in a flux generation simulation, set to
input_options.md:        CUDA FFS only. Default: False; in a flux generation simulation that
input_options.md:        CUDA FFS only. Mandatory if die_on_unexpected_master is True; the
examples/PERSISTENCE_LENGTH/NEW_TOPOLOGY/input_persistences:#CUDA_list = verlet
examples/PERSISTENCE_LENGTH/NEW_TOPOLOGY/input_persistences:#CUDA_device = 0
examples/PERSISTENCE_LENGTH/input_persistences:#CUDA_list = verlet
examples/PERSISTENCE_LENGTH/input_persistences:#CUDA_device = 0
examples/PERSISTENCE_LENGTH/README.md:In order to compute the persistence length, one needs a large number of decorrelated states, so this simulation takes a while.  It can be sped up by running on a GPU rather than the default CPU.
examples/NEW_RELAX_PROCEDURE/README:interaction is that it also works on CUDA, and works in the same way for both
examples/NEW_RELAX_PROCEDURE/README:anymore, as this new procedure should be much faster, also works on CUDA and
examples/METADYNAMICS/metad_interface.py:                    N_walkers=1, use_seq_GPUs=False, p_dict={},
examples/METADYNAMICS/metad_interface.py:                        if use_seq_GPUs:
examples/METADYNAMICS/metad_interface.py:                            input_file["CUDA_device"] = str(w.index)
examples/METADYNAMICS/metad_interface.py:    parser.add_argument("--use_sequential_GPUs", action="store_true", help="Each walker will try to use a dedicated GPU: the first one will attempt to use GPU 1, the second one GPU 2, etc.")
examples/METADYNAMICS/metad_interface.py:    use_seq_GPUs = args.use_sequential_GPUs
examples/METADYNAMICS/metad_interface.py:                    N_walkers=N_walkers, use_seq_GPUs=use_seq_GPUs,
examples/METADYNAMICS/SINGLE_HELIX_BENDING/do.run_CUDA.sh:python3 ../metad_interface.py --A 0.5 --sigma 0.05 --tau 5000 --dX 0.001 --N_walkers 4 --dT 20 --dim 1 --p_fname locs.meta --ratio 0 --angle 0 --Niter 20000 --xmin 0 --xmax 20 --T "295 K" base_CUDA
examples/METADYNAMICS/SINGLE_HELIX_BENDING/base_CUDA/input:backend = CUDA
examples/METADYNAMICS/SINGLE_HELIX_BENDING/base_CUDA/input:CUDA_list = verlet
examples/METADYNAMICS/README.md:The script is heavily based on the code developed by [Kaufhold *et al.*](https://zenodo.org/record/6326800) (see [References](#references)). However, differently from their code this interface uses oxpy, oxDNA's Python bindings, which allows for a much greater flexibility. In addition, directly accessing (and manipulating) oxDNA's data structures can sizeably improve performances, especially when simulations are run on GPUs.
examples/METADYNAMICS/README.md:When used on CUDA-powered simulations, by default the interface launches the processes without any indication about the device that should be used, and in fact will honour the `CUDA_device` key if found in the base input file. Therefore, if GPUs are not set to run in compute mode "EXCLUSIVE_PROCESS", all the walkers will use the same GPU (the one specified by `CUDA_device` or, if this is not present in the base input file, the first available GPU). The `--use_sequential_GPUs` switch can be used to tell each walker to run on a distinct GPU: walker 0 will run on device 0, walker 1 on device 1, *etc.*
examples/METADYNAMICS/README.md:1. Distance between a pair of sets of nucleotides (also available on GPUs).
examples/METADYNAMICS/README.md:The example can be run on CPUs or GPUs with the `do.run_CPU.sh` and `do.run_CUDA.sh` scripts, respectively. Note that for the small system studied here we include a CUDA example for the sake of completeness, since using GPUs will result in much slower sampling compared to CPUs. The script will launch 6 CPU (or 4 GPU) simultaneous walkers which share a bias. The bias (as specified in the `locs.meta` file) is the distance between the centres of mass of nucleotides 30, 31, 32, 29, 28, 27 and 56, 58, 59, 0, 1, 2, that is, the centres of mass of the initial and final bits of the duplex.
examples/CUDA_EXAMPLE/input:backend = CUDA
examples/CUDA_EXAMPLE/input:# if no CUDA_device is specified, then oxDNA will try to 
examples/CUDA_EXAMPLE/input:#CUDA_device = 3
examples/CUDA_EXAMPLE/input:CUDA_list = verlet
examples/CUDA_EXAMPLE/input:CUDA_sort_every = 0
examples/CUDA_EXAMPLE/README.txt:CUDA example
examples/CUDA_EXAMPLE/README.txt:This example runs a short GPU-enabled simulation a system composed by 2048 double strands 
examples/CUDA_EXAMPLE/README.txt:(32768 nucleotides). Note that you need to compile oxDNA with CUDA support (add the flag
examples/CUDA_EXAMPLE/README.txt:-DCUDA=ON to the cmake command) and have a working CUDA installation (CUDA >= 3.2 required). 
examples/FFS_example/FFS_CUDA/input_flux:backend = CUDA
examples/FFS_example/FFS_CUDA/input_flux:CUDA_list = verlet
examples/FFS_example/FFS_CUDA/input_flux:#CUDA_device = 4
examples/FFS_example/FFS_CUDA/README:|CUDA Forward Flux Sampling|
examples/FFS_example/FFS_CUDA/README:# these must be set to use the CUDA FFS backend
examples/FFS_example/FFS_CUDA/README:backend = CUDA
examples/FFS_example/FFS_CUDA/README:There are two modes for the CUDA FFS simulation:
examples/FFS_example/FFS_CUDA/README:NB UNLIKE THE CPU CODE, the GPU implementation does not add columns with the order parameter values to the step-by-step stdout
examples/FFS_example/FFS_CUDA/README:Unlike the CPU FFS backend, the CUDA FFS backend supports the nearly-bonded order parameter as well as the hydrogen bonded and minimum distance order parameters. The nearly-bonded order parameter counts the number of specified nucleotide pairs for which the following is true: At least 6 of the 7 factors that enter into the hydrogen bonding energy term are non-zero. This means that a pair may be counted as nearly bonded only if all but one of the factors is non-zero or if none are zero, in which case there is a finite hydrogen bonding energy. Note that by this definition of being nearly-bonded, every pair that would normally be considered hydrogen bonded is also nearly-bonded. To use the nearly-bonded order parameter, an entry for a hydrogen bond order parameter should be made in the order parameter file, with the energy cutoff set to 64. e.g.
examples/FFS_example/FFS_CUDA/input_shoot:backend = CUDA
examples/FFS_example/FFS_CUDA/input_shoot:CUDA_list = verlet
examples/FFS_example/FFS_CUDA/input_shoot:#CUDA_device = 4
examples/OXPY_REMD/README.md:    CUDA_device
examples/OXPY_REMD/README.md:- For CUDA runs the script assumes that the number of available GPUs is at least equal to the number of
examples/OXPY_REMD/remd.py:    input["CUDA_device"] = str(rank)
examples/OXPY_REMD/input_md:CUDA_list = verlet
examples/OXPY_REMD/input_md:CUDA_sort_every = 0
examples/HAIRPIN_CLOSING_TIME/input:#CUDA_list = verlet
examples/HAIRPIN_CLOSING_TIME/input:#CUDA_device = 0
.cproject:									<listOptionValue builtIn="false" value="/usr/local/cuda/include"/>
.cproject:									<listOptionValue builtIn="false" value="/home/lorenzo/NVIDIA_GPU_Computing_SDK/C/common/inc"/>
.cproject:									<listOptionValue builtIn="false" value="/usr/local/cuda/include"/>
.cproject:									<listOptionValue builtIn="false" value="/home/lorenzo/NVIDIA_GPU_Computing_SDK/C/common/inc"/>
.cproject:			<storageModule buildSystemId="org.eclipse.cdt.managedbuilder.core.configurationDataProvider" id="0.840946717.209146630.421345064" moduleId="org.eclipse.cdt.core.settings" name="CUDA">
.cproject:				<configuration artifactName="binmix" buildProperties="" description="" id="0.840946717.209146630.421345064" name="CUDA" parent="org.eclipse.cdt.build.core.prefbase.cfg">
.cproject:							<builder buildPath="${workspace_loc:/oxDNA}/cuda" id="cdt.managedbuild.builder.gnu.cross.483897514" keepEnvironmentInBuildfile="false" managedBuildOn="false" name="Gnu Make Builder" superClass="cdt.managedbuild.builder.gnu.cross"/>
.cproject:		<configuration configurationName="CUDA">
legacy/UTILS/process_data/Makefile:# change these two lines to match your CUDA installation
legacy/UTILS/process_data/Makefile:DEFINES = -DNOCUDA 
legacy/UTILS/process_data/ProcessData_Backend.cpp:	this->_is_CUDA_sim = false;
legacy/UTILS/process_data/ProcessData_Backend.cpp:		// if we are performing a CUDA simulation then we don't need CPU verlet lists
legacy/UTILS/cadnano_interface.py:def add_slice(cuda_system, vhelix, begin, end, nodes, strands, pos, dir, perp, rot, helix_angles, strand_type, use_seq, seqs):
legacy/UTILS/cadnano_interface.py:    cuda_system.add_strand(new_strands[strand_type].get_slice(begin_slice, end_slice), check_overlap = False)
legacy/UTILS/cadnano_interface.py:    return cuda_system
legacy/UTILS/origami_utils.py:    def __init__(self, system=False, cad2cuda_file = False, visibility = False):
legacy/UTILS/origami_utils.py:        if cad2cuda_file:
legacy/UTILS/origami_utils.py:            self.get_cad2cudadna(cad2cuda_file, visibility = visibility)
legacy/UTILS/origami_utils.py:            for (vhelix, vbase), (strand1, nucs1) in self._cad2cudadna._scaf.iteritems():
legacy/UTILS/origami_utils.py:                    (strand2, nucs2) = self._cad2cudadna._stap[vhelix, vbase]
legacy/UTILS/origami_utils.py:            self.vhelix_indices = sorted(list(set([key[0] for key in self._cad2cudadna._scaf.keys()] + [key[0] for key in self._cad2cudadna._stap.keys()])))
legacy/UTILS/origami_utils.py:            self.vbase_indices = sorted(list(set([key[1] for key in self._cad2cudadna._scaf.keys()] + [key[1] for key in self._cad2cudadna._stap.keys()])))
legacy/UTILS/origami_utils.py:            for (vh, vb) in iter(self._cad2cudadna._scaf):
legacy/UTILS/origami_utils.py:            for (vh, vb) in iter(self._cad2cudadna._stap):
legacy/UTILS/origami_utils.py:            self._cad2cudadna = {}
legacy/UTILS/origami_utils.py:#            print self._cad2cudadna._stap[0,ii]
legacy/UTILS/origami_utils.py:        if self._cad2cudadna == {}:
legacy/UTILS/origami_utils.py:            base.Logger.log("get_corners: build cad2cudadna property first", base.Logger.CRITICAL)
legacy/UTILS/origami_utils.py:        if self._cad2cudadna:
legacy/UTILS/origami_utils.py:                    strand, nucs = self._cad2cudadna._scaf[(vhelix, vbase)]
legacy/UTILS/origami_utils.py:                    strand, nucs = self._cad2cudadna._stap[(vhelix, vbase)]
legacy/UTILS/origami_utils.py:                    strand, nucs = self._cad2cudadna._scaf[(vhelix, vbase)]
legacy/UTILS/origami_utils.py:                    strand, nucs = self._cad2cudadna._stap[(vhelix, vbase)]
legacy/UTILS/origami_utils.py:                strand, nucs1 = self._cad2cudadna._scaf[(vhelix, vbase)]
legacy/UTILS/origami_utils.py:                strand, nucs2 = self._cad2cudadna._stap[(vhelix, vbase)]
legacy/UTILS/origami_utils.py:            base.Logger.log("no cadnano to cudadna file detected, using old and possibly wrong get_nucleotides function", base.Logger.WARNING)
legacy/UTILS/origami_utils.py:            for x in iter(self._cad2cudadna._scaf):
legacy/UTILS/origami_utils.py:    def get_cad2cudadna(self, infile, visibility = False):
legacy/UTILS/origami_utils.py:            self._cad2cudadna = data[0]
legacy/UTILS/origami_utils.py:                            for vh, vb in self._cad2cudadna._scaf.keys():
legacy/UTILS/origami_utils.py:                                    vis_c2cdna._scaf[(vh, vb)] = self._cad2cudadna._scaf[(vh, vb)]
legacy/UTILS/origami_utils.py:                            for vh, vb in self._cad2cudadna._stap.keys():
legacy/UTILS/origami_utils.py:                                    vis_c2cdna._stap[(vh, vb)] = self._cad2cudadna._stap[(vh, vb)]
legacy/UTILS/origami_utils.py:                        for vh, vb in self._cad2cudadna._scaf.keys():
legacy/UTILS/origami_utils.py:                                vis_c2cdna._scaf[(vh, vb)] = self._cad2cudadna._scaf[(vh, vb)]
legacy/UTILS/origami_utils.py:                        for vh, vb in self._cad2cudadna._stap.keys():
legacy/UTILS/origami_utils.py:                                vis_c2cdna._stap[(vh, vb)] = self._cad2cudadna._stap[(vh, vb)]
legacy/UTILS/origami_utils.py:                    self._cad2cudadna = vis_c2cdna
legacy/UTILS/origami_utils.py:            self._cad2cudadna = data
legacy/UTILS/origami_utils.py:                        strand, nucs = self._cad2cudadna._stap[(vhelix, current_vbase)]
legacy/UTILS/origami_utils.py:                strand, nucs = self._cad2cudadna._stap[(vhelix, i)]
legacy/UTILS/origami_utils.py:                strand, nucs = self._cad2cudadna._stap[(vh,vb)]
legacy/UTILS/origami_utils.py:                strand, nucs = self._cad2cudadna._scaf[(vh,vb)]
legacy/UTILS/origami_utils.py:                cstrand, [cnuc] = self._cad2cudadna._stap[(vh,vb)]
legacy/UTILS/origami_utils.py:                nstrand, [nnuc] = self._cad2cudadna._stap[(vhn,vb)]
legacy/UTILS/origami_utils.py:                cstrand, [cnuc] = self._cad2cudadna._scaf[(vh,vb)]
legacy/UTILS/origami_utils.py:                nstrand, [nnuc] = self._cad2cudadna._scaf[(vhn,vb)]
legacy/UTILS/origami_utils.py:        assumes no insertions/deletions when using cad2cudadna
legacy/UTILS/origami_utils.py:            strandid, nucids = self._cad2cudadna._scaf[(vh, vb)]
legacy/UTILS/origami_utils.py:            strandid, nucids = self._cad2cudadna._stap[(vh, vb)]
legacy/UTILS/parse_options.py:CATEGORIES["CUDA"] = []
legacy/UTILS/parse_options.py:CATEGORIES["CUDA"] = [
legacy/UTILS/parse_options.py:                      "CUDA/CUDAForces.h",
legacy/UTILS/parse_options.py:                      "CUDA/CUDAForces.h",
legacy/UTILS/parse_options.py:                      "CUDA/*/*.h"
legacy/UTILS/parse_options.py:                     "CUDA/Backends/FFS_MD_CUDAMixedBackend.h"
legacy/UTILS/parse_options.py:        "CUDA" : True,
.gitignore:cuda/
.project:					<value>dbg=1 nocuda=1</value>
src/Managers/SimManager.h: * @brief Manages a simulation, be it MC, MD, on GPU or on CPU.
src/Utilities/Timings.cpp:#ifdef NOCUDA
src/Utilities/Timings.cpp:#include <cuda_runtime_api.h>
src/Utilities/Timings.cpp:#define SYNCHRONIZE() cudaDeviceSynchronize()
src/Interactions/InteractionFactory.cpp:		// in order to avoid small mismatches between potential energies computed on the GPU and 
src/Interactions/InteractionFactory.cpp:		if(backend.compare("CUDA") == 0) return std::make_shared<DNAInteraction_nomesh>();
src/Interactions/InteractionFactory.cpp:		if(backend.compare("CUDA") == 0) return std::make_shared<DNA2Interaction_nomesh>();
src/Interactions/TEPInteraction.h:	bool _is_on_cuda;
src/Interactions/TEPInteraction.cpp:	_is_on_cuda = false;
src/Interactions/TEPInteraction.cpp:	// check whether it's on CUDA, so that if unimplemented features are used oxDNA dies swollen
src/Interactions/TEPInteraction.cpp:	if(backend == "CUDA")
src/Interactions/TEPInteraction.cpp:		_is_on_cuda = true;
src/Interactions/TEPInteraction.cpp:	if(setNonNegativeNumber(&inp, "TEP_th_b_0_default", &_th_b_0_default, 0, "_th_b_0_default - default value for the equilibrium bending angle") && backend == "CUDA") {
src/Interactions/TEPInteraction.cpp:		throw oxDNAException("can't set TEP_th_b_0_default when on CUDA - non-zero equilibrium bending angle not implemented on CUDA.");
src/Interactions/TEPInteraction.cpp:	if(setNonNegativeNumber(&inp, "TEP_beta_0_default", &_beta_0_default, 0, "_beta_0_default - default value for the equilibrium bending direction") && backend == "CUDA") {
src/Interactions/TEPInteraction.cpp:		throw oxDNAException("can't set TEP_beta_0_default when on CUDA - non-zero equilibrium bending angle not implemented on CUDA.");
src/Interactions/TEPInteraction.cpp:						if(_is_on_cuda)
src/Interactions/TEPInteraction.cpp:							throw oxDNAException(" Setting th_b_0 in the topology file is not implemented on CUDA");
src/Interactions/TEPInteraction.cpp:						if(_is_on_cuda)
src/Interactions/TEPInteraction.cpp:							throw oxDNAException(" Setting beta_0 in the topology file is not implemented on CUDA");
src/CUDA/CUDAForces.h: * CUDAForces.h
src/CUDA/CUDAForces.h:#ifndef CUDAFORCES_H_
src/CUDA/CUDAForces.h:#define CUDAFORCES_H_
src/CUDA/CUDAForces.h:#include "CUDAUtils.h"
src/CUDA/CUDAForces.h:#define CUDA_TRAP_NO_FORCE -1
src/CUDA/CUDAForces.h:#define CUDA_TRAP_CONSTANT 0
src/CUDA/CUDAForces.h:#define CUDA_TRAP_MUTUAL 1
src/CUDA/CUDAForces.h:#define CUDA_TRAP_MOVING 2
src/CUDA/CUDAForces.h:#define CUDA_REPULSION_PLANE 3
src/CUDA/CUDAForces.h:#define CUDA_REPULSION_PLANE_MOVING 4
src/CUDA/CUDAForces.h:#define CUDA_TRAP_MOVING_LOWDIM 5
src/CUDA/CUDAForces.h:#define CUDA_LJ_WALL 6
src/CUDA/CUDAForces.h:#define CUDA_REPULSIVE_SPHERE 7
src/CUDA/CUDAForces.h:#define CUDA_CONSTANT_RATE_TORQUE 8
src/CUDA/CUDAForces.h:#define CUDA_GENERIC_CENTRAL_FORCE 9
src/CUDA/CUDAForces.h:#define CUDA_LJ_CONE 10
src/CUDA/CUDAForces.h:#define CUDA_REPULSIVE_SPHERE_SMOOTH 11
src/CUDA/CUDAForces.h:#define CUDA_REPULSIVE_ELLIPSOID 12
src/CUDA/CUDAForces.h:#define CUDA_COM_FORCE 13
src/CUDA/CUDAForces.h:#define CUDA_LR_COM_TRAP 14
src/CUDA/CUDAForces.h:#define CUDA_YUKAWA_SPHERE 15
src/CUDA/CUDAForces.h:#define CUDA_ATTRACTION_PLANE 16
src/CUDA/CUDAForces.h: * @brief CUDA version of a ConstantRateForce.
src/CUDA/CUDAForces.h:void init_ConstantRateForce_from_CPU(constant_rate_force *cuda_force, ConstantRateForce *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_TRAP_CONSTANT;
src/CUDA/CUDAForces.h:	cuda_force->F0 = cpu_force->_F0;
src/CUDA/CUDAForces.h:	cuda_force->dir_as_centre = cpu_force->dir_as_centre;
src/CUDA/CUDAForces.h:	cuda_force->rate = cpu_force->_rate;
src/CUDA/CUDAForces.h:	cuda_force->x = cpu_force->_direction.x;
src/CUDA/CUDAForces.h:	cuda_force->y = cpu_force->_direction.y;
src/CUDA/CUDAForces.h:	cuda_force->z = cpu_force->_direction.z;
src/CUDA/CUDAForces.h: * @brief CUDA version of a MutualTrap.
src/CUDA/CUDAForces.h:void init_MutualTrap_from_CPU(mutual_trap *cuda_force, MutualTrap *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_TRAP_MUTUAL;
src/CUDA/CUDAForces.h:	cuda_force->rate = cpu_force->_rate;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->stiff_rate = cpu_force->_stiff_rate;
src/CUDA/CUDAForces.h:	cuda_force->r0 = cpu_force->_r0;
src/CUDA/CUDAForces.h:	cuda_force->p_ind = cpu_force->_p_ptr->index;
src/CUDA/CUDAForces.h:	cuda_force->PBC = cpu_force->PBC;
src/CUDA/CUDAForces.h: * @brief CUDA version of a MovingTrap.
src/CUDA/CUDAForces.h:void init_MovingTrap_from_CPU(moving_trap *cuda_force, MovingTrap *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_TRAP_MOVING;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->rate = cpu_force->_rate;
src/CUDA/CUDAForces.h:	cuda_force->pos0 = make_float3(cpu_force->_pos0.x, cpu_force->_pos0.y, cpu_force->_pos0.z);
src/CUDA/CUDAForces.h:	cuda_force->dir = make_float3(cpu_force->_direction.x, cpu_force->_direction.y, cpu_force->_direction.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of a LowdimMovingTrap.
src/CUDA/CUDAForces.h:void init_LowdimMovingTrap_from_CPU(lowdim_moving_trap *cuda_force, LowdimMovingTrap *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_TRAP_MOVING_LOWDIM;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->rate = cpu_force->_rate;
src/CUDA/CUDAForces.h:	cuda_force->pos0 = make_float3(cpu_force->_pos0.x, cpu_force->_pos0.y, cpu_force->_pos0.z);
src/CUDA/CUDAForces.h:	cuda_force->dir = make_float3(cpu_force->_direction.x, cpu_force->_direction.y, cpu_force->_direction.z);
src/CUDA/CUDAForces.h:	cuda_force->visX = cpu_force->_visX;
src/CUDA/CUDAForces.h:	cuda_force->visY = cpu_force->_visY;
src/CUDA/CUDAForces.h:	cuda_force->visZ = cpu_force->_visZ;
src/CUDA/CUDAForces.h: * @brief CUDA version of a RepulsionPlane.
src/CUDA/CUDAForces.h:void init_RepulsionPlane_from_CPU(repulsion_plane *cuda_force, RepulsionPlane *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_REPULSION_PLANE;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->position = cpu_force->_position;
src/CUDA/CUDAForces.h:	cuda_force->dir = make_float3(cpu_force->_direction.x, cpu_force->_direction.y, cpu_force->_direction.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of an AttractionPlane.
src/CUDA/CUDAForces.h:void init_AttractionPlane_from_CPU(attraction_plane *cuda_force, AttractionPlane *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_ATTRACTION_PLANE;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->position = cpu_force->_position;
src/CUDA/CUDAForces.h:	cuda_force->dir = make_float3(cpu_force->_direction.x, cpu_force->_direction.y, cpu_force->_direction.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of a RepulsionPlaneMoving.
src/CUDA/CUDAForces.h:void init_RepulsionPlaneMoving_from_CPU(repulsion_plane_moving *cuda_force, RepulsionPlaneMoving *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_REPULSION_PLANE_MOVING;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->dir = make_float3(cpu_force->_direction.x, cpu_force->_direction.y, cpu_force->_direction.z);
src/CUDA/CUDAForces.h:	cuda_force->low_idx = cpu_force->low_idx;
src/CUDA/CUDAForces.h:	cuda_force->high_idx = cpu_force->high_idx;
src/CUDA/CUDAForces.h: * @brief CUDA version of a RepulsiveSphere.
src/CUDA/CUDAForces.h:void init_RepulsiveSphere_from_CPU(repulsive_sphere *cuda_force, RepulsiveSphere *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_REPULSIVE_SPHERE;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->rate = cpu_force->_rate;
src/CUDA/CUDAForces.h:	cuda_force->r0 = cpu_force->_r0;
src/CUDA/CUDAForces.h:	cuda_force->r_ext = cpu_force->_r_ext;
src/CUDA/CUDAForces.h:	cuda_force->centre = make_float3(cpu_force->_center.x, cpu_force->_center.y, cpu_force->_center.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of a RepulsiveSphereSmooth.
src/CUDA/CUDAForces.h:void init_RepulsiveSphereSmooth_from_CPU(repulsive_sphere_smooth *cuda_force, RepulsiveSphereSmooth *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_REPULSIVE_SPHERE_SMOOTH;
src/CUDA/CUDAForces.h:	cuda_force->r0 = cpu_force->_r0;
src/CUDA/CUDAForces.h:	cuda_force->r_ext = cpu_force->_r_ext;
src/CUDA/CUDAForces.h:	cuda_force->smooth = cpu_force->_smooth;
src/CUDA/CUDAForces.h:	cuda_force->alpha = cpu_force->_alpha;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->centre = make_float3(cpu_force->_center.x, cpu_force->_center.y, cpu_force->_center.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of an LJWall.
src/CUDA/CUDAForces.h:void init_LJWall_from_CPU(LJ_wall *cuda_force, LJWall *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_LJ_WALL;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->position = cpu_force->_position;
src/CUDA/CUDAForces.h:	cuda_force->n = cpu_force->_n;
src/CUDA/CUDAForces.h:	cuda_force->cutoff = cpu_force->_cutoff;
src/CUDA/CUDAForces.h:	cuda_force->sigma = cpu_force->_sigma;
src/CUDA/CUDAForces.h:	cuda_force->dir = make_float3(cpu_force->_direction.x, cpu_force->_direction.y, cpu_force->_direction.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of a ConstantRateTorque.
src/CUDA/CUDAForces.h:void init_ConstantRateTorque_from_CPU(constant_rate_torque *cuda_force, ConstantRateTorque *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_CONSTANT_RATE_TORQUE;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->F0 = cpu_force->_F0;
src/CUDA/CUDAForces.h:	cuda_force->rate = cpu_force->_rate;
src/CUDA/CUDAForces.h:	cuda_force->center = make_float3(cpu_force->_center.x, cpu_force->_center.y, cpu_force->_center.z);
src/CUDA/CUDAForces.h:	cuda_force->pos0 = make_float3(cpu_force->_pos0.x, cpu_force->_pos0.y, cpu_force->_pos0.z);
src/CUDA/CUDAForces.h:	cuda_force->axis = make_float3(cpu_force->_axis.x, cpu_force->_axis.y, cpu_force->_axis.z);
src/CUDA/CUDAForces.h:	cuda_force->mask = make_float3(cpu_force->_mask.x, cpu_force->_mask.y, cpu_force->_mask.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of a GenericCentralForce.
src/CUDA/CUDAForces.h:void init_GenericCentralForce_from_CPU(generic_constant_force *cuda_force, GenericCentralForce *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_GENERIC_CENTRAL_FORCE;
src/CUDA/CUDAForces.h:	cuda_force->F0 = cpu_force->_F0;
src/CUDA/CUDAForces.h:	cuda_force->inner_cut_off_sqr = cpu_force->inner_cut_off_sqr;
src/CUDA/CUDAForces.h:	cuda_force->outer_cut_off_sqr = cpu_force->outer_cut_off_sqr;
src/CUDA/CUDAForces.h:	cuda_force->x = cpu_force->center.x;
src/CUDA/CUDAForces.h:	cuda_force->y = cpu_force->center.y;
src/CUDA/CUDAForces.h:	cuda_force->z = cpu_force->center.z;
src/CUDA/CUDAForces.h: * @brief CUDA version of an LJCone.
src/CUDA/CUDAForces.h:void init_LJCone_from_CPU(LJ_cone *cuda_force, LJCone *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_LJ_CONE;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->n = cpu_force->_n;
src/CUDA/CUDAForces.h:	cuda_force->cutoff = cpu_force->_cutoff;
src/CUDA/CUDAForces.h:	cuda_force->sigma = cpu_force->_sigma;
src/CUDA/CUDAForces.h:	cuda_force->alpha = cpu_force->_alpha;
src/CUDA/CUDAForces.h:	cuda_force->sin_alpha = cpu_force->_sin_alpha;
src/CUDA/CUDAForces.h:	cuda_force->dir = make_float3(cpu_force->_direction.x, cpu_force->_direction.y, cpu_force->_direction.z);
src/CUDA/CUDAForces.h:	cuda_force->pos0 = make_float3(cpu_force->_pos0.x, cpu_force->_pos0.y, cpu_force->_pos0.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of a RepulsiveEllipsoid.
src/CUDA/CUDAForces.h:void init_RepulsiveEllipsoid_from_CPU(repulsive_ellipsoid *cuda_force, RepulsiveEllipsoid *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_REPULSIVE_ELLIPSOID;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->centre = make_float3(cpu_force->_centre.x, cpu_force->_centre.y, cpu_force->_centre.z);
src/CUDA/CUDAForces.h:	cuda_force->r_1 = make_float3(cpu_force->_r_1.x, cpu_force->_r_1.y, cpu_force->_r_1.z);
src/CUDA/CUDAForces.h:	cuda_force->r_2 = make_float3(cpu_force->_r_2.x, cpu_force->_r_2.y, cpu_force->_r_2.z);
src/CUDA/CUDAForces.h: * @brief CUDA version of a COMForce.
src/CUDA/CUDAForces.h:void init_COMForce_from_CPU(COM_force *cuda_force, COMForce *cpu_force, bool first_time) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_COM_FORCE;
src/CUDA/CUDAForces.h:	cuda_force->stiff = cpu_force->_stiff;
src/CUDA/CUDAForces.h:	cuda_force->r0 = cpu_force->_r0;
src/CUDA/CUDAForces.h:	cuda_force->rate = cpu_force->_rate;
src/CUDA/CUDAForces.h:	cuda_force->n_com = cpu_force->_com_list.size();
src/CUDA/CUDAForces.h:	cuda_force->n_ref = cpu_force->_ref_list.size();
src/CUDA/CUDAForces.h:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&cuda_force->com_indexes, sizeof(int) * local_com_indexes.size()));
src/CUDA/CUDAForces.h:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&cuda_force->ref_indexes, sizeof(int) * local_ref_indexes.size()));
src/CUDA/CUDAForces.h:	CUDA_SAFE_CALL(cudaMemcpy(cuda_force->com_indexes, local_com_indexes.data(), sizeof(int) * local_com_indexes.size(), cudaMemcpyHostToDevice));
src/CUDA/CUDAForces.h:	CUDA_SAFE_CALL(cudaMemcpy(cuda_force->ref_indexes, local_ref_indexes.data(), sizeof(int) * local_ref_indexes.size(), cudaMemcpyHostToDevice));
src/CUDA/CUDAForces.h: * @brief CUDA version of a COMForce.
src/CUDA/CUDAForces.h:void init_LTCOMTrap_from_CPU(lt_com_trap *cuda_force, LTCOMTrap *cpu_force, bool first_time) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_LR_COM_TRAP;
src/CUDA/CUDAForces.h:	cuda_force->xmin = cpu_force->xmin;
src/CUDA/CUDAForces.h:	cuda_force->xmax = cpu_force->xmax;
src/CUDA/CUDAForces.h:	cuda_force->N_grid = cpu_force->N_grid;
src/CUDA/CUDAForces.h:	cuda_force->dX = cpu_force->dX;
src/CUDA/CUDAForces.h:	cuda_force->mode = cpu_force->_mode;
src/CUDA/CUDAForces.h:	cuda_force->PBC = cpu_force->PBC;
src/CUDA/CUDAForces.h:	cuda_force->p1a_size = cpu_force->_p1a_ptr.size();
src/CUDA/CUDAForces.h:	cuda_force->p2a_size = cpu_force->_p2a_ptr.size();
src/CUDA/CUDAForces.h:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&cuda_force->p1a, sizeof(int) * local_p1a.size()));
src/CUDA/CUDAForces.h:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&cuda_force->p2a, sizeof(int) * local_p2a.size()));
src/CUDA/CUDAForces.h:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number>(&cuda_force->potential_grid, sizeof(c_number) * local_grid.size()));
src/CUDA/CUDAForces.h:	CUDA_SAFE_CALL(cudaMemcpy(cuda_force->p1a, local_p1a.data(), sizeof(int) * local_p1a.size(), cudaMemcpyHostToDevice));
src/CUDA/CUDAForces.h:	CUDA_SAFE_CALL(cudaMemcpy(cuda_force->p2a, local_p2a.data(), sizeof(int) * local_p2a.size(), cudaMemcpyHostToDevice));
src/CUDA/CUDAForces.h:	CUDA_SAFE_CALL(cudaMemcpy(cuda_force->potential_grid, local_grid.data(), sizeof(c_number) * local_grid.size(), cudaMemcpyHostToDevice));
src/CUDA/CUDAForces.h: * @brief CUDA version of a YukawaSphere.
src/CUDA/CUDAForces.h:void init_YukawaSphere_from_CPU(Yukawa_sphere *cuda_force, YukawaSphere *cpu_force) {
src/CUDA/CUDAForces.h:	cuda_force->type = CUDA_YUKAWA_SPHERE;
src/CUDA/CUDAForces.h:	cuda_force->center = make_float3(cpu_force->_center.x, cpu_force->_center.y, cpu_force->_center.z);
src/CUDA/CUDAForces.h:	cuda_force->radius = cpu_force->_radius;
src/CUDA/CUDAForces.h:	cuda_force->epsilon = cpu_force->_epsilon;
src/CUDA/CUDAForces.h:	cuda_force->sigma = cpu_force->_sigma;
src/CUDA/CUDAForces.h:	cuda_force->WCA_n = cpu_force->_WCA_n;
src/CUDA/CUDAForces.h:	cuda_force->WCA_cutoff = cpu_force->_WCA_cutoff;
src/CUDA/CUDAForces.h:	cuda_force->debye_length = cpu_force->_debye_length;
src/CUDA/CUDAForces.h:	cuda_force->debye_A = cpu_force->_debye_A;
src/CUDA/CUDAForces.h:	cuda_force->cutoff = cpu_force->_cutoff;
src/CUDA/CUDAForces.h: * @brief Used internally by CUDA classes to provide an inheritance-like mechanism for external forces.
src/CUDA/CUDAForces.h:union CUDA_trap {
src/CUDA/CUDAForces.h:#endif /* CUDAFORCES_H_ */
src/CUDA/Interactions/CUDATEPInteraction.h: * CUDATEPInteraction.h
src/CUDA/Interactions/CUDATEPInteraction.h:#ifndef CUDATEPINTERACTION_H_
src/CUDA/Interactions/CUDATEPInteraction.h:#define CUDATEPINTERACTION_H_
src/CUDA/Interactions/CUDATEPInteraction.h:#include "CUDABaseInteraction.h"
src/CUDA/Interactions/CUDATEPInteraction.h: * @brief CUDA implementation of the {@link TEPInteraction TEP interaction}.
src/CUDA/Interactions/CUDATEPInteraction.h:class CUDATEPInteraction: public CUDABaseInteraction, public TEPInteraction {
src/CUDA/Interactions/CUDATEPInteraction.h:	CUDATEPInteraction();
src/CUDA/Interactions/CUDATEPInteraction.h:	virtual ~CUDATEPInteraction();
src/CUDA/Interactions/CUDATEPInteraction.h:	void cuda_init(int N) override;
src/CUDA/Interactions/CUDATEPInteraction.h:	c_number get_cuda_rcut() {
src/CUDA/Interactions/CUDATEPInteraction.h:	void compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box);
src/CUDA/Interactions/CUDATEPInteraction.h:#endif /* CUDATEPINTERACTION_H_ */
src/CUDA/Interactions/CUDA_DNA.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number rsqr = CUDA_DOT(r, r);
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number t4 = CUDA_LRACOS(CUDA_DOT(n3z, n5z));
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number cost5 = CUDA_DOT(n5z, rstackdir);
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number t5 = CUDA_LRACOS(cost5);
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number cost6 = -CUDA_DOT(n3z, rstackdir);
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number t6 = CUDA_LRACOS(cost6);
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number cosphi1 = CUDA_DOT(n5y, rbackref) / rbackrefmod;
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number cosphi2 = CUDA_DOT(n3y, rbackref) / rbackrefmod;
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number ra2 = CUDA_DOT(rstackdir, n5y);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number ra1 = CUDA_DOT(rstackdir, n5x);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number rb1 = CUDA_DOT(rstackdir, n3x);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number a2b1 = CUDA_DOT(n5y, n3x);
src/CUDA/Interactions/CUDA_DNA.cuh:		ra2 = CUDA_DOT(rstackdir, n3y);
src/CUDA/Interactions/CUDA_DNA.cuh:		rb1 = CUDA_DOT(rstackdir, n5x);
src/CUDA/Interactions/CUDA_DNA.cuh:		a2b1 = CUDA_DOT(n3y, n5x);
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost2 = -CUDA_DOT(b1, rhydrodir);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t2 = CUDA_LRACOS(cost2);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost3 = CUDA_DOT(a1, rhydrodir);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t3 = CUDA_LRACOS(cost3);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost7 = -CUDA_DOT(rhydrodir, b3);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t7 = CUDA_LRACOS(cost7);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost8 = CUDA_DOT(rhydrodir, a3);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t8 = CUDA_LRACOS(cost8);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost2 = -CUDA_DOT(b1, rcstackdir);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t2 = CUDA_LRACOS(cost2);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost3 = CUDA_DOT(a1, rcstackdir);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t3 = CUDA_LRACOS(cost3);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost7 = -CUDA_DOT(rcstackdir, b3);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t7 = CUDA_LRACOS(cost7);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost8 = CUDA_DOT(rcstackdir, a3);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t8 = CUDA_LRACOS(cost8);
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number rstackmodsqr = CUDA_DOT(rstack, rstack);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost5 = CUDA_DOT(a3, rstackdir);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t5 = CUDA_LRACOS(cost5);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number cost6 = -CUDA_DOT(b3, rstackdir);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t6 = CUDA_LRACOS(cost6);
src/CUDA/Interactions/CUDA_DNA.cuh:			cosphi3 = CUDA_DOT(rstackdir, (_cross(rbackbonerefdir, a1)));
src/CUDA/Interactions/CUDA_DNA.cuh:				c_number a2b1 = CUDA_DOT(a2, b1);
src/CUDA/Interactions/CUDA_DNA.cuh:				c_number a3b1 = CUDA_DOT(a3, b1);
src/CUDA/Interactions/CUDA_DNA.cuh:				c_number ra1 = CUDA_DOT(rstackdir, a1);
src/CUDA/Interactions/CUDA_DNA.cuh:				c_number ra2 = CUDA_DOT(rstackdir, a2);
src/CUDA/Interactions/CUDA_DNA.cuh:				c_number ra3 = CUDA_DOT(rstackdir, a3);
src/CUDA/Interactions/CUDA_DNA.cuh:				c_number rb1 = CUDA_DOT(rstackdir, b1);
src/CUDA/Interactions/CUDA_DNA.cuh:__global__ void dna_forces_edge_nonbonded(const c_number4 __restrict__ *poss, const GPU_quat __restrict__ *orientations, c_number4 __restrict__ *forces,
src/CUDA/Interactions/CUDA_DNA.cuh:		bool use_debye_huckel, bool use_oxDNA2_coaxial_stacking, bool update_st, CUDAStressTensor *st, const CUDABox *box) {
src/CUDA/Interactions/CUDA_DNA.cuh:	if(CUDA_DOT(dT, dT) > (c_number) 0.f) LR_atomicAddXYZ(&(torques[from_index]), dT);
src/CUDA/Interactions/CUDA_DNA.cuh:	if(CUDA_DOT(dF, dF) > (c_number) 0.f) {
src/CUDA/Interactions/CUDA_DNA.cuh:			CUDAStressTensor p_st;
src/CUDA/Interactions/CUDA_DNA.cuh:	if(CUDA_DOT(dT, dT) > (c_number) 0.f) LR_atomicAddXYZ(&(torques[to_index]), dT);
src/CUDA/Interactions/CUDA_DNA.cuh:__global__ void dna_forces_edge_bonded(const c_number4 __restrict__ *poss, const GPU_quat __restrict__ *orientations, c_number4 __restrict__ *forces,
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number mbf_finf, bool update_st, CUDAStressTensor *st) {
src/CUDA/Interactions/CUDA_DNA.cuh:	CUDAStressTensor p_st;
src/CUDA/Interactions/CUDA_DNA.cuh:__global__ void dna_forces(const c_number4 __restrict__ *poss, const GPU_quat __restrict__ *orientations, c_number4 __restrict__ *forces,
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number mbf_finf, bool update_st, CUDAStressTensor *st, const CUDABox *box) {
src/CUDA/Interactions/CUDA_DNA.cuh:	CUDAStressTensor p_st;
src/CUDA/Interactions/CUDA_DNA.cuh:__global__ void hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDABox *box) {
src/CUDA/Interactions/CUDA_DNA.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Interactions/CUDA_DNA.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t2 = CUDA_LRACOS(-CUDA_DOT(b1, rhydrodir));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t3 = CUDA_LRACOS(CUDA_DOT(a1, rhydrodir));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t7 = CUDA_LRACOS(-CUDA_DOT(rhydrodir, b3));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t8 = CUDA_LRACOS(CUDA_DOT(rhydrodir, a3));
src/CUDA/Interactions/CUDA_DNA.cuh:__global__ void near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDABox *box) {
src/CUDA/Interactions/CUDA_DNA.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Interactions/CUDA_DNA.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Interactions/CUDA_DNA.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t2 = CUDA_LRACOS(-CUDA_DOT(b1, rhydrodir));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t3 = CUDA_LRACOS(CUDA_DOT(a1, rhydrodir));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t7 = CUDA_LRACOS(-CUDA_DOT(rhydrodir, b3));
src/CUDA/Interactions/CUDA_DNA.cuh:		c_number t8 = CUDA_LRACOS(CUDA_DOT(rhydrodir, a3));
src/CUDA/Interactions/CUDA_DNA.cuh:__global__ void dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDABox *box) {
src/CUDA/Interactions/CUDA_DNA.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Interactions/CUDA_DNA.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Interactions/CUDABaseInteraction.cu: * CUDABaseInteraction.cu
src/CUDA/Interactions/CUDABaseInteraction.cu:#include "CUDABaseInteraction.h"
src/CUDA/Interactions/CUDABaseInteraction.cu:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Interactions/CUDABaseInteraction.cu:CUDABaseInteraction::CUDABaseInteraction() {
src/CUDA/Interactions/CUDABaseInteraction.cu:CUDABaseInteraction::~CUDABaseInteraction() {
src/CUDA/Interactions/CUDABaseInteraction.cu:		if(_d_edge_forces != NULL) CUDA_SAFE_CALL(cudaFree(_d_edge_forces));
src/CUDA/Interactions/CUDABaseInteraction.cu:		if(_d_edge_torques != NULL) CUDA_SAFE_CALL(cudaFree(_d_edge_torques));
src/CUDA/Interactions/CUDABaseInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_st));
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::_sum_edge_forces(c_number4 *d_forces) {
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::_sum_edge_forces_torques(c_number4 *d_forces, c_number4 *d_torques) {
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::get_cuda_settings(input_file &inp) {
src/CUDA/Interactions/CUDABaseInteraction.cu:	getInputInt(&inp, "CUDA_update_stress_tensor_every", &update_st_every, 0);
src/CUDA/Interactions/CUDABaseInteraction.cu:			throw oxDNAException("The selected CUDA interaction is not compatible with 'use_edge = true'");
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::cuda_init(int N) {
src/CUDA/Interactions/CUDABaseInteraction.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_edge_forces, size));
src/CUDA/Interactions/CUDABaseInteraction.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_edge_torques, size));
src/CUDA/Interactions/CUDABaseInteraction.cu:		CUDA_SAFE_CALL(cudaMemset(_d_edge_forces, 0, size));
src/CUDA/Interactions/CUDABaseInteraction.cu:		CUDA_SAFE_CALL(cudaMemset(_d_edge_torques, 0, size));
src/CUDA/Interactions/CUDABaseInteraction.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_st, N * sizeof(CUDAStressTensor)));
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::set_launch_cfg(CUDA_kernel_cfg &launch_cfg) {
src/CUDA/Interactions/CUDABaseInteraction.cu:	__device__ CUDAStressTensor operator()(const c_number4 &v) const {
src/CUDA/Interactions/CUDABaseInteraction.cu:		return CUDAStressTensor(
src/CUDA/Interactions/CUDABaseInteraction.cu:StressTensor CUDABaseInteraction::CPU_stress_tensor(c_number4 *vels) {
src/CUDA/Interactions/CUDABaseInteraction.cu:	thrust::device_ptr<CUDAStressTensor> t_st = thrust::device_pointer_cast(_d_st);
src/CUDA/Interactions/CUDABaseInteraction.cu:	CUDAStressTensor st_sum = thrust::reduce(t_st, t_st + _N, CUDAStressTensor());
src/CUDA/Interactions/CUDABaseInteraction.cu:	st_sum += thrust::transform_reduce(t_vels, t_vels + _N, vel_to_st(), CUDAStressTensor(), thrust::plus<CUDAStressTensor>());
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDABaseInteraction.cu:	throw oxDNAException("On CUDA, FFS is only implemented for the DNA and RNA interactions");
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::_near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDABaseInteraction.cu:	throw oxDNAException("On CUDA, FFS is only implemented for the DNA and RNA interactions");
src/CUDA/Interactions/CUDABaseInteraction.cu:void CUDABaseInteraction::_dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDA_kernel_cfg _ffs_dist_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDABaseInteraction.cu:	throw oxDNAException("On CUDA, FFS is only implemented for the DNA and RNA interactions");
src/CUDA/Interactions/CUDAPatchyInteraction.h: * CUDAPatchyInteraction.h
src/CUDA/Interactions/CUDAPatchyInteraction.h:#ifndef CUDAPATCHYINTERACTION_H_
src/CUDA/Interactions/CUDAPatchyInteraction.h:#define CUDAPATCHYINTERACTION_H_
src/CUDA/Interactions/CUDAPatchyInteraction.h:#include "CUDABaseInteraction.h"
src/CUDA/Interactions/CUDAPatchyInteraction.h:#define CUDA_MAX_PATCHES 5
src/CUDA/Interactions/CUDAPatchyInteraction.h: * @brief CUDA implementation of the {@link PatchyInteraction patchy interaction}.
src/CUDA/Interactions/CUDAPatchyInteraction.h:class CUDAPatchyInteraction: public CUDABaseInteraction, public PatchyInteraction {
src/CUDA/Interactions/CUDAPatchyInteraction.h:	CUDAPatchyInteraction();
src/CUDA/Interactions/CUDAPatchyInteraction.h:	virtual ~CUDAPatchyInteraction();
src/CUDA/Interactions/CUDAPatchyInteraction.h:	void cuda_init(int N) override;
src/CUDA/Interactions/CUDAPatchyInteraction.h:	c_number get_cuda_rcut() {
src/CUDA/Interactions/CUDAPatchyInteraction.h:	void compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box);
src/CUDA/Interactions/CUDAPatchyInteraction.h:#endif /* CUDAPATCHYINTERACTION_H_ */
src/CUDA/Interactions/CUDALJInteraction.cu: * CUDALJInteraction.cu
src/CUDA/Interactions/CUDALJInteraction.cu:#include "CUDALJInteraction.h"
src/CUDA/Interactions/CUDALJInteraction.cu:#include "CUDA_LJ.cuh"
src/CUDA/Interactions/CUDALJInteraction.cu:#include "../Lists/CUDASimpleVerletList.h"
src/CUDA/Interactions/CUDALJInteraction.cu:#include "../Lists/CUDANoList.h"
src/CUDA/Interactions/CUDALJInteraction.cu:CUDALJInteraction::CUDALJInteraction() {
src/CUDA/Interactions/CUDALJInteraction.cu:CUDALJInteraction::~CUDALJInteraction() {
src/CUDA/Interactions/CUDALJInteraction.cu:void CUDALJInteraction::get_settings(input_file &inp) {
src/CUDA/Interactions/CUDALJInteraction.cu:void CUDALJInteraction::cuda_init(int N) {
src/CUDA/Interactions/CUDALJInteraction.cu:	CUDABaseInteraction::cuda_init(N);
src/CUDA/Interactions/CUDALJInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
src/CUDA/Interactions/CUDALJInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_LJ_n, &this->_n, 3 * sizeof(int)));
src/CUDA/Interactions/CUDALJInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &this->_n_forces, sizeof(int)));
src/CUDA/Interactions/CUDALJInteraction.cu:void CUDALJInteraction::compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box) {
src/CUDA/Interactions/CUDATEPInteraction.cu: * CUDATEPInteraction.cu
src/CUDA/Interactions/CUDATEPInteraction.cu:#include "CUDATEPInteraction.h"
src/CUDA/Interactions/CUDATEPInteraction.cu:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Interactions/CUDATEPInteraction.cu:	c_number rnorm = CUDA_DOT(r, r);
src/CUDA/Interactions/CUDATEPInteraction.cu:	c_number cosine = CUDA_DOT(up, uq);
src/CUDA/Interactions/CUDATEPInteraction.cu:__device__ void _bonded_particle_particle(c_number4 &n3_pos, c_number4 &up, c_number4 &fp, c_number4 &vp, c_number4 &n5_pos, c_number4 &uq, c_number4 &fq, c_number4 &vq, c_number4 &F, c_number4 &T, CUDABox *box, c_number kb1, c_number kb2, c_number tu, c_number tk, c_number kt_pref, bool alignment_only = false) {
src/CUDA/Interactions/CUDATEPInteraction.cu:	c_number rmod = sqrtf(CUDA_DOT(r, r));
src/CUDA/Interactions/CUDATEPInteraction.cu:		c_number M = CUDA_DOT(fp, fq) + CUDA_DOT(vp, vq);
src/CUDA/Interactions/CUDATEPInteraction.cu:		c_number L = 1.f + CUDA_DOT(up, uq);
src/CUDA/Interactions/CUDATEPInteraction.cu:		 F.w += MD_kb[0]*(1.f - CUDA_DOT(up, uq))*kb1;*/
src/CUDA/Interactions/CUDATEPInteraction.cu:	c_number4 force = MD_ka[0] * (ba_up - ba_tp * CUDA_DOT(ba_up, ba_tp) / SQR(rmod)) / rmod;
src/CUDA/Interactions/CUDATEPInteraction.cu:	F.w += MD_ka[0] * (1.f - CUDA_DOT(ba_up, ba_tp) / rmod);
src/CUDA/Interactions/CUDATEPInteraction.cu:	c_number scalar = CUDA_DOT(vector, versor);
src/CUDA/Interactions/CUDATEPInteraction.cu:__global__ void TEP_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, int *matrix_neighs, int *number_neighs, LR_bonds *bonds, CUDABox *box, c_number *kb1_pref, c_number *kb2_pref, c_number *xk_bending, c_number *xu_bending, c_number *kt_pref, c_number4 *o_vects, c_number4 *w_vects, llint step) {
src/CUDA/Interactions/CUDATEPInteraction.cu:#include "../Lists/CUDASimpleVerletList.h"
src/CUDA/Interactions/CUDATEPInteraction.cu:#include "../Lists/CUDANoList.h"
src/CUDA/Interactions/CUDATEPInteraction.cu:CUDATEPInteraction::CUDATEPInteraction() {
src/CUDA/Interactions/CUDATEPInteraction.cu:CUDATEPInteraction::~CUDATEPInteraction() {
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(_d_kb1_pref != NULL) CUDA_SAFE_CALL(cudaFree(_d_kb1_pref));
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(_d_kb2_pref != NULL) CUDA_SAFE_CALL(cudaFree(_d_kb2_pref));
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(_d_xk_bending != NULL) CUDA_SAFE_CALL(cudaFree(_d_xk_bending));
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(_d_xu_bending != NULL) CUDA_SAFE_CALL(cudaFree(_d_xu_bending));
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(_d_kt_pref != NULL) CUDA_SAFE_CALL(cudaFree(_d_kt_pref));
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(_d_o_vects != NULL) CUDA_SAFE_CALL(cudaFree(_d_o_vects));
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(_d_w_vects != NULL) CUDA_SAFE_CALL(cudaFree(_d_w_vects));
src/CUDA/Interactions/CUDATEPInteraction.cu:void CUDATEPInteraction::get_settings(input_file &inp) {
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(getInputInt(&inp, "CUDA_sort_every", &sort_every, 0) == KEY_FOUND) {
src/CUDA/Interactions/CUDATEPInteraction.cu:	if(this->_prefer_harmonic_over_fene) throw oxDNAException("The 'prefer_harmonic_over_fene' option is not compatible with CUDA");
src/CUDA/Interactions/CUDATEPInteraction.cu:void CUDATEPInteraction::cuda_init(int N) {
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDABaseInteraction::cuda_init(N);
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_kb1_pref, k_size));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_kb2_pref, k_size));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_xk_bending, k_size));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_xu_bending, k_size));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_kt_pref, k_size));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_kb1_pref, this->_kb1_pref, k_size, cudaMemcpyHostToDevice));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_kb2_pref, this->_kb2_pref, k_size, cudaMemcpyHostToDevice));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_xk_bending, this->_xk_bending, k_size, cudaMemcpyHostToDevice));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_xu_bending, this->_xu_bending, k_size, cudaMemcpyHostToDevice));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_kt_pref, this->_kt_pref, k_size, cudaMemcpyHostToDevice));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_o_vects, 2 * sizeof(c_number4)));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_w_vects, 2 * sizeof(c_number4)));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_o_vects, h_o_vects, 2 * sizeof(c_number4), cudaMemcpyHostToDevice));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_w_vects, h_w_vects, 2 * sizeof(c_number4), cudaMemcpyHostToDevice));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_o_modulus, o_modulus, 2 * sizeof(float)));
src/CUDA/Interactions/CUDATEPInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
src/CUDA/Interactions/CUDATEPInteraction.cu:void CUDATEPInteraction::compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box) {
src/CUDA/Interactions/CUDA_RNA.cuh:struct CUDAModel {
src/CUDA/Interactions/CUDA_RNA.cuh:__constant__ CUDAModel rnamodel;
src/CUDA/Interactions/CUDA_RNA.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number rsqr = CUDA_DOT(r, r);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number t4 = CUDA_LRACOS(CUDA_DOT(n3z, n5z));
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number cost5 = CUDA_DOT(n5z, rstackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number t5 = CUDA_LRACOS(cost5);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number cost6 = -CUDA_DOT(n3z, rstackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number t6 = CUDA_LRACOS(cost6);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number cosphi1 = CUDA_DOT(n5y, rbackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number cosphi2 = CUDA_DOT(n3y, rbackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number costB1 = -CUDA_DOT(rbackdir, n5bbvector_3);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number costB2 = -CUDA_DOT(rbackdir, n3bbvector_5);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number tB1 = CUDA_LRACOS(costB1);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number tB2 = CUDA_LRACOS(costB2);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost2 = -CUDA_DOT(b1, rhydrodir);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t2 = CUDA_LRACOS(cost2);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost3 = CUDA_DOT(a1, rhydrodir);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t3 = CUDA_LRACOS(cost3);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost7 = -CUDA_DOT(rhydrodir, b3);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t7 = CUDA_LRACOS(cost7);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost8 = CUDA_DOT(rhydrodir, a3);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t8 = CUDA_LRACOS(cost8);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost2 = -CUDA_DOT(b1, rcstackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t2 = CUDA_LRACOS(cost2);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost3 = CUDA_DOT(a1, rcstackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t3 = CUDA_LRACOS(cost3);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost7 = -CUDA_DOT(rcstackdir, b3);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t7 = CUDA_LRACOS(cost7);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost8 = CUDA_DOT(rcstackdir, a3);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t8 = CUDA_LRACOS(cost8);
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number rstackmodsqr = CUDA_DOT(rstack, rstack);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost5 = CUDA_DOT(a3, rstackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t5 = CUDA_LRACOS(cost5);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cost6 = -CUDA_DOT(b3, rstackdir);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t6 = CUDA_LRACOS(cost6);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cosphi3 = CUDA_DOT(rstackdir, (_cross(rbackbonedir, a1)));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number cosphi4 = CUDA_DOT(rstackdir, (_cross(rbackbonedir, b1)));
src/CUDA/Interactions/CUDA_RNA.cuh:			c_number rb_dot_a1rs = CUDA_DOT(rbackbonedir, a1rs);
src/CUDA/Interactions/CUDA_RNA.cuh:			c_number rs_dot_rba1 = CUDA_DOT(rstackdir, rba1);
src/CUDA/Interactions/CUDA_RNA.cuh:			c_number rb_dot_b1rs = CUDA_DOT(rbackbonedir, b1rs);
src/CUDA/Interactions/CUDA_RNA.cuh:			c_number rs_dot_rbb1 = CUDA_DOT(rstackdir, rbb1);
src/CUDA/Interactions/CUDA_RNA.cuh:__global__ void rna_forces_edge_nonbonded(const c_number4 __restrict__ *poss, const GPU_quat __restrict__ *orientations,
src/CUDA/Interactions/CUDA_RNA.cuh:		int n_edges, const int *is_strand_end, bool average, bool use_debye_huckel, bool mismatch_repulsion, CUDABox *box) {
src/CUDA/Interactions/CUDA_RNA.cuh:	if(CUDA_DOT(dT, dT) > (c_number) 0.f) LR_atomicAddXYZ(&(torques[from_index]), dT);
src/CUDA/Interactions/CUDA_RNA.cuh:	if(CUDA_DOT(dF, dF) > (c_number) 0.f) {
src/CUDA/Interactions/CUDA_RNA.cuh:	if(CUDA_DOT(dT, dT) > (c_number) 0.f) LR_atomicAddXYZ(&(torques[to_index]), dT);
src/CUDA/Interactions/CUDA_RNA.cuh:__global__ void rna_forces_edge_bonded(const c_number4 __restrict__ *poss, const GPU_quat __restrict__ *orientations,
src/CUDA/Interactions/CUDA_RNA.cuh:__global__ void rna_forces(const c_number4 __restrict__ *poss, const GPU_quat __restrict__ *orientations, c_number4 __restrict__ *forces,
src/CUDA/Interactions/CUDA_RNA.cuh:		bool use_debye_huckel, bool mismatch_repulsion, bool use_mbf, c_number mbf_xmax, c_number mbf_finf, CUDABox *box) {
src/CUDA/Interactions/CUDA_RNA.cuh:__global__ void rna_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDABox *box) {
src/CUDA/Interactions/CUDA_RNA.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Interactions/CUDA_RNA.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t2 = CUDA_LRACOS(-CUDA_DOT(b1, rhydrodir));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t3 = CUDA_LRACOS(CUDA_DOT(a1, rhydrodir));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t7 = CUDA_LRACOS(-CUDA_DOT(rhydrodir, b3));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t8 = CUDA_LRACOS(CUDA_DOT(rhydrodir, a3));
src/CUDA/Interactions/CUDA_RNA.cuh:__global__ void rna_near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDABox *box) {
src/CUDA/Interactions/CUDA_RNA.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Interactions/CUDA_RNA.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Interactions/CUDA_RNA.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t2 = CUDA_LRACOS(-CUDA_DOT(b1, rhydrodir));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t3 = CUDA_LRACOS(CUDA_DOT(a1, rhydrodir));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t7 = CUDA_LRACOS(-CUDA_DOT(rhydrodir, b3));
src/CUDA/Interactions/CUDA_RNA.cuh:		c_number t8 = CUDA_LRACOS(CUDA_DOT(rhydrodir, a3));
src/CUDA/Interactions/CUDA_RNA.cuh:__global__ void rna_dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDABox *box) {
src/CUDA/Interactions/CUDA_RNA.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Interactions/CUDA_RNA.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Interactions/CUDABaseInteraction.h: * CUDABaseInteraction.h
src/CUDA/Interactions/CUDABaseInteraction.h:#ifndef CUDABASEINTERACTION_H_
src/CUDA/Interactions/CUDABaseInteraction.h:#define CUDABASEINTERACTION_H_
src/CUDA/Interactions/CUDABaseInteraction.h:#include "../CUDAUtils.h"
src/CUDA/Interactions/CUDABaseInteraction.h:#include "../Lists/CUDABaseList.h"
src/CUDA/Interactions/CUDABaseInteraction.h:#include "../cuda_utils/CUDABox.h"
src/CUDA/Interactions/CUDABaseInteraction.h: * @brief Abstract class providing an interface for CUDA-based interactions.
src/CUDA/Interactions/CUDABaseInteraction.h:class CUDABaseInteraction {
src/CUDA/Interactions/CUDABaseInteraction.h:	CUDA_kernel_cfg _launch_cfg;
src/CUDA/Interactions/CUDABaseInteraction.h:	CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg;
src/CUDA/Interactions/CUDABaseInteraction.h:	CUDA_kernel_cfg _ffs_dist_precalc_kernel_cfg;
src/CUDA/Interactions/CUDABaseInteraction.h:	CUDA_kernel_cfg _ffs_hb_eval_kernel_cfg;
src/CUDA/Interactions/CUDABaseInteraction.h:	CUDA_kernel_cfg _ffs_dist_eval_kernel_cfg;
src/CUDA/Interactions/CUDABaseInteraction.h:	CUDAStressTensor *_d_st = nullptr, *_h_st = nullptr;
src/CUDA/Interactions/CUDABaseInteraction.h:	CUDABaseInteraction();
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual ~CUDABaseInteraction();
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual void get_cuda_settings(input_file &inp);
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual void cuda_init(int N);
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual c_number get_cuda_rcut() = 0;
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual void sync_GPU() {}
src/CUDA/Interactions/CUDABaseInteraction.h:	void set_launch_cfg(CUDA_kernel_cfg &launch_cfg);
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual void compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox *d_box) = 0;
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual void _hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg hb_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual void _near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg hb_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDABaseInteraction.h:	virtual void _dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDA_kernel_cfg dist_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDABaseInteraction.h:#endif /* CUDABASEINTERACTION_H_ */
src/CUDA/Interactions/CUDARNAInteraction.h: * CUDARNAInteraction.h
src/CUDA/Interactions/CUDARNAInteraction.h:#ifndef CUDARNAINTERACTION_H_
src/CUDA/Interactions/CUDARNAInteraction.h:#define CUDARNAINTERACTION_H_
src/CUDA/Interactions/CUDARNAInteraction.h:#include "CUDABaseInteraction.h"
src/CUDA/Interactions/CUDARNAInteraction.h: * @brief CUDA implementation of the oxRNA model, as provided by RNAInteraction.
src/CUDA/Interactions/CUDARNAInteraction.h:class CUDARNAInteraction: public CUDABaseInteraction, public RNAInteraction {
src/CUDA/Interactions/CUDARNAInteraction.h:	CUDARNAInteraction();
src/CUDA/Interactions/CUDARNAInteraction.h:	virtual ~CUDARNAInteraction();
src/CUDA/Interactions/CUDARNAInteraction.h:	void cuda_init(int N) override;
src/CUDA/Interactions/CUDARNAInteraction.h:	c_number get_cuda_rcut() {
src/CUDA/Interactions/CUDARNAInteraction.h:	void compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box);
src/CUDA/Interactions/CUDARNAInteraction.h:	void _hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg hb_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDARNAInteraction.h:	void _near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg hb_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDARNAInteraction.h:	void _dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDA_kernel_cfg dist_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDARNAInteraction.h:#endif /* CUDARNAINTERACTION_H_ */
src/CUDA/Interactions/CUDAInteractionFactory.h: * CUDAInteractionFactory.h
src/CUDA/Interactions/CUDAInteractionFactory.h:#ifndef CUDAINTERACTIONFACTORY_H_
src/CUDA/Interactions/CUDAInteractionFactory.h:#define CUDAINTERACTIONFACTORY_H_
src/CUDA/Interactions/CUDAInteractionFactory.h:#include "CUDABaseInteraction.h"
src/CUDA/Interactions/CUDAInteractionFactory.h: * @brief Static factory class. Its only public method builds a {@link CUDABaseInteraction CUDA interaction} as specified in the input file.
src/CUDA/Interactions/CUDAInteractionFactory.h:class CUDAInteractionFactory {
src/CUDA/Interactions/CUDAInteractionFactory.h:	CUDAInteractionFactory();
src/CUDA/Interactions/CUDAInteractionFactory.h:	virtual ~CUDAInteractionFactory();
src/CUDA/Interactions/CUDAInteractionFactory.h:	static std::shared_ptr<CUDABaseInteraction> make_interaction(input_file &inp);
src/CUDA/Interactions/CUDAInteractionFactory.h:#endif /* CUDAINTERACTIONFACTORY_H_ */
src/CUDA/Interactions/CUDAPatchyInteraction.cu: * CUDAPatchyInteraction.cu
src/CUDA/Interactions/CUDAPatchyInteraction.cu:#include "CUDAPatchyInteraction.h"
src/CUDA/Interactions/CUDAPatchyInteraction.cu:#include "CUDA_Patchy.cuh"
src/CUDA/Interactions/CUDAPatchyInteraction.cu:#include "../Lists/CUDASimpleVerletList.h"
src/CUDA/Interactions/CUDAPatchyInteraction.cu:#include "../Lists/CUDANoList.h"
src/CUDA/Interactions/CUDAPatchyInteraction.cu:CUDAPatchyInteraction::CUDAPatchyInteraction() {
src/CUDA/Interactions/CUDAPatchyInteraction.cu:CUDAPatchyInteraction::~CUDAPatchyInteraction() {
src/CUDA/Interactions/CUDAPatchyInteraction.cu:void CUDAPatchyInteraction::get_settings(input_file &inp) {
src/CUDA/Interactions/CUDAPatchyInteraction.cu:void CUDAPatchyInteraction::cuda_init(int N) {
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	CUDABaseInteraction::cuda_init(N);
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patches, &this->_N_patches, sizeof(int)));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	if(this->_is_binary) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N_patches, &this->_N_patches_B, sizeof(int), sizeof(int)));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_patch_rcut, &f_copy, sizeof(float)));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_patch_pow_alpha, &f_copy, sizeof(float)));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	float4 base_patches[CUDA_MAX_PATCHES];
src/CUDA/Interactions/CUDAPatchyInteraction.cu:			c_number factor = 0.5 / sqrt(CUDA_DOT(base_patches[j], base_patches[j]));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_base_patches, base_patches, sizeof(float4)*n_patches, i*sizeof(float4)*CUDA_MAX_PATCHES));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	if(this->_N_patches > CUDA_MAX_PATCHES) throw oxDNAException("CUDA supports only particles with up to %d patches", CUDA_MAX_PATCHES);
src/CUDA/Interactions/CUDAPatchyInteraction.cu:	if(this->_use_edge) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &this->_n_forces, sizeof(int)));
src/CUDA/Interactions/CUDAPatchyInteraction.cu:void CUDAPatchyInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box) {
src/CUDA/Interactions/CUDAInteractionFactory.cu: * CUDAInteractionFactory.cpp
src/CUDA/Interactions/CUDAInteractionFactory.cu:#include "CUDAInteractionFactory.h"
src/CUDA/Interactions/CUDAInteractionFactory.cu:#include "CUDADNAInteraction.h"
src/CUDA/Interactions/CUDAInteractionFactory.cu:#include "CUDALJInteraction.h"
src/CUDA/Interactions/CUDAInteractionFactory.cu:#include "CUDAPatchyInteraction.h"
src/CUDA/Interactions/CUDAInteractionFactory.cu:#include "CUDATEPInteraction.h"
src/CUDA/Interactions/CUDAInteractionFactory.cu:#include "CUDARNAInteraction.h"
src/CUDA/Interactions/CUDAInteractionFactory.cu:CUDAInteractionFactory::CUDAInteractionFactory() {
src/CUDA/Interactions/CUDAInteractionFactory.cu:CUDAInteractionFactory::~CUDAInteractionFactory() {
src/CUDA/Interactions/CUDAInteractionFactory.cu:std::shared_ptr<CUDABaseInteraction> CUDAInteractionFactory::make_interaction(input_file &inp) {
src/CUDA/Interactions/CUDAInteractionFactory.cu:	if(!inter_type.compare("DNA") || !inter_type.compare("DNA_nomesh") || !inter_type.compare("DNA2")) return std::make_shared<CUDADNAInteraction>();
src/CUDA/Interactions/CUDAInteractionFactory.cu:	else if(!inter_type.compare("RNA") || !inter_type.compare("RNA2")  ) return std::make_shared<CUDARNAInteraction>();
src/CUDA/Interactions/CUDAInteractionFactory.cu:	else if(!inter_type.compare("LJ")) return std::make_shared<CUDALJInteraction>();
src/CUDA/Interactions/CUDAInteractionFactory.cu:	else if(!inter_type.compare("patchy")) return std::make_shared<CUDAPatchyInteraction>();
src/CUDA/Interactions/CUDAInteractionFactory.cu:	else if(inter_type.compare("TEP") == 0) return std::make_shared<CUDATEPInteraction>();
src/CUDA/Interactions/CUDAInteractionFactory.cu:		std::string cuda_name(inter_type);
src/CUDA/Interactions/CUDAInteractionFactory.cu:		cuda_name = "CUDA" + cuda_name;
src/CUDA/Interactions/CUDAInteractionFactory.cu:		std::shared_ptr<CUDABaseInteraction> res = std::dynamic_pointer_cast<CUDABaseInteraction>(PluginManager::instance()->get_interaction(cuda_name));
src/CUDA/Interactions/CUDAInteractionFactory.cu:			throw oxDNAException ("CUDA interaction '%s' not found. Aborting", cuda_name.c_str());
src/CUDA/Interactions/CUDADNAInteraction.cu: * CUDADNAInteraction.cu
src/CUDA/Interactions/CUDADNAInteraction.cu:#include "CUDADNAInteraction.h"
src/CUDA/Interactions/CUDADNAInteraction.cu:#include "CUDA_DNA.cuh"
src/CUDA/Interactions/CUDADNAInteraction.cu:#include "../Lists/CUDASimpleVerletList.h"
src/CUDA/Interactions/CUDADNAInteraction.cu:#include "../Lists/CUDANoList.h"
src/CUDA/Interactions/CUDADNAInteraction.cu:CUDADNAInteraction::CUDADNAInteraction() {
src/CUDA/Interactions/CUDADNAInteraction.cu:CUDADNAInteraction::~CUDADNAInteraction() {
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_is_strand_end));
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::get_settings(input_file &inp) {
src/CUDA/Interactions/CUDADNAInteraction.cu:			// we don't need the F4_... terms as the macros are used in the CUDA_DNA.cuh file; this doesn't apply for the F2_K term
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::cuda_init(int N) {
src/CUDA/Interactions/CUDADNAInteraction.cu:	CUDABaseInteraction::cuda_init(N);
src/CUDA/Interactions/CUDADNAInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_hb_multi, &f_copy, sizeof(float)));
src/CUDA/Interactions/CUDADNAInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
src/CUDA/Interactions/CUDADNAInteraction.cu:	if(this->_use_edge) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &this->_n_forces, sizeof(int)));
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_RC, &_debye_huckel_RC, sizeof(float)));
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_RHIGH, &_debye_huckel_RHIGH, sizeof(float)));
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_prefactor, &_debye_huckel_prefactor, sizeof(float)));
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_B, &_debye_huckel_B, sizeof(float)));
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_minus_kappa, &_minus_kappa, sizeof(float)));
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_half_charged_ends, &_debye_huckel_half_charged_ends, sizeof(bool)));
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::_on_T_update() {
src/CUDA/Interactions/CUDADNAInteraction.cu:	cuda_init(_N);
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::_init_strand_ends(LR_bonds *d_bonds) {
src/CUDA/Interactions/CUDADNAInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_is_strand_end, sizeof(int) * _N));
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::compute_forces(CUDABaseList *lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box) {
src/CUDA/Interactions/CUDADNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemset(_d_st, 0, _N * sizeof(CUDAStressTensor)));
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::_near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDADNAInteraction.cu:void CUDADNAInteraction::_dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDA_kernel_cfg _ffs_dist_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDALJInteraction.h: * CUDALJInteraction.h
src/CUDA/Interactions/CUDALJInteraction.h:#ifndef CUDALJINTERACTION_H_
src/CUDA/Interactions/CUDALJInteraction.h:#define CUDALJINTERACTION_H_
src/CUDA/Interactions/CUDALJInteraction.h:#include "CUDABaseInteraction.h"
src/CUDA/Interactions/CUDALJInteraction.h: * @brief CUDA implementation of the {@link LJInteraction Lennard-Jones interaction}.
src/CUDA/Interactions/CUDALJInteraction.h:class CUDALJInteraction: public CUDABaseInteraction, public LJInteraction {
src/CUDA/Interactions/CUDALJInteraction.h:	CUDALJInteraction();
src/CUDA/Interactions/CUDALJInteraction.h:	virtual ~CUDALJInteraction();
src/CUDA/Interactions/CUDALJInteraction.h:	void cuda_init(int N) override;
src/CUDA/Interactions/CUDALJInteraction.h:	c_number get_cuda_rcut() {
src/CUDA/Interactions/CUDALJInteraction.h:	void compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box);
src/CUDA/Interactions/CUDALJInteraction.h:#endif /* CUDALJINTERACTION_H_ */
src/CUDA/Interactions/CUDADNAInteraction.h: * CUDADNAInteraction.h
src/CUDA/Interactions/CUDADNAInteraction.h:#ifndef CUDADNAINTERACTION_H_
src/CUDA/Interactions/CUDADNAInteraction.h:#define CUDADNAINTERACTION_H_
src/CUDA/Interactions/CUDADNAInteraction.h:#include "CUDABaseInteraction.h"
src/CUDA/Interactions/CUDADNAInteraction.h: * @brief CUDA implementation of the oxDNA model, as provided by DNAInteraction.
src/CUDA/Interactions/CUDADNAInteraction.h:class CUDADNAInteraction: public CUDABaseInteraction, public DNAInteraction {
src/CUDA/Interactions/CUDADNAInteraction.h:	CUDADNAInteraction();
src/CUDA/Interactions/CUDADNAInteraction.h:	virtual ~CUDADNAInteraction();
src/CUDA/Interactions/CUDADNAInteraction.h:	void cuda_init(int N) override;
src/CUDA/Interactions/CUDADNAInteraction.h:	c_number get_cuda_rcut() {
src/CUDA/Interactions/CUDADNAInteraction.h:	void compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_qorientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box);
src/CUDA/Interactions/CUDADNAInteraction.h:	void _hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg hb_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDADNAInteraction.h:	void _near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg hb_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDADNAInteraction.h:	void _dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDA_kernel_cfg dist_kernel_cfg, CUDABox*d_box);
src/CUDA/Interactions/CUDADNAInteraction.h:#endif /* CUDADNAINTERACTION_H_ */
src/CUDA/Interactions/CUDA_Patchy.cuh:__constant__ float4 MD_base_patches[2][CUDA_MAX_PATCHES];
src/CUDA/Interactions/CUDA_Patchy.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Interactions/CUDA_Patchy.cuh:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &a1, c_number4 &a2, c_number4 &a3, c_number4 &b1, c_number4 &b2, c_number4 &b3, c_number4 &F, c_number4 &torque, CUDABox *box) {
src/CUDA/Interactions/CUDA_Patchy.cuh:	c_number sqr_r = CUDA_DOT(r, r);
src/CUDA/Interactions/CUDA_Patchy.cuh:			c_number dist = CUDA_DOT(patch_dist, patch_dist);
src/CUDA/Interactions/CUDA_Patchy.cuh:__global__ void patchy_forces_edge(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, edge_bond *edge_list, int n_edges, CUDABox *box) {
src/CUDA/Interactions/CUDA_Patchy.cuh:	GPU_quat po = orientations[b.from];
src/CUDA/Interactions/CUDA_Patchy.cuh:	GPU_quat qo = orientations[b.to];
src/CUDA/Interactions/CUDA_Patchy.cuh:__global__ void patchy_forces(c_number4 *poss, GPU_quat *orientations, c_number4 *forces, c_number4 *torques, int *matrix_neighs, int *number_neighs, CUDABox *box) {
src/CUDA/Interactions/CUDA_Patchy.cuh:	GPU_quat po = orientations[IND];
src/CUDA/Interactions/CUDA_Patchy.cuh:			GPU_quat qo = orientations[k_index];
src/CUDA/Interactions/CUDARNAInteraction.cu: * CUDARNAInteraction.cu
src/CUDA/Interactions/CUDARNAInteraction.cu:#include "CUDARNAInteraction.h"
src/CUDA/Interactions/CUDARNAInteraction.cu:#include "CUDA_RNA.cuh"
src/CUDA/Interactions/CUDARNAInteraction.cu:#include "../Lists/CUDASimpleVerletList.h"
src/CUDA/Interactions/CUDARNAInteraction.cu:#include "../Lists/CUDANoList.h"
src/CUDA/Interactions/CUDARNAInteraction.cu://this function is necessary, as CUDA does not allow to define a constant memory for a class
src/CUDA/Interactions/CUDARNAInteraction.cu:void copy_Model_to_CUDAModel(Model& model_from, CUDAModel& model_to) {
src/CUDA/Interactions/CUDARNAInteraction.cu:CUDARNAInteraction::CUDARNAInteraction() {
src/CUDA/Interactions/CUDARNAInteraction.cu:CUDARNAInteraction::~CUDARNAInteraction() {
src/CUDA/Interactions/CUDARNAInteraction.cu:		CUDA_SAFE_CALL(cudaFree(_d_is_strand_end));
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::get_settings(input_file &inp) {
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::cuda_init(int N) {
src/CUDA/Interactions/CUDARNAInteraction.cu:	CUDABaseInteraction::cuda_init(N);
src/CUDA/Interactions/CUDARNAInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_hb_multi, &f_copy, sizeof(float)));
src/CUDA/Interactions/CUDARNAInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &N, sizeof(int)));
src/CUDA/Interactions/CUDARNAInteraction.cu:	CUDAModel cudamodel;
src/CUDA/Interactions/CUDARNAInteraction.cu:	copy_Model_to_CUDAModel(*(model), cudamodel);
src/CUDA/Interactions/CUDARNAInteraction.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(rnamodel, &cudamodel, sizeof(CUDAModel)));
src/CUDA/Interactions/CUDARNAInteraction.cu:	if(_use_edge) CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_n_forces, &_n_forces, sizeof(int)));
src/CUDA/Interactions/CUDARNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_RC, &_debye_huckel_RC, sizeof(float)));
src/CUDA/Interactions/CUDARNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_RHIGH, &_debye_huckel_RHIGH, sizeof(float)));
src/CUDA/Interactions/CUDARNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_prefactor, &_debye_huckel_prefactor, sizeof(float)));
src/CUDA/Interactions/CUDARNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_B, &_debye_huckel_B, sizeof(float)));
src/CUDA/Interactions/CUDARNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_minus_kappa, &_minus_kappa, sizeof(float)));
src/CUDA/Interactions/CUDARNAInteraction.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dh_half_charged_ends, &_debye_huckel_half_charged_ends, sizeof(bool)));
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::_on_T_update() {
src/CUDA/Interactions/CUDARNAInteraction.cu:	cuda_init(_N);
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::_init_strand_ends(LR_bonds *d_bonds) {
src/CUDA/Interactions/CUDARNAInteraction.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_is_strand_end, sizeof(int) * _N));
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::compute_forces(CUDABaseList*lists, c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_forces, c_number4 *d_torques, LR_bonds *d_bonds, CUDABox*d_box) {
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::_near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb, CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDARNAInteraction.cu:void CUDARNAInteraction::_dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads, CUDA_kernel_cfg _ffs_dist_precalc_kernel_cfg, CUDABox*d_box) {
src/CUDA/Interactions/CUDA_LJ.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Interactions/CUDA_LJ.cuh:__device__ void _particle_particle_interaction(c_number4 &ppos, c_number4 &qpos, c_number4 &F, CUDABox *box) {
src/CUDA/Interactions/CUDA_LJ.cuh:	c_number sqr_r = CUDA_DOT(r, r);
src/CUDA/Interactions/CUDA_LJ.cuh:__global__ void lj_forces_edge(c_number4 *poss, c_number4 *forces, edge_bond *edge_list, int n_edges, CUDABox *box) {
src/CUDA/Interactions/CUDA_LJ.cuh:__global__ void lj_forces(c_number4 *poss, c_number4 *forces, int *matrix_neighs, int *number_neighs, CUDABox *box) {
src/CUDA/CUDA_base_interactions.h:#ifndef CUDA_BASE_INTERACTIONS_H_
src/CUDA/CUDA_base_interactions.h:#define CUDA_BASE_INTERACTIONS_H_
src/CUDA/CUDA_base_interactions.h:	c_number rsqr = CUDA_DOT(r, r);
src/CUDA/CUDAUtils.h: * @file    CUDAUtils.h
src/CUDA/CUDAUtils.h:#ifndef GPUUTILS_H_
src/CUDA/CUDAUtils.h:#define GPUUTILS_H_
src/CUDA/CUDAUtils.h:#include "cuda_defs.h"
src/CUDA/CUDAUtils.h:#include "cuda_utils/helper_cuda.h"
src/CUDA/CUDAUtils.h:* @brief Static class. It stores many utility functions used by CUDA classes. It could probably be turned into a namespace...
src/CUDA/CUDAUtils.h:class GpuUtils {
src/CUDA/CUDAUtils.h:	static c_number4 sum_c_number4_on_GPU(c_number4 *dv, int N);
src/CUDA/CUDAUtils.h:	static double sum_c_number4_to_double_on_GPU(c_number4 *dv, int N);
src/CUDA/CUDAUtils.h:	static cudaError_t LR_cudaMalloc(T **devPtr, size_t size);
src/CUDA/CUDAUtils.h:	static void init_texture_object(cudaTextureObject_t *obj, cudaChannelFormatDesc format, T *dev_ptr, size_t size) {
src/CUDA/CUDAUtils.h:		cudaResourceDesc res_desc_eps;
src/CUDA/CUDAUtils.h:		res_desc_eps.resType = cudaResourceTypeLinear;
src/CUDA/CUDAUtils.h:		cudaTextureDesc tex_desc_eps;
src/CUDA/CUDAUtils.h:		tex_desc_eps.readMode = cudaReadModeElementType;
src/CUDA/CUDAUtils.h:		CUDA_SAFE_CALL(cudaCreateTextureObject(obj, &res_desc_eps, &tex_desc_eps, NULL));
src/CUDA/CUDAUtils.h:cudaError_t GpuUtils::LR_cudaMalloc(T **devPtr, size_t size) {
src/CUDA/CUDAUtils.h:	OX_LOG(Logger::LOG_DEBUG, "Allocating %lld bytes (%.2lf MB) on the GPU", size, size / 1000000.0);
src/CUDA/CUDAUtils.h:	GpuUtils::_allocated_dev_mem += size;
src/CUDA/CUDAUtils.h:	return cudaMalloc((void **) devPtr, size);
src/CUDA/CUDAUtils.h:#endif /* GPUUTILS_H_ */
src/CUDA/cuda_defs.h: * cuda_defs.h
src/CUDA/cuda_defs.h:#ifndef SRC_CUDA_CUDA_DEFS_H_
src/CUDA/cuda_defs.h:#define SRC_CUDA_CUDA_DEFS_H_
src/CUDA/cuda_defs.h:/// CUDA_SAFE_CALL replacement for backwards compatibility (CUDA < 5.0)
src/CUDA/cuda_defs.h:#define CUDA_SAFE_CALL(call)                                  \
src/CUDA/cuda_defs.h:    cudaError_t err = call;                                   \
src/CUDA/cuda_defs.h:    if (err != cudaSuccess) {                                 \
src/CUDA/cuda_defs.h:      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
src/CUDA/cuda_defs.h:             cudaGetErrorString(err));                        \
src/CUDA/cuda_defs.h:/// CUT_CHECK_ERROR replacement for backwards compatibility (CUDA < 5.0)
src/CUDA/cuda_defs.h:#define CUT_CHECK_ERROR(x) getLastCudaError(x);
src/CUDA/cuda_defs.h:#define CUDA_LRACOS(x) (((x) >= (c_number)1) ? (c_number) 0 : ((x) <= (c_number)-1) ? (c_number) PI : acosf(x))
src/CUDA/cuda_defs.h:#define CUDA_DOT(a, b) (a.x*b.x + a.y*b.y + a.z*b.z)
src/CUDA/cuda_defs.h:		CUDA_SAFE_CALL(cudaMemcpyToSymbol((dest), val, (size)*sizeof(float)));\
src/CUDA/cuda_defs.h:		CUDA_SAFE_CALL(cudaMemcpyToSymbol((dest), &tmp, sizeof(float)));\
src/CUDA/cuda_defs.h:#include <cuda.h>
src/CUDA/cuda_defs.h:#include <cuda_runtime_api.h>
src/CUDA/cuda_defs.h: * @brief Utility struct used by CUDA class to store information about kernel configurations.
src/CUDA/cuda_defs.h:typedef struct CUDA_kernel_cfg {
src/CUDA/cuda_defs.h:} CUDA_kernel_cfg;
src/CUDA/cuda_defs.h:#ifdef CUDA_DOUBLE_PRECISION
src/CUDA/cuda_defs.h:using GPU_quat = double4;
src/CUDA/cuda_defs.h:using GPU_quat = float4;
src/CUDA/cuda_defs.h: * @brief Used to store the stress tensor on GPUs
src/CUDA/cuda_defs.h:typedef struct __align__(16) CUDAStressTensor {
src/CUDA/cuda_defs.h:	__device__ __host__ CUDAStressTensor() : e{0} {
src/CUDA/cuda_defs.h:	__device__ __host__ CUDAStressTensor(c_number e0, c_number e1, c_number e2, c_number e3, c_number e4, c_number e5) :
src/CUDA/cuda_defs.h:	__device__ __host__ inline CUDAStressTensor operator+(const CUDAStressTensor &other) const {
src/CUDA/cuda_defs.h:		return CUDAStressTensor(
src/CUDA/cuda_defs.h:	__device__ __host__ inline void operator+=(const CUDAStressTensor &other) {
src/CUDA/cuda_defs.h:} CUDAStressTensor;
src/CUDA/cuda_defs.h:#endif /* SRC_CUDA_CUDA_DEFS_H_ */
src/CUDA/Thermostats/CUDALangevinThermostat.h: * CUDALangevinThermostat.h
src/CUDA/Thermostats/CUDALangevinThermostat.h:#ifndef CUDALANGEVINTHERMOSTAT_H_
src/CUDA/Thermostats/CUDALangevinThermostat.h:#define CUDALANGEVINTHERMOSTAT_H_
src/CUDA/Thermostats/CUDALangevinThermostat.h:#include "CUDABaseThermostat.h"
src/CUDA/Thermostats/CUDALangevinThermostat.h:#include "../cuda_utils/cuda_device_utils.h"
src/CUDA/Thermostats/CUDALangevinThermostat.h:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Thermostats/CUDALangevinThermostat.h:class CUDALangevinThermostat: public CUDABaseThermostat, public LangevinThermostat {
src/CUDA/Thermostats/CUDALangevinThermostat.h:	CUDALangevinThermostat();
src/CUDA/Thermostats/CUDALangevinThermostat.h:	virtual ~CUDALangevinThermostat();
src/CUDA/Thermostats/CUDALangevinThermostat.h:	virtual void apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step);
src/CUDA/Thermostats/CUDALangevinThermostat.h:#endif /* CUDALANGEVINTHERMOSTAT_H_ */
src/CUDA/Thermostats/CUDANoThermostat.cu: * CUDANoThermostat.cu
src/CUDA/Thermostats/CUDANoThermostat.cu:#include "CUDANoThermostat.h"
src/CUDA/Thermostats/CUDANoThermostat.cu:CUDANoThermostat::CUDANoThermostat() :
src/CUDA/Thermostats/CUDANoThermostat.cu:CUDANoThermostat::~CUDANoThermostat() {
src/CUDA/Thermostats/CUDANoThermostat.cu:void CUDANoThermostat::apply_cuda(c_number4 *d_poss, GPU_quat *d_orientationss, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
src/CUDA/Thermostats/CUDABussiThermostat.h: * CUDABussiThermostat.h
src/CUDA/Thermostats/CUDABussiThermostat.h:#ifndef CUDABUSSITHERMOSTAT_H_
src/CUDA/Thermostats/CUDABussiThermostat.h:#define CUDABUSSITHERMOSTAT_H_
src/CUDA/Thermostats/CUDABussiThermostat.h:#include "CUDABaseThermostat.h"
src/CUDA/Thermostats/CUDABussiThermostat.h:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Thermostats/CUDABussiThermostat.h: * @brief CUDA implementation of the {@link BussiThermostat thermostat} by Bussi et al.
src/CUDA/Thermostats/CUDABussiThermostat.h:class CUDABussiThermostat: public CUDABaseThermostat, public BussiThermostat {
src/CUDA/Thermostats/CUDABussiThermostat.h:	CUDABussiThermostat();
src/CUDA/Thermostats/CUDABussiThermostat.h:	virtual ~CUDABussiThermostat();
src/CUDA/Thermostats/CUDABussiThermostat.h:	virtual void apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step);
src/CUDA/Thermostats/CUDABussiThermostat.h:#endif /* CUDABUSSITHERMOSTAT_H_ */
src/CUDA/Thermostats/CUDABussiThermostat.cu: * CUDABussiThermostat.cpp
src/CUDA/Thermostats/CUDABussiThermostat.cu:#include "CUDABussiThermostat.h"
src/CUDA/Thermostats/CUDABussiThermostat.cu:		res.w = 0.5f * CUDA_DOT(a, a);
src/CUDA/Thermostats/CUDABussiThermostat.cu:		res.w = 0.5f * CUDA_DOT(v_rel, v_rel);
src/CUDA/Thermostats/CUDABussiThermostat.cu:CUDABussiThermostat::CUDABussiThermostat() :
src/CUDA/Thermostats/CUDABussiThermostat.cu:				CUDABaseThermostat(),
src/CUDA/Thermostats/CUDABussiThermostat.cu:CUDABussiThermostat::~CUDABussiThermostat() {
src/CUDA/Thermostats/CUDABussiThermostat.cu:void CUDABussiThermostat::get_settings(input_file &inp) {
src/CUDA/Thermostats/CUDABussiThermostat.cu:	CUDABaseThermostat::get_cuda_settings(inp);
src/CUDA/Thermostats/CUDABussiThermostat.cu:void CUDABussiThermostat::init() {
src/CUDA/Thermostats/CUDABussiThermostat.cu:bool CUDABussiThermostat::would_activate(llint curr_step) {
src/CUDA/Thermostats/CUDABussiThermostat.cu:void CUDABussiThermostat::apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
src/CUDA/Thermostats/CUDAThermostatFactory.h: * CUDAThermostatFactory.h
src/CUDA/Thermostats/CUDAThermostatFactory.h:#ifndef CUDATHERMOSTATFACTORY_H_
src/CUDA/Thermostats/CUDAThermostatFactory.h:#define CUDATHERMOSTATFACTORY_H_
src/CUDA/Thermostats/CUDAThermostatFactory.h:#include "CUDABaseThermostat.h"
src/CUDA/Thermostats/CUDAThermostatFactory.h:#include "../cuda_utils/CUDABox.h"
src/CUDA/Thermostats/CUDAThermostatFactory.h: * @brief CUDA Thermostat factory class.
src/CUDA/Thermostats/CUDAThermostatFactory.h:class CUDAThermostatFactory {
src/CUDA/Thermostats/CUDAThermostatFactory.h:	CUDAThermostatFactory() = delete;
src/CUDA/Thermostats/CUDAThermostatFactory.h:	virtual ~CUDAThermostatFactory() = delete;
src/CUDA/Thermostats/CUDAThermostatFactory.h:	 * @brief Method that returns a pointer to a CUDABaseThermostat object
src/CUDA/Thermostats/CUDAThermostatFactory.h:	 * @return a pointer to a CUDA thermostat Object, which must be of a class derived
src/CUDA/Thermostats/CUDAThermostatFactory.h:	static std::shared_ptr<CUDABaseThermostat> make_thermostat(input_file &inp, BaseBox * box);
src/CUDA/Thermostats/CUDAThermostatFactory.h:#endif /* CUDATHERMOSTATFACTORY_H_ */
src/CUDA/Thermostats/CUDASRDThermostat.cu: * CUDASRDThermostat.cu
src/CUDA/Thermostats/CUDASRDThermostat.cu:#include "CUDASRDThermostat.h"
src/CUDA/Thermostats/CUDASRDThermostat.cu:#include "CUDA_SRD.cuh"
src/CUDA/Thermostats/CUDASRDThermostat.cu:#include "../CUDAUtils.h"
src/CUDA/Thermostats/CUDASRDThermostat.cu:CUDASRDThermostat::CUDASRDThermostat(BaseBox *box) :
src/CUDA/Thermostats/CUDASRDThermostat.cu:				CUDABaseThermostat(),
src/CUDA/Thermostats/CUDASRDThermostat.cu:CUDASRDThermostat::~CUDASRDThermostat() {
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_counters_cells));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_cells));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_poss));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_vels));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_cells_dp));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_reduced_cells_dp));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_reduce_keys));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFree(_d_reduced_cells_keys));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaFreeHost(_d_cell_overflow));
src/CUDA/Thermostats/CUDASRDThermostat.cu:void CUDASRDThermostat::get_settings(input_file &inp) {
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDABaseThermostat::get_cuda_settings(inp);
src/CUDA/Thermostats/CUDASRDThermostat.cu:void CUDASRDThermostat::init() {
src/CUDA/Thermostats/CUDASRDThermostat.cu:	// copy constant values to the GPU
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(box_side, &f_copy, sizeof(float)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(rcell, &f_copy, sizeof(float)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(m_small, &f_copy, sizeof(float)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(sqrt_m_small, &f_copy, sizeof(float)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dt, &f_copy, sizeof(float)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(N_tot, &_N_tot, sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(N_solvent, &this->_N_particles, sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(N_cells_side, &this->_N_cells_side, sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(max_N_per_cell, &_max_N_per_cell, sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	// allocate memory on the GPU
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_poss, _N_tot * sizeof(c_number4)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_vels, _N_tot * sizeof(c_number4)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_counters_cells, this->_N_cells * sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_cells, this->_N_cells * _max_N_per_cell * sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_cells_dp, this->_N_cells * _max_N_per_cell * sizeof(c_number4)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_reduced_cells_dp, this->_N_cells * sizeof(c_number4)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_reduce_keys, this->_N_cells * _max_N_per_cell * sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_reduced_cells_keys, this->_N_cells * sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMallocHost(&_d_cell_overflow, sizeof(bool), cudaHostAllocDefault));
src/CUDA/Thermostats/CUDASRDThermostat.cu:bool CUDASRDThermostat::would_activate(llint curr_step) {
src/CUDA/Thermostats/CUDASRDThermostat.cu:void CUDASRDThermostat::apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemset(_d_counters_cells, 0, this->_N_cells * sizeof(int)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemset(_d_cells_dp, 0, this->_N_cells * _max_N_per_cell * sizeof(c_number4)));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_poss + this->_N_particles, d_poss, _N_vec_size, cudaMemcpyDeviceToDevice));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_vels + this->_N_particles, d_vels, _N_vec_size, cudaMemcpyDeviceToDevice));
src/CUDA/Thermostats/CUDASRDThermostat.cu:	//GpuUtils::print_device_array<c_number4>(_d_cells_dp, this->_N_cells*_max_N_per_cell);
src/CUDA/Thermostats/CUDASRDThermostat.cu:	//GpuUtils::print_device_array<c_number4>(_d_cells_dp, this->_N_cells*_max_N_per_cell);
src/CUDA/Thermostats/CUDASRDThermostat.cu:	//GpuUtils::print_device_array<c_number4>(_d_reduced_cells_dp, this->_N_cells);
src/CUDA/Thermostats/CUDASRDThermostat.cu:	CUDA_SAFE_CALL(cudaMemcpy(d_vels, _d_vels + this->_N_particles, _N_vec_size, cudaMemcpyDeviceToDevice));
src/CUDA/Thermostats/CUDABaseThermostat.cu: * CUDABaseThermostat.cu
src/CUDA/Thermostats/CUDABaseThermostat.cu:#include "CUDABaseThermostat.h"
src/CUDA/Thermostats/CUDABaseThermostat.cu:CUDABaseThermostat::CUDABaseThermostat() :
src/CUDA/Thermostats/CUDABaseThermostat.cu:CUDABaseThermostat::~CUDABaseThermostat() {
src/CUDA/Thermostats/CUDABaseThermostat.cu:	if(_d_rand_state != NULL) CUDA_SAFE_CALL(cudaFree(_d_rand_state));
src/CUDA/Thermostats/CUDABaseThermostat.cu:void CUDABaseThermostat::get_cuda_settings(input_file &inp) {
src/CUDA/Thermostats/CUDABaseThermostat.cu:void CUDABaseThermostat::_setup_rand(int N) {
src/CUDA/Thermostats/CUDABaseThermostat.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<curandState>(&_d_rand_state, N * sizeof(curandState)));
src/CUDA/Thermostats/CUDALangevinThermostat.cu: * CUDALangevinThermostat.cu
src/CUDA/Thermostats/CUDALangevinThermostat.cu:#include "CUDALangevinThermostat.h"
src/CUDA/Thermostats/CUDALangevinThermostat.cu:		//Could define operators for GPU_quat
src/CUDA/Thermostats/CUDALangevinThermostat.cu:CUDALangevinThermostat::CUDALangevinThermostat() :
src/CUDA/Thermostats/CUDALangevinThermostat.cu:				CUDABaseThermostat(),
src/CUDA/Thermostats/CUDALangevinThermostat.cu:CUDALangevinThermostat::~CUDALangevinThermostat() {
src/CUDA/Thermostats/CUDALangevinThermostat.cu:void CUDALangevinThermostat::get_settings(input_file &inp) {
src/CUDA/Thermostats/CUDALangevinThermostat.cu:	CUDABaseThermostat::get_cuda_settings(inp);
src/CUDA/Thermostats/CUDALangevinThermostat.cu:void CUDALangevinThermostat::init() {
src/CUDA/Thermostats/CUDALangevinThermostat.cu:bool CUDALangevinThermostat::would_activate(llint curr_step) {
src/CUDA/Thermostats/CUDALangevinThermostat.cu:void CUDALangevinThermostat::apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
src/CUDA/Thermostats/CUDABrownianThermostat.h: * CUDABrownianThermostat.h
src/CUDA/Thermostats/CUDABrownianThermostat.h:#ifndef CUDABROWNIANTHERMOSTAT_H_
src/CUDA/Thermostats/CUDABrownianThermostat.h:#define CUDABROWNIANTHERMOSTAT_H_
src/CUDA/Thermostats/CUDABrownianThermostat.h:#include "CUDABaseThermostat.h"
src/CUDA/Thermostats/CUDABrownianThermostat.h: * @brief CUDA implementation of a {@link BrownianThermostat brownian thermostat}.
src/CUDA/Thermostats/CUDABrownianThermostat.h:class CUDABrownianThermostat: public CUDABaseThermostat, public BrownianThermostat {
src/CUDA/Thermostats/CUDABrownianThermostat.h:	CUDABrownianThermostat();
src/CUDA/Thermostats/CUDABrownianThermostat.h:	virtual ~CUDABrownianThermostat();
src/CUDA/Thermostats/CUDABrownianThermostat.h:	virtual void apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step);
src/CUDA/Thermostats/CUDABrownianThermostat.h:#endif /* CUDABROWNIANTHERMOSTAT_H_ */
src/CUDA/Thermostats/CUDABaseThermostat.h: * CUDABaseThermostat.h
src/CUDA/Thermostats/CUDABaseThermostat.h:#ifndef CUDABASETHERMOSTAT_H_
src/CUDA/Thermostats/CUDABaseThermostat.h:#define CUDABASETHERMOSTAT_H_
src/CUDA/Thermostats/CUDABaseThermostat.h:#include "../CUDAUtils.h"
src/CUDA/Thermostats/CUDABaseThermostat.h: * @brief Abstract class for CUDA thermostats
src/CUDA/Thermostats/CUDABaseThermostat.h: * to implement the apply() method. The best practice to write CUDA thermostats would be to first
src/CUDA/Thermostats/CUDABaseThermostat.h: * write a CPU thermostat and then let the CUDA class inherit from it, like CUDABrownianThermostat does.
src/CUDA/Thermostats/CUDABaseThermostat.h:class CUDABaseThermostat: public virtual IBaseThermostat {
src/CUDA/Thermostats/CUDABaseThermostat.h:	CUDA_kernel_cfg _launch_cfg;
src/CUDA/Thermostats/CUDABaseThermostat.h:	CUDABaseThermostat();
src/CUDA/Thermostats/CUDABaseThermostat.h:	virtual ~CUDABaseThermostat();
src/CUDA/Thermostats/CUDABaseThermostat.h:	virtual void get_cuda_settings(input_file &inp);
src/CUDA/Thermostats/CUDABaseThermostat.h:	virtual void apply_cuda(c_number4 *d_pos, GPU_quat *d_orientations, c_number4 *d_vel, c_number4 *d_L, llint curr_step) = 0;
src/CUDA/Thermostats/CUDABaseThermostat.h:#endif /* CUDABASETHERMOSTAT_H_ */
src/CUDA/Thermostats/CUDABrownianThermostat.cu: * CUDABrownianThermostat.cpp
src/CUDA/Thermostats/CUDABrownianThermostat.cu:#include "CUDABrownianThermostat.h"
src/CUDA/Thermostats/CUDABrownianThermostat.cu:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Thermostats/CUDABrownianThermostat.cu:CUDABrownianThermostat::CUDABrownianThermostat() :
src/CUDA/Thermostats/CUDABrownianThermostat.cu:				CUDABaseThermostat(),
src/CUDA/Thermostats/CUDABrownianThermostat.cu:CUDABrownianThermostat::~CUDABrownianThermostat() {
src/CUDA/Thermostats/CUDABrownianThermostat.cu:void CUDABrownianThermostat::get_settings(input_file &inp) {
src/CUDA/Thermostats/CUDABrownianThermostat.cu:	CUDABaseThermostat::get_cuda_settings(inp);
src/CUDA/Thermostats/CUDABrownianThermostat.cu:void CUDABrownianThermostat::init() {
src/CUDA/Thermostats/CUDABrownianThermostat.cu:bool CUDABrownianThermostat::would_activate(llint curr_step) {
src/CUDA/Thermostats/CUDABrownianThermostat.cu:void CUDABrownianThermostat::apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step) {
src/CUDA/Thermostats/CUDA_SRD.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Thermostats/CUDASRDThermostat.h: * CUDASRDThermostat.h
src/CUDA/Thermostats/CUDASRDThermostat.h:#ifndef CUDASRDTHERMOSTAT_H_
src/CUDA/Thermostats/CUDASRDThermostat.h:#define CUDASRDTHERMOSTAT_H_
src/CUDA/Thermostats/CUDASRDThermostat.h:#include "CUDABaseThermostat.h"
src/CUDA/Thermostats/CUDASRDThermostat.h: * @brief CUDA implementation of a {@link SRDThermostat SRD thermostat}.
src/CUDA/Thermostats/CUDASRDThermostat.h:class CUDASRDThermostat: public CUDABaseThermostat, public SRDThermostat {
src/CUDA/Thermostats/CUDASRDThermostat.h:	CUDASRDThermostat(BaseBox * box);
src/CUDA/Thermostats/CUDASRDThermostat.h:	virtual ~CUDASRDThermostat();
src/CUDA/Thermostats/CUDASRDThermostat.h:	virtual void apply_cuda(c_number4 *d_poss, GPU_quat *d_orientations, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step);
src/CUDA/Thermostats/CUDASRDThermostat.h:#endif /* CUDASRDTHERMOSTAT_H_ */
src/CUDA/Thermostats/CUDANoThermostat.h: * CUDANoThermostat.h
src/CUDA/Thermostats/CUDANoThermostat.h:#ifndef CUDANOTHERMOSTAT_H_
src/CUDA/Thermostats/CUDANoThermostat.h:#define CUDANOTHERMOSTAT_H_
src/CUDA/Thermostats/CUDANoThermostat.h:#include "CUDABaseThermostat.h"
src/CUDA/Thermostats/CUDANoThermostat.h:class CUDANoThermostat: public CUDABaseThermostat, public NoThermostat {
src/CUDA/Thermostats/CUDANoThermostat.h:	CUDANoThermostat();
src/CUDA/Thermostats/CUDANoThermostat.h:	virtual ~CUDANoThermostat();
src/CUDA/Thermostats/CUDANoThermostat.h:	virtual void apply_cuda(c_number4 *d_poss, GPU_quat *d_orientationss, c_number4 *d_vels, c_number4 *d_Ls, llint curr_step);
src/CUDA/Thermostats/CUDANoThermostat.h:#endif /* CUDANOTHERMOSTAT_H_ */
src/CUDA/Thermostats/CUDAThermostatFactory.cu:#include "CUDAThermostatFactory.h"
src/CUDA/Thermostats/CUDAThermostatFactory.cu:#include "CUDANoThermostat.h"
src/CUDA/Thermostats/CUDAThermostatFactory.cu:#include "CUDABrownianThermostat.h"
src/CUDA/Thermostats/CUDAThermostatFactory.cu:#include "CUDALangevinThermostat.h"
src/CUDA/Thermostats/CUDAThermostatFactory.cu:#include "CUDASRDThermostat.h"
src/CUDA/Thermostats/CUDAThermostatFactory.cu:#include "CUDABussiThermostat.h"
src/CUDA/Thermostats/CUDAThermostatFactory.cu:std::shared_ptr<CUDABaseThermostat> CUDAThermostatFactory::make_thermostat(input_file &inp, BaseBox * box) {
src/CUDA/Thermostats/CUDAThermostatFactory.cu:		return std::make_shared<CUDABrownianThermostat>();
src/CUDA/Thermostats/CUDAThermostatFactory.cu:		return std::make_shared<CUDABrownianThermostat>();
src/CUDA/Thermostats/CUDAThermostatFactory.cu:		return std::make_shared<CUDABussiThermostat>();
src/CUDA/Thermostats/CUDAThermostatFactory.cu:		return std::make_shared<CUDALangevinThermostat>();
src/CUDA/Thermostats/CUDAThermostatFactory.cu:		return std::make_shared<CUDASRDThermostat>(box);
src/CUDA/Thermostats/CUDAThermostatFactory.cu:		return std::make_shared<CUDANoThermostat>();
src/CUDA/Thermostats/CUDAThermostatFactory.cu:	else throw oxDNAException("Invalid CUDA thermostat '%s'", thermostat_type);
src/CUDA/CUDA_sort.cuh: * CUDA_sort.cuh
src/CUDA/CUDA_sort.cuh:#ifndef CUDA_SORT_H_
src/CUDA/CUDA_sort.cuh:#define CUDA_SORT_H_
src/CUDA/CUDA_sort.cuh:#include "CUDAUtils.h"
src/CUDA/CUDA_sort.cuh:__global__ void permute_particles(int *sorted_hindex, int *inv_sorted_hindex, c_number4 *poss, c_number4 *vels, c_number4 *Ls, GPU_quat *orientations, LR_bonds *bonds, int *particles_to_mols, c_number4 *buff_poss, c_number4 *buff_vels, c_number4 *buff_Ls, GPU_quat *buff_orientations, LR_bonds *buff_bonds, int *buff_particles_to_mols);
src/CUDA/CUDA_sort.cuh:#endif /* CUDA_SORT_H_ */
src/CUDA/cuda_utils/CUDABox.h: * CUDABox.h
src/CUDA/cuda_utils/CUDABox.h:#ifndef CUDABOX_H_
src/CUDA/cuda_utils/CUDABox.h:#define CUDABOX_H_
src/CUDA/cuda_utils/CUDABox.h:#include "../CUDAUtils.h"
src/CUDA/cuda_utils/CUDABox.h:class CUDABox {
src/CUDA/cuda_utils/CUDABox.h:	CUDABox() :
src/CUDA/cuda_utils/CUDABox.h:	CUDABox(const CUDABox &b) {
src/CUDA/cuda_utils/CUDABox.h:	~CUDABox() {
src/CUDA/cuda_utils/CUDABox.h:	void set_CUDA_from_CPU(BaseBox *box) {
src/CUDA/cuda_utils/CUDABox.h:	void set_CPU_from_CUDA(BaseBox *box) {
src/CUDA/cuda_utils/CUDABox.h:#endif /* CUDABOX_H_ */
src/CUDA/cuda_utils/helper_cuda.h: * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
src/CUDA/cuda_utils/helper_cuda.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
src/CUDA/cuda_utils/helper_cuda.h:// These are CUDA Helper functions for initialization and error checking
src/CUDA/cuda_utils/helper_cuda.h:#ifndef HELPER_CUDA_H
src/CUDA/cuda_utils/helper_cuda.h:#define HELPER_CUDA_H
src/CUDA/cuda_utils/helper_cuda.h:// definitions for cuda 4.0
src/CUDA/cuda_utils/helper_cuda.h:#if CUDA_VERSION < 4010
src/CUDA/cuda_utils/helper_cuda.h:#define CUDA_ERROR_ASSERT 710
src/CUDA/cuda_utils/helper_cuda.h:#define CUDA_ERROR_TOO_MANY_PEERS 711
src/CUDA/cuda_utils/helper_cuda.h:#define CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED 712
src/CUDA/cuda_utils/helper_cuda.h:#define CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED 713
src/CUDA/cuda_utils/helper_cuda.h:// refer the CUDA examples for examples of the needed CUDA headers, which may change depending
src/CUDA/cuda_utils/helper_cuda.h:// on which CUDA functions are used.
src/CUDA/cuda_utils/helper_cuda.h:// CUDA Runtime error messages
src/CUDA/cuda_utils/helper_cuda.h:inline static const char *_cudaGetErrorEnum(cudaError_t error)
src/CUDA/cuda_utils/helper_cuda.h:        case cudaSuccess:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaSuccess";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorMissingConfiguration:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorMissingConfiguration";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorMemoryAllocation:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorMemoryAllocation";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInitializationError:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInitializationError";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorLaunchFailure:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorLaunchFailure";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorPriorLaunchFailure:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorPriorLaunchFailure";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorLaunchTimeout:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorLaunchTimeout";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorLaunchOutOfResources:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorLaunchOutOfResources";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidDeviceFunction:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidDeviceFunction";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidConfiguration:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidConfiguration";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidDevice:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidDevice";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidValue:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidValue";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidPitchValue:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidPitchValue";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidSymbol:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidSymbol";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorMapBufferObjectFailed:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorMapBufferObjectFailed";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorUnmapBufferObjectFailed:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorUnmapBufferObjectFailed";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidHostPointer:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidHostPointer";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidDevicePointer:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidDevicePointer";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidTexture:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidTexture";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidTextureBinding:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidTextureBinding";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidChannelDescriptor:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidChannelDescriptor";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidMemcpyDirection:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidMemcpyDirection";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorAddressOfConstant:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorAddressOfConstant";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorTextureFetchFailed:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorTextureFetchFailed";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorTextureNotBound:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorTextureNotBound";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorSynchronizationError:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorSynchronizationError";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidFilterSetting:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidFilterSetting";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidNormSetting:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidNormSetting";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorMixedDeviceExecution:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorMixedDeviceExecution";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorCudartUnloading:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorCudartUnloading";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorUnknown:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorUnknown";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorNotYetImplemented:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorNotYetImplemented";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorMemoryValueTooLarge:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorMemoryValueTooLarge";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidResourceHandle:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidResourceHandle";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorNotReady:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorNotReady";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInsufficientDriver:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInsufficientDriver";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorSetOnActiveProcess:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorSetOnActiveProcess";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidSurface:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidSurface";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorNoDevice:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorNoDevice";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorECCUncorrectable:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorECCUncorrectable";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorSharedObjectSymbolNotFound:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorSharedObjectSymbolNotFound";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorSharedObjectInitFailed:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorSharedObjectInitFailed";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorUnsupportedLimit:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorUnsupportedLimit";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorDuplicateVariableName:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorDuplicateVariableName";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorDuplicateTextureName:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorDuplicateTextureName";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorDuplicateSurfaceName:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorDuplicateSurfaceName";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorDevicesUnavailable:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorDevicesUnavailable";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorInvalidKernelImage:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorInvalidKernelImage";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorNoKernelImageForDevice:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorNoKernelImageForDevice";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorIncompatibleDriverContext:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorIncompatibleDriverContext";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorPeerAccessAlreadyEnabled:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorPeerAccessAlreadyEnabled";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorPeerAccessNotEnabled:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorPeerAccessNotEnabled";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorDeviceAlreadyInUse:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorDeviceAlreadyInUse";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorProfilerDisabled:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorProfilerDisabled";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorProfilerNotInitialized:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorProfilerNotInitialized";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorProfilerAlreadyStarted:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorProfilerAlreadyStarted";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorProfilerAlreadyStopped:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorProfilerAlreadyStopped";
src/CUDA/cuda_utils/helper_cuda.h:#if __CUDA_API_VERSION >= 0x4000
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorAssert:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorAssert";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorTooManyPeers:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorTooManyPeers";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorHostMemoryAlreadyRegistered:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorHostMemoryAlreadyRegistered";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorHostMemoryNotRegistered:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorHostMemoryNotRegistered";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorStartupFailure:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorStartupFailure";
src/CUDA/cuda_utils/helper_cuda.h:        case cudaErrorApiFailureBase:
src/CUDA/cuda_utils/helper_cuda.h:            return "cudaErrorApiFailureBase";
src/CUDA/cuda_utils/helper_cuda.h:#ifdef __cuda_cuda_h__
src/CUDA/cuda_utils/helper_cuda.h:// CUDA Driver API errors
src/CUDA/cuda_utils/helper_cuda.h:inline static const char *_cudaGetErrorEnum(CUresult error)
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_SUCCESS:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_SUCCESS";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_INVALID_VALUE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_INVALID_VALUE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_OUT_OF_MEMORY:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_OUT_OF_MEMORY";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NOT_INITIALIZED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NOT_INITIALIZED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_DEINITIALIZED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_DEINITIALIZED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_PROFILER_DISABLED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_PROFILER_DISABLED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NO_DEVICE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NO_DEVICE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_INVALID_DEVICE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_INVALID_DEVICE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_INVALID_IMAGE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_INVALID_IMAGE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_INVALID_CONTEXT:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_INVALID_CONTEXT";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_MAP_FAILED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_MAP_FAILED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_UNMAP_FAILED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_UNMAP_FAILED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_ARRAY_IS_MAPPED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_ARRAY_IS_MAPPED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_ALREADY_MAPPED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_ALREADY_MAPPED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NO_BINARY_FOR_GPU:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NO_BINARY_FOR_GPU";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_ALREADY_ACQUIRED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_ALREADY_ACQUIRED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NOT_MAPPED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NOT_MAPPED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_ECC_UNCORRECTABLE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_ECC_UNCORRECTABLE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_UNSUPPORTED_LIMIT:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_UNSUPPORTED_LIMIT";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_INVALID_SOURCE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_INVALID_SOURCE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_FILE_NOT_FOUND:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_FILE_NOT_FOUND";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_OPERATING_SYSTEM:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_OPERATING_SYSTEM";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_INVALID_HANDLE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_INVALID_HANDLE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NOT_FOUND:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NOT_FOUND";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_NOT_READY:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_NOT_READY";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_LAUNCH_FAILED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_LAUNCH_FAILED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_LAUNCH_TIMEOUT:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_LAUNCH_TIMEOUT";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_ASSERT:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_ASSERT";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_TOO_MANY_PEERS:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_TOO_MANY_PEERS";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
src/CUDA/cuda_utils/helper_cuda.h:        case CUDA_ERROR_UNKNOWN:
src/CUDA/cuda_utils/helper_cuda.h:            return "CUDA_ERROR_UNKNOWN";
src/CUDA/cuda_utils/helper_cuda.h:static const char *_cudaGetErrorEnum(cublasStatus_t error)
src/CUDA/cuda_utils/helper_cuda.h:static const char *_cudaGetErrorEnum(cufftResult error)
src/CUDA/cuda_utils/helper_cuda.h:static const char *_cudaGetErrorEnum(cusparseStatus_t error)
src/CUDA/cuda_utils/helper_cuda.h:static const char *_cudaGetErrorEnum(curandStatus_t error)
src/CUDA/cuda_utils/helper_cuda.h:static const char *_cudaGetErrorEnum(NppStatus error)
src/CUDA/cuda_utils/helper_cuda.h:        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
src/CUDA/cuda_utils/helper_cuda.h:            return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
src/CUDA/cuda_utils/helper_cuda.h:        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
src/CUDA/cuda_utils/helper_cuda.h:                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
src/CUDA/cuda_utils/helper_cuda.h:                std::string msg("CUDA error at ");
src/CUDA/cuda_utils/helper_cuda.h:                msg += _cudaGetErrorEnum(result);
src/CUDA/cuda_utils/helper_cuda.h:// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
src/CUDA/cuda_utils/helper_cuda.h:#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
src/CUDA/cuda_utils/helper_cuda.h:// This will output the proper error string when calling cudaGetLastError
src/CUDA/cuda_utils/helper_cuda.h:#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
src/CUDA/cuda_utils/helper_cuda.h:inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
src/CUDA/cuda_utils/helper_cuda.h:    cudaError_t err = cudaGetLastError();
src/CUDA/cuda_utils/helper_cuda.h:    if (cudaSuccess != err)
src/CUDA/cuda_utils/helper_cuda.h:        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
src/CUDA/cuda_utils/helper_cuda.h:                file, line, errorMessage, (int)err, cudaGetErrorString(err));
src/CUDA/cuda_utils/helper_cuda.h:// Beginning of GPU Architecture definitions
src/CUDA/cuda_utils/helper_cuda.h:    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
src/CUDA/cuda_utils/helper_cuda.h:    sSMtoCores nGpuArchCoresPerSM[] =
src/CUDA/cuda_utils/helper_cuda.h:    while (nGpuArchCoresPerSM[index].SM != -1)
src/CUDA/cuda_utils/helper_cuda.h:        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
src/CUDA/cuda_utils/helper_cuda.h:            return nGpuArchCoresPerSM[index].Cores;
src/CUDA/cuda_utils/helper_cuda.h:    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
src/CUDA/cuda_utils/helper_cuda.h:    return nGpuArchCoresPerSM[7].Cores;
src/CUDA/cuda_utils/helper_cuda.h:// end of GPU Architecture definitions
src/CUDA/cuda_utils/helper_cuda.h:#ifdef __CUDA_RUNTIME_H__
src/CUDA/cuda_utils/helper_cuda.h:// General GPU Device CUDA Initialization
src/CUDA/cuda_utils/helper_cuda.h:inline int gpuDeviceInit(int devID)
src/CUDA/cuda_utils/helper_cuda.h:    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
src/CUDA/cuda_utils/helper_cuda.h:        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
src/CUDA/cuda_utils/helper_cuda.h:        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
src/CUDA/cuda_utils/helper_cuda.h:        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
src/CUDA/cuda_utils/helper_cuda.h:    cudaDeviceProp deviceProp;
src/CUDA/cuda_utils/helper_cuda.h:    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
src/CUDA/cuda_utils/helper_cuda.h:    if (deviceProp.computeMode == cudaComputeModeProhibited)
src/CUDA/cuda_utils/helper_cuda.h:        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
src/CUDA/cuda_utils/helper_cuda.h:        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
src/CUDA/cuda_utils/helper_cuda.h:    checkCudaErrors(cudaSetDevice(devID));
src/CUDA/cuda_utils/helper_cuda.h:    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);
src/CUDA/cuda_utils/helper_cuda.h:// This function returns the best GPU (with maximum GFLOPS)
src/CUDA/cuda_utils/helper_cuda.h:inline int gpuGetMaxGflopsDeviceId()
src/CUDA/cuda_utils/helper_cuda.h:    cudaDeviceProp deviceProp;
src/CUDA/cuda_utils/helper_cuda.h:    cudaGetDeviceCount(&device_count);
src/CUDA/cuda_utils/helper_cuda.h:    // Find the best major SM Architecture GPU device
src/CUDA/cuda_utils/helper_cuda.h:        cudaGetDeviceProperties(&deviceProp, current_device);
src/CUDA/cuda_utils/helper_cuda.h:        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
src/CUDA/cuda_utils/helper_cuda.h:        if (deviceProp.computeMode != cudaComputeModeProhibited)
src/CUDA/cuda_utils/helper_cuda.h:    // Find the best CUDA capable GPU device
src/CUDA/cuda_utils/helper_cuda.h:        cudaGetDeviceProperties(&deviceProp, current_device);
src/CUDA/cuda_utils/helper_cuda.h:        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
src/CUDA/cuda_utils/helper_cuda.h:        if (deviceProp.computeMode != cudaComputeModeProhibited)
src/CUDA/cuda_utils/helper_cuda.h:                // If we find GPU with SM major > 2, search only these
src/CUDA/cuda_utils/helper_cuda.h:// Initialization code to find the best CUDA Device
src/CUDA/cuda_utils/helper_cuda.h:inline int findCudaDevice(int argc, const char **argv)
src/CUDA/cuda_utils/helper_cuda.h:    cudaDeviceProp deviceProp;
src/CUDA/cuda_utils/helper_cuda.h:            devID = gpuDeviceInit(devID);
src/CUDA/cuda_utils/helper_cuda.h:        devID = gpuGetMaxGflopsDeviceId();
src/CUDA/cuda_utils/helper_cuda.h:        checkCudaErrors(cudaSetDevice(devID));
src/CUDA/cuda_utils/helper_cuda.h:        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
src/CUDA/cuda_utils/helper_cuda.h:        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
src/CUDA/cuda_utils/helper_cuda.h:// General check for CUDA GPU SM Capabilities
src/CUDA/cuda_utils/helper_cuda.h:inline bool checkCudaCapabilities(int major_version, int minor_version)
src/CUDA/cuda_utils/helper_cuda.h:    cudaDeviceProp deviceProp;
src/CUDA/cuda_utils/helper_cuda.h:    checkCudaErrors(cudaGetDevice(&dev));
src/CUDA/cuda_utils/helper_cuda.h:    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
src/CUDA/cuda_utils/helper_cuda.h:        printf("No GPU device was found that can support CUDA compute capability %d.%d.\n", major_version, minor_version);
src/CUDA/cuda_utils/helper_cuda.h:// end of CUDA Helper Functions
src/CUDA/cuda_utils/CUDA_lr_common.cuh: * CUDA_lr_common.cuh
src/CUDA/cuda_utils/CUDA_lr_common.cuh:#ifndef CUDA_LR_COMMON
src/CUDA/cuda_utils/CUDA_lr_common.cuh:#define CUDA_LR_COMMON
src/CUDA/cuda_utils/CUDA_lr_common.cuh:#include "../CUDAUtils.h"
src/CUDA/cuda_utils/CUDA_lr_common.cuh:__device__ void _update_stress_tensor(CUDAStressTensor &st, const c_number4 &r, const c_number4 &force) {
src/CUDA/cuda_utils/CUDA_lr_common.cuh:__forceinline__ __device__ void get_vectors_from_quat(const GPU_quat &q, c_number4 &a1, c_number4 &a2, c_number4 &a3) {
src/CUDA/cuda_utils/CUDA_lr_common.cuh:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
src/CUDA/cuda_utils/CUDA_lr_common.cuh:__forceinline__ __device__ void LR_atomicAddST(CUDAStressTensor *dst, const CUDAStressTensor &delta) {
src/CUDA/cuda_utils/CUDA_lr_common.cuh:// Necessary to for calculating the torque without storing a separate GPU_matrix on the GPU. Since we have the a1, a2, and a3 vectors anyway, I don't think this is costly. This step might be avoidable if torque and angular momentum were also calculated and stored as quaternions.
src/CUDA/cuda_utils/CUDA_lr_common.cuh:// Necessary to for calculating the torque without storing a separate GPU_matrix on the GPU. Since we have the a1, a2, and a3 vectors anyway, I don't think this is costly. This step might be avoidable if torque and angular momentum were also calculated and stored as quaternions.
src/CUDA/cuda_utils/CUDA_lr_common.cuh:#endif /* CUDA_LR_COMMON */
src/CUDA/cuda_utils/cuda_device_utils.cu:#include "cuda_device_utils.h"
src/CUDA/cuda_utils/cuda_device_utils.cu:#include <cuda.h>
src/CUDA/cuda_utils/cuda_device_utils.cu:#include <cuda_runtime_api.h>
src/CUDA/cuda_utils/cuda_device_utils.cu:	if(cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
src/CUDA/cuda_utils/cuda_device_utils.cu:		fprintf(stderr, "cudaGetDeviceCount FAILED, CUDA Driver and Runtime CUDA Driver and Runtime version may be mismatched, exiting.\n");
src/CUDA/cuda_utils/cuda_device_utils.cu:cudaDeviceProp get_current_device_prop() {
src/CUDA/cuda_utils/cuda_device_utils.cu:	cudaGetDevice(&curr_dev);
src/CUDA/cuda_utils/cuda_device_utils.cu:cudaDeviceProp get_device_prop(int device) {
src/CUDA/cuda_utils/cuda_device_utils.cu:	cudaDeviceProp deviceProp;
src/CUDA/cuda_utils/cuda_device_utils.cu:	cudaGetDeviceProperties(&deviceProp, device);
src/CUDA/cuda_utils/cuda_device_utils.h: * cuda_device_info.h
src/CUDA/cuda_utils/cuda_device_utils.h:#ifndef CUDA_DEVICE_INFO_H_
src/CUDA/cuda_utils/cuda_device_utils.h:#define CUDA_DEVICE_INFO_H_
src/CUDA/cuda_utils/cuda_device_utils.h:cudaDeviceProp get_current_device_prop();
src/CUDA/cuda_utils/cuda_device_utils.h:cudaDeviceProp get_device_prop(int device);
src/CUDA/cuda_utils/cuda_device_utils.h:#endif /* CUDA_DEVICE_INFO_H_ */
src/CUDA/cuda_utils/helper_string.h: * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
src/CUDA/cuda_utils/helper_string.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
src/CUDA/cuda_utils/helper_string.h:// CUDA Utility Helper Functions
src/CUDA/cuda_utils/helper_string.h:        "./7_CUDALibraries/",                       // "/7_CUDALibraries/" subdir
src/CUDA/cuda_utils/helper_string.h:        "../C/7_CUDALibraries/<executable_name>/data/",  // up 1 in tree, "/7_CUDALibraries/<executable_name>/" subdir
src/CUDA/cuda_utils/helper_string.h:        "../7_CUDALibraries/<executable_name>/data/",    // up 1 in tree, "/7_CUDALibraries/<executable_name>/" subdir
src/CUDA/cuda_utils/helper_string.h:        "../../7_CUDALibraries/<executable_name>/data/", // up 2 in tree, "/7_CUDALibraries/<executable_name>/" subdir
src/CUDA/cuda_utils/helper_string.h:        "../../../7_CUDALibraries/<executable_name>/data/", // up 3 in tree, "/7_CUDALibraries/<executable_name>/" subdir
src/CUDA/Backends/MD_CUDABackend.h: * @file    MD_CUDABackend.h
src/CUDA/Backends/MD_CUDABackend.h:#ifndef MD_CUDABACKEND_H_
src/CUDA/Backends/MD_CUDABackend.h:#define MD_CUDABACKEND_H_
src/CUDA/Backends/MD_CUDABackend.h:#include <cuda.h>
src/CUDA/Backends/MD_CUDABackend.h:#include <cuda_runtime_api.h>
src/CUDA/Backends/MD_CUDABackend.h:#include "CUDABaseBackend.h"
src/CUDA/Backends/MD_CUDABackend.h:#include "../CUDAUtils.h"
src/CUDA/Backends/MD_CUDABackend.h:#include "../Thermostats/CUDABrownianThermostat.h"
src/CUDA/Backends/MD_CUDABackend.h:#include "../cuda_utils/cuda_device_utils.h"
src/CUDA/Backends/MD_CUDABackend.h:#include "../Lists/CUDANoList.h"
src/CUDA/Backends/MD_CUDABackend.h:#include "../Lists/CUDASimpleVerletList.h"
src/CUDA/Backends/MD_CUDABackend.h:union CUDA_trap;
src/CUDA/Backends/MD_CUDABackend.h: * @brief Manages a MD simulation on GPU with CUDA.
src/CUDA/Backends/MD_CUDABackend.h:class MD_CUDABackend: public MDBackend, public CUDABaseBackend{
src/CUDA/Backends/MD_CUDABackend.h:	int *_h_gpu_index, *_h_cpu_index;
src/CUDA/Backends/MD_CUDABackend.h:	std::shared_ptr<CUDABaseThermostat> _cuda_thermostat;
src/CUDA/Backends/MD_CUDABackend.h:	bool _cuda_barostat_always_refresh = false;
src/CUDA/Backends/MD_CUDABackend.h:	std::shared_ptr<CUDABrownianThermostat> _cuda_barostat_thermostat;
src/CUDA/Backends/MD_CUDABackend.h:	CUDA_trap *_d_ext_forces;
src/CUDA/Backends/MD_CUDABackend.h:	virtual void _gpu_to_host();
src/CUDA/Backends/MD_CUDABackend.h:	virtual void _host_to_gpu();
src/CUDA/Backends/MD_CUDABackend.h:	virtual void _init_CUDA_MD_symbols();
src/CUDA/Backends/MD_CUDABackend.h:	MD_CUDABackend();
src/CUDA/Backends/MD_CUDABackend.h:	virtual ~MD_CUDABackend();
src/CUDA/Backends/MD_CUDABackend.h:#endif /* MD_CUDABACKEND_H_ */
src/CUDA/Backends/CUDA_mixed.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Backends/CUDA_mixed.cuh:__device__ GPU_quat_double _get_updated_orientation(LR_double4 &L, GPU_quat_double &old_o) {
src/CUDA/Backends/CUDA_mixed.cuh:	double norm = sqrt(CUDA_DOT(L, L));
src/CUDA/Backends/CUDA_mixed.cuh:	GPU_quat_double R = {0.5*L.x*sintheta*winv, 0.5*L.y*sintheta*winv, 0.5*L.z*sintheta*winv, qw};
src/CUDA/Backends/CUDA_mixed.cuh:__global__ void first_step_mixed(float4 __restrict__ *poss, GPU_quat __restrict__ *orientations, LR_double4 __restrict__ *possd,
src/CUDA/Backends/CUDA_mixed.cuh:		GPU_quat_double __restrict__ *orientationsd, const float4 __restrict__ *list_poss, LR_double4 __restrict__ *velsd,
src/CUDA/Backends/CUDA_mixed.cuh:		GPU_quat_double new_o = _get_updated_orientation(L, orientationsd[IND]);
src/CUDA/Backends/CUDA_mixed.cuh:		GPU_quat new_of = {(float)new_o.x, (float)new_o.y, (float)new_o.z, (float)new_o.w};
src/CUDA/Backends/CUDA_mixed.cuh:__global__ void float4_to_LR_double4(GPU_quat *src, GPU_quat_double *dest) {
src/CUDA/Backends/CUDA_mixed.cuh:	GPU_quat tmp = src[IND];
src/CUDA/Backends/CUDA_mixed.cuh:	GPU_quat_double res = {(double)tmp.x, (double)tmp.y, (double)tmp.z, (double)tmp.w};
src/CUDA/Backends/CUDA_mixed.cuh:__global__ void LR_double4_to_float4(GPU_quat_double *src, GPU_quat *dest) {
src/CUDA/Backends/CUDA_mixed.cuh:	GPU_quat_double tmp = src[IND];
src/CUDA/Backends/CUDA_mixed.cuh:	GPU_quat res = {(float)tmp.x, (float)tmp.y, (float)tmp.z, (float)tmp.w};
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu: * FFS_MD_CUDAMixedBackend.cu
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:#include "FFS_MD_CUDAMixedBackend.h"
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:#include "FFS_CUDA_MD.cuh"
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:FFS_MD_CUDAMixedBackend::FFS_MD_CUDAMixedBackend() :
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:				CUDAMixedBackend() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:FFS_MD_CUDAMixedBackend::~FFS_MD_CUDAMixedBackend() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_hb_pairs1));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_hb_pairs2));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_dist_pairs1));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_dist_pairs2));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_op_dists));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_hb_energies));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_dist_region_lens));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_dist_region_rows));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_hb_region_lens));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_hb_region_rows));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_hb_cutoffs));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(_d_region_is_nearhb));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_free_master_conditions(vector<master_condition> master_conditions) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_free_simple_conditions(SimpleConditions sc) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_dist_cond_lens));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_dist_cond_rows));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_hb_cond_lens));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_hb_cond_rows));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_hb_cond_mags));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_hb_cond_types));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_nearhb_cond_lens));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_nearhb_cond_rows));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_nearhb_cond_mags));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_nearhb_cond_types));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_dist_cond_mags));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_dist_cond_types));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaFree(sc.d_ffs_stop));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::sim_step() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDAMixedBackend::sim_step();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::get_settings(input_file &inp) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDAMixedBackend::get_settings(inp);
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::init() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDAMixedBackend::init();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_hb_region_rows, _n_hb_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_hb_region_lens, _n_hb_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_dist_region_rows, _n_dist_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_dist_region_lens, _n_dist_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_hb_pairs1, _n_hb_pairs * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_hb_pairs2, _n_hb_pairs * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<float>(&_d_hb_energies, _n_hb_pairs * sizeof(float)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<bool>(&_d_nearhb_states, _n_hb_pairs * sizeof(bool)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_dist_pairs1, _n_dist_pairs * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_dist_pairs2, _n_dist_pairs * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<float>(&_d_op_dists, _n_dist_pairs * sizeof(float)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<float>(&_d_hb_cutoffs, _n_hb_regions * sizeof(float)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<bool>(&_d_region_is_nearhb, _n_hb_regions * sizeof(bool)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_dist_region_lens, h_dist_region_lens, _n_dist_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_dist_region_rows, h_dist_region_rows, _n_dist_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_hb_region_lens, h_hb_region_lens, _n_hb_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_hb_region_rows, h_hb_region_rows, _n_hb_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_hb_pairs1, h_hb_pairs1, _n_hb_pairs * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_hb_pairs2, h_hb_pairs2, _n_hb_pairs * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_dist_pairs1, h_dist_pairs1, _n_dist_pairs * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_dist_pairs2, h_dist_pairs2, _n_dist_pairs * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_hb_cutoffs, h_hb_cutoffs, _n_hb_regions * sizeof(float), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_region_is_nearhb, _h_region_is_nearhb, _n_hb_regions * sizeof(bool), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	cudaDeviceSynchronize();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:SimpleConditions FFS_MD_CUDAMixedBackend::_get_simple_conditions(std::vector<parsed_condition> conditions, const char type[256]) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<float>(&sc.d_dist_cond_mags, sc.dist_cond_len * sizeof(float)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_dist_cond_types, sc.dist_cond_len * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_dist_cond_lens, _n_dist_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_dist_cond_rows, _n_dist_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<float>(&sc.d_hb_cond_mags, sc.hb_cond_len * sizeof(float)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_hb_cond_types, sc.hb_cond_len * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_hb_cond_lens, _n_hb_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_hb_cond_rows, _n_hb_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<float>(&sc.d_nearhb_cond_mags, sc.nearhb_cond_len * sizeof(float)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_nearhb_cond_types, sc.nearhb_cond_len * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_nearhb_cond_lens, _n_hb_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&sc.d_nearhb_cond_rows, _n_hb_regions * sizeof(int)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<bool>(&sc.d_ffs_stop, sc.stop_length * sizeof(bool)));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_hb_cond_mags, sc.h_hb_cond_mags, sc.hb_cond_len * sizeof(float), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_hb_cond_types, sc.h_hb_cond_types, sc.hb_cond_len * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_hb_cond_lens, sc.h_hb_cond_lens, _n_hb_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_hb_cond_rows, sc.h_hb_cond_rows, _n_hb_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_nearhb_cond_mags, sc.h_nearhb_cond_mags, sc.nearhb_cond_len * sizeof(float), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_nearhb_cond_types, sc.h_nearhb_cond_types, sc.nearhb_cond_len * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_nearhb_cond_lens, sc.h_nearhb_cond_lens, _n_hb_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_nearhb_cond_rows, sc.h_nearhb_cond_rows, _n_hb_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_dist_cond_mags, sc.h_dist_cond_mags, sc.dist_cond_len * sizeof(float), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_dist_cond_types, sc.h_dist_cond_types, sc.dist_cond_len * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_dist_cond_lens, sc.h_dist_cond_lens, _n_dist_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.d_dist_cond_rows, sc.h_dist_cond_rows, _n_dist_regions * sizeof(int), cudaMemcpyHostToDevice));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_init_CUDA_MD_symbols() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	// this line tells the mixed backend to initialize the constants needed by its kernels (which are in CUDA_mixed.cuh), so that sim_step will work properly
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDAMixedBackend::_init_CUDA_MD_symbols();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:bool FFS_MD_CUDAMixedBackend::_read_conditions(const char *fname, const char *condition_set_type, std::vector<parsed_condition> *conditions) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:bool FFS_MD_CUDAMixedBackend::_read_conditions(const char *fname, const char *condition_set_type, std::vector<master_condition> *conditions) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_master_conditions_prepare_simple_conditions(vector<master_condition> *master_conditions, const char *fname) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_init_ffs_from_file(const char *fname) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_init_ffs_kernel_config(CUDA_kernel_cfg *kernel_cfg, int total_threads) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_eval_order_parameter_states() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	_cuda_interaction->_hb_op_precalc(_d_poss, _d_orientations, _d_hb_pairs1, _d_hb_pairs2, _d_hb_energies, _n_hb_pairs, _d_region_is_nearhb, _ffs_hb_precalc_kernel_cfg, _d_cuda_box);
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	_cuda_interaction->_near_hb_op_precalc(_d_poss, _d_orientations, _d_hb_pairs1, _d_hb_pairs2, _d_nearhb_states, _n_hb_pairs, _d_region_is_nearhb, _ffs_hb_precalc_kernel_cfg, _d_cuda_box);
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	_cuda_interaction->_dist_op_precalc(_d_poss, _d_orientations, _d_dist_pairs1, _d_dist_pairs2, _d_op_dists, _n_dist_pairs, _ffs_dist_precalc_kernel_cfg, _d_cuda_box);
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	cudaDeviceSynchronize();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_eval_stop_conditions(SimpleConditions sc) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	// check the values of the order parameters against the stopping conditions given -- do it on the GPU to reduce device-host data transfer
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	cudaDeviceSynchronize();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(sc.h_ffs_stop, sc.d_ffs_stop, sc.stop_length * sizeof(bool), cudaMemcpyDeviceToHost));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:int FFS_MD_CUDAMixedBackend::_test_crossing(SimpleConditions sc, bool suppress_logging) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:bool FFS_MD_CUDAMixedBackend::_test_master_condition(master_condition master_condition) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:bool FFS_MD_CUDAMixedBackend::_test_master_conditions(vector<master_condition> master_conditions) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_log_master_state(master_condition master_condition) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_handle_unexpected_master() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:bool FFS_MD_CUDAMixedBackend::_check_stop() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::_prepare_configuration(char *conf_str) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:int *FFS_MD_CUDAMixedBackend::_get_2D_rows(int rows_len, int *lens) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:char * FFS_MD_CUDAMixedBackend::get_op_state_str(void) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_hb_energies, _d_hb_energies, _n_hb_pairs * sizeof(float), cudaMemcpyDeviceToHost));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_op_dists, _d_op_dists, _n_dist_pairs * sizeof(float), cudaMemcpyDeviceToHost));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_nearhb_states, _d_nearhb_states, _n_hb_pairs * sizeof(bool), cudaMemcpyDeviceToHost));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::sprintf_names_and_values(char *str) {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	// copy order parameter states from GPU
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_hb_energies, _d_hb_energies, _n_hb_pairs * sizeof(float), cudaMemcpyDeviceToHost));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_op_dists, _d_op_dists, _n_dist_pairs * sizeof(float), cudaMemcpyDeviceToHost));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_nearhb_states, _d_nearhb_states, _n_hb_pairs * sizeof(bool), cudaMemcpyDeviceToHost));
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:void FFS_MD_CUDAMixedBackend::print_observables() {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.cu:	CUDAMixedBackend::print_observables();
src/CUDA/Backends/MD_CUDAMixedBackend.h: * MD_CUDAMixedBackend.h
src/CUDA/Backends/MD_CUDAMixedBackend.h:#ifndef MD_CUDAMIXEDBACKEND_H_
src/CUDA/Backends/MD_CUDAMixedBackend.h:#define MD_CUDAMIXEDBACKEND_H_
src/CUDA/Backends/MD_CUDAMixedBackend.h:#include "MD_CUDABackend.h"
src/CUDA/Backends/MD_CUDAMixedBackend.h:// this compilation unit is compiled only when the CUDA backend is compiled with CUDA_DOUBLE set to OFF
src/CUDA/Backends/MD_CUDAMixedBackend.h:using GPU_quat_double = double4;
src/CUDA/Backends/MD_CUDAMixedBackend.h: * @brief CUDA backend with mixed precision for MD simulations.
src/CUDA/Backends/MD_CUDAMixedBackend.h: * This class is a regular MD backend written to be almost as fast as a MD_CUDABackend<float, float4>
src/CUDA/Backends/MD_CUDAMixedBackend.h: * and, at the same time, almost as reliable as MD_CUDABackend<double, LR_double4> when it comes
src/CUDA/Backends/MD_CUDAMixedBackend.h:class CUDAMixedBackend: public MD_CUDABackend {
src/CUDA/Backends/MD_CUDAMixedBackend.h:	GPU_quat_double *_d_orientationsd;
src/CUDA/Backends/MD_CUDAMixedBackend.h:	void _quat_double_to_quat_float(GPU_quat_double *src, GPU_quat *dest);
src/CUDA/Backends/MD_CUDAMixedBackend.h:	void _quat_float_to_quat_double(GPU_quat *src, GPU_quat_double *dest);
src/CUDA/Backends/MD_CUDAMixedBackend.h:	void _init_CUDA_MD_symbols() override;
src/CUDA/Backends/MD_CUDAMixedBackend.h:	CUDAMixedBackend();
src/CUDA/Backends/MD_CUDAMixedBackend.h:	virtual ~CUDAMixedBackend();
src/CUDA/Backends/MD_CUDAMixedBackend.h:#endif /* MD_CUDAMIXEDBACKEND_H_ */
src/CUDA/Backends/FFS_CUDA_MD.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Backends/FFS_CUDA_MD.cuh:__global__ void hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, float *hb_energies, int n_threads, bool *region_is_nearhb) {
src/CUDA/Backends/FFS_CUDA_MD.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Backends/FFS_CUDA_MD.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Backends/FFS_CUDA_MD.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t2 = CUDA_LRACOS(-CUDA_DOT(b1, rhydrodir));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t3 = CUDA_LRACOS(CUDA_DOT(a1, rhydrodir));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t7 = CUDA_LRACOS(-CUDA_DOT(rhydrodir, b3));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t8 = CUDA_LRACOS(CUDA_DOT(rhydrodir, a3));
src/CUDA/Backends/FFS_CUDA_MD.cuh:__global__ void near_hb_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, bool *nearly_bonded_array, int n_threads, bool *region_is_nearhb) {
src/CUDA/Backends/FFS_CUDA_MD.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Backends/FFS_CUDA_MD.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Backends/FFS_CUDA_MD.cuh:	c_number rhydromodsqr = CUDA_DOT(rhydro, rhydro);
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t1 = CUDA_LRACOS(-CUDA_DOT(a1, b1));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t2 = CUDA_LRACOS(-CUDA_DOT(b1, rhydrodir));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t3 = CUDA_LRACOS(CUDA_DOT(a1, rhydrodir));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t4 = CUDA_LRACOS(CUDA_DOT(a3, b3));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t7 = CUDA_LRACOS(-CUDA_DOT(rhydrodir, b3));
src/CUDA/Backends/FFS_CUDA_MD.cuh:		c_number t8 = CUDA_LRACOS(CUDA_DOT(rhydrodir, a3));
src/CUDA/Backends/FFS_CUDA_MD.cuh:__global__ void dist_op_precalc(c_number4 *poss, GPU_quat *orientations, int *op_pairs1, int *op_pairs2, c_number *op_dists, int n_threads) {
src/CUDA/Backends/FFS_CUDA_MD.cuh:	GPU_quat po = orientations[pind];
src/CUDA/Backends/FFS_CUDA_MD.cuh:	GPU_quat qo = orientations[qind];
src/CUDA/Backends/CUDABaseBackend.cu: * CUDABaseBackend.cpp
src/CUDA/Backends/CUDABaseBackend.cu:#include "CUDABaseBackend.h"
src/CUDA/Backends/CUDABaseBackend.cu:#include "../Lists/CUDAListFactory.h"
src/CUDA/Backends/CUDABaseBackend.cu:#include "../Interactions/CUDAInteractionFactory.h"
src/CUDA/Backends/CUDABaseBackend.cu:CUDABaseBackend::CUDABaseBackend() :
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_lists = NULL;
src/CUDA/Backends/CUDABaseBackend.cu:	_d_cuda_box = NULL;
src/CUDA/Backends/CUDABaseBackend.cu:	GpuUtils::reset_allocated_mem();
src/CUDA/Backends/CUDABaseBackend.cu:CUDABaseBackend::~CUDABaseBackend() {
src/CUDA/Backends/CUDABaseBackend.cu:	if(_cuda_lists != NULL) {
src/CUDA/Backends/CUDABaseBackend.cu:		_cuda_lists->clean();
src/CUDA/Backends/CUDABaseBackend.cu:		delete _cuda_lists;
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_poss));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_bonds));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_orientations));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_list_poss));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(cudaFreeHost(_d_are_lists_old));
src/CUDA/Backends/CUDABaseBackend.cu:			CUDA_SAFE_CALL(cudaFree(_d_hindex));
src/CUDA/Backends/CUDABaseBackend.cu:			CUDA_SAFE_CALL(cudaFree(_d_sorted_hindex));
src/CUDA/Backends/CUDABaseBackend.cu:			CUDA_SAFE_CALL(cudaFree(_d_inv_sorted_hindex));
src/CUDA/Backends/CUDABaseBackend.cu:			CUDA_SAFE_CALL(cudaFree(_d_buff_poss));
src/CUDA/Backends/CUDABaseBackend.cu:			CUDA_SAFE_CALL(cudaFree(_d_buff_bonds));
src/CUDA/Backends/CUDABaseBackend.cu:			CUDA_SAFE_CALL(cudaFree(_d_buff_orientations));
src/CUDA/Backends/CUDABaseBackend.cu:void CUDABaseBackend::_host_to_gpu() {
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_poss, _h_poss, _vec_size, cudaMemcpyHostToDevice));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_bonds, _h_bonds, _bonds_size, cudaMemcpyHostToDevice));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_orientations, _h_orientations, _orient_size, cudaMemcpyHostToDevice));
src/CUDA/Backends/CUDABaseBackend.cu:	_h_cuda_box.set_CUDA_from_CPU(CONFIG_INFO->box);
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_cuda_box, &_h_cuda_box, sizeof(CUDABox), cudaMemcpyHostToDevice));
src/CUDA/Backends/CUDABaseBackend.cu:void CUDABaseBackend::_gpu_to_host() {
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_poss, _d_poss, _vec_size, cudaMemcpyDeviceToHost));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_bonds, _d_bonds, _bonds_size, cudaMemcpyDeviceToHost));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_orientations, _d_orientations, _orient_size, cudaMemcpyDeviceToHost));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(&_h_cuda_box, _d_cuda_box, sizeof(CUDABox), cudaMemcpyDeviceToHost));
src/CUDA/Backends/CUDABaseBackend.cu:	_h_cuda_box.set_CPU_from_CUDA(CONFIG_INFO->box);
src/CUDA/Backends/CUDABaseBackend.cu:void CUDABaseBackend::get_settings(input_file &inp) {
src/CUDA/Backends/CUDABaseBackend.cu:	if(getInputInt(&inp, "CUDA_device", &_device_number, 0) == KEY_NOT_FOUND) {
src/CUDA/Backends/CUDABaseBackend.cu:		OX_LOG(Logger::LOG_INFO, "CUDA device not specified");
src/CUDA/Backends/CUDABaseBackend.cu:		OX_LOG(Logger::LOG_INFO, "Using CUDA device %d", _device_number);
src/CUDA/Backends/CUDABaseBackend.cu:	if(getInputInt(&inp, "CUDA_sort_every", &_sort_every, 0) == KEY_NOT_FOUND) {
src/CUDA/Backends/CUDABaseBackend.cu:		OX_LOG(Logger::LOG_INFO, "CUDA sort_every not specified, using 0");
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_interaction = CUDAInteractionFactory::make_interaction(inp);
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_interaction->get_settings(inp);
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_interaction->get_cuda_settings(inp);
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_lists = CUDAListFactory::make_list(inp);
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_lists->get_settings(inp);
src/CUDA/Backends/CUDABaseBackend.cu:		throw oxDNAException("The CUDA backend does not support reloading checkpoints, owing to its intrinsically stochastic nature");
src/CUDA/Backends/CUDABaseBackend.cu:void CUDABaseBackend::_choose_device() {
src/CUDA/Backends/CUDABaseBackend.cu:	cudaDeviceProp tryprop;
src/CUDA/Backends/CUDABaseBackend.cu:		cudaSetDevice(trydev);
src/CUDA/Backends/CUDABaseBackend.cu:		cudaError_t result = cudaMalloc((void **) &dummyptr, (size_t) sizeof(int));
src/CUDA/Backends/CUDABaseBackend.cu:		cudaFree(dummyptr);
src/CUDA/Backends/CUDABaseBackend.cu:		if(result == cudaSuccess) {
src/CUDA/Backends/CUDABaseBackend.cu:			cudaGetLastError();
src/CUDA/Backends/CUDABaseBackend.cu:void CUDABaseBackend::init_cuda() {
src/CUDA/Backends/CUDABaseBackend.cu:	if(cudaSetDevice(_device_number) != cudaSuccess || cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) != cudaSuccess) {
src/CUDA/Backends/CUDABaseBackend.cu:	_h_cuda_box.set_CUDA_from_CPU(CONFIG_INFO->box);
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_interaction->cuda_init(N);
src/CUDA/Backends/CUDABaseBackend.cu:	_orient_size = sizeof(GPU_quat) * N;
src/CUDA/Backends/CUDABaseBackend.cu:	// GPU memory allocations
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_poss, _vec_size));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<LR_bonds>(&_d_bonds, _bonds_size));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<GPU_quat>(&_d_orientations, _orient_size));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_list_poss, _vec_size));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<CUDABox>(&_d_cuda_box, sizeof(CUDABox)));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMallocHost(&_d_are_lists_old, sizeof(bool), cudaHostAllocDefault));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_list_poss, _vec_size));
src/CUDA/Backends/CUDABaseBackend.cu:	CUDA_SAFE_CALL(cudaMemset(_d_list_poss, 0, _vec_size));
src/CUDA/Backends/CUDABaseBackend.cu:	_h_orientations = new GPU_quat[N];
src/CUDA/Backends/CUDABaseBackend.cu:	_init_CUDA_kernel_cfgs();
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_lists->init(N, _cuda_interaction->get_cuda_rcut(), &_h_cuda_box, _d_cuda_box);
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_hindex, N * sizeof(int)));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_sorted_hindex, N * sizeof(int)));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_inv_sorted_hindex, N * sizeof(int)));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_buff_poss, _vec_size));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<LR_bonds>(&_d_buff_bonds, _bonds_size));
src/CUDA/Backends/CUDABaseBackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<GPU_quat>(&_d_buff_orientations, _orient_size));
src/CUDA/Backends/CUDABaseBackend.cu:void CUDABaseBackend::_init_CUDA_kernel_cfgs() {
src/CUDA/Backends/CUDABaseBackend.cu:	_cuda_interaction->set_launch_cfg(_particles_kernel_cfg);
src/CUDA/Backends/CUDABaseBackend.cu:void CUDABaseBackend::_sort_index() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu: * MD_CUDAMixedBackend.cu
src/CUDA/Backends/MD_CUDAMixedBackend.cu:#include "MD_CUDAMixedBackend.h"
src/CUDA/Backends/MD_CUDAMixedBackend.cu:#include "CUDA_mixed.cuh"
src/CUDA/Backends/MD_CUDAMixedBackend.cu:CUDAMixedBackend::CUDAMixedBackend() : MD_CUDABackend() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:CUDAMixedBackend::~CUDAMixedBackend(){
src/CUDA/Backends/MD_CUDAMixedBackend.cu:		CUDA_SAFE_CALL( cudaFree(_d_possd) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:		CUDA_SAFE_CALL( cudaFree(_d_orientationsd) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:		CUDA_SAFE_CALL( cudaFree(_d_velsd) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:		CUDA_SAFE_CALL( cudaFree(_d_Lsd) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::init() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	MD_CUDABackend::init();
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	_orient_sized = ((size_t) N()) * sizeof(GPU_quat_double);
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<LR_double4>(&_d_possd, _vec_sized) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<LR_double4>(&_d_velsd, _vec_sized) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<LR_double4>(&_d_Lsd, _vec_sized) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<GPU_quat_double>(&_d_orientationsd, _orient_sized) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_init_CUDA_MD_symbols() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	MD_CUDABackend::_init_CUDA_MD_symbols();
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_sqr_verlet_skin, &f_copy, sizeof(float)) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dt, &f_copy, sizeof(float)) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_N, &myN, sizeof(int)) );
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_float4_to_LR_double4(float4 *src, LR_double4 *dest) {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_LR_double4_to_float4(LR_double4 *src, float4 *dest) {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_quat_float_to_quat_double(GPU_quat *src, GPU_quat_double *dest) {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_quat_double_to_quat_float(GPU_quat_double *src, GPU_quat *dest) {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_first_step() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_rescale_positions(float4 new_Ls, float4 old_Ls) {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	MD_CUDABackend::_rescale_positions(new_Ls, old_Ls);
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_sort_particles() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	MD_CUDABackend::_sort_particles();
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_forces_second_step() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	_cuda_interaction->compute_forces(_cuda_lists, _d_poss, _d_orientations, _d_forces, _d_torques, _d_bonds, _d_cuda_box);
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::apply_simulation_data_changes() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	MD_CUDABackend::apply_simulation_data_changes();
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::apply_changes_to_simulation_data() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	MD_CUDABackend::apply_changes_to_simulation_data();
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_thermalize() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	if(_cuda_thermostat->would_activate(current_step())) {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:		MD_CUDABackend::_thermalize();
src/CUDA/Backends/MD_CUDAMixedBackend.cu:void CUDAMixedBackend::_update_stress_tensor() {
src/CUDA/Backends/MD_CUDAMixedBackend.cu:	MD_CUDABackend::_update_stress_tensor();
src/CUDA/Backends/MD_CUDABackend.cu: * MD_CUDABackend.cpp
src/CUDA/Backends/MD_CUDABackend.cu:#include "MD_CUDABackend.h"
src/CUDA/Backends/MD_CUDABackend.cu:#include "../CUDAForces.h"
src/CUDA/Backends/MD_CUDABackend.cu:#include "CUDA_MD.cuh"
src/CUDA/Backends/MD_CUDABackend.cu:#include "../CUDA_base_interactions.h"
src/CUDA/Backends/MD_CUDABackend.cu:#include "../Thermostats/CUDAThermostatFactory.h"
src/CUDA/Backends/MD_CUDABackend.cu:MD_CUDABackend::MD_CUDABackend() :
src/CUDA/Backends/MD_CUDABackend.cu:				CUDABaseBackend(),
src/CUDA/Backends/MD_CUDABackend.cu:	_h_gpu_index = _h_cpu_index = nullptr;
src/CUDA/Backends/MD_CUDABackend.cu:	// on CUDA the timers need to be told to explicitly synchronise on the GPU
src/CUDA/Backends/MD_CUDABackend.cu:MD_CUDABackend::~MD_CUDABackend() {
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_particles_to_mols));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_vels));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_Ls));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_forces));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_torques));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_molecular_coms));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_buff_vels));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_buff_Ls));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaFree(_d_buff_particles_to_mols));
src/CUDA/Backends/MD_CUDABackend.cu:	if(_h_gpu_index != nullptr) {
src/CUDA/Backends/MD_CUDABackend.cu:		delete[] _h_gpu_index;
src/CUDA/Backends/MD_CUDABackend.cu:			CUDA_SAFE_CALL(cudaFree(_d_ext_forces));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_host_to_gpu() {
src/CUDA/Backends/MD_CUDABackend.cu:	CUDABaseBackend::_host_to_gpu();
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_particles_to_mols, _h_particles_to_mols.data(), sizeof(int) * N(), cudaMemcpyHostToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_vels, _h_vels, _vec_size, cudaMemcpyHostToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_Ls, _h_Ls, _vec_size, cudaMemcpyHostToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_gpu_to_host() {
src/CUDA/Backends/MD_CUDABackend.cu:	CUDABaseBackend::_gpu_to_host();
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_particles_to_mols.data(), _d_particles_to_mols, sizeof(int) * N(), cudaMemcpyDeviceToHost));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_vels, _d_vels, _vec_size, cudaMemcpyDeviceToHost));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_Ls, _d_Ls, _vec_size, cudaMemcpyDeviceToHost));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_forces, _d_forces, _vec_size, cudaMemcpyDeviceToHost));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_h_torques, _d_torques, _vec_size, cudaMemcpyDeviceToHost));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_apply_external_forces_changes() {
src/CUDA/Backends/MD_CUDABackend.cu:			throw oxDNAException("External forces and CUDA_sort_every > 0 are not compatible");
src/CUDA/Backends/MD_CUDABackend.cu:		static std::vector<CUDA_trap> h_ext_forces(N() * MAX_EXT_FORCES);
src/CUDA/Backends/MD_CUDABackend.cu:			CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<CUDA_trap >(&_d_ext_forces, N() * MAX_EXT_FORCES * sizeof(CUDA_trap)));
src/CUDA/Backends/MD_CUDABackend.cu:				CUDA_trap *cuda_force = &(h_ext_forces[j * N() + i]);
src/CUDA/Backends/MD_CUDABackend.cu:					init_ConstantRateForce_from_CPU(&cuda_force->constant, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_MutualTrap_from_CPU(&cuda_force->mutual, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_MovingTrap_from_CPU(&cuda_force->moving, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_LowdimMovingTrap_from_CPU(&cuda_force->lowdim, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_RepulsionPlane_from_CPU(&cuda_force->repulsionplane, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_AttractionPlane_from_CPU(&cuda_force->attractionplane, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_RepulsionPlaneMoving_from_CPU(&cuda_force->repulsionplanemoving, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_RepulsiveSphere_from_CPU(&cuda_force->repulsivesphere, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_RepulsiveSphereSmooth_from_CPU(&cuda_force->repulsivespheresmooth, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_LJWall_from_CPU(&cuda_force->ljwall, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_ConstantRateTorque_from_CPU(&cuda_force->constantratetorque, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_GenericCentralForce_from_CPU(&cuda_force->genericconstantforce, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_LJCone_from_CPU(&cuda_force->ljcone, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_RepulsiveEllipsoid_from_CPU(&cuda_force->repulsiveellipsoid, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:					init_COMForce_from_CPU(&cuda_force->comforce, p_force, first_time);
src/CUDA/Backends/MD_CUDABackend.cu:					init_LTCOMTrap_from_CPU(&cuda_force->ltcomtrap, p_force, first_time);
src/CUDA/Backends/MD_CUDABackend.cu:					init_YukawaSphere_from_CPU(&cuda_force->yukawasphere, p_force);
src/CUDA/Backends/MD_CUDABackend.cu:							"forces are supported on CUDA at the moment.\n");
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaMemcpy(_d_ext_forces, h_ext_forces.data(), N() * MAX_EXT_FORCES * sizeof(CUDA_trap), cudaMemcpyHostToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::apply_changes_to_simulation_data() {
src/CUDA/Backends/MD_CUDABackend.cu:		int gpu_index = _h_gpu_index[i];
src/CUDA/Backends/MD_CUDABackend.cu:		BaseParticle *p = _particles[gpu_index];
src/CUDA/Backends/MD_CUDABackend.cu:		_h_poss[i].w = GpuUtils::int_as_float((p->btype << 22) | ((~msk) & p->index));
src/CUDA/Backends/MD_CUDABackend.cu:		int mybtype = (GpuUtils::float_as_int(_h_poss[i].w)) >> 22;
src/CUDA/Backends/MD_CUDABackend.cu:		int myindex = (GpuUtils::float_as_int(_h_poss[i].w)) & (~msk);
src/CUDA/Backends/MD_CUDABackend.cu:			throw oxDNAException("Could not treat the type (A, C, G, T or something specific) of particle %d; On CUDA, integer base types cannot be larger than 511 or smaller than -511");
src/CUDA/Backends/MD_CUDABackend.cu:			throw oxDNAException("Could not treat the index of particle %d; remember that on CUDA the maximum c_number of particles is 2^21", p->index);
src/CUDA/Backends/MD_CUDABackend.cu:	_host_to_gpu();
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_interaction->sync_GPU();
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::apply_simulation_data_changes() {
src/CUDA/Backends/MD_CUDABackend.cu:	_gpu_to_host();
src/CUDA/Backends/MD_CUDABackend.cu:		int newindex = ((GpuUtils::float_as_int(_h_poss[i].w)) & (~msk));
src/CUDA/Backends/MD_CUDABackend.cu:		_h_gpu_index[i] = newindex;
src/CUDA/Backends/MD_CUDABackend.cu:		p->btype = (GpuUtils::float_as_int(_h_poss[i].w)) >> 22;
src/CUDA/Backends/MD_CUDABackend.cu:			int n3index = ((GpuUtils::float_as_int(_h_poss[_h_bonds[i].n3].w)) & (~msk));
src/CUDA/Backends/MD_CUDABackend.cu:			int n5index = ((GpuUtils::float_as_int(_h_poss[_h_bonds[i].n5].w)) & (~msk));
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_interaction->sync_host();
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_init_CUDA_MD_symbols() {
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_sqr_verlet_skin, &f_copy, sizeof(float)));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_dt, &f_copy, sizeof(float)));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(MD_N, &myN, sizeof(int)));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_first_step() {
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_rescale_molecular_positions(c_number4 new_Ls, c_number4 old_Ls, bool recompute_coms) {
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaMemset(_d_molecular_coms, 0, sizeof(c_number4) * _molecules.size()));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_rescale_positions(c_number4 new_Ls, c_number4 old_Ls) {
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_apply_barostat() {
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_interaction->compute_forces(_cuda_lists, _d_poss, _d_orientations, _d_forces, _d_torques, _d_bonds, _d_cuda_box);
src/CUDA/Backends/MD_CUDABackend.cu:	double old_energy = GpuUtils::sum_c_number4_to_double_on_GPU(_d_forces, N()) / 2.;
src/CUDA/Backends/MD_CUDABackend.cu:	c_number old_V = _h_cuda_box.V();
src/CUDA/Backends/MD_CUDABackend.cu:	c_number4 old_Ls = _h_cuda_box.box_sides();
src/CUDA/Backends/MD_CUDABackend.cu:	_h_cuda_box.change_sides(new_Ls.x, new_Ls.y, new_Ls.z);
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_cuda_box, &_h_cuda_box, sizeof(CUDABox), cudaMemcpyHostToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_lists->update(_d_poss, _d_list_poss, _d_bonds);
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_interaction->compute_forces(_cuda_lists, _d_poss, _d_orientations, _d_forces, _d_torques, _d_bonds, _d_cuda_box);
src/CUDA/Backends/MD_CUDABackend.cu:	double new_energy = GpuUtils::sum_c_number4_to_double_on_GPU(_d_forces, N()) / 2.;
src/CUDA/Backends/MD_CUDABackend.cu:	c_number new_V = _h_cuda_box.V();
src/CUDA/Backends/MD_CUDABackend.cu:		_h_cuda_box.change_sides(old_Ls.x, old_Ls.y, old_Ls.z);
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(cudaMemcpy(_d_cuda_box, &_h_cuda_box, sizeof(CUDABox), cudaMemcpyHostToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:		_cuda_lists->update(_d_poss, _d_list_poss, _d_bonds);
src/CUDA/Backends/MD_CUDABackend.cu:	if(_cuda_barostat_always_refresh) {
src/CUDA/Backends/MD_CUDABackend.cu:		_cuda_barostat_thermostat->apply_cuda(_d_poss, _d_orientations, _d_vels, _d_Ls, current_step());
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_forces_second_step() {
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_interaction->compute_forces(_cuda_lists, _d_poss, _d_orientations, _d_forces, _d_torques, _d_bonds, _d_cuda_box);
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_set_external_forces() {
src/CUDA/Backends/MD_CUDABackend.cu:		(_d_poss, _d_orientations, _d_ext_forces, _d_forces, _d_torques, current_step(), _max_ext_forces, _d_cuda_box);
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_sort_particles() {
src/CUDA/Backends/MD_CUDABackend.cu:	CUDABaseBackend::_sort_index();
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_orientations, _d_buff_orientations, _orient_size, cudaMemcpyDeviceToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_poss, _d_buff_poss, _vec_size, cudaMemcpyDeviceToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_bonds, _d_buff_bonds, _bonds_size, cudaMemcpyDeviceToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_vels, _d_buff_vels, _vec_size, cudaMemcpyDeviceToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_Ls, _d_buff_Ls, _vec_size, cudaMemcpyDeviceToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_particles_to_mols, _d_buff_particles_to_mols, sizeof(int) * N(), cudaMemcpyDeviceToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_thermalize() {
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_thermostat->apply_cuda(_d_poss, _d_orientations, _d_vels, _d_Ls, current_step());
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::_update_stress_tensor() {
src/CUDA/Backends/MD_CUDABackend.cu:		_interaction->set_stress_tensor(_cuda_interaction->CPU_stress_tensor(_d_vels));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::sim_step() {
src/CUDA/Backends/MD_CUDABackend.cu:			_cuda_lists->update(_d_poss, _d_list_poss, _d_bonds);
src/CUDA/Backends/MD_CUDABackend.cu:		c_number energy = GpuUtils::sum_c_number4_to_double_on_GPU(_d_forces, N());
src/CUDA/Backends/MD_CUDABackend.cu:		_backend_info = Utils::sformat("\tCUDA_energy: %lf", energy / (2. * N()));
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::get_settings(input_file &inp) {
src/CUDA/Backends/MD_CUDABackend.cu:	CUDABaseBackend::get_settings(inp);
src/CUDA/Backends/MD_CUDABackend.cu:	getInputBool(&inp, "CUDA_avoid_cpu_calculations", &_avoid_cpu_calculations, 0);
src/CUDA/Backends/MD_CUDABackend.cu:	getInputBool(&inp, "CUDA_barostat_always_refresh", &_cuda_barostat_always_refresh, 0);
src/CUDA/Backends/MD_CUDABackend.cu:	getInputBool(&inp, "CUDA_print_energy", &_print_energy, 0);
src/CUDA/Backends/MD_CUDABackend.cu:	getInputInt(&inp, "CUDA_update_stress_tensor_every", &_update_st_every, 0);
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_thermostat = CUDAThermostatFactory::make_thermostat(inp, _box.get());
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_thermostat->get_settings(inp);
src/CUDA/Backends/MD_CUDABackend.cu:		_cuda_barostat_thermostat = std::make_shared<CUDABrownianThermostat>();
src/CUDA/Backends/MD_CUDABackend.cu:		_cuda_barostat_thermostat->get_settings(*inp_file);
src/CUDA/Backends/MD_CUDABackend.cu:void MD_CUDABackend::init() {
src/CUDA/Backends/MD_CUDABackend.cu:	CUDABaseBackend::init_cuda();
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_vels, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_Ls, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_forces, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_torques, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_particles_to_mols, sizeof(int) * N()));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_mol_sizes, sizeof(int) * _molecules.size()));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_molecular_coms, sizeof(c_number4) * _molecules.size()));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemset(_d_forces, 0, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemset(_d_torques, 0, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:	// initialise the GPU array containing the size of the molecules
src/CUDA/Backends/MD_CUDABackend.cu:	CUDA_SAFE_CALL(cudaMemcpy(_d_mol_sizes, mol_sizes.data(), sizeof(int) * _molecules.size(), cudaMemcpyHostToDevice));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_buff_vels, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<c_number4>(&_d_buff_Ls, _vec_size));
src/CUDA/Backends/MD_CUDABackend.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&_d_buff_particles_to_mols, sizeof(int) * N()));
src/CUDA/Backends/MD_CUDABackend.cu:	_h_gpu_index = new int[N()];
src/CUDA/Backends/MD_CUDABackend.cu:		_h_gpu_index[i] = i;
src/CUDA/Backends/MD_CUDABackend.cu:	_init_CUDA_MD_symbols();
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_thermostat->set_seed(lrand48());
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_thermostat->init();
src/CUDA/Backends/MD_CUDABackend.cu:		_cuda_barostat_thermostat->set_seed(lrand48());
src/CUDA/Backends/MD_CUDABackend.cu:		_cuda_barostat_thermostat->init();
src/CUDA/Backends/MD_CUDABackend.cu:	OX_LOG(Logger::LOG_INFO, "Allocated CUDA memory: %.2lf MBs", GpuUtils::get_allocated_mem_mb());
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_lists->update(_d_poss, _d_list_poss, _d_bonds);
src/CUDA/Backends/MD_CUDABackend.cu:	_cuda_interaction->compute_forces(_cuda_lists, _d_poss, _d_orientations, _d_forces, _d_torques, _d_bonds, _d_cuda_box);
src/CUDA/Backends/MD_CUDABackend.cu:		_interaction->set_stress_tensor(_cuda_interaction->CPU_stress_tensor(_d_vels));
src/CUDA/Backends/CUDABaseBackend.h: * @file    CUDABaseBackend.h
src/CUDA/Backends/CUDABaseBackend.h:#ifndef CUDABASEBACKEND_H_
src/CUDA/Backends/CUDABaseBackend.h:#define CUDABASEBACKEND_H_
src/CUDA/Backends/CUDABaseBackend.h:#include <cuda.h>
src/CUDA/Backends/CUDABaseBackend.h:#include <cuda_runtime_api.h>
src/CUDA/Backends/CUDABaseBackend.h:#include "../cuda_utils/cuda_device_utils.h"
src/CUDA/Backends/CUDABaseBackend.h:#include "../cuda_utils/CUDABox.h"
src/CUDA/Backends/CUDABaseBackend.h:#include "../CUDAUtils.h"
src/CUDA/Backends/CUDABaseBackend.h:#include "../Lists/CUDABaseList.h"
src/CUDA/Backends/CUDABaseBackend.h:#include "../CUDA_sort.cuh"
src/CUDA/Backends/CUDABaseBackend.h:#include "../Interactions/CUDABaseInteraction.h"
src/CUDA/Backends/CUDABaseBackend.h: * @brief Basic simulation backend on CUDA. All CUDA backends should inherit from this class as well as from a regular CPU backend
src/CUDA/Backends/CUDABaseBackend.h: * This class does not actually do any computation but provides basic CUDA facilities.
src/CUDA/Backends/CUDABaseBackend.h:[CUDA_device = <int> (CUDA-enabled device to run the simulation on. If it is not specified or it is given a negative c_number, a suitable device will be automatically chosen.)]
src/CUDA/Backends/CUDABaseBackend.h:[CUDA_sort_every = <int> (sort particles according to a 3D Hilbert curve every CUDA_sort_every time steps. This will greatly enhnance performances for some types of interaction. Defaults to 0, which disables sorting.)]
src/CUDA/Backends/CUDABaseBackend.h:[threads_per_block = <int> (c_number of threads per block on the CUDA grid. defaults to 2 * the size of a warp.)]
src/CUDA/Backends/CUDABaseBackend.h:class CUDABaseBackend {
src/CUDA/Backends/CUDABaseBackend.h:	cudaDeviceProp _device_prop;
src/CUDA/Backends/CUDABaseBackend.h:	CUDA_kernel_cfg _particles_kernel_cfg;
src/CUDA/Backends/CUDABaseBackend.h:	CUDABaseList*_cuda_lists;
src/CUDA/Backends/CUDABaseBackend.h:	CUDABox _h_cuda_box, *_d_cuda_box;
src/CUDA/Backends/CUDABaseBackend.h:	GPU_quat *_d_buff_orientations;
src/CUDA/Backends/CUDABaseBackend.h:	GPU_quat *_d_orientations, *_h_orientations;
src/CUDA/Backends/CUDABaseBackend.h:	std::shared_ptr<CUDABaseInteraction> _cuda_interaction = nullptr;
src/CUDA/Backends/CUDABaseBackend.h:	virtual void _host_to_gpu();
src/CUDA/Backends/CUDABaseBackend.h:	virtual void _gpu_to_host();
src/CUDA/Backends/CUDABaseBackend.h:	virtual void _init_CUDA_kernel_cfgs();
src/CUDA/Backends/CUDABaseBackend.h:	 * @brief This handy method automatically selects a CUDA device to run the simulation on if there are no user-specified devices.
src/CUDA/Backends/CUDABaseBackend.h:	CUDABaseBackend();
src/CUDA/Backends/CUDABaseBackend.h:	virtual ~CUDABaseBackend();
src/CUDA/Backends/CUDABaseBackend.h:	virtual void init_cuda();
src/CUDA/Backends/CUDABaseBackend.h:#endif /* CUDABASEBACKEND_H_ */
src/CUDA/Backends/CUDA_MD.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Backends/CUDA_MD.cuh:__device__ GPU_quat _get_updated_orientation(c_number4 &L, GPU_quat &old_o) {
src/CUDA/Backends/CUDA_MD.cuh:	GPU_quat R = {(c_number) 0.5f*L.x*sintheta*winv, (c_number) 0.5f*L.y*sintheta*winv, (c_number) 0.5f*L.z*sintheta*winv, qw};
src/CUDA/Backends/CUDA_MD.cuh:__global__ void first_step(c_number4 *poss, GPU_quat *orientations, c_number4 *list_poss, c_number4 *vels, c_number4 *Ls, c_number4 *forces, c_number4 *torques, bool *are_lists_old) {
src/CUDA/Backends/CUDA_MD.cuh:	GPU_quat qold_o = orientations[IND];
src/CUDA/Backends/CUDA_MD.cuh:__global__ void set_external_forces(c_number4 *poss, GPU_quat *orientations, CUDA_trap *ext_forces, c_number4 *forces, c_number4 *torques, llint step, int max_ext_forces, CUDABox *box) {
src/CUDA/Backends/CUDA_MD.cuh:		CUDA_trap extF = ext_forces[MD_N[0] * i + IND];
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_TRAP_CONSTANT: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_TRAP_MUTUAL: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_TRAP_MOVING: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_TRAP_MOVING_LOWDIM: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_REPULSION_PLANE: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_ATTRACTION_PLANE: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_REPULSION_PLANE_MOVING: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_REPULSIVE_SPHERE: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_REPULSIVE_SPHERE_SMOOTH: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_LJ_WALL: {
src/CUDA/Backends/CUDA_MD.cuh:				c_number distance = CUDA_DOT(extF.ljwall.dir, ppos) + extF.ljwall.position;
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_CONSTANT_RATE_TORQUE: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_GENERIC_CENTRAL_FORCE: {
src/CUDA/Backends/CUDA_MD.cuh:				c_number dist_sqr = CUDA_DOT(dir, dir);
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_LJ_CONE: {
src/CUDA/Backends/CUDA_MD.cuh:				c_number d_along_axis = CUDA_DOT(v_from_apex, extF.ljcone.dir);
src/CUDA/Backends/CUDA_MD.cuh:				c_number beta = CUDA_LRACOS(CUDA_DOT(v_from_apex, v_along_axis)/(d_from_apex*d_along_axis));
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_REPULSIVE_ELLIPSOID: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_COM_FORCE: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_LR_COM_TRAP: {
src/CUDA/Backends/CUDA_MD.cuh:			case CUDA_YUKAWA_SPHERE: {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: * FFS_MD_CUDAMixedBackend.h
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:#ifndef FFS_MD_CUDAMIXEDBACKEND_H_
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:#define FFS_MD_CUDAMIXEDBACKEND_H_
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:#include "MD_CUDAMixedBackend.h"
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: * @brief CUDA backend with forward flux sampling capability
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: * This class is derived from the CUDA MD mixed backend with 
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: * on CUDA only.
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: backend = CUDA (For CUDA FFS -- NB unlike the CPU implementation, the CUDA implementation does not print extra columns with the current order parameter values whenever the energy is printed)
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: ffs_file = <string> (path to the file with the simulation stopping conditions. Optionally, one may use 'master conditions' (CUDA FFS only), which allow one to more easily handle very high dimensional order parameters. See the EXAMPLES/CUDA_FFS/README file for more information)
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [ffs_generate_flux = <bool> (CUDA FFS only. Default: False; if False, the simulation will run until a stopping condition is reached; if True, a flux generation simulation will be run, in which case reaching a condition will cause a configuration to be saved but will not terminate the simulation. In the stopping condition file, the conditions must be labelled forward1, forward2, ... (for the forward conditions); and backward1, backward2, ... (for the backward conditions), ... instead of condition1, condition2, ... . To get standard flux generation, set the forward and backward conditions to correspond to crossing the same interface (and use conditions corresponding to different interfaces for Tom's flux generation). As with the single shooting run mode, the name of the condition crossed will be printed to stderr each time.)]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [gen_flux_save_every = <integer> (CUDA FFS only. Mandatory if ffs_generate_flux is True; save a configuration for 1 in every N forward crossings)]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [gen_flux_total_crossings = <integer> (CUDA FFS only. Mandatory if ffs_generate_flux is True; stop the simulation after N crossings achieved)]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [gen_flux_conf_prefix = <string> (CUDA FFS only. Mandatory if ffs_generate_flux is True; the prefix used for the file names of configurations corresponding to the saved forward crossings. Counting starts at zero so the 3rd crossing configuration will be saved as MY_PREFIX_N2.dat)]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [gen_flux_debug = <bool> (CUDA FFS only. Default: False; In a flux generation simulation, set to true to save backward-crossing configurations for debugging)]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [check_initial_state = <bool> (CUDA FFS only. Default: False; in a flux generation simulation, set to true to turn on initial state checking. In this mode an initial configuration that crosses the forward conditions after only 1 step will cause the code to complain and exit. Useful for checking that a flux generation simulation does not start out of the A-state)]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [die_on_unexpected_master = <bool> (CUDA FFS only. Default: False; in a flux generation simulation that uses master conditions, set to true to cause the simulation to die if any master conditions except master_forward1 or master_backward1 are reached. Useful for checking that a flux generation simulation does not enter any unwanted free energy basins (i.e. other than the initial state and the desired final state))]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h: [unexpected_master_prefix = <string> (CUDA FFS only. Mandatory if die_on_unexpected_master is True; the prefix used for the file names of configurations corresponding to reaching any unexpected master conditions (see die_on_unexpected_master).)]
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:class FFS_MD_CUDAMixedBackend: public CUDAMixedBackend {
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	CUDA_kernel_cfg _ffs_hb_precalc_kernel_cfg;
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	CUDA_kernel_cfg _ffs_dist_precalc_kernel_cfg;
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	CUDA_kernel_cfg _ffs_hb_eval_kernel_cfg;
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	CUDA_kernel_cfg _ffs_dist_eval_kernel_cfg;
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	 * @param kernel_cfg pointer to CUDA kernel config
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	void _init_ffs_kernel_config(CUDA_kernel_cfg *kernel_cfg, int total_threads);
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	 * @brief Make sure the required CUDA constants are available to the kernels that run the simulation and check the stopping conditions
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	void _init_CUDA_MD_symbols();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	FFS_MD_CUDAMixedBackend();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:	virtual ~FFS_MD_CUDAMixedBackend();
src/CUDA/Backends/FFS_MD_CUDAMixedBackend.h:#endif /* FFS_MD_CUDAMIXEDBACKEND_H_ */
src/CUDA/CUDAUtils.cu: * GpuUtils.cpp
src/CUDA/CUDAUtils.cu:#include "CUDAUtils.h"
src/CUDA/CUDAUtils.cu:size_t GpuUtils::_allocated_dev_mem = 0;
src/CUDA/CUDAUtils.cu:void GpuUtils::print_device_array(T *v, int N) {
src/CUDA/CUDAUtils.cu:	cudaDeviceSynchronize();
src/CUDA/CUDAUtils.cu:void GpuUtils::check_device_thresold(T *v, int N, int t) {
src/CUDA/CUDAUtils.cu:	cudaDeviceSynchronize();
src/CUDA/CUDAUtils.cu:c_number4 GpuUtils::sum_c_number4_on_GPU(c_number4 *dv, int N) {
src/CUDA/CUDAUtils.cu:double GpuUtils::sum_c_number4_to_double_on_GPU(c_number4 *dv, int N) {
src/CUDA/Lists/CUDABaseList.h: * @file    CUDABaseList.h
src/CUDA/Lists/CUDABaseList.h:#ifndef CUDABASELIST_H_
src/CUDA/Lists/CUDABaseList.h:#define CUDABASELIST_H_
src/CUDA/Lists/CUDABaseList.h:#include "../CUDAUtils.h"
src/CUDA/Lists/CUDABaseList.h:#include "../cuda_utils/CUDABox.h"
src/CUDA/Lists/CUDABaseList.h: * @brief Abstract class for list-based force computing on CUDA
src/CUDA/Lists/CUDABaseList.h:class CUDABaseList {
src/CUDA/Lists/CUDABaseList.h:	CUDABox *_h_cuda_box, *_d_cuda_box;
src/CUDA/Lists/CUDABaseList.h:	CUDABaseList() :
src/CUDA/Lists/CUDABaseList.h:					_h_cuda_box(nullptr),
src/CUDA/Lists/CUDABaseList.h:					_d_cuda_box(nullptr) {
src/CUDA/Lists/CUDABaseList.h:	virtual ~CUDABaseList() {
src/CUDA/Lists/CUDABaseList.h:	virtual void init(int N, c_number rcut, CUDABox *h_cuda_box, CUDABox *d_cuda_box) {
src/CUDA/Lists/CUDABaseList.h:		_h_cuda_box = h_cuda_box;
src/CUDA/Lists/CUDABaseList.h:		_d_cuda_box = d_cuda_box;
src/CUDA/Lists/CUDABaseList.h:#endif /* CUDABASELIST_H_ */
src/CUDA/Lists/CUDAListFactory.cu: * CUDAListFactory.cpp
src/CUDA/Lists/CUDAListFactory.cu:#include "CUDAListFactory.h"
src/CUDA/Lists/CUDAListFactory.cu:#include "CUDANoList.h"
src/CUDA/Lists/CUDAListFactory.cu:#include "CUDASimpleVerletList.h"
src/CUDA/Lists/CUDAListFactory.cu:#include "CUDABinVerletList.h"
src/CUDA/Lists/CUDAListFactory.cu:CUDABaseList* CUDAListFactory::make_list(input_file &inp) {
src/CUDA/Lists/CUDAListFactory.cu:	if(getInputString(&inp, "CUDA_list", list_type, 0) == KEY_NOT_FOUND || !strcmp("verlet", list_type)) {
src/CUDA/Lists/CUDAListFactory.cu:		return new CUDASimpleVerletList();
src/CUDA/Lists/CUDAListFactory.cu:		return new CUDANoList();
src/CUDA/Lists/CUDAListFactory.cu:		return new CUDABinVerletList();
src/CUDA/Lists/CUDAListFactory.cu:		throw oxDNAException("CUDA_list '%s' is not supported", list_type);
src/CUDA/Lists/CUDASimpleVerletList.h: * @file    CUDASimpleVerletList.h
src/CUDA/Lists/CUDASimpleVerletList.h:#ifndef CUDASIMPLEVERLETLIST_H_
src/CUDA/Lists/CUDASimpleVerletList.h:#define CUDASIMPLEVERLETLIST_H_
src/CUDA/Lists/CUDASimpleVerletList.h:#include "CUDABaseList.h"
src/CUDA/Lists/CUDASimpleVerletList.h:#include "../CUDAUtils.h"
src/CUDA/Lists/CUDASimpleVerletList.h: * @brief CUDA implementation of a {@link VerletList Verlet list}.
src/CUDA/Lists/CUDASimpleVerletList.h:class CUDASimpleVerletList: public CUDABaseList {
src/CUDA/Lists/CUDASimpleVerletList.h:	cudaTextureObject_t _counters_cells_tex = 0;
src/CUDA/Lists/CUDASimpleVerletList.h:	CUDA_kernel_cfg _cells_kernel_cfg;
src/CUDA/Lists/CUDASimpleVerletList.h:	CUDASimpleVerletList();
src/CUDA/Lists/CUDASimpleVerletList.h:	virtual ~CUDASimpleVerletList();
src/CUDA/Lists/CUDASimpleVerletList.h:	void init(int N, c_number rcut, CUDABox *h_cuda_box, CUDABox *d_cuda_box);
src/CUDA/Lists/CUDASimpleVerletList.h:#endif /* CUDASIMPLEVERLETLIST_H_ */
src/CUDA/Lists/CUDA_bin_verlet.cuh: * CUDA_verlet.cu
src/CUDA/Lists/CUDA_bin_verlet.cuh:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Lists/CUDABinVerletList.h: * CUDABinVerletList.h
src/CUDA/Lists/CUDABinVerletList.h:#ifndef CUDABINVERLETLIST_H_
src/CUDA/Lists/CUDABinVerletList.h:#define CUDABINVERLETLIST_H_
src/CUDA/Lists/CUDABinVerletList.h:#include "CUDASimpleVerletList.h"
src/CUDA/Lists/CUDABinVerletList.h:class CUDABinVerletList: public CUDASimpleVerletList {
src/CUDA/Lists/CUDABinVerletList.h:	CUDA_kernel_cfg _cells_kernel_cfg;
src/CUDA/Lists/CUDABinVerletList.h:	void _init_CUDA_verlet_symbols();
src/CUDA/Lists/CUDABinVerletList.h:	CUDABinVerletList();
src/CUDA/Lists/CUDABinVerletList.h:	virtual ~CUDABinVerletList();
src/CUDA/Lists/CUDABinVerletList.h:	void init(int N, c_number rcut, CUDABox*h_cuda_box, CUDABox*d_cuda_box);
src/CUDA/Lists/CUDABinVerletList.h:#endif /* CUDABINVERLETLIST_H_ */
src/CUDA/Lists/CUDANoList.cu: * CUDANoList.cpp
src/CUDA/Lists/CUDANoList.cu:#include "CUDANoList.h"
src/CUDA/Lists/CUDANoList.cu:CUDANoList::CUDANoList() {
src/CUDA/Lists/CUDANoList.cu:CUDANoList::~CUDANoList() {
src/CUDA/Lists/CUDANoList.cu:void CUDANoList::get_settings(input_file &inp) {
src/CUDA/Lists/CUDANoList.cu:		throw oxDNAException("'CUDA_list = no' and 'use_edge = true' are incompatible");
src/CUDA/Lists/CUDA_simple_verlet.cuh: * CUDA_verlet.cu
src/CUDA/Lists/CUDA_simple_verlet.cuh:__device__ void update_cell_neigh_list(cudaTextureObject_t counters_cells_tex, c_number4 *poss, int cell_ind, int *cells, c_number4 r, int *neigh, int &N_neigh, LR_bonds b, CUDABox *box) {
src/CUDA/Lists/CUDA_simple_verlet.cuh:__global__ void simple_update_neigh_list(cudaTextureObject_t counters_cells_tex, c_number4 *poss, c_number4 *list_poss, int *cells, int *matrix_neighs, int *c_number_neighs, LR_bonds *bonds, CUDABox *box) {
src/CUDA/Lists/CUDA_simple_verlet.cuh:__global__ void simple_fill_cells(c_number4 *poss, int *cells, int *counters_cells, bool *cell_overflow, CUDABox *box) {
src/CUDA/Lists/CUDA_simple_verlet.cuh:__device__ void edge_update_cell_neigh_list(cudaTextureObject_t counters_cells_tex, c_number4 *poss, int cell_ind, int *cells, c_number4 &r, int *neigh, int &N_n, LR_bonds b, int &N_n_no_doubles, CUDABox *box) {
src/CUDA/Lists/CUDA_simple_verlet.cuh:__global__ void edge_update_neigh_list(cudaTextureObject_t counters_cells_tex, c_number4 *poss, c_number4 *list_poss, int *cells, int *matrix_neighs, int *nn, int *nn_no_doubles, LR_bonds *bonds, CUDABox *box) {
src/CUDA/Lists/CUDASimpleVerletList.cu: * CUDASimpleVerletList.cu
src/CUDA/Lists/CUDASimpleVerletList.cu:#include "CUDASimpleVerletList.h"
src/CUDA/Lists/CUDASimpleVerletList.cu:#include "CUDA_simple_verlet.cuh"
src/CUDA/Lists/CUDASimpleVerletList.cu:#include "../cuda_utils/CUDA_lr_common.cuh"
src/CUDA/Lists/CUDASimpleVerletList.cu:CUDASimpleVerletList::CUDASimpleVerletList() {
src/CUDA/Lists/CUDASimpleVerletList.cu:CUDASimpleVerletList::~CUDASimpleVerletList() {
src/CUDA/Lists/CUDASimpleVerletList.cu:void CUDASimpleVerletList::clean() {
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(_d_cells));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(_d_counters_cells));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(d_matrix_neighs));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(d_number_neighs));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFreeHost(_d_cell_overflow));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(d_edge_list));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(_d_number_neighs_no_doubles));
src/CUDA/Lists/CUDASimpleVerletList.cu:void CUDASimpleVerletList::get_settings(input_file &inp) {
src/CUDA/Lists/CUDASimpleVerletList.cu:__global__ void count_N_in_cells(c_number4 *poss, uint *counters_cells, int N_cells_side[3], int N, CUDABox box) {
src/CUDA/Lists/CUDASimpleVerletList.cu:int CUDASimpleVerletList::_largest_N_in_cells(c_number4 *poss, c_number min_cell_size) {
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaMallocHost(&N_cells_side, sizeof(int) * 3, cudaHostAllocDefault));
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaMalloc(&counters_cells, (size_t ) N_cells * sizeof(uint)));
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaMemset(counters_cells, 0, N_cells * sizeof(uint)));
src/CUDA/Lists/CUDASimpleVerletList.cu:		(poss, counters_cells, N_cells_side, N, *_h_cuda_box);
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaFreeHost(N_cells_side));
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaFree(counters_cells));
src/CUDA/Lists/CUDASimpleVerletList.cu:void CUDASimpleVerletList::_compute_N_cells_side(int N_cells_side[3], number min_cell_size) {
src/CUDA/Lists/CUDASimpleVerletList.cu:	c_number4 box_sides_n4 = _h_cuda_box->box_sides();
src/CUDA/Lists/CUDASimpleVerletList.cu:	c_number max_factor = pow(2. * _N / _h_cuda_box->V(), 1. / 3.);
src/CUDA/Lists/CUDASimpleVerletList.cu:void CUDASimpleVerletList::_init_cells(c_number4 *poss) {
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(_d_cells));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaFree(_d_counters_cells));
src/CUDA/Lists/CUDASimpleVerletList.cu:		cudaDestroyTextureObject(_counters_cells_tex);
src/CUDA/Lists/CUDASimpleVerletList.cu:		OX_DEBUG("Re-allocating cells on GPU, from %d to %d\n", _old_N_cells, _N_cells);
src/CUDA/Lists/CUDASimpleVerletList.cu:			CUDA_SAFE_CALL(cudaMalloc(&poss, (size_t ) N * sizeof(c_number4)));
src/CUDA/Lists/CUDASimpleVerletList.cu:			CUDA_SAFE_CALL(cudaMemcpy(poss, host_positions.data(), sizeof(c_number4) * N, cudaMemcpyHostToDevice));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_counters_cells, (size_t ) _N_cells * sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_cells, (size_t ) _N_cells * _max_N_per_cell * sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(verlet_N_cells_side, _N_cells_side, 3 * sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(cudaMemcpyToSymbol(verlet_max_N_per_cell, &_max_N_per_cell, sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:			CUDA_SAFE_CALL(cudaFree(poss));
src/CUDA/Lists/CUDASimpleVerletList.cu:			cudaDestroyTextureObject(_counters_cells_tex);
src/CUDA/Lists/CUDASimpleVerletList.cu:		GpuUtils::init_texture_object(&_counters_cells_tex, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned), _d_counters_cells, _N_cells);
src/CUDA/Lists/CUDASimpleVerletList.cu:void CUDASimpleVerletList::init(int N, c_number rcut, CUDABox *h_cuda_box, CUDABox *d_cuda_box) {
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDABaseList::init(N, rcut, h_cuda_box, d_cuda_box);
src/CUDA/Lists/CUDASimpleVerletList.cu:	OX_LOG(Logger::LOG_INFO, "CUDA Cells mem: %.2lf MBs, lists mem: %.2lf MBs", (double) _N_cells*(1 + _max_N_per_cell) * sizeof(int)/1048576., (double) _N * (1 + _max_neigh) * sizeof(int)/1048576.);
src/CUDA/Lists/CUDASimpleVerletList.cu:	OX_LOG(Logger::LOG_INFO, "CUDA max_neigh: %d, max_N_per_cell: %d, N_cells: %d (per side: %d %d %d)", _max_neigh, _max_N_per_cell, _N_cells, _N_cells_side[0], _N_cells_side[1], _N_cells_side[2]);
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&d_number_neighs, (size_t ) _N * sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&d_matrix_neighs, (size_t ) _N * _max_neigh * sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaMallocHost(&_d_cell_overflow, sizeof(bool), cudaHostAllocDefault));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&d_edge_list, (size_t ) _N * _max_neigh * sizeof(edge_bond)));
src/CUDA/Lists/CUDASimpleVerletList.cu:		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc(&_d_number_neighs_no_doubles, (size_t ) (_N + 1) * sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(verlet_sqr_rverlet, &f_copy, sizeof(float)));
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(verlet_N, &_N, sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:std::vector<int> CUDASimpleVerletList::is_large(c_number4 *data) {
src/CUDA/Lists/CUDASimpleVerletList.cu:void CUDASimpleVerletList::update(c_number4 *poss, c_number4 *list_poss, LR_bonds *bonds) {
src/CUDA/Lists/CUDASimpleVerletList.cu:	CUDA_SAFE_CALL(cudaMemset(_d_counters_cells, 0, _N_cells * sizeof(int)));
src/CUDA/Lists/CUDASimpleVerletList.cu:		(poss, _d_cells, _d_counters_cells, _d_cell_overflow, _d_cuda_box);
src/CUDA/Lists/CUDASimpleVerletList.cu:	cudaDeviceSynchronize();
src/CUDA/Lists/CUDASimpleVerletList.cu:			(_counters_cells_tex, poss, list_poss, _d_cells, d_matrix_neighs, d_number_neighs, _d_number_neighs_no_doubles, bonds, _d_cuda_box);
src/CUDA/Lists/CUDASimpleVerletList.cu:		// thrust operates on the GPU
src/CUDA/Lists/CUDASimpleVerletList.cu:			(_counters_cells_tex, poss, list_poss, _d_cells, d_matrix_neighs, d_number_neighs, bonds, _d_cuda_box);
src/CUDA/Lists/CUDANoList.h: * @file    CUDANoList.h
src/CUDA/Lists/CUDANoList.h:#ifndef CUDANOLIST_H_
src/CUDA/Lists/CUDANoList.h:#define CUDANOLIST_H_
src/CUDA/Lists/CUDANoList.h:#include "CUDABaseList.h"
src/CUDA/Lists/CUDANoList.h: * @brief Implements a O(N^2) type of simulation (each particle interact with each other) with CUDA.
src/CUDA/Lists/CUDANoList.h:class CUDANoList: public CUDABaseList{
src/CUDA/Lists/CUDANoList.h:	CUDANoList();
src/CUDA/Lists/CUDANoList.h:	virtual ~CUDANoList();
src/CUDA/Lists/CUDANoList.h:#endif /* CUDANOLIST_H_ */
src/CUDA/Lists/CUDABinVerletList.cu: * CUDABinVerletList.cu
src/CUDA/Lists/CUDABinVerletList.cu:#include "CUDABinVerletList.h"
src/CUDA/Lists/CUDABinVerletList.cu:#include "CUDA_bin_verlet.cuh"
src/CUDA/Lists/CUDABinVerletList.cu:CUDABinVerletList::CUDABinVerletList() :
src/CUDA/Lists/CUDABinVerletList.cu:CUDABinVerletList::~CUDABinVerletList() {
src/CUDA/Lists/CUDABinVerletList.cu:void CUDABinVerletList::_init_CUDA_verlet_symbols() {
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bverlet_box_side, &f_copy, sizeof(float)));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bverlet_N, &this->_N, sizeof(int)));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bverlet_N_cells_side, this->_N_cells_side, 3 * sizeof(int)));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bverlet_max_N_per_cell, this->_max_N_per_cell, 3 * sizeof(int)));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bverlet_cells_offset, this->_cells_offset, 3 * sizeof(int)));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bverlet_counters_offset, this->_counters_offset, 3 * sizeof(int)));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(bverlet_AO_mixture, &_AO_mixture, sizeof(bool)));
src/CUDA/Lists/CUDABinVerletList.cu:void CUDABinVerletList::get_settings(input_file &inp) {
src/CUDA/Lists/CUDABinVerletList.cu:	CUDASimpleVerletList::get_settings(inp);
src/CUDA/Lists/CUDABinVerletList.cu:void CUDABinVerletList::init(int N, c_number rcut, CUDABox*h_cuda_box, CUDABox*d_cuda_box) {
src/CUDA/Lists/CUDABinVerletList.cu:	CUDABaseList::init(N, rcut, h_cuda_box, d_cuda_box);
src/CUDA/Lists/CUDABinVerletList.cu:	c_number4 box_sides = h_cuda_box->box_sides();
src/CUDA/Lists/CUDABinVerletList.cu:	if(box_sides.x != box_sides.y || box_sides.y != box_sides.z) throw oxDNAException("CUDA_list = bin_verlet can work only with cubic boxes");
src/CUDA/Lists/CUDABinVerletList.cu:	c_number density = N / h_cuda_box->V();
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&this->d_number_neighs, c_number_mem));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&this->d_matrix_neighs, matrix_mem));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&this->_d_counters_cells, _counters_mem));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<int>(&this->_d_cells, cells_mem));
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMallocHost(&this->_d_cell_overflow, sizeof(bool), cudaHostAllocDefault));
src/CUDA/Lists/CUDABinVerletList.cu:	_init_CUDA_verlet_symbols();
src/CUDA/Lists/CUDABinVerletList.cu:void CUDABinVerletList::update(c_number4 *poss, c_number4 *list_poss, LR_bonds *bonds) {
src/CUDA/Lists/CUDABinVerletList.cu:	CUDA_SAFE_CALL(cudaMemset(this->_d_counters_cells, 0, _counters_mem));
src/CUDA/Lists/CUDABinVerletList.cu:	cudaDeviceSynchronize();
src/CUDA/Lists/CUDABinVerletList.cu:void CUDABinVerletList::clean() {
src/CUDA/Lists/CUDAListFactory.h: * CUDAListFactory.h
src/CUDA/Lists/CUDAListFactory.h:#ifndef CUDALISTFACTORY_H_
src/CUDA/Lists/CUDAListFactory.h:#define CUDALISTFACTORY_H_
src/CUDA/Lists/CUDAListFactory.h:#include "CUDABaseList.h"
src/CUDA/Lists/CUDAListFactory.h: * @brief Static factory class. Its only public method builds a {@link CUDABaseList CUDA list}.
src/CUDA/Lists/CUDAListFactory.h:[CUDA_list = no|verlet (Neighbour lists for CUDA simulations. Defaults to 'no'.)]
src/CUDA/Lists/CUDAListFactory.h:class CUDAListFactory {
src/CUDA/Lists/CUDAListFactory.h:	CUDAListFactory() = delete;
src/CUDA/Lists/CUDAListFactory.h:	virtual ~CUDAListFactory() = delete;
src/CUDA/Lists/CUDAListFactory.h:	static CUDABaseList *make_list(input_file &inp);
src/CUDA/Lists/CUDAListFactory.h:#endif /* CUDALISTFACTORY_H_ */
src/CUDA/CUDA_sort.cu: * Cuda++	_sort.cu
src/CUDA/CUDA_sort.cu:#include "CUDA_sort.cuh"
src/CUDA/CUDA_sort.cu:__global__ void permute_particles(int *sorted_hindex, int *inv_sorted_hindex, c_number4 *poss, c_number4 *vels, c_number4 *Ls, GPU_quat *orientations, LR_bonds *bonds, int *particles_to_mols, c_number4 *buff_poss, c_number4 *buff_vels, c_number4 *buff_Ls, GPU_quat *buff_orientations, LR_bonds *buff_bonds, int *buff_particles_to_mols) {
src/CUDA/CUDA_sort.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(hilb_N, &N, sizeof(int)));
src/CUDA/CUDA_sort.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(hilb_depth, &depth, sizeof(int)));
src/CUDA/CUDA_sort.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(hilb_box_side, &box_side, sizeof(float)));
src/CUDA/CUDA_sort.cu:	CUDA_SAFE_CALL(cudaMemcpyToSymbol(hilb_N_unsortable, &N_unsortable, sizeof(int)));
src/CMakeLists.txt:IF(CUDA)
src/CMakeLists.txt:	FIND_PACKAGE("CUDA" REQUIRED)
src/CMakeLists.txt:	INCLUDE_DIRECTORIES(${CUDA_INSTALL_PREFIX}/include/)
src/CMakeLists.txt:	LINK_DIRECTORIES(${CUDA_INSTALL_PREFIX}/lib)
src/CMakeLists.txt:	IF(CUDA_COMMON_ARCH)
src/CMakeLists.txt:		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Common)
src/CMakeLists.txt:	ELSE(CUDA_COMMON_ARCH)
src/CMakeLists.txt:		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
src/CMakeLists.txt:	ENDIF(CUDA_COMMON_ARCH)
src/CMakeLists.txt:	MESSAGE(STATUS "Generating code for the following CUDA architectures: ${ARCH_FLAGS_readable}")
src/CMakeLists.txt:	LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
src/CMakeLists.txt:	MESSAGE(STATUS "${CUDA_NVCC_FLAGS}")
src/CMakeLists.txt:		SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo)
src/CMakeLists.txt:		SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo -G)
src/CMakeLists.txt:	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -use_fast_math --compiler-options "-ffast-math -O3" -Xcudafe "--diag_suppress=unrecognized_gcc_pragma")
src/CMakeLists.txt:		CUDA/CUDA_sort.cu
src/CMakeLists.txt:		CUDA/CUDAUtils.cu
src/CMakeLists.txt:		CUDA/cuda_utils/cuda_device_utils.cu
src/CMakeLists.txt:		CUDA/Backends/CUDABaseBackend.cu
src/CMakeLists.txt:		CUDA/Backends/MD_CUDABackend.cu
src/CMakeLists.txt:		CUDA/Interactions/CUDABaseInteraction.cu
src/CMakeLists.txt:		CUDA/Interactions/CUDADNAInteraction.cu
src/CMakeLists.txt:		CUDA/Interactions/CUDALJInteraction.cu
src/CMakeLists.txt:		CUDA/Interactions/CUDARNAInteraction.cu
src/CMakeLists.txt:		CUDA/Interactions/CUDAPatchyInteraction.cu
src/CMakeLists.txt:		CUDA/Interactions/CUDATEPInteraction.cu
src/CMakeLists.txt:		CUDA/Interactions/CUDAInteractionFactory.cu
src/CMakeLists.txt:		CUDA/Lists/CUDAListFactory.cu
src/CMakeLists.txt:		CUDA/Lists/CUDANoList.cu
src/CMakeLists.txt:		CUDA/Lists/CUDASimpleVerletList.cu
src/CMakeLists.txt:		CUDA/Lists/CUDABinVerletList.cu
src/CMakeLists.txt:		CUDA/Thermostats/CUDABaseThermostat.cu
src/CMakeLists.txt:		CUDA/Thermostats/CUDANoThermostat.cu
src/CMakeLists.txt:		CUDA/Thermostats/CUDAThermostatFactory.cu
src/CMakeLists.txt:		CUDA/Thermostats/CUDABrownianThermostat.cu
src/CMakeLists.txt:		CUDA/Thermostats/CUDASRDThermostat.cu
src/CMakeLists.txt:		CUDA/Thermostats/CUDALangevinThermostat.cu
src/CMakeLists.txt:		CUDA/Thermostats/CUDABussiThermostat.cu
src/CMakeLists.txt:	IF(CUDA_DOUBLE)
src/CMakeLists.txt:		ADD_DEFINITIONS(-DCUDA_DOUBLE_PRECISION)
src/CMakeLists.txt:			CUDA/Backends/MD_CUDAMixedBackend.cu 
src/CMakeLists.txt:			CUDA/Backends/FFS_MD_CUDAMixedBackend.cu
src/CMakeLists.txt:	CUDA_ADD_LIBRARY(${lib_name} SHARED ${common_SOURCES})
src/CMakeLists.txt:	CUDA_ADD_EXECUTABLE(${exe_name} ${oxDNA_SOURCES})
src/CMakeLists.txt:	CUDA_ADD_EXECUTABLE(DNAnalysis ${DNAnalysis_SOURCES} Utilities/Timings.cpp)
src/CMakeLists.txt:	ADD_DEFINITIONS(-DNOCUDA)
src/CMakeLists.txt:ENDIF(CUDA)
src/Forces/BaseForce.h: * We have to change scope policy due to the fact that GPU classes
src/Forces/BaseForce.h:	 * to the GPU memory
src/Backends/BackendFactory.h:backend = CPU|CUDA (simulation backend. Defaults to CPU)
src/Backends/BackendFactory.h:[backend_precision = float|double|mixed (Precision at which calculateions are carried out. The mixed precision is only available on CUDA. Defaults to double.)]
src/Backends/BackendFactory.h:[sim_type = MD|MC|VMMC|FFS_MD (Type of the simulation. Supported types are Molecular Dynamics, Monte Carlo, Virtual Move Monte Carlo and Forward Flux Sampling. The first and last ones are also available on CUDA. Defaults to MD.)]
src/Backends/Thermostats/SRDThermostat.cpp:	_is_cuda = false;
src/Backends/Thermostats/SRDThermostat.cpp:	if(!_is_cuda) {
src/Backends/Thermostats/SRDThermostat.cpp:	_is_cuda = (strcmp(backend, "CUDA") == 0);
src/Backends/Thermostats/SRDThermostat.cpp:	if(!_is_cuda && _cells == nullptr) {
src/Backends/Thermostats/SRDThermostat.cpp:	if(_is_cuda) throw oxDNAException("The apply method of the SRD thermostat has been called on the CPU on a CUDA-enabled simulation. This should not happen.");
src/Backends/Thermostats/SRDThermostat.h:	bool _is_cuda;
src/Backends/Thermostats/ThermostatFactory.h:[thermostat = no|refresh|brownian|langevin|srd|bussi (Select the simulation thermostat for MD simulations. 'no' means constant-energy simulations. 'refresh' is the Andersen thermostat. 'brownian' is an Anderson-like thermostat that refreshes momenta of randomly chosen particles. 'langevin' implements a regular Langevin thermostat. 'srd' is an (experimental) implementation of a stochastic rotational dynamics algorithm, 'bussi' is the Bussi-Donadio-Parrinello thermostat. 'no', 'brownian' and 'bussi' are also available on CUDA. Defaults to 'no'.)]
src/Backends/BackendFactory.cpp:#ifndef NOCUDA
src/Backends/BackendFactory.cpp:#include "../CUDA/Backends/MD_CUDABackend.h"
src/Backends/BackendFactory.cpp:#ifndef CUDA_DOUBLE_PRECISION
src/Backends/BackendFactory.cpp:#include "../CUDA/Backends/MD_CUDAMixedBackend.h"
src/Backends/BackendFactory.cpp:#include "../CUDA/Backends/FFS_MD_CUDAMixedBackend.h"
src/Backends/BackendFactory.cpp:#ifndef NOCUDA
src/Backends/BackendFactory.cpp:		else if(backend_opt == "CUDA") {
src/Backends/BackendFactory.cpp:#ifndef CUDA_DOUBLE_PRECISION
src/Backends/BackendFactory.cpp:				new_backend = new CUDAMixedBackend();
src/Backends/BackendFactory.cpp:				new_backend = new MD_CUDABackend();
src/Backends/BackendFactory.cpp:				new_backend = new MD_CUDABackend();
src/Backends/BackendFactory.cpp:			OX_LOG(Logger::LOG_INFO, "CUDA backend precision: %s", backend_prec.c_str());
src/Backends/BackendFactory.cpp:#ifndef NOCUDA
src/Backends/BackendFactory.cpp:#ifndef CUDA_DOUBLE_PRECISION
src/Backends/BackendFactory.cpp:		else if(backend_opt == "CUDA") {
src/Backends/BackendFactory.cpp:				new_backend = new FFS_MD_CUDAMixedBackend();
src/Backends/BackendFactory.cpp:				throw oxDNAException("Backend precision '%s' for FFS simulations with CUDA is not supported", backend_prec.c_str());

```
