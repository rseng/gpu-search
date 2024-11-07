# https://github.com/JLBLine/WODEN

```console
setup.py:    description = 'GPU-accelerated code to simulate radio-frequency interferometric observations',
docs/sphinx/installation/installation.rst:``WODEN`` is built for speed and only works with a GPU. Currently, you need either an NVIDIA GPU to use ``CUDA`` functionality, or something else that can use ``HIP`` (likely an AMD GPU). ``CUDA`` is tried and tested, whereas ``HIP`` is new in version 2.2 and not well tested. Furthermore, ``WODEN`` has only been tested to run on linux, specifically Ubuntu 16.04 up to 24.04. This does however include the _`Windows Subsystem for Linux 2 (WSL 2)`., so you can technically run in on Windows kinda.
docs/sphinx/installation/installation.rst:- Garrawarla (Pawsey) CUDA
docs/sphinx/installation/installation.rst:- OzStar (Swinburne University) CUDA
docs/sphinx/installation/installation.rst:- Ngarrgu Tindebeek (Swinburne University) CUDA
docs/sphinx/installation/installation.rst:- ``WODEN/templates/install_woden_nt.sh`` for a CUDA build on Ngarrgu Tindebeek
docs/sphinx/installation/installation.rst:- Either **NVIDIA CUDA** - https://developer.nvidia.com/cuda-downloads
docs/sphinx/installation/installation.rst:- or **AMD ROCm** - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
docs/sphinx/installation/installation.rst:+ **NVIDIA CUDA** - https://developer.nvidia.com/cuda-downloads. Best used if you have an NVIDIA GPU. I typically download the runfile option, which you run as::
docs/sphinx/installation/installation.rst:  $ sudo sh cuda_11.2.2_460.32.03_linux.run ##your version will likely be different
docs/sphinx/installation/installation.rst:  but I do NOT install the drivers at this point, as I'll already have drivers. Up to you and how your system works. Also, don't ignore the step of adding something like ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64`` to your ``~/.bashrc``, or your system won't find ``CUDA``.
docs/sphinx/installation/installation.rst:+ **AMD ROCm** - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/::
docs/sphinx/installation/installation.rst:  I don't have an AMD GPU, so I've never done this. Fingers crossed the linked instructions work for you!
docs/sphinx/installation/installation.rst:+ **mwa_hyperbeam** - https://github.com/MWATelescope/mwa_hyperbeam - ``mwa_hyperbeam`` is the go-to package for calculating the MWA Fully Embedded Element (FEE) primary beam model. At the time of writing (23/03/2022), we'll have to install and compile from source to get the CUDA code that we want to link to. We should be able to install release versions in the future. For now, you'll first need to install ``rust``, the language the library is written in. I followed the installation guide at https://www.rust-lang.org/tools/install, which for me on Ubuntu just means running::
docs/sphinx/installation/installation.rst:  Once that's installed, I run the following commands for a CUDA installation (you can choose where to install it, I'm just putting where I happened to do it this time round)::
docs/sphinx/installation/installation.rst:  $ export HYPERDRIVE_CUDA_COMPUTE=60 ##your compute capability
docs/sphinx/installation/installation.rst:  $ cargo build --locked --release --features=cuda,hdf5-static
docs/sphinx/installation/installation.rst:  .. note:: ``export HYPERDRIVE_CUDA_COMPUTE=60`` is not essential as the compiler should be smart enough, but you *might* get a speed boost but setting the correct architecture. This of course depends on your GPU; see 'Machine specifics' below on how to work out your architecture.
docs/sphinx/installation/installation.rst:  If you have an AMD GPU, replace the last two lines with something like::
docs/sphinx/installation/installation.rst:  where again the value of ``HYPERBEAM_HIP_ARCH`` depends on what kind of GPU you have.
docs/sphinx/installation/installation.rst:Compiling ``WODEN`` ``C/CUDA`` code
docs/sphinx/installation/installation.rst:you have a newer NVIDIA GPU, you should be able to simply run::
docs/sphinx/installation/installation.rst:.. warning:: Even if the code compiled, if your GPU has a compute capability < 5.1, newer versions of ``nvcc`` won't compile code that will work. You'll get error messages like "No kernel image available". Check out how to fix that in 'Machine specifics' below.
docs/sphinx/installation/installation.rst:All NVIDIA GPUs have a specific compute capability, which relates to their internal architecture. You can tell the compiler which architecture to compile for, which in theory should make compilation quicker, and ensure the code runs correctly on your GPU. You can find out the compute value here (https://developer.nvidia.com/cuda-gpus), and pass it to CMake via setting the ``CUDAARCHS`` environment variable (https://cmake.org/cmake/help/latest/envvar/CUDAARCHS.html) BEFORE you run the call to ``cmake``::
docs/sphinx/installation/installation.rst:  $ export CUDAARCHS=60
docs/sphinx/installation/installation.rst:.. warning:: For newer ``CUDA`` versions, some compute capabilities are deprecated, so the compiler leaves them out by default. For example, using ``CUDA`` version 11.2, compute capabilities 3.5 to 5.0 are ignored. If you card has a compute capability of 5.0, you **must** include the flag ``-DCUDA_ARCH=5.0``, otherwise the `nvcc` compiler will not create an executable capable of running on your device.
docs/sphinx/installation/installation.rst:If you need to pass extra flags to your CUDA compiler, you can do so by adding something like the following (noting that all CMake flags start with ``-D``)::
docs/sphinx/installation/installation.rst:  -DCMAKE_CUDA_FLAGS="-Dsomeflag"
docs/sphinx/installation/installation.rst:If you have an AMD GPU, you can compile the ``HIP`` code instead of the ``CUDA`` code. This is a new feature in ``WODEN`` and not as well tested. You can compile the ``HIP`` code by setting the ``USE_HIP`` flag to ``ON`` when you run ``cmake`` (you'll still need to link )::
docs/sphinx/installation/installation.rst:Similarly to ``CUDA``, you can set a ``HIP`` architecture. To find out which one you need, try::
docs/sphinx/installation/installation.rst:OK, we've compiled the C/GPU libraries; now to install the ``WODEN`` Python package and executables. You can do this by running::
docs/sphinx/installation/installation.rst:For CUDA
docs/sphinx/installation/installation.rst:Fair warning, this is a new option, and hasn't been heavily tested. I have successfully run it on a number of clusters (via singularity). Which version you pull depends on your GPU. If you have an NVIDIA GPU, you need to work out what your compute capability is, and pull the appropriate image. Say you have an NVIDIA V100 card, you have a compute capacity of 7.0, so you'd pull the image like this::
docs/sphinx/installation/installation.rst:  $ docker pull jlbline/woden-2.3:cuda-70
docs/sphinx/installation/installation.rst:  $ docker run -it --gpus all woden-2.3:cuda-70 \
docs/sphinx/installation/installation.rst:where the ``--gpus all`` means the docker instance can see your GPUs. The environment variables point to somewhere to keep your ``astropy`` outputs, which is useful if you're running somewhere you're not admin (like on a cluster). There must be a better way to do this but I'm a ``docker`` noob.
docs/sphinx/installation/installation.rst:The only HIP image I've made is for the Setonix cluster, and is based on a Pawsey specific image https://quay.io/repository/pawsey/rocm-mpich-base?tab=tags&tag=latest. You can pull it like this::
docs/sphinx/installation/installation.rst:For CUDA
docs/sphinx/installation/installation.rst:  $ singularity build woden-2.3-70.sif docker://jlbline/woden-2.3:cuda-70
docs/sphinx/installation/installation.rst:Similarly to the ``docker`` image, ``--nv`` means use the NVIDIA GPUs, and ``--home`` sets a specific location to treat as home if you're not on a local machine.
docs/sphinx/installation/installation.rst:.. warning:: EVERYTHING on the internet will tell you to use the ``--rocm`` flag. This WILL NOT WORK with the Setonix based image, because of shenanigans. So leave it be.
docs/sphinx/API_reference/python_code/wodenpy/use_libwoden.rst:Ways to interact with the C/GPU via ``ctypes``.
docs/sphinx/API_reference/API_index.rst:``GPU`` code
docs/sphinx/API_reference/API_index.rst:All GPU code is either compiled as ``CUDA`` or ``HIP`` code. This is decided at
docs/sphinx/API_reference/API_index.rst:compilation, depending on whether ``-D__NVCC__`` (for ``CUDA``) or ``-D__HIPCC__``
docs/sphinx/API_reference/API_index.rst:(for ``HIP``) was passed to the compiler. Various macros found in ``WODEN/include/gpu_macros.h``
docs/sphinx/API_reference/API_index.rst:are used to drop in the correct GPU calls depending on the language requested.
docs/sphinx/API_reference/API_index.rst:``WODEN/include/gpucomplex.h``:
docs/sphinx/API_reference/API_index.rst:  GPU_code/calculate_visibilities
docs/sphinx/API_reference/API_index.rst:  GPU_code/gpu_macros
docs/sphinx/API_reference/API_index.rst:  GPU_code/gpucomplex
docs/sphinx/API_reference/API_index.rst:  GPU_code/fundamental_coords
docs/sphinx/API_reference/API_index.rst:  GPU_code/primary_beam_gpu
docs/sphinx/API_reference/API_index.rst:  GPU_code/source_components
docs/sphinx/API_reference/GPU_code/primary_beam_gpu.rst:``primary_beam_gpu``
docs/sphinx/API_reference/GPU_code/primary_beam_gpu.rst:API documentation for ``primary_beam_gpu.cpp``.
docs/sphinx/API_reference/GPU_code/primary_beam_gpu.rst:.. doxygenfile:: primary_beam_gpu.h
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:``gpu_macros``
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:API documentation for ``gpu_macros.h``. 
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:.. doxygenfile:: gpu_macros.h
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:See below for a table of the macros employed in ``gpucomplex.h``. Note that many
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:of the GPU functions are wrapped in the ``GPUErrorCheck`` function (documented 
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:above as ``docGPUErrorCheck``), which checks for errors and exits with a message
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:``-D__HIPCC__`` at compilations determines whether ``CUDA`` or ``HIP`` functions
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:.. list-table:: GPU function macros
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMalloc
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMalloc
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuHostAlloc
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaHostAlloc
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuHostAllocDefault
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaHostAllocDefault
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMemcpy
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMemcpy
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMemcpyAsync
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMemcpyAsync
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMemset
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMemset
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuDeviceSynchronize
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaDeviceSynchronize
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMemcpyDeviceToHost
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMemcpyDeviceToHost
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMemcpyHostToDevice
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMemcpyHostToDevice
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMemcpyDeviceToDevice
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMemcpyDeviceToDevice
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuFree
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaFree
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuHostFree
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaFreeHost
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuStream_t
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaStream_t
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuStreamCreate
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaStreamCreate
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuStreamDestroy
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaStreamDestroy
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuEventCreate
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaEventCreate
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuGetDeviceCount
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaGetDeviceCount
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuGetLastError
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaGetLastError
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMemGetInfo
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMemGetInfo
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuMallocHost
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaMallocHost
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuFreeHost
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaFreeHost
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuGetDeviceProperties
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaGetDeviceProperties
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuDeviceProp
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaDeviceProp
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuPeekAtLastError
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:     - cudaPeekAtLastError
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCreal
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCrealf
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCimag
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCimagf
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCadd
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCmul
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCdiv
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuConj
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCsub
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCabs
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCaddf
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCsubf
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCmulf
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuCdivf
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuDoubleComplex
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - gpuFloatComplex
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - make_gpuDoubleComplex
docs/sphinx/API_reference/GPU_code/gpu_macros.rst:   * - make_gpuFloatComplex
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:``gpucomplex``
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:This header is in the ``RTS``, and contains useful CUDA operators. All credit to
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:as my `gpuUserComplex`` def, which allows ``float`` or ``double`` to be
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:though in the below it says ``typedef gpuFloatComplex gupuUserComplex``, this
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:depends on compilation). This code has also been updated with GPU macros, so
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:it can be used for both CUDA and HIP code in conjunction with the
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:``gpu_macros.h`` header.
docs/sphinx/API_reference/GPU_code/gpucomplex.rst:.. doxygenfile:: gpucomplex.h
docs/sphinx/index.rst:  Before ``WODEN`` version 1.4.0, in the output `uvfits` files, the first polarisation (usually called XX) was derived from North-South dipoles, as is the labelling convention according to the IAU. However, most `uvfits` users I've met, as well as the data out of the MWA telescope, define XX as East-West. So although the internal labelling and mathematics within the C/CUDA code is to IAU spec, by default, ``run_woden.py`` now writes out XX as East-West and YY as North-South. From version 1.4.0, a header value of ``IAUORDER = F`` will appear, with ``F`` meaning IAU ordering is False, so the polarisations go EW-EW, NS-NS, EW-NS, NS-EW. If ``IAUORDER = T``, the order is NS-NS, EW-EW, NS-EW, EW-NS. If there is no ``IAUORDER`` at all, assume ``IAUORDER = T``.
docs/sphinx/index.rst:``WODEN`` is C / CUDA code designed to be able to simulate low-frequency radio interferometric data. It is written to be simplistic and *fast* to allow all-sky simulations. Although ``WODEN`` was primarily written to simulate Murchinson Widefield Array (MWA, `Tingay et al. 2013`_) visibilities, it is becoming less instrument-specific as time goes on. `WODEN` outputs `uvfits` files.
docs/sphinx/code_graphs/code_graphs.rst:Ok, here is an attempt to map out the structure of ``WODEN``. These call graphs split into two sets: one for the Python side, the other for the C/GPU side. They are generated using the ``WODEN/docs/sphinx/code_graphs/run_make_graph.sh``, via the ``pyan`` Python module, and ``Doxygen``.
docs/sphinx/code_graphs/code_graphs.rst:C/GPU Call Graphs
docs/sphinx/code_graphs/code_graphs.rst:Eventually, ``run_woden.py`` calls the C/GPU function ``calculate_visibilities``. This is the call graph for ``calculate_visibilities``; note that ``dot`` has truncated some boxes (which are rendered red), as it has a maximum width. Scroll further for a second graph that starts the function ``source_component_common``, which includes the missing calls.
docs/sphinx/testing/cmake_testing.rst:which will pull in the relevant ``CMake-codecov`` dependencies. This allows us to track code coverage for the ``python`` and ``C`` code (no free tools exist for ``CUDA`` at the time of writing, boooo).
docs/sphinx/testing/cmake_testing.rst:for a different file from ``WODEN/src``. Within each test directory, there are separate files for testing different functions, which include the function name. As an example, the directory ``WODEN/cmake_testing/GPU_code/fundamental_coords`` contains tests for the file ``WODEN/src/fundamental_coords.cpp``, and contains test files that test the following functions::
docs/sphinx/testing/cmake_testing.rst:  cmake_testing/GPU_code/fundamental_coords/test_lmn_coords.c -> src/fundamental_coords.cpp::kern_calc_lmn
docs/sphinx/testing/cmake_testing.rst:  cmake_testing/GPU_code/fundamental_coords/test_uvw_coords.c -> src/fundamental_coords.cpp::kern_calc_uvw
docs/sphinx/testing/cmake_testing.rst:The ``C`` and ``GPU`` functions are tested using the `Unity`_ library, which has useful functions like::
docs/sphinx/testing/cmake_testing.rst:``GPU`` code tests:
docs/sphinx/testing/cmake_testing.rst:   cmake_testing/primary_beam_gpu
docs/sphinx/testing/cmake_testing.rst:.. note:: To be able to test ``GPU`` functions that are designed to work solely in GPU memory, it's necessary to write wrapper functions that allocate GPU memory, pass the data into the ``GPU`` code to be tested, and then copy the results back into host memory. I've kept these 'intermediate' test functions inside the ``*.cpp`` files that contain the code being tested, as it's not straight forward / performance degrading to have them in separate files. On casual inspection it looks like there are many functions in the ``*.cpp`` files I haven't written tests for, but the extra functions are there *because* of testing. Sigh.
docs/sphinx/testing/testing.rst:There are two ways to test WODEN. The first is to use the CMake testing suite which is more for developing ``WODEN``, and contains unit/integration tests for ``Python``, ``C``, and ``GPU`` code. The second is via the ``test_installation`` directory which contains a number of commands that can be used to test the functionality of the local installation of ``WODEN``. See below for details.
docs/sphinx/testing/script_testing.rst:.. note:: I've tried to make these tests computationally low, so they need < 1.5 GB of GPU RAM, and < 8 GB system RAM. To do this, I've had to set ``--precision=float`` for some of the tests, to keep the memory requirements down. Running all the simulations will need about 600 MB of storage, with the imaging adding a further 800 MB (for a total of < 1.5 GB storage). The simulations should take < 2 minutes on most GPUs, and imaging less that 10 minutes for most CPUs (far less for fancier CPUs). I ran these tests fine on my laptop which has an Intel i7 2.8 GHz CPU, 16 GB system RAM, and an NVIDIA GeForce 940MX card with 2 GB RAM.
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:``primary_beam_gpu``
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:Tests for the functions in ``WODEN/src/primary_beam_gpu.cu``. These functions
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:This calls ``primary_beam_gpu::test_kern_gaussian_beam``, which in turn
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:tests ``primary_beam_gpu::kern_gaussian_beam``, the kernel that calculates
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:This calls ``primary_beam_gpu::test_analytic_dipole_beam``, which in turn
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:tests ``primary_beam_gpu::calculate_analytic_dipole_beam``, code that copies
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:az/za angles into GPU memory, calculates an analytic dipole response toward
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:those directions, and then frees the az/za coords from GPU memory.
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:This calls ``primary_beam_gpu::test_calculate_MWA_analytic_beam``, which calls
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:``primary_beam_gpu::calculate_MWA_analytic_beam``, which calculates an
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:``cmake_testing/primary_beam_gpu/run_header_setup_and_plots.ipynb``. Along with
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:This calls ``primary_beam_gpu::test_run_hyperbeam_gpu``, which calls
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:``primary_beam_gpu::run_hyperbeam_gpu``, which is a wrapper around
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:This calls ``primary_beam_gpu::test_run_hyperbeam_gpu``, which calls
docs/sphinx/testing/cmake_testing/primary_beam_gpu.rst:``primary_beam_gpu::run_hyperbeam_gpu``, which is a wrapper around `mwa_hyperbeam`_ to calculate the MWA FEE beam. Unlike ``test_run_hyperbeam.c`` however, we used
docs/sphinx/testing/cmake_testing/wodenpy/use_libwoden.rst:This is a collection of modules that interact with the C/CUDA code via ``ctypes``. Much of the code here defined ``ctypes`` ``structures`` classes that are equivalent to a ``C`` ``struct``. This allows us to pass data back and forth between ``Python`` and ``C``. Much of the functionality in ``wodenpy.use_libwoden`` is tested by other unit tests that use the aforementioned classes. The tests here attempt to fill in any gaps.
docs/sphinx/testing/cmake_testing/wodenpy/skymodel.rst:``create_skymodel_chunk_map`` makes a map of how to split a given number of sky model components into bit size chunks that fit onto the GPU, and also fit in RAM. This splitting is done based on the number of visibilities being simulated and user settings. Test by running with seven different combinations of inputs, varying the numbers of points, gaussians, shapelets, maximum visibilities, number time steps and frequencies. Test against a separate set of test functions that produce the desired outputs.
docs/sphinx/testing/cmake_testing/calculate_visibilities.rst:to all ``CUDA`` functionality in ``WODEN``. It takes in simulations settings and
docs/sphinx/testing/cmake_testing/calculate_visibilities.rst:GPU correctly. More rigorous testing of this functionality is included in other
docs/sphinx/testing/cmake_testing/scripts/run_woden.rst:``run_woden.py`` is the main ``WODEN`` executable. It should parse input command arguments, gather metafits and sky model information, calculate observational parameters, launch GPU simulation code, and output a ``uvfits`` file.
docs/sphinx/testing/cmake_testing/scripts/concat_woden_uvfits.rst:``concat_woden_uvfits.py`` is supposed to concatenate multiple coarse band ``uvfits`` into one (aka 24 coarse bands of 32 channels in one ``uvfits`` of length 384 channel). This is useful to simulate across 24 GPUs but combine outputs into one ``uvfits`` that can be input to calibration software.
docs/sphinx/testing/cmake_testing/primary_beam.rst:setup primary beam settings, ready to calculate beam responses on the GPU.
docs/sphinx/testing/cmake_testing/fundamental_coords.rst:versions are tested within a tolerance of 1e-16 (this test compares the ``CUDA``
docs/sphinx/testing/cmake_testing/source_components.rst:I assert the ``CUDA`` outputs must be within an absolute tolerance of 1e-7 for
docs/sphinx/testing/cmake_testing/source_components.rst:The values are copied into a ``source_t`` struct, passed through the ``CUDA``
docs/sphinx/testing/cmake_testing/source_components.rst:everything, and the cyan squares are what is output by the ``CUDA`` code.
docs/sphinx/testing/cmake_testing/source_components.rst:I assert the ``CUDA`` code output must match the ``C`` code output to
docs/sphinx/testing/cmake_testing/source_components.rst:test the ``CUDA`` code against. I assert the ``CUDA`` code output must match the
docs/sphinx/testing/cmake_testing/source_components.rst:to test the ``CUDA`` code against.
docs/sphinx/testing/cmake_testing/source_components.rst:performed by ``C`` code, which the outputs of the GPU code are tested against.
docs/sphinx/scripts/run_woden.rst:This is the main ``WODEN`` executable. It takes command line arguments, creates and array layout, visibility containers, read sky models, and launches GPU code to calculate the visibilities. Finally, it writes the outputs to ``uvfits`` files.
docs/sphinx/conf.py:##Add these in so the c++ acknowledges CUDA specific attributes
docs/sphinx/examples/fornaxA_sim.rst:with the "float" simulation taking 59 seconds on my GPU, the "double" taking 144 seconds, and the image looking like:
docs/sphinx/examples/MWA_EoR1_sim.rst:Running this took 55 mins 46 secs seconds on my GPU (running with the ``--precision=float`` flag runs in 10 min 39 sec). I've reduced the time and frequency resolution as specified in the ``metafits`` file to keep the size of the outputs smaller on your machine. If you wanted to run the full resolution data of this observation, (2s, 40kHz), you can just remove the ``--num_freq_channels, --num_time_steps, --freq_res, --time_res`` arguments.
docs/sphinx/examples/example_simulations.rst:.. note:: For all simulation times reported in the below, I used a single NVIDIA GeForce GTX 1080 Ti with 12 GB of RAM.
docs/sphinx/examples/example_simulations.rst:.. warning:: If you have a GPU with small amounts of RAM (say 2GB) some of these simulations won't work to DOUBLE precision, you won't have enough memory. You can add the ``--precision=float`` argument to switch to a lower memory requirement (for the loss of accuracy).
docs/sphinx/operating_principles/visibility_calcs.rst:  Before ``WODEN`` version 1.4.0, in the output `uvfits` files, the first polarisation (usually called XX) was derived from North-South dipoles, as is the labelling convention according to the IAU. However, most `uvfits` users I've met, as well as the data out of the MWA telescope, define XX as East-West. So although the internal labelling and mathematics within the C/CUDA code is to IAU spec, by default, ``run_woden.py`` now writes out XX as East-West and YY as North-South. From version 1.4.0, a header value of ``IAUORDER = F`` will appear, with ``F`` meaning IAU ordering is False, so the polarisations go EW-EW, NS-NS, EW-NS, NS-EW. If ``IAUORDER = T``, the order is NS-NS, EW-EW, NS-EW, EW-NS. If there is no ``IAUORDER`` at all, assume ``IAUORDER = T``.
docs/sphinx/operating_principles/run_Gaussian_beam.c://External CUDA code being linked in
docs/sphinx/operating_principles/run_RTS_analy_beam.py:##Only feed az/za above the horizon to save on CUDA memory
docs/sphinx/operating_principles/run_RTS_analy_beam.py:##write out the az/za to feed into the C/CUDA code
docs/sphinx/operating_principles/run_RTS_analy_beam.py:# ##compile the C/CUDA code
docs/sphinx/operating_principles/run_RTS_analy_beam.py:# ##run the C/CUDA code
docs/sphinx/operating_principles/run_RTS_analy_beam.py:##read in outputs from C/CUDA code
docs/sphinx/operating_principles/run_Gaussian_beam.py:##Only feed az/za above the horizon to save on CUDA memory
docs/sphinx/operating_principles/run_Gaussian_beam.py:##write out the az/za to feed into the C/CUDA code
docs/sphinx/operating_principles/run_EDA2_beam.py:##Only feed az/za above the horizon to save on CUDA memory
docs/sphinx/operating_principles/run_EDA2_beam.py:##write out the az/za to feed into the C/CUDA code
docs/sphinx/operating_principles/make_run_RTS_analy_beam.sh:  -L$WODEN_DIR -lwodenCUDA_double -lwodenC_double \
docs/sphinx/operating_principles/frequency_wording.rst:As many coarse bands as needed can be run, allowing for straight-forward splitting of the simulation across multiple GPUs. We'll look at an example of that later.  If you run with following arguments::
docs/sphinx/operating_principles/plot_hyperbeam_ctest_outputs.py:used_az, used_za, gx_re, gx_im, Dx_re, Dx_im, Dy_re, Dy_im, gy_re, gy_im, freqs = np.loadtxt(f"../../../build/cmake_testing/primary_beam_cuda/hyperbeam_zenith_200_rot_double.txt",unpack=True)
docs/sphinx/operating_principles/plot_hyperbeam_ctest_outputs.py:gx_re, gx_im, Dx_re, Dx_im, Dy_re, Dy_im, gy_re, gy_im, freqs = np.loadtxt(f"../../../build/cmake_testing/primary_beam_cuda/hyperbeam_interp_delays1_freqs1.txt",unpack=True)
docs/sphinx/operating_principles/plot_hyperbeam_ctest_outputs.py:gx_re, gx_im, Dx_re, Dx_im, Dy_re, Dy_im, gy_re, gy_im, freqs = np.loadtxt(f"../../../build/cmake_testing/primary_beam_cuda/hyperbeam_interp_delays4_freqs4.txt",unpack=True)
docs/sphinx/operating_principles/make_run_Gaussian_beam.sh:  -L$WODEN_DIR -lwodenCUDA -lwodenC \
docs/sphinx/operating_principles/make_run_EDA2_beam.sh:  -L$WODEN_DIR -lwodenCUDA -lwodenC \
docs/sphinx/operating_principles/run_RTS_analy_beam.c://External CUDA code we're linking in
wodenpy/skymodel/chunk_sky_model.py:        ##Splitting POINTs and GAUSSIANS into lovely chunks that our GPU can chew
wodenpy/skymodel/chunk_sky_model.py:    ##GPU memory TODO don't pray, submit a warning?
wodenpy/skymodel/read_skymodel.py:    resultant can be passed into C/CUDA code to calculate visibilities.
wodenpy/skymodel/read_skymodel.py:        to the C/CUDA structs.
wodenpy/skymodel/read_skymodel.py:        A source catalogue that can be used by C/CUDA code to calculate visibilities.
wodenpy/skymodel/read_fits_skymodel.py:    ##some GPU code to not bother using V when calculating XX, YY, XY, YX
wodenpy/skymodel/read_fits_skymodel.py:    that can be used by C/CUDA code to calculate visibilities. Uses the
wodenpy/skymodel/read_fits_skymodel.py:        A class containing all the ctypes structures with the correct precision, needed for the C/CUDA code.
wodenpy/skymodel/read_fits_skymodel.py:        A source catalogue that can be used by C/CUDA code to calculate visibilities.
wodenpy/skymodel/read_fits_skymodel.py:    ##of each source and be fed straight into C/CUDA
wodenpy/use_libwoden/create_woden_struct_classes.py:        these are used to match the `-DUSE_DOUBLE` flag in the C/GPU code.
wodenpy/use_libwoden/skymodel_structs.py:    struct in the C/CUDA code. Created dynamically based on the `precision`,
wodenpy/use_libwoden/skymodel_structs.py:        the C and CUDA code in libwoden_float.so or libwoden_double.so.
wodenpy/use_libwoden/skymodel_structs.py:                    ##used to hold l,m,n coords on the GPU
wodenpy/use_libwoden/skymodel_structs.py:    used by the C and CUDA code in libwoden_double.so. The `Components` class
wodenpy/use_libwoden/skymodel_structs.py:        the C and CUDA code in libwoden_double.so
wodenpy/use_libwoden/skymodel_structs.py:    `source_catalogue_t` struct, used by the C and CUDA code. The `Source` class
wodenpy/use_libwoden/skymodel_structs.py:        the C and CUDA code in libwoden_float.so
wodenpy/use_libwoden/skymodel_structs.py:    Sets up `chunked_source`, which will be passed to C/CUDA. Allocates the
wodenpy/use_libwoden/skymodel_structs.py:    ##This is grabbing all the lovely things calculated by the GPU
wodenpy/use_libwoden/use_libwoden.py:    """Load in the WODEN C and CUDA code via a dynamic library, with the
wodenpy/use_libwoden/use_libwoden.py:        to the C/CUDA structs. Should have been initialised with the correct
wodenpy/use_libwoden/use_libwoden.py:        The C wrapper function `run_woden`, which runs the C/CUDA code.
wodenpy/use_libwoden/woden_settings.py:    struct in the C/CUDA code. Created dynamically based on the `precision`,
wodenpy/use_libwoden/woden_settings.py:        the C and CUDA code in libwoden_float.so or libwoden_double.so.
wodenpy/use_libwoden/woden_settings.py:       Populated ctype struct that can be passed into the C/CUDA code
wodenpy/use_libwoden/visibility_set.py:    struct in the C/CUDA code. Created dynamically based on the `precision`,
wodenpy/use_libwoden/visibility_set.py:        the C and CUDA code in libwoden_float.so
wodenpy/use_libwoden/visibility_set.py:    ##This is grabbing all the lovely things calculated by the GPU
wodenpy/use_libwoden/visibility_set.py:        ##GPU, we don't have to store copies of u,v,w for every frequency like
wodenpy/use_libwoden/visibility_set.py:        ##we did when writing to file. Will be a moderate save on GPU memory
wodenpy/uvfits/wodenpy_uvfits.py:    Will only work for data as ordered as coming out of the WODEN C/CUDA code
wodenpy/uvfits/wodenpy_uvfits.py:        By default, the visibilities out of the GPU/C code have 
wodenpy/uvfits/wodenpy_uvfits.py:    ##Data out of WODEN C/CUDA code is in IAU pol order, so do nothing
wodenpy/wodenpy_setup/run_setup.py:                "any simulation to be split across multiple GPUs as separate "
wodenpy/wodenpy_setup/run_setup.py:    ##args.dipamps only is carried into the C/CUDA code, so always turn on
coverage_outputs/README.md:At the moment this is slightly stone-henge, due to the fact that CUDA is not supported for code coverage, and you can't automate a GPU test for free. For now, users must run the unit tests locally (using `ctest`, see [ReadTheDocs](https://woden.readthedocs.io/en/latest/testing/cmake_testing.html) for instructions). They can then run `source create_cov_reports.sh`, which will grab the outputs of the `C` tests from `ctest` and covert them into an appropriate formats, as well as running the `python` tests using `coverage`, which also generates appropriate outputs. To get `coverage`, do something like:
coverage_outputs/README.md:It's possible in the future that we can separate out the CUDA tests from the C/python tests, and automate that whole part, to automagically generate the whole "codecov" report.
README.md:[![Documentation Status](https://readthedocs.org/projects/woden/badge/?version=latest)](https://woden.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/JLBLine/WODEN/branch/master/graph/badge.svg?token=Q3JFCI5GOC)](https://codecov.io/gh/JLBLine/WODEN) _*note code coverage only applies to `python3` and `C` code, `CUDA` code is not currently supported by free code coverage software_
README.md:`WODEN` is C / GPU code designed to be able to simulate low-frequency radio interferometric data. It is written to be simplistic and *fast* to allow all-sky simulations. Although `WODEN` was primarily written to simulate Murchinson Widefield Array ([MWA, Tingay et al. 2013](https://doi.org/10.1017/pasa.2012.007)) visibilities, it is capable of bespoke array layouts, and has a number of primary beam options. `WODEN` outputs `uvfits` files. The `WODEN` documentation lives [here on readthedocs](https://woden.readthedocs.io/en/latest/). If your internet has broken and you have already installed `WODEN`, you can build a local copy by navigating into `WODEN/docs/sphinx` and running `make html` (you'll need to have `doxygen` installed).
README.md: - Thanks to Marcin Sokolowski, and code from the [PaCER Blink project](https://github.com/PaCER-BLINK-Project), there is nominal support for AMD GPUs via ROCm. I've managed to compile and run on the Pawsey Setonix cluster, but this is the only AMD GPU I have access to. If you have an AMD GPU, please test and let me know how you go!
README.md: - There are some installation examples in ``WODEN/templates/`` for a CUDA and an AMD installation on superclusters. Hopefully you can adapt these to your needs if the Docker images don't work for you.
README.md: - CUDA arch flag on compilation is changed to setting an environment variable `CUDAARCHS` to use `CMakes` `CMAKE_CUDA_ARCHITECTURES` feature.
README.md: - The `C/CUDA` code that remains is called directly from `python` now, meaning the `.json` and `.dat` files are no longer needed
README.md:Please be aware, before ``WODEN`` version 1.4.0, in the output `uvfits` files, the first polarisation (usually called XX) was derived from North-South dipoles, as is the labelling convention according to the IAU. However, most `uvfits` users I've met, as well as the data out of the MWA telescope, define XX as East-West. So although the internal labelling and mathematics within the C/GPU code is to IAU spec, by default, ``run_woden.py`` now writes out XX as East-West and YY as North-South. From version 1.4.0, a header value of ``IAUORDER = F`` will appear, with ``F`` meaning IAU ordering is False, so the polarisations go EW-EW, NS-NS, EW-NS, NS-EW. If ``IAUORDER = T``, the order is NS-NS, EW-EW, NS-EW, EW-NS. If there is no ``IAUORDER`` at all, assume ``IAUORDER = T``.
README.md:The quickest way is to use a docker image (for a CUDA arch of 7.5 for example):
README.md:docker pull docker://jlbline/woden-2.3:cuda-75
README.md:- `jlbline/woden-2.3:cuda-60` - tested on Swinburne OzStar
README.md:- `jlbline/woden-2.3:cuda-80` - tested on Swinburne Ngarrgu Tindebeek
README.md:- Either NVIDIA CUDA - https://developer.nvidia.com/cuda-downloads
README.md:- or AMD ROCm - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
include/hyperbeam_error.h://Going to be calling this code in both C, and CUDA, so stick the
include/gpucomplex.h:  CUDA operators for complex numbers.
include/gpucomplex.h:  For some reason, these aren't defined in the CUDA headers form NVIDIA
include/gpucomplex.h:#include "gpu_macros.h"
include/gpucomplex.h:typedef gpuDoubleComplex gpuUserComplex;
include/gpucomplex.h:typedef gpuFloatComplex gpuUserComplex;
include/gpucomplex.h:inline __device__ gpuFloatComplex operator-( gpuFloatComplex a ) {
include/gpucomplex.h:  return( make_gpuFloatComplex( -a.x, -a.y ) );
include/gpucomplex.h:inline __device__ void operator+=( gpuFloatComplex &a, const gpuFloatComplex b ) {
include/gpucomplex.h:  a = gpuCaddf( a, b );
include/gpucomplex.h:inline __device__ gpuFloatComplex operator+( const gpuFloatComplex a, const gpuFloatComplex b ) {
include/gpucomplex.h:  return( gpuCaddf( a, b ) );
include/gpucomplex.h:inline __device__ gpuFloatComplex operator+( const gpuFloatComplex a, const float r ) {
include/gpucomplex.h:  return( make_gpuFloatComplex( a.x+r, a.y ) );
include/gpucomplex.h:inline __device__ gpuFloatComplex operator+( const float r, const gpuFloatComplex z ) {
include/gpucomplex.h:inline __device__ void operator-=( gpuFloatComplex &a, const gpuFloatComplex b ) {
include/gpucomplex.h:  a = gpuCsubf( a, b );
include/gpucomplex.h:inline __device__ gpuFloatComplex operator-( const gpuFloatComplex a, const gpuFloatComplex b ) {
include/gpucomplex.h:  return( gpuCsubf( a, b ) );
include/gpucomplex.h:inline __device__ void operator*=( gpuFloatComplex &a, const gpuFloatComplex b ) {
include/gpucomplex.h:  a = gpuCmulf( a, b );
include/gpucomplex.h:inline __device__ gpuFloatComplex operator*( const gpuFloatComplex a, const gpuFloatComplex b ) {
include/gpucomplex.h:  return( gpuCmulf( a, b ) );
include/gpucomplex.h:inline __device__ void operator*=( gpuFloatComplex &z, const float r ) {
include/gpucomplex.h:inline __device__ gpuFloatComplex operator*( const gpuFloatComplex z, const float r ) {
include/gpucomplex.h:  gpuFloatComplex temp;
include/gpucomplex.h:inline __device__ gpuFloatComplex operator*( const float r, const gpuFloatComplex z ) {
include/gpucomplex.h:inline __device__ gpuFloatComplex operator/( const gpuFloatComplex a, const gpuFloatComplex b ) {
include/gpucomplex.h:  return( gpuCdivf( a, b ) );
include/gpucomplex.h:inline __device__ gpuFloatComplex operator/( const float r, const gpuFloatComplex b ) {
include/gpucomplex.h:  return( gpuCdivf( make_gpuFloatComplex( r, 0 ), b ) );
include/gpucomplex.h:inline __device__ gpuFloatComplex operator/( const gpuFloatComplex a, const float r ) {
include/gpucomplex.h:  return( make_gpuFloatComplex( a.x / r, a.y / r ) );
include/gpucomplex.h:inline __device__ gpuFloatComplex gpuComplexExp( const gpuFloatComplex z ) {
include/gpucomplex.h:  float x = gpuCrealf( z );
include/gpucomplex.h:  float y = gpuCimagf( z );
include/gpucomplex.h:  gpuFloatComplex temp = make_gpuFloatComplex( cosf(y), sinf(y) );
include/gpucomplex.h:inline __device__ gpuFloatComplex U1polar( const float theta ) {
include/gpucomplex.h:  gpuFloatComplex z;
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator-( gpuDoubleComplex a ) {
include/gpucomplex.h:  return( make_gpuDoubleComplex( -a.x, -a.y ) );
include/gpucomplex.h:inline __device__ void operator+=( gpuDoubleComplex &a, const gpuDoubleComplex b ) {
include/gpucomplex.h:  a = gpuCadd( a, b );
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator+( const gpuDoubleComplex a, const gpuDoubleComplex b ) {
include/gpucomplex.h:  return( gpuCadd( a, b ) );
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator+( const gpuDoubleComplex a, const double r ) {
include/gpucomplex.h:  return( make_gpuDoubleComplex( a.x+r, a.y ) );
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator+( const double r, const gpuDoubleComplex z ) {
include/gpucomplex.h:inline __device__ void operator-=( gpuDoubleComplex &a, const gpuDoubleComplex b ) {
include/gpucomplex.h:  a = gpuCsub( a, b );
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator-( const gpuDoubleComplex a, const gpuDoubleComplex b ) {
include/gpucomplex.h:  return( gpuCsub( a, b ) );
include/gpucomplex.h:inline __device__ void operator*=( gpuDoubleComplex &a, const gpuDoubleComplex b ) {
include/gpucomplex.h:  a = gpuCmul( a, b );
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator*( const gpuDoubleComplex a, const gpuDoubleComplex b ) {
include/gpucomplex.h:  return( gpuCmul( a, b ) );
include/gpucomplex.h:inline __device__ void operator*=( gpuDoubleComplex &z, const double r ) {
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator*( const gpuDoubleComplex z, const double r ) {
include/gpucomplex.h:  gpuDoubleComplex temp;
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator*( const double r, const gpuDoubleComplex z ) {
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator/( const gpuDoubleComplex a, const gpuDoubleComplex b ) {
include/gpucomplex.h:  return( gpuCdiv( a, b ) );
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator/( const double r, const gpuDoubleComplex b ) {
include/gpucomplex.h:  return( gpuCdiv( make_gpuDoubleComplex( r, 0 ), b ) );
include/gpucomplex.h:inline __device__ gpuDoubleComplex operator/( const gpuDoubleComplex a, const double r ) {
include/gpucomplex.h:  return( make_gpuDoubleComplex( a.x / r, a.y / r ) );
include/gpucomplex.h:inline __device__ gpuDoubleComplex gpuComplexExp( const gpuDoubleComplex z ) {
include/gpucomplex.h:  double x = gpuCreal( z );
include/gpucomplex.h:  double y = gpuCimag( z );
include/gpucomplex.h:  gpuDoubleComplex temp = make_gpuDoubleComplex( cos(y), sin(y) );
include/gpucomplex.h:inline __device__ gpuDoubleComplex U1polar( const double theta ) {
include/gpucomplex.h:  gpuDoubleComplex z;
include/gpucomplex.h://! Make a CUDA complex to the precision set during compilation
include/gpucomplex.h:inline __device__ gpuUserComplex make_gpuUserComplex( user_precision_t real,
include/gpucomplex.h:  gpuUserComplex z;
include/woden_struct_defs.h:  //itself will have the d_ label if doing things on the GPU
include/woden_struct_defs.h:    struct FEEBeamGpu *gpu_fee_beam; /*!< Single initialised hyperbeam device model for desired pointing */
include/calculate_visibilities.h:  through the CUDA code, and grab everything back off the device and onto the
include/calculate_visibilities.h: `cropped_sky_models`, and `woden_settings`, run the GPU simulation, and store
include/calculate_visibilities.h:and the simulation settings described in `woden_settings` to run the GPU
include/visibility_set.h://Going to be calling this code in both C and CUDA, so stick the
include/visibility_set.h:injest into CUDA kernels as a single axis. `channel_frequencies` will list
include/gpu_macros.h:  A bunch of macros to choose either CUDA or HIP for GPU operations. Needs the compiler to define `__NVCC__` or `__HIPCC__` to choose the correct functionality. Also includes a macro to check for errors in GPU operations, and a macro to run a GPU kernel and check for errors.
include/gpu_macros.h:#define __GPU__
include/gpu_macros.h:// bool gpu_support() { return true;}
include/gpu_macros.h:// automatically when using "gpu*" calls.
include/gpu_macros.h:#define gpuError_t cudaError_t
include/gpu_macros.h:#define gpuSuccess cudaSuccess
include/gpu_macros.h:#define gpuGetErrorString cudaGetErrorString
include/gpu_macros.h:#include <cuda_runtime.h>
include/gpu_macros.h:#define gpuError_t hipError_t
include/gpu_macros.h:#define gpuSuccess hipSuccess
include/gpu_macros.h:#define gpuGetErrorString hipGetErrorString
include/gpu_macros.h:#define GPU_CHECK_ERROR(X)({\
include/gpu_macros.h:    if(X != gpuSuccess){\
include/gpu_macros.h:        fprintf(stderr, "GPU error (%s:%d): %s\n", __FILE__ , __LINE__ , gpuGetErrorString(X));\
include/gpu_macros.h:Used within `GPUErrorCheck`. If true, exit if an error is found */
include/gpu_macros.h:@brief Take a GPU error message (code), and checks whether an error
include/gpu_macros.h:user, along with the decoded CUDA error message. Uses `file` and `line` to
include/gpu_macros.h:@param[in] code Error message out of CUDA call (e.g. cudaMalloc)
include/gpu_macros.h:@param[in] abort If true, exit the CUDA code when an error is found
include/gpu_macros.h:inline void GPUErrorCheck(const char *message, gpuError_t code, const char *file, int line, bool abort=EXITERROR){
include/gpu_macros.h:  if (code != gpuSuccess) {
include/gpu_macros.h:    fprintf(stderr,"GPU ERROR %s: %s\n %s:%d\n",
include/gpu_macros.h:                    message, gpuGetErrorString(code), file, line);
include/gpu_macros.h:      printf("GPU IS EXITING\n");
include/gpu_macros.h:#define cudaCheckErrors(msg) \
include/gpu_macros.h:        cudaError_t __err = cudaGetLastError(); \
include/gpu_macros.h:        if (__err != cudaSuccess) { \
include/gpu_macros.h:                msg, cudaGetErrorString(__err), \
include/gpu_macros.h:#define gpuMalloc(...) GPUErrorCheck("cudaMalloc", cudaMalloc(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuHostAlloc(...) GPUErrorCheck("cudaHostAlloc", cudaHostAlloc(__VA_ARGS__, 0),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuHostAllocDefault cudaHostAllocDefault
include/gpu_macros.h:#define gpuMemcpy(...) GPUErrorCheck("cudaMemcpy", cudaMemcpy(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuMemcpyAsync(...) GPUErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuMemset(...) GPUErrorCheck("cudaMemset", cudaMemset(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:// #define gpuDeviceSynchronize(...) GPUErrorCheck("cudaDeviceSynchronize", cudaDeviceSynchronize(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuDeviceSynchronize cudaDeviceSynchronize
include/gpu_macros.h:#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
include/gpu_macros.h:#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
include/gpu_macros.h:#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
include/gpu_macros.h:#define gpuFree(...) GPUErrorCheck("cudaFree", cudaFree(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuHostFree(...) GPUErrorCheck("cudaFreeHost", cudaFreeHost(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuStream_t cudaStream_t
include/gpu_macros.h:#define gpuStreamCreate(...) GPUErrorCheck("cudaStreamCreate", cudaStreamCreate(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuStreamDestroy(...) GPUErrorCheck("cudaStreamDestroy", cudaStreamDestroy(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuEventCreate(...) GPUErrorCheck("cudaEventCreate", cudaEventCreate(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuGetDeviceCount(...) GPUErrorCheck("cudaGetDeviceCount", cudaGetDeviceCount(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuGetLastError cudaGetLastError
include/gpu_macros.h:#define gpuMemGetInfo(...) GPUErrorCheck("cudaMemGetInfo", cudaMemGetInfo(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuMallocHost(...) GPUErrorCheck("cudaMallocHost", cudaMallocHost(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:// #define gpuCheckErrors(...) cudaCheckErrors(__VA_ARGS__)
include/gpu_macros.h:#define gpuFreeHost(...) GPUErrorCheck(" cudaFreeHost",  cudaFreeHost(__VA_ARGS__),__FILE__, __LINE__ )
include/gpu_macros.h:#define gpuGetDeviceProperties(...) cudaGetDeviceProperties(__VA_ARGS__)
include/gpu_macros.h:#define gpuDeviceProp cudaDeviceProp
include/gpu_macros.h:#define gpuPeekAtLastError cudaPeekAtLastError
include/gpu_macros.h:#define gpuCreal cuCreal
include/gpu_macros.h:#define gpuCrealf cuCrealf
include/gpu_macros.h:#define gpuCimag cuCimag
include/gpu_macros.h:#define gpuCimagf cuCimagf
include/gpu_macros.h:#define gpuCadd  cuCadd
include/gpu_macros.h:#define gpuCmul  cuCmul
include/gpu_macros.h:#define gpuCdiv  cuCdiv
include/gpu_macros.h:#define gpuConj  cuConj
include/gpu_macros.h:#define gpuCsub  cuCsub
include/gpu_macros.h:#define gpuCabs  cuCabs
include/gpu_macros.h:#define gpuCaddf cuCaddf
include/gpu_macros.h:#define gpuCsubf cuCsubf
include/gpu_macros.h:#define gpuCmulf cuCmulf
include/gpu_macros.h:#define gpuCdivf cuCdivf
include/gpu_macros.h:#define gpuDoubleComplex cuDoubleComplex
include/gpu_macros.h:#define gpuFloatComplex cuFloatComplex
include/gpu_macros.h:#define make_gpuDoubleComplex make_cuDoubleComplex
include/gpu_macros.h:#define make_gpuFloatComplex make_cuFloatComplex
include/gpu_macros.h:/*inline int num_available_gpus()
include/gpu_macros.h:    int num_gpus;
include/gpu_macros.h:    gpuGetDeviceCount(&num_gpus);
include/gpu_macros.h:    return num_gpus;
include/gpu_macros.h:#define gpuMalloc(...) GPUErrorCheck("hipMalloc", hipMalloc(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuHostAlloc(...) GPUErrorCheck("hipHostMalloc", hipHostMalloc(__VA_ARGS__, 0),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuHostAllocDefault 0
include/gpu_macros.h:#define gpuMemcpy(...) GPUErrorCheck("hipMemcpy", hipMemcpy(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuMemcpyAsync(...) GPUErrorCheck("hipMemcpyAsync", hipMemcpyAsync(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuMemset(...) GPUErrorCheck("hipMemset", hipMemset(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:// #define gpuDeviceSynchronize(...) GPUErrorCheck("hipDeviceSynchronize",hipDeviceSynchronize(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuDeviceSynchronize hipDeviceSynchronize
include/gpu_macros.h:#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
include/gpu_macros.h:#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
include/gpu_macros.h:#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
include/gpu_macros.h:#define gpuFree(...) GPUErrorCheck("hipFree", hipFree(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuHostFree(...) GPUErrorCheck("hipHostFree", hipHostFree(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuStream_t hipStream_t
include/gpu_macros.h:#define gpuStreamCreate(...) GPUErrorCheck("hipStreamCreate", hipStreamCreate(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuStreamDestroy(...) GPUErrorCheck("hipStreamDestroy", hipStreamDestroy(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuEventCreate(...) GPUErrorCheck("hipEventCreate", hipEventCreate(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuGetDeviceCount(...) GPUErrorCheck("hipGetDeviceCount", hipGetDeviceCount(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuGetLastError hipGetLastError
include/gpu_macros.h:#define gpuMemGetInfo(...) GPUErrorCheck("hipMemGetInfo", hipMemGetInfo(__VA_ARGS__),__FILE__, __LINE__)
include/gpu_macros.h:#define gpuMallocHost(...) GPUErrorCheck("hipHostMalloc", hipHostMalloc(__VA_ARGS__, 0),__FILE__, __LINE__) // TODO : double check this may be temporary only
include/gpu_macros.h:// #define gpuCheckErrors(...) hipCheckErrors(__VA_ARGS__)
include/gpu_macros.h:#define gpuFreeHost(...)  GPUErrorCheck( "hipFreeHost", hipFreeHost(__VA_ARGS__),__FILE__, __LINE__ )
include/gpu_macros.h:#define gpuGetDeviceProperties(...) GPUErrorCheck( "hipGetDeviceProperties", hipGetDeviceProperties(__VA_ARGS__),__FILE__, __LINE__ )
include/gpu_macros.h:#define gpuDeviceProp hipDeviceProp_t
include/gpu_macros.h:#define gpuPeekAtLastError hipPeekAtLastError
include/gpu_macros.h:#define gpuCreal hipCreal
include/gpu_macros.h:#define gpuCrealf hipCrealf
include/gpu_macros.h:#define gpuCimag hipCimag
include/gpu_macros.h:#define gpuCimagf hipCimagf
include/gpu_macros.h:#define gpuCadd  hipCadd
include/gpu_macros.h:#define gpuCmul  hipCmul
include/gpu_macros.h:#define gpuCdiv  hipCdiv
include/gpu_macros.h:#define gpuConj  hipConj
include/gpu_macros.h:#define gpuCsub  hipCsub
include/gpu_macros.h:#define gpuCabs  hipCabs
include/gpu_macros.h:#define gpuCaddf hipCaddf
include/gpu_macros.h:#define gpuCsubf hipCsubf
include/gpu_macros.h:#define gpuCmulf hipCmulf
include/gpu_macros.h:#define gpuCdivf hipCdivf
include/gpu_macros.h:#define gpuDoubleComplex hipDoubleComplex
include/gpu_macros.h:#define gpuFloatComplex  hipFloatComplex
include/gpu_macros.h:#define make_gpuDoubleComplex make_hipDoubleComplex
include/gpu_macros.h:#define make_gpuFloatComplex make_hipFloatComplex
include/gpu_macros.h:#define gpuCheckLastError(...) GPUErrorCheck(gpuGetLastError())
include/gpu_macros.h:// bool gpu_support() { return false;}
include/gpu_macros.h:// inline int num_available_gpus(){ return 0; } 
include/gpu_macros.h:@brief Takes a GPU kernel, runs it with given arguments, and passes results
include/gpu_macros.h:`gpuErrorCheckKernel` then passes the string `message` on to `GPUErrorCheck`,
include/gpu_macros.h:checks the errors from both `gpuGetLastError()` and `gpuDeviceSynchronize()`
include/gpu_macros.h:          gpuErrorCheckKernel("Call to fancy_kernel",
include/gpu_macros.h:#define gpuErrorCheckKernel(message, kernel, grid, threads, ...) \
include/gpu_macros.h:  GPUErrorCheck(message, gpuGetLastError(), __FILE__, __LINE__); \
include/gpu_macros.h:  GPUErrorCheck(message, gpuDeviceSynchronize(), __FILE__, __LINE__);
include/gpu_macros.h://   gpuDeviceSynchronize();
include/gpu_macros.h:@brief NOTE the actual function is `GPUErrorCheck`, but for some goddam reason Doxygen refuses to document it when inside a conditional so I've made a copy here. Take a GPU error message (code), and checks whether an error
include/gpu_macros.h:user, along with the decoded CUDA error message. Uses `file` and `line` to
include/gpu_macros.h:@param[in] code Error message out of CUDA call (e.g. cudaMalloc)
include/gpu_macros.h:@param[in] abort If true, exit the CUDA code when an error is found. Defaults to True.
include/gpu_macros.h:inline void docGPUErrorCheck(const char *message, gpuError_t code, const char *file, int line, bool abort=EXITERROR){
include/primary_beam_gpu.h:#include "gpucomplex.h"
include/primary_beam_gpu.h:           gpuUserComplex *d_primay_beam_J00, gpuUserComplex *d_primay_beam_J11);
include/primary_beam_gpu.h:           gpuUserComplex *d_primay_beam_J00, gpuUserComplex *d_primay_beam_J11);
include/primary_beam_gpu.h:           gpuUserComplex * d_beam_X, gpuUserComplex * d_beam_Y);
include/primary_beam_gpu.h:           gpuUserComplex *d_primay_beam_J00, gpuUserComplex *d_primay_beam_J11);
include/primary_beam_gpu.h:     gpuUserComplex *d_primay_beam_J00, gpuUserComplex *d_primay_beam_J11);
include/primary_beam_gpu.h:`primary_beam_gpu::calculate_RTS_MWA_analytic_beam`)
include/primary_beam_gpu.h:           gpuUserComplex * gx, gpuUserComplex * Dx,
include/primary_beam_gpu.h:           gpuUserComplex * Dy, gpuUserComplex * gy);
include/primary_beam_gpu.h:@details Kernel calls `primary_beam_gpu::RTS_MWA_beam`. The MWA primary beam
include/primary_beam_gpu.h:`primary_beam_gpu::calculate_RTS_MWA_analytic_beam`)
include/primary_beam_gpu.h:           gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
include/primary_beam_gpu.h:           gpuUserComplex *d_Dys, gpuUserComplex *d_gys);
include/primary_beam_gpu.h:@details Uses the kernel `primary_beam_gpu::kern_RTS_analytic_MWA_beam`.
include/primary_beam_gpu.h:`primary_beam_gpu::RTS_MWA_beam``.
include/primary_beam_gpu.h:     gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
include/primary_beam_gpu.h:     gpuUserComplex *d_Dys, gpuUserComplex *d_gys);
include/primary_beam_gpu.h:`*gpu_fee_beam`. NOTE that the azs, zas need to increment by component
include/primary_beam_gpu.h:@details Calls `mwa_hyperbeam::fee_calc_jones_gpu_device` to calculate the beam
include/primary_beam_gpu.h:`struct FEEBeamGpu *gpu_fee_beam` object (initialised using
include/primary_beam_gpu.h:`mwa_hyperbeam::new_gpu_fee_beam`), which in turn needs a
include/primary_beam_gpu.h:d_primay_beam_J* arrays using the kernel `primary_beam_gpu::kern_map_hyperbeam_gains`.
include/primary_beam_gpu.h:@param[in] *gpu_fee_beam An initialised `mwa_hyperbeam` `struct FEEBeamGpu`
include/primary_beam_gpu.h:extern "C" void run_hyperbeam_gpu(int num_components,
include/primary_beam_gpu.h:           struct FEEBeamGpu *gpu_fee_beam,
include/primary_beam_gpu.h:           gpuUserComplex *d_primay_beam_J00,
include/primary_beam_gpu.h:           gpuUserComplex *d_primay_beam_J01,
include/primary_beam_gpu.h:           gpuUserComplex *d_primay_beam_J10,
include/primary_beam_gpu.h:           gpuUserComplex *d_primay_beam_J11);
include/source_components.h:  gpuUserComplex *d_gxs = NULL; /*!< Device copy of North-South Beam gain values
include/source_components.h:  gpuUserComplex *d_Dxs = NULL; /*!< Device copy of North-South Beam leakage values
include/source_components.h:  gpuUserComplex *d_Dys = NULL; /*!< Device copy of East-West Beam leakage values
include/source_components.h:  gpuUserComplex *d_gys = NULL; /*!< Device copy of East-West Beam gain values
include/source_components.h:@return `visi`, a `gpuUserComplex` of the visibility
include/source_components.h:__device__ gpuUserComplex calc_measurement_equation(user_precision_t *d_us,
include/source_components.h:__device__ void apply_beam_gains_stokesIQUV(gpuUserComplex g1x, gpuUserComplex D1x,
include/source_components.h:          gpuUserComplex D1y, gpuUserComplex g1y,
include/source_components.h:          gpuUserComplex g2x, gpuUserComplex D2x,
include/source_components.h:          gpuUserComplex D2y, gpuUserComplex g2y,
include/source_components.h:          gpuUserComplex visi_component,
include/source_components.h:          gpuUserComplex * visi_XX, gpuUserComplex * visi_XY,
include/source_components.h:          gpuUserComplex * visi_YX, gpuUserComplex * visi_YY);
include/source_components.h:__device__ void apply_beam_gains_stokesI(gpuUserComplex g1x, gpuUserComplex D1x,
include/source_components.h:          gpuUserComplex D1y, gpuUserComplex g1y,
include/source_components.h:          gpuUserComplex g2x, gpuUserComplex D2x,
include/source_components.h:          gpuUserComplex D2y, gpuUserComplex g2y,
include/source_components.h:          gpuUserComplex visi_component,
include/source_components.h:          gpuUserComplex * visi_XX, gpuUserComplex * visi_XY,
include/source_components.h:          gpuUserComplex * visi_YX, gpuUserComplex * visi_YY);
include/source_components.h:           gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
include/source_components.h:           gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
include/source_components.h:           gpuUserComplex * g1x, gpuUserComplex * D1x,
include/source_components.h:           gpuUserComplex * D1y, gpuUserComplex * g1y,
include/source_components.h:           gpuUserComplex * g2x, gpuUserComplex * D2x,
include/source_components.h:           gpuUserComplex * D2y, gpuUserComplex * g2y);
include/source_components.h:           gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
include/source_components.h:           gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
include/source_components.h:           gpuUserComplex * g1x, gpuUserComplex * D1x,
include/source_components.h:           gpuUserComplex * D1y, gpuUserComplex * g1y,
include/source_components.h:           gpuUserComplex * g2x, gpuUserComplex * D2x,
include/source_components.h:           gpuUserComplex * D2y, gpuUserComplex * g2y);
include/source_components.h:    gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
include/source_components.h:    gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
include/source_components.h:    gpuUserComplex visi_component,
include/source_components.h:    gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
include/source_components.h:    gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
include/source_components.h:    gpuUserComplex visi_component,
include/source_components.h:    gpuMalloc( (void**)&d_components->extrap_stokesI, num_comps*num_freqs*sizeof(double) );
include/source_components.h:    gpuMalloc( (void**)&d_components->extrap_stokesQ, num_comps*num_freqs*sizeof(double) );
include/source_components.h:    gpuMalloc( (void**)&d_components->extrap_stokesU, num_comps*num_freqs*sizeof(double) );
include/source_components.h:    gpuMalloc( (void**)&d_components->extrap_stokesV, num_comps*num_freqs*sizeof(double) );
include/source_components.h:`chunk_sky_model::remap_source_for_gpu`. The remapping has a memory cost,
include/source_components.h:`chunk_sky_model::remap_source_for_gpu`
include/source_components.h:as filled by `source_components::copy_chunked_source_to_GPU`
include/source_components.h:void copy_components_to_GPU(source_t *chunked_source, source_t *d_chunked_source,
include/source_components.h:copied across to the GPU is (no empty pointers arrays are copied across)
include/source_components.h:source_t * copy_chunked_source_to_GPU(source_t *chunked_source);
include/source_components.h:   gpuFree d_components->extrap_stokesI );
include/source_components.h:   gpuFree d_components->extrap_stokesQ );
include/source_components.h:   gpuFree d_components->extrap_stokesU );
include/source_components.h:   gpuFree d_components->extrap_stokesV );
include/fundamental_coords.h:@brief Internally to WODEN C/CUDA, all cross-correlations are stored first,
CMakeLists.txt:##Need to find CUDA compiler
CMakeLists.txt:# check_language(CUDA)
CMakeLists.txt:option(USE_HIP "Enable the GPU acceleration for AMD GPUs." OFF)
CMakeLists.txt:##default to using CUDA to be somewhat backwards compatible
CMakeLists.txt:option(USE_CUDA "Enable the GPU acceleration for NVIDIA GPUs." ON)
CMakeLists.txt:  set(USE_CUDA OFF)
CMakeLists.txt:if(USE_CUDA)
CMakeLists.txt:    message("USING CUDA : USE_CUDA=ON")
CMakeLists.txt:    project(woden VERSION 2.3 LANGUAGES CXX C CUDA)
CMakeLists.txt:    set(CMAKE_CUDA_STANDARD 11)
CMakeLists.txt:    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 ")
CMakeLists.txt:    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__NVCC__ ")
CMakeLists.txt:    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
CMakeLists.txt:    if (NOT DEFINED ENV{CUDAARCHS})
CMakeLists.txt:        message("WARNING : CUDAARCHS not set, so CMAKE_CUDA_ARCHITECTURES has defaulted to whatever CMake has set, which is ${CMAKE_CUDA_ARCHITECTURES}")
CMakeLists.txt:        message("INFO : CUDAARCHS was set to $ENV{CUDAARCHS}, so now CMAKE_CUDA_ARCHITECTURES is set to ${CMAKE_CUDA_ARCHITECTURES}")
CMakeLists.txt:    ##TODO, remove this if -DGPU_TARGETS="gfx1032;gfx1035" works
CMakeLists.txt:file(GLOB GPUSOURCES "src/*.cpp")
CMakeLists.txt:if(USE_CUDA)
CMakeLists.txt:    set_source_files_properties( ${GPUSOURCES} PROPERTIES LANGUAGE CUDA)
CMakeLists.txt:    ##Generate library for the CUDA code
CMakeLists.txt:    add_library(wodenGPU_float SHARED ${GPUSOURCES})
CMakeLists.txt:    ##Compile CUDA code with all warnings
CMakeLists.txt:    if(USE_CUDA)
CMakeLists.txt:       target_compile_options(wodenGPU_float PRIVATE --compiler-options -Wall)
CMakeLists.txt:       target_compile_options(wodenGPU_float PRIVATE -Wall)
CMakeLists.txt:    ##Add the total WODEN library, linking in the CUDA code
CMakeLists.txt:    target_link_libraries(woden_float PUBLIC wodenGPU_float ${CC_LINKLIBS})
CMakeLists.txt:    ##Generate library for the CUDA code
CMakeLists.txt:    add_library(wodenGPU_double SHARED ${GPUSOURCES})
CMakeLists.txt:    ##Compile CUDA code with all warnings
CMakeLists.txt:    if(USE_CUDA)
CMakeLists.txt:       target_compile_options(wodenGPU_double PRIVATE -DDOUBLE_PRECISION --compiler-options -Wall)
CMakeLists.txt:        target_compile_options(wodenGPU_double PRIVATE -DDOUBLE_PRECISION -Wall)
CMakeLists.txt:    ##Add the total WODEN library, linking in the CUDA code
CMakeLists.txt:    target_link_libraries(woden_double PUBLIC wodenGPU_double ${CC_LINKLIBS})
CMakeLists.txt:    ##Generate library for the CUDA code
CMakeLists.txt:    add_library(wodenGPU_float SHARED ${GPUSOURCES})
CMakeLists.txt:    ##Compile CUDA code with all warnings
CMakeLists.txt:    if(USE_CUDA)
CMakeLists.txt:        target_compile_options(wodenGPU_float PRIVATE -G --compiler-options -g)
CMakeLists.txt:        target_compile_options(wodenGPU_float PRIVATE -G -g)
CMakeLists.txt:    ##Add the total WODEN library, linking in the CUDA code
CMakeLists.txt:    target_link_libraries(woden_float PUBLIC wodenGPU_float ${CC_LINKLIBS})
CMakeLists.txt:    ##Generate library for the CUDA code
CMakeLists.txt:    add_library(wodenGPU_double SHARED ${GPUSOURCES})
CMakeLists.txt:    ##Compile CUDA code with all warnings
CMakeLists.txt:    target_compile_options(wodenGPU_double PRIVATE -DDOUBLE_PRECISION
CMakeLists.txt:    if(USE_CUDA)
CMakeLists.txt:        target_compile_options(wodenGPU_double PRIVATE -DDOUBLE_PRECISION
CMakeLists.txt:        target_compile_options(wodenGPU_double PRIVATE -DDOUBLE_PRECISION -G -g)
CMakeLists.txt:    ##Add the total WODEN library, linking in the CUDA code
CMakeLists.txt:    target_link_libraries(woden_double PUBLIC wodenGPU_double ${CC_LINKLIBS})
scripts/run_woden.py:"""Wrapper script to run the GPU WODEN code. Author: J.L.B. Line
scripts/run_woden.py:    This function runs WODEN C/CUDA code on a separate thread, processing source catalogues from a queue until the queue is empty.
scripts/run_woden.py:        to the C/CUDA structs.
scripts/run_woden.py:    like reading in the sky model and running the GPU code in parallel; the
scripts/run_woden.py:        ##C/CUDA library and return the `run_woden` function
scripts/run_woden.py:        ### and running GPU code at the same time. Means we can limit the
scripts/run_woden.py:        ##setup a queue system to run out GPU code at the same time as the
docker/Dockerfile_cuda:#Base image including CUDA to write everything on to
docker/Dockerfile_cuda:FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
docker/Dockerfile_cuda:ARG CUDA_ARCH
docker/Dockerfile_cuda:ENV HYPERDRIVE_CUDA_COMPUTE=${CUDA_ARCH}
docker/Dockerfile_cuda:ENV CUDAARCHS=${CUDA_ARCH}
docker/Dockerfile_cuda:RUN echo "Building for CUDA_ARCH=$CUDA_ARCH"
docker/Dockerfile_cuda:RUN echo "HYPERDRIVE_CUDA_COMPUTE=$HYPERDRIVE_CUDA_COMPUTE"
docker/Dockerfile_cuda:RUN echo "CUDAARCHS=$CUDAARCHS"
docker/Dockerfile_cuda:  && /opt/cargo/bin/cargo build --release --features=cuda,cuda-static \
docker/Dockerfile_cuda:## clone the release verion of WODEN and complile the C/CUDA code
docker/Dockerfile_cuda:##that is user GPU dependent. But we can install them.
docker/make_docker_image.sh:##Build some CUDA versions. Could we just do this with a multi-arch flag?
docker/make_docker_image.sh:    docker build --progress=plain --build-arg="CUDA_ARCH=${arch}" -t jlbline/woden-2.3:cuda-$arch -f Dockerfile_cuda .
docker/make_docker_image.sh:    docker push jlbline/woden-2.3:cuda-$arch
docker/make_docker_image.sh:##of the CUDA stuff above if they can work out how to do it.
docker/run_docker.sh:# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
docker/run_docker.sh:docker run -it --gpus all woden-2.1 \
docker/Dockerfile_setonix:ARG ROCM_VER=5.7.3
docker/Dockerfile_setonix:# - rocm5.4.6-mpich3.4.3-ubuntu22
docker/Dockerfile_setonix:# - rocm5.6.0-mpich3.4.3-ubuntu22
docker/Dockerfile_setonix:# - rocm5.6.1-mpich3.4.3-ubuntu22
docker/Dockerfile_setonix:# - rocm5.7.3-mpich3.4.3-ubuntu22
docker/Dockerfile_setonix:# - rocm6.0.2-mpich3.4.3-ubuntu22
docker/Dockerfile_setonix:# - rocm6.1-mpich3.4.3-ubuntu22
docker/Dockerfile_setonix:#Base image ROCm
docker/Dockerfile_setonix:FROM quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu22
docker/Dockerfile_setonix:## clone the release verion of WODEN and complile the C/CUDA code
docker/Dockerfile_setonix:##that is user GPU dependent. But we can install them.
templates/install_woden_setonix.sh:# salloc --nodes=1 --partition=gpu --account=${PAWSEY_PROJECT}-gpu -t 00:30:00 --gres=gpu:1
templates/install_woden_setonix.sh:module load rocm/5.7.3
templates/install_woden_setonix.sh:git checkout compile_both_CUDA_and_HIP
templates/install_woden_nt.sh:module load cuda/12.0.0
templates/install_woden_nt.sh:export HYPERDRIVE_CUDA_COMPUTE=80
templates/install_woden_nt.sh:cargo build --locked --release --features=cuda,hdf5-static
templates/install_woden_nt.sh:cmake .. -DUSE_CUDA=ON \
templates/install_woden_nt.sh:# sinteractive --time 1:00:00 --nodes 1 --cpus-per-task 2 --mem=50gb --gres=gpu:1 --account=oz048
templates/install_woden_nt.sh:# module load cuda/12.0.0
.gitignore:cmake_testing/GPU_code/FEE_primary_beam_gpu/beam_plots
.gitignore:cmake_testing/GPU_code/FEE_primary_beam_gpu/hyperbeam*
.gitignore:cmake_testing/GPU_code/FEE_primary_beam_gpu/MWA_FEE_multifreq_gains*
.gitignore:cmake_testing/GPU_code/FEE_primary_beam_gpu/*.png
.gitignore:cmake_testing/GPU_code/primary_beam_gpu/__pycache__
.gitignore:cmake_testing/GPU_code/primary_beam_gpu/hyperbeam*
.gitignore:cmake_testing/GPU_code/primary_beam_gpu/old_delete
.gitignore:cmake_testing/GPU_code/source_components/plots
.gitignore:cmake_testing/GPU_code/source_components/*.png
.gitignore:cmake_testing/GPU_code/primary_beam_gpu/*.png
.gitignore:cmake_testing/GPU_code/primary_beam_gpu/plots
joss_paper/paper.bib:keywords = {GPU computing,astronomy data analysis,giant radio galaxies,reionisation},
joss_paper/paper.md:title: '`WODEN`: A CUDA-enabled package to simulate low-frequency radio interferometric data'
joss_paper/paper.md:  - CUDA
joss_paper/paper.md:The core functionality of `WODEN` is written in CUDA as interferometric simulations are computationally intensive but embarrassingly parallel. The performance of CUDA allows for large-scale simulations to be run including emission from all directions in the sky. This is paramount for interferometers with a wide field of view such as the Murchison Widefield Array [MWA, @Tingay2013]. A Python wrapper is used to take advantage of community packages such as [astropy](https://www.astropy.org/) [@astropy2013; @astropy2018] and [pyerfa](https://pypi.org/project/pyerfa/) [@pyerfa] and to present a user-friendly interface to `WODEN`. Those simulating MWA observations can use the MWA `metafits` file to quickly feed in observational parameters to `WODEN` to match real data.
joss_paper/paper.md:Under this discrete sky formalism, upwards of $j\ge25\times10^6$ components can be required to achieve the angular resolution required. Furthermore, $u,v,w$ are time and frequency dependent, so to sample in frequency of order 500 times and 100 samples in time, there are of order $10^{12}$ visibility calculations to make. This makes CUDA acceleration paramount.
joss_paper/paper.md:Alternative approaches to interferometric simulations exist, such as [pyuvsim](https://github.com/RadioAstronomySoftwareGroup/pyuvsim) [@Lanman2019], which sacrifices speed for excellent precision, and [RIMEz](https://github.com/upenneor/rimez), which decomposes the sky into spherical harmonics rather than discrete points. `WODEN` was designed with the Australian MWA Epoch of Reionisation (EoR) processing pipeline in mind, which uses a calibration and foreground removal software called the `RTS` [@Mitchell2008] in search of signals from the very first stars [see @Yoshiura2021 for a recent use of this pipeline]. The `RTS` creates a sky model using the same formalism above, however the code is not optimised enough to handle the volume of sources to simulate the entire sky. To test the `RTS` method of sky generation, we therefore needed a fast and discretised method. Another excellent CUDA accelerated simulation package, [OSKAR](https://github.com/OxfordSKA/OSKAR) [@OSKAR], addresses these two points. However, the `RTS` also generates parts of the sky model via shapelets [see @Line2020 for an overview], which `OSKAR` cannot. Furthermore, in real data, the precession/nutation of the Earth's rotational axis causes sources to move from the sky coordinates as specified in the RA, DEC J2000 coordinate system. The `RTS` is designed to undo this precession/nutation, and so a simulation fed into the `RTS` should *contain* precession. `WODEN` adds in this precession using the same method as the `RTS` to be consistent. This unique combination of CUDA, shapelet foregrounds, the MWA FEE primary beam, along with source precession, created the need for `WODEN`. These effects should not preclude other calibration packages from using `WODEN` outputs however, meaning `WODEN` is not limited to feeding data into the `RTS` alone.
joss_paper/paper.md:As 32 and 64 bit precision calculations are performed in physically different parts of an NVIDIA GPU, with cards typically having less double precision hardware that single, the `woden_double` version is slower that the `woden_float`. Each card will show a different slow-down between the two modes. As a test, I ran a simulation using a catalogue of over 300,000 sources. The number of sources above the horizon and the simulation settings used are listed in Table \ref{tab:benchmark_sim}, along with the speed difference between the `woden_float` and `woden_double` executables for two different NVIDIA GPU cards.
joss_paper/paper.md:Given this > 5 times slow down on a desktop card, having the option to toggle between `woden_float` and `woden_double` allows quick experimentation using `woden_float` and longer science-quality runs with `woden_double`. Luckily, for cards like the V100, the slowdown is around 1.3. Note that these simulations can easily be broken up and run across multiple GPUs if available, reducing the real time taken to complete the simulations.
joss_paper/paper.md:I acknowledge direct contributions from Tony Farlie (who taught me how pointer arithmetic works in `C`) and contributions from Bart Pindor and Daniel Mitchell (through their work in the `RTS` and through advising me on `CUDA`). I would like to thank Chris Jordan who acted as a sounding board as I learned `C` and `CUDA`. Finally, I would like to thank both Matthew Kolopanis and Paul La Plante for reviewing the code and giving useful suggestions on how to improve the code.
cmake_testing/C_code/visibility_set/CMakeLists.txt:  DEFINE_COMP_FLAGS(${PRECISION} C_FLAGS CUDA_FLAGS C_COVER_FLAGS)
cmake_testing/C_code/primary_beam/CMakeLists.txt:  DEFINE_COMP_FLAGS(${PRECISION} C_FLAGS CUDA_FLAGS C_COVER_FLAGS)
cmake_testing/wodenpy/skymodel/test_create_skymodel_chunk_map.py:        ##GPU memory TODO don't pray, submit a warning?
cmake_testing/wodenpy/skymodel/read_skymodel_common.py:        ##as in the GPU code, that index will reference where we stick extrapolated
cmake_testing/wodenpy/use_libwoden/CMakeLists.txt:  DEFINE_COMP_FLAGS(${PRECISION} C_FLAGS CUDA_FLAGS C_COVER_FLAGS)
cmake_testing/CMakeLists.txt:if(USE_CUDA)
cmake_testing/CMakeLists.txt:  ##Flags to make CUDA code compile in WODEN float or double modes
cmake_testing/CMakeLists.txt:  set(FLOAT_GPU_FLAGS -g -G --compiler-options -Wall )
cmake_testing/CMakeLists.txt:  set(DOUBLE_GPU_FLAGS -DDOUBLE_PRECISION -g -G --compiler-options -Wall -D__NVCC__)
cmake_testing/CMakeLists.txt:  ##Flags to make CUDA code compile in WODEN float or double modes
cmake_testing/CMakeLists.txt:  set(FLOAT_GPU_FLAGS -g -Wall )
cmake_testing/CMakeLists.txt:  set(DOUBLE_GPU_FLAGS -DDOUBLE_PRECISION -g -Wall -D__HIPCC__)
cmake_testing/CMakeLists.txt:function(DEFINE_COMP_FLAGS PRECISION C_FLAGS GPU_FLAGS C_COVER_FLAGS)
cmake_testing/CMakeLists.txt:    set(${GPU_FLAGS} ${FLOAT_GPU_FLAGS} PARENT_SCOPE)
cmake_testing/CMakeLists.txt:    set(${GPU_FLAGS} ${DOUBLE_GPU_FLAGS} PARENT_SCOPE)
cmake_testing/CMakeLists.txt:## Test CUDA code
cmake_testing/CMakeLists.txt:add_subdirectory(GPU_code)
cmake_testing/GPU_code/CMakeLists.txt:add_subdirectory(primary_beam_gpu)
cmake_testing/GPU_code/fundamental_coords/test_lmn_coords.c://External CUDA code we're linking in
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:  DEFINE_COMP_FLAGS(${PRECISION} C_FLAGS GPU_FLAGS C_COVER_FLAGS)
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:  if(USE_CUDA)
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:              PROPERTIES LANGUAGE CUDA)
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:  add_library(fundamental_coordsGPU_${PRECISION} SHARED
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:  target_compile_options(fundamental_coordsGPU_${PRECISION} PRIVATE ${GPU_FLAGS})
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:      fundamental_coordsGPU_${PRECISION}
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:  add_test(GPU_test_lmn_coords_${PRECISION} test_lmn_coords_${PRECISION}_app)
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:      fundamental_coordsGPU_${PRECISION}
cmake_testing/GPU_code/fundamental_coords/CMakeLists.txt:  add_test(GPU_test_uvw_coords_${PRECISION} test_uvw_coords_${PRECISION}_app)
cmake_testing/GPU_code/fundamental_coords/test_uvw_coords.c:CUDA code we are linking in
cmake_testing/GPU_code/fundamental_coords/test_uvw_coords.c:    //Run the CUDA code via fundamental_coords::test_kern_calc_uvw
cmake_testing/GPU_code/fundamental_coords/test_uvw_coords.c:    //Run the CUDA code via fundamental_coords::test_kern_calc_uvw
cmake_testing/GPU_code/fundamental_coords/test_uvw_coords.c:     //Run the CUDA code via fundamental_coords::test_kern_calc_uvw_shapelet
cmake_testing/GPU_code/source_components/test_kern_calc_autos_multiants.c://External CUDA code being linked in
cmake_testing/GPU_code/source_components/test_apply_beam_gains.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_apply_beam_gains.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_kern_calc_autos.c://External CUDA code being linked in
cmake_testing/GPU_code/source_components/test_update_sum_visis_multiants.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_update_sum_visis_multiants.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_extrap_stokes.py:    ctest_data = np.loadtxt("../../../build/cmake_testing/GPU_code/source_components/test_extrap_stokes.txt")
cmake_testing/GPU_code/source_components/test_extrap_stokes.py:        axs[list_ind].plot(extrap_freqs/1e+6, extrap_I, 'cs', ms=6, mfc='none', label='GPU extrap')
cmake_testing/GPU_code/source_components/test_kern_calc_visi_shape.c:  //Container for many arrays to feed the GPU
cmake_testing/GPU_code/source_components/CMakeLists.txt:  DEFINE_COMP_FLAGS(${PRECISION} C_FLAGS GPU_FLAGS C_COVER_FLAGS)
cmake_testing/GPU_code/source_components/CMakeLists.txt:  if(USE_CUDA)
cmake_testing/GPU_code/source_components/CMakeLists.txt:              "${CMAKE_SOURCE_DIR}/src/primary_beam_gpu.cpp" PROPERTIES LANGUAGE CUDA)
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_library(source_componentsGPU_${PRECISION} SHARED
cmake_testing/GPU_code/source_components/CMakeLists.txt:              "${CMAKE_SOURCE_DIR}/src/primary_beam_gpu.cpp")
cmake_testing/GPU_code/source_components/CMakeLists.txt:  target_compile_options(source_componentsGPU_${PRECISION} PRIVATE ${GPU_FLAGS})
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_extrap_stokes_${PRECISION} test_extrap_stokes_${PRECISION}_app)
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_calc_measurement_equation_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_apply_beam_gains_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_get_beam_gains_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_source_component_common_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_kern_calc_visi_point_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_kern_calc_visi_gauss_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_kern_calc_visi_shape_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_update_sum_visis_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_kern_calc_autos_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_get_beam_gains_two_antennas_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_update_sum_visis_multiants_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:      source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_source_component_common_multiants_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  source_componentsGPU_${PRECISION}
cmake_testing/GPU_code/source_components/CMakeLists.txt:  add_test(GPU_test_kern_calc_autos_multiants_${PRECISION}
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c:    status = new_gpu_fee_beam(beam_settings->fee_beam,
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c:                               &beam_settings->gpu_fee_beam);
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c:      handle_hyperbeam_error(__FILE__, __LINE__, "new_gpu_fee_beam");
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c:  //that copies components from CPU to GPU needs these extra fields defined
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c:    free_gpu_fee_beam(beam_settings->gpu_fee_beam);
cmake_testing/GPU_code/source_components/test_source_component_common_multiants.c:    free_gpu_fee_beam(beam_settings->gpu_fee_beam);
cmake_testing/GPU_code/source_components/test_calc_measurement_equation.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_calc_measurement_equation.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_calc_measurement_equation.c:      //Ensure visis are set to zero before going into CUDA code
cmake_testing/GPU_code/source_components/test_calc_measurement_equation.c:    //Run the CUDA code
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.c://Take input parameters and test whether GPU outputs match expectations
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.c:  //Container for many arrays to feed the GPU
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.c:  //Container for many arrays to feed the GPU
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.c:  //Container for many arrays to feed the GPU
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.c:  //Container for many arrays to feed the GPU
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.h://external CUDA code used for testng
cmake_testing/GPU_code/source_components/test_kern_calc_visi_common.h://Take input parameters and test whether GPU outputs match expectations
cmake_testing/GPU_code/source_components/test_get_beam_gains.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_get_beam_gains.c:  //Run the CUDA code and get some results
cmake_testing/GPU_code/source_components/common_testing_functions.h://using the CPU code to compare to the GPU
cmake_testing/GPU_code/source_components/test_extrap_stokes.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_extrap_stokes.c:  //copying over to the GPU
cmake_testing/GPU_code/source_components/test_extrap_stokes.c:  //The generic function that copies sky models from CPU to GPU needs
cmake_testing/GPU_code/source_components/test_extrap_stokes.c:  // //Run the CUDA code
cmake_testing/GPU_code/source_components/test_update_sum_visis.c:// #define __GPU__
cmake_testing/GPU_code/source_components/test_update_sum_visis.c://Depends on whether using NVIDIA or AMD, and if double float or double precision
cmake_testing/GPU_code/source_components/test_update_sum_visis.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_update_sum_visis.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_update_sum_visis.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_update_sum_visis.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_get_beam_gains_two_antennas.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_get_beam_gains_two_antennas.c:  // //Run the CUDA code
cmake_testing/GPU_code/source_components/plot_extrap_stokes.py:    ctest_data = np.loadtxt("../../../build/cmake_testing/GPU_code/source_components/test_extrap_stokes.txt")
cmake_testing/GPU_code/source_components/plot_extrap_stokes.py:        axs[0,0].loglog(extrap_freqs/1e+6, extrap_I, 'C1o', mfc='none', label='GPU Stokes I')
cmake_testing/GPU_code/source_components/plot_extrap_stokes.py:        axs[0,1].plot(extrap_freqs/1e+6, extrap_Q, 'C1o', mfc='none', label='GPU Stokes Q')
cmake_testing/GPU_code/source_components/plot_extrap_stokes.py:        axs[1,0].plot(extrap_freqs/1e+6, extrap_U, 'C1o', mfc='none', label='GPU Stokes U')
cmake_testing/GPU_code/source_components/plot_extrap_stokes.py:            axs[1,1].semilogx(extrap_freqs/1e+6, extrap_V, 'C1o', mfc='none', label='GPU Stokes V')
cmake_testing/GPU_code/source_components/plot_extrap_stokes.py:            axs[1,1].loglog(extrap_freqs/1e+6, extrap_V, 'C1o', mfc='none', label='GPU Stokes V')
cmake_testing/GPU_code/source_components/test_source_component_common.c://External CUDA code we're linking in
cmake_testing/GPU_code/source_components/test_source_component_common.c:    status = new_gpu_fee_beam(beam_settings->fee_beam,
cmake_testing/GPU_code/source_components/test_source_component_common.c:                               &beam_settings->gpu_fee_beam);
cmake_testing/GPU_code/source_components/test_source_component_common.c:      handle_hyperbeam_error(__FILE__, __LINE__, "new_gpu_fee_beam");
cmake_testing/GPU_code/source_components/test_source_component_common.c:  //that copies components from CPU to GPU needs these extra fields defined
cmake_testing/GPU_code/source_components/test_source_component_common.c:  //Run the CUDA code
cmake_testing/GPU_code/source_components/test_source_component_common.c:    free_gpu_fee_beam(beam_settings->gpu_fee_beam);
cmake_testing/GPU_code/source_components/test_source_component_common.c:    free_gpu_fee_beam(beam_settings->gpu_fee_beam);
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_nobeam.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_nobeam.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_multibeams.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_multibeams.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_multibeams.c://External CUDA code we're linking in
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwaanalybeam.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwaanalybeam.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwaanalybeam.c://External CUDA code we're linking in
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    DEFINE_COMP_FLAGS(${PRECISION} C_FLAGS GPU_FLAGS C_COVER_FLAGS)
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    if(USE_CUDA)
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:              "${CMAKE_SOURCE_DIR}/src/primary_beam_gpu.cpp" PROPERTIES LANGUAGE CUDA)
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    ##First add necessary CUDA code to a library
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_library(calculate_visibilitiesGPU_${PRECISION} SHARED
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:                "${CMAKE_SOURCE_DIR}/src/primary_beam_gpu.cpp")
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    target_compile_options(calculate_visibilitiesGPU_${PRECISION} PRIVATE ${GPU_FLAGS})
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:        calculate_visibilitiesGPU_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_test(GPU_test_calculate_visibilities_nobeam_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:        calculate_visibilitiesGPU_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_test(GPU_test_calculate_visibilities_gaussbeam_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:        calculate_visibilitiesGPU_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_test(GPU_test_calculate_visibilities_edabeam_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:        calculate_visibilitiesGPU_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_test(GPU_test_calculate_visibilities_mwafeebeam_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:        calculate_visibilitiesGPU_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_test(GPU_test_calculate_visibilities_mwafeebeaminterp_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:        calculate_visibilitiesGPU_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_test(GPU_test_calculate_visibilities_mwaanalybeam_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    calculate_visibilitiesGPU_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/CMakeLists.txt:    add_test(GPU_test_calculate_visibilities_multibeams_${PRECISION}
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_edabeam.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_edabeam.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_common.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_common.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_common.c:// //External CUDA code we're linking in
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_common.c:  printf("Calling GPU\n");
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_common.c:  printf("GPU has finished\n");
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_common.h://External CUDA code we're linking in
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_gaussbeam.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_gaussbeam.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwafeebeam.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwafeebeam.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwafeebeam.c://External CUDA code we're linking in
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwafeebeaminterp.c:to all CUDA functionality in WODEN. We'll test here in one baseline, frequency,
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwafeebeaminterp.c:in different test suites, so really just test that the correct CUDA functions
cmake_testing/GPU_code/calculate_visibilities/test_calculate_visibilities_mwafeebeaminterp.c://External CUDA code we're linking in
cmake_testing/GPU_code/primary_beam_gpu/test_MWA_analytic.c://External CUDA code we're linking in
cmake_testing/GPU_code/primary_beam_gpu/test_MWA_analytic.c:  //Run the CUDA code
cmake_testing/GPU_code/primary_beam_gpu/test_MWA_analytic.c:  //Run the CUDA code
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:// //External CUDA code we're linking in
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:extern void test_run_hyperbeam_gpu(int num_components,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:           struct FEEBeamGpu *gpu_fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:  struct FEEBeamGpu *gpu_fee_beam;
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:  status = new_gpu_fee_beam(fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:                             &gpu_fee_beam);
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:    handle_hyperbeam_error(__FILE__, __LINE__, "new_gpu_fee_beam");
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:  test_run_hyperbeam_gpu(num_components,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:             gpu_fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_multi_antennas.c:  free_gpu_fee_beam(gpu_fee_beam);
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  DEFINE_COMP_FLAGS(${PRECISION} C_FLAGS GPU_FLAGS C_COVER_FLAGS)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  if(USE_CUDA)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:              "${CMAKE_SOURCE_DIR}/src/primary_beam_gpu.cpp" PROPERTIES LANGUAGE CUDA)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  add_library(primary_beam_gpuGPU_${PRECISION} SHARED
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:              "${CMAKE_SOURCE_DIR}/src/primary_beam_gpu.cpp"
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  target_compile_options(primary_beam_gpuGPU_${PRECISION} PRIVATE ${GPU_FLAGS})
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:      primary_beam_gpuGPU_${PRECISION}
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  add_test(GPU_test_gaussian_beam_${PRECISION} test_gaussian_beam_${PRECISION}_app)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:      primary_beam_gpuGPU_${PRECISION}
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  add_test(GPU_test_analytic_dipole_beam_${PRECISION} test_analytic_dipole_beam_${PRECISION}_app)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:      primary_beam_gpuGPU_${PRECISION}
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  add_test(GPU_test_MWA_analytic_${PRECISION} test_MWA_analytic_${PRECISION}_app)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:      primary_beam_gpuGPU_${PRECISION}
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  add_test(GPU_test_run_hyperbeam_${PRECISION} test_run_hyperbeam_${PRECISION}_app)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:      primary_beam_gpuGPU_${PRECISION}
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:  add_test(GPU_test_run_hyperbeam_interp_${PRECISION} test_run_hyperbeam_interp_${PRECISION}_app)
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:        primary_beam_gpuGPU_${PRECISION}
cmake_testing/GPU_code/primary_beam_gpu/CMakeLists.txt:    add_test(GPU_test_run_hyperbeam_multi_antennas_${PRECISION} test_run_hyperbeam_multi_antennas_${PRECISION}_app)
cmake_testing/GPU_code/primary_beam_gpu/make_some_beam_plots.py:            filename = f"../../../build/cmake_testing/CUDA_code/primary_beam_cuda/hyperbeam_{delay_name}_{int(freq/1e+6)}_two_ants_rot_double.txt"
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:// //External CUDA code we're linking in
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:extern void test_run_hyperbeam_gpu(int num_components,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:           struct FEEBeamGpu *gpu_fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:  struct FEEBeamGpu *gpu_fee_beam;
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:  status = new_gpu_fee_beam(fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:                             &gpu_fee_beam);
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:    handle_hyperbeam_error(__FILE__, __LINE__, "new_gpu_fee_beam");
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:  test_run_hyperbeam_gpu(num_components,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:             gpu_fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam.c:  free_gpu_fee_beam(gpu_fee_beam);
cmake_testing/GPU_code/primary_beam_gpu/test_analytic_dipole_beam.c://External CUDA code we're linking in
cmake_testing/GPU_code/primary_beam_gpu/test_analytic_dipole_beam.c:  //Run the CUDA code
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:// //External CUDA code we're linking in
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:extern void test_run_hyperbeam_gpu(int num_components,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:           struct FEEBeamGpu *gpu_fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:  struct FEEBeamGpu *gpu_fee_beam;
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:  status = new_gpu_fee_beam(fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:                          &gpu_fee_beam);
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:    handle_hyperbeam_error(__FILE__, __LINE__, "new_gpu_fee_beam");
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:  test_run_hyperbeam_gpu(NUM_COMPS,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:             gpu_fee_beam,
cmake_testing/GPU_code/primary_beam_gpu/test_run_hyperbeam_interp.c:  free_gpu_fee_beam(gpu_fee_beam);
cmake_testing/GPU_code/primary_beam_gpu/comp_MWA_analytic_to_FEE.c://External CUDA code we're linking in
cmake_testing/GPU_code/primary_beam_gpu/comp_MWA_analytic_to_FEE.c:extern void test_RTS_CUDA_FEE_beam(int num_components,
cmake_testing/GPU_code/primary_beam_gpu/comp_MWA_analytic_to_FEE.c:    test_RTS_CUDA_FEE_beam(num_azza,
cmake_testing/GPU_code/primary_beam_gpu/comp_MWA_analytic_to_FEE.c:    test_RTS_CUDA_FEE_beam(num_azza,
cmake_testing/GPU_code/primary_beam_gpu/test_gaussian_beam.c://External CUDA code we're linking in
src/woden.c://Main GPU executable to link in
src/woden.c:    //Launch the GPU code
src/woden.c:    printf("GPU calls for band %d finished\n",band_num );
src/primary_beam_gpu.cpp:#include "primary_beam_gpu.h"
src/primary_beam_gpu.cpp:#include "gpu_macros.h"
src/primary_beam_gpu.cpp:           gpuUserComplex *d_g1xs, gpuUserComplex *d_g1ys) {
src/primary_beam_gpu.cpp:    d_g1xs[beam_ind] = make_gpuUserComplex(d_beam_real, d_beam_imag);
src/primary_beam_gpu.cpp:    d_g1ys[beam_ind] = make_gpuUserComplex(d_beam_real, d_beam_imag);
src/primary_beam_gpu.cpp:           gpuUserComplex *d_g1xs, gpuUserComplex *d_g1ys){
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_has, num_beam_hadec*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy( d_beam_has, beam_has,
src/primary_beam_gpu.cpp:                      num_beam_hadec*sizeof(double), gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_decs, num_beam_hadec*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy( d_beam_decs, beam_decs,
src/primary_beam_gpu.cpp:                      num_beam_hadec*sizeof(double), gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_ls, num_beam_hadec*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_ms, num_beam_hadec*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_ns, num_beam_hadec*sizeof(double)) );
src/primary_beam_gpu.cpp:  gpuErrorCheckKernel("kern_calc_lmn", kern_calc_lmn, grid, threads,
src/primary_beam_gpu.cpp:  gpuErrorCheckKernel("kern_gaussian_beam",kern_gaussian_beam, grid, threads,
src/primary_beam_gpu.cpp:  gpuFree( d_beam_ns );
src/primary_beam_gpu.cpp:  gpuFree( d_beam_ms );
src/primary_beam_gpu.cpp:  gpuFree( d_beam_ls );
src/primary_beam_gpu.cpp:  gpuFree( d_beam_decs );
src/primary_beam_gpu.cpp:  gpuFree( d_beam_has );
src/primary_beam_gpu.cpp:           gpuUserComplex * d_beam_X, gpuUserComplex * d_beam_Y) {
src/primary_beam_gpu.cpp:  gpuUserComplex tempX;
src/primary_beam_gpu.cpp:  gpuUserComplex tempY;
src/primary_beam_gpu.cpp:           gpuUserComplex *d_g1xs, gpuUserComplex *d_g1ys) {
src/primary_beam_gpu.cpp:    gpuUserComplex d_beam_X, d_beam_Y;
src/primary_beam_gpu.cpp:    gpuUserComplex d_beam_norm_X, d_beam_norm_Y;
src/primary_beam_gpu.cpp:    gpuUserComplex normed_X = d_beam_X;
src/primary_beam_gpu.cpp:    gpuUserComplex normed_Y = d_beam_Y;
src/primary_beam_gpu.cpp:     gpuUserComplex *d_g1xs, gpuUserComplex *d_g1ys){
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_azs, num_beam_azza*sizeof(user_precision_t)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_azs, azs, num_beam_azza*sizeof(user_precision_t),
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_zas, num_beam_azza*sizeof(user_precision_t)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_zas, zas, num_beam_azza*sizeof(user_precision_t),
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  gpuErrorCheckKernel("kern_analytic_dipole_beam",
src/primary_beam_gpu.cpp:  ( gpuFree(d_azs) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_zas) );
src/primary_beam_gpu.cpp:           gpuUserComplex * gx, gpuUserComplex * Dx,
src/primary_beam_gpu.cpp:           gpuUserComplex * Dy, gpuUserComplex * gy) {
src/primary_beam_gpu.cpp:  gpuUserComplex x_dip;
src/primary_beam_gpu.cpp:  gpuUserComplex y_dip;
src/primary_beam_gpu.cpp:  gpuUserComplex gx_dip = {0.0, 0.0};
src/primary_beam_gpu.cpp:  gpuUserComplex Dx_dip = {0.0, 0.0};
src/primary_beam_gpu.cpp:  gpuUserComplex Dy_dip = {0.0, 0.0};
src/primary_beam_gpu.cpp:  gpuUserComplex gy_dip = {0.0, 0.0};
src/primary_beam_gpu.cpp:  gpuUserComplex pgx = gx_dip * rot0 * ground_plane_div_dipoles;
src/primary_beam_gpu.cpp:  gpuUserComplex pDx = Dx_dip * rot1 * ground_plane_div_dipoles;
src/primary_beam_gpu.cpp:  gpuUserComplex pDy = Dy_dip * rot2 * ground_plane_div_dipoles;
src/primary_beam_gpu.cpp:  gpuUserComplex pgy = gy_dip * rot3 * ground_plane_div_dipoles;
src/primary_beam_gpu.cpp:           gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_Dys, gpuUserComplex *d_gys) {
src/primary_beam_gpu.cpp:    gpuUserComplex gx;
src/primary_beam_gpu.cpp:    gpuUserComplex Dx;
src/primary_beam_gpu.cpp:    gpuUserComplex Dy;
src/primary_beam_gpu.cpp:    gpuUserComplex gy;
src/primary_beam_gpu.cpp:     gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
src/primary_beam_gpu.cpp:     gpuUserComplex *d_Dys, gpuUserComplex *d_gys){
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_azs, num_coords*sizeof(user_precision_t)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_azs, azs, num_coords*sizeof(user_precision_t),
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_zas, num_coords*sizeof(user_precision_t)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_zas, zas, num_coords*sizeof(user_precision_t),
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_has,
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_beam_has, beam_has,
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_decs,
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_beam_decs, beam_decs,
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  //Copy over to the GPU
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_metre_delays, NUM_DIPOLES*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_metre_delays, metre_delays, NUM_DIPOLES*sizeof(double),
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:  gpuErrorCheckKernel("kern_RTS_analytic_MWA_beam",
src/primary_beam_gpu.cpp:  ( gpuFree(d_azs) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_zas) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_metre_delays) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_beam_has) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_beam_decs) );
src/primary_beam_gpu.cpp:           gpuUserComplex *d_gxs,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_Dxs,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_Dys,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_gys) {
src/primary_beam_gpu.cpp:    gpuUserComplex d_beam_J00;
src/primary_beam_gpu.cpp:    gpuUserComplex d_beam_J01;
src/primary_beam_gpu.cpp:    gpuUserComplex d_beam_J10;
src/primary_beam_gpu.cpp:    gpuUserComplex d_beam_J11;
src/primary_beam_gpu.cpp:extern "C" void run_hyperbeam_gpu(int num_components,
src/primary_beam_gpu.cpp:           struct FEEBeamGpu *gpu_fee_beam,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_gxs,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_Dxs,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_Dys,
src/primary_beam_gpu.cpp:           gpuUserComplex *d_gys){
src/primary_beam_gpu.cpp:    ( gpuMalloc( (void**)&(d_jones),
src/primary_beam_gpu.cpp:    ( gpuMalloc( (void**)&(d_jones),
src/primary_beam_gpu.cpp:  int num_unique_fee_freqs = get_num_unique_fee_freqs(gpu_fee_beam);
src/primary_beam_gpu.cpp:  // int num_unique_fee_tiles = get_num_unique_fee_tiles(gpu_fee_beam);
src/primary_beam_gpu.cpp:  tile_map = get_fee_tile_map(gpu_fee_beam);
src/primary_beam_gpu.cpp:  freq_map = get_fee_freq_map(gpu_fee_beam);
src/primary_beam_gpu.cpp:  //Copy the tile and freq maps to the GPU
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&(d_tile_map),
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&(d_freq_map),
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_tile_map, tile_map,
src/primary_beam_gpu.cpp:            num_beams*sizeof(int32_t), gpuMemcpyHostToDevice ) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_freq_map, freq_map,
src/primary_beam_gpu.cpp:            num_freqs*sizeof(int32_t), gpuMemcpyHostToDevice ) );
src/primary_beam_gpu.cpp:      status = fee_calc_jones_gpu_device(gpu_fee_beam,
src/primary_beam_gpu.cpp:      gpuErrorCheckKernel("kern_map_hyperbeam_gains",
src/primary_beam_gpu.cpp:    status = fee_calc_jones_gpu_device(gpu_fee_beam,
src/primary_beam_gpu.cpp:      gpuErrorCheckKernel("kern_map_hyperbeam_gains",
src/primary_beam_gpu.cpp:    handle_hyperbeam_error(__FILE__, __LINE__, "fee_calc_jones_gpu_device");
src/primary_beam_gpu.cpp:    // printf("Something went wrong running fee_calc_jones_gpu_device\n");
src/primary_beam_gpu.cpp:  ( gpuFree(d_jones) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_gxs,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_Dxs,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_Dys,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_gys,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_freqs, num_freqs*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_freqs, freqs, num_freqs*sizeof(double),
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:       (gpuUserComplex*)d_gxs, (gpuUserComplex*)d_Dxs,
src/primary_beam_gpu.cpp:       (gpuUserComplex*)d_Dys, (gpuUserComplex*)d_gys);
src/primary_beam_gpu.cpp:  ( gpuMemcpy(gxs, d_gxs,
src/primary_beam_gpu.cpp:       gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(Dxs, d_Dxs,
src/primary_beam_gpu.cpp:       gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(Dys, d_Dys,
src/primary_beam_gpu.cpp:       gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(gys, d_gys,
src/primary_beam_gpu.cpp:       gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_gxs) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_Dxs) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_Dys) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_gys) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_freqs) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_analy_beam_X,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_analy_beam_Y,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_freqs, num_freqs*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_freqs, freqs, num_freqs*sizeof(double),
src/primary_beam_gpu.cpp:                      gpuMemcpyHostToDevice) );
src/primary_beam_gpu.cpp:      (gpuUserComplex *)d_analy_beam_X, (gpuUserComplex *)d_analy_beam_Y);
src/primary_beam_gpu.cpp:  ( gpuMemcpy(analy_beam_X, d_analy_beam_X,
src/primary_beam_gpu.cpp:             gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(analy_beam_Y, d_analy_beam_Y,
src/primary_beam_gpu.cpp:             gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_analy_beam_X) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_analy_beam_Y) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_freqs) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_ls, num_beam_hadec*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_beam_ls, beam_ls,
src/primary_beam_gpu.cpp:                           num_beam_hadec*sizeof(double), gpuMemcpyHostToDevice ) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_beam_ms, num_beam_hadec*sizeof(double)) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_beam_ms, beam_ms,
src/primary_beam_gpu.cpp:                           num_beam_hadec*sizeof(double), gpuMemcpyHostToDevice ) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_freqs, num_freqs*sizeof(double) ) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_freqs, freqs,
src/primary_beam_gpu.cpp:                           num_freqs*sizeof(double), gpuMemcpyHostToDevice ) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_g1xs,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_g1ys,
src/primary_beam_gpu.cpp:  gpuErrorCheckKernel("kern_gaussian_beam",
src/primary_beam_gpu.cpp:                        (gpuUserComplex *)d_g1xs,
src/primary_beam_gpu.cpp:                        (gpuUserComplex *)d_g1ys);
src/primary_beam_gpu.cpp:  ( gpuMemcpy(primay_beam_J00, d_g1xs,
src/primary_beam_gpu.cpp:                 num_freqs*num_beam_hadec*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(primay_beam_J11, d_g1ys,
src/primary_beam_gpu.cpp:                 num_freqs*num_beam_hadec*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_beam_ls ) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_beam_ms ) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_freqs ) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_g1xs ) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_g1ys ) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_freqs, num_freqs*sizeof(double) ) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(d_freqs, freqs,
src/primary_beam_gpu.cpp:                           num_freqs*sizeof(double), gpuMemcpyHostToDevice ) );
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_g1xs,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_g1ys,
src/primary_beam_gpu.cpp:                         (gpuUserComplex *)d_g1xs,
src/primary_beam_gpu.cpp:                         (gpuUserComplex *)d_g1ys);
src/primary_beam_gpu.cpp:  ( gpuMemcpy(primay_beam_J00, d_g1xs,
src/primary_beam_gpu.cpp:                 num_freqs*num_beam_hadec*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy(primay_beam_J11, d_g1ys,
src/primary_beam_gpu.cpp:                 num_freqs*num_beam_hadec*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_freqs ) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_g1xs ) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_g1ys ) );
src/primary_beam_gpu.cpp:extern "C" void test_run_hyperbeam_gpu(int num_components,
src/primary_beam_gpu.cpp:           struct FEEBeamGpu *gpu_fee_beam,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_gxs,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_Dxs,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_Dys,
src/primary_beam_gpu.cpp:  ( gpuMalloc( (void**)&d_gys,
src/primary_beam_gpu.cpp:  run_hyperbeam_gpu(num_components,
src/primary_beam_gpu.cpp:             gpu_fee_beam,
src/primary_beam_gpu.cpp:             (gpuUserComplex *)d_gxs,
src/primary_beam_gpu.cpp:             (gpuUserComplex *)d_Dxs,
src/primary_beam_gpu.cpp:             (gpuUserComplex *)d_Dys,
src/primary_beam_gpu.cpp:             (gpuUserComplex *)d_gys);
src/primary_beam_gpu.cpp:  ( gpuMemcpy( primay_beam_J00, d_gxs,
src/primary_beam_gpu.cpp:                      gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy( primay_beam_J01, d_Dxs,
src/primary_beam_gpu.cpp:                      gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy( primay_beam_J10, d_Dys,
src/primary_beam_gpu.cpp:                      gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuMemcpy( primay_beam_J11, d_gys,
src/primary_beam_gpu.cpp:                      gpuMemcpyDeviceToHost) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_gxs) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_Dxs) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_Dys) );
src/primary_beam_gpu.cpp:  ( gpuFree(d_gys) );
src/fundamental_coords.cpp:#include "gpu_macros.h"
src/fundamental_coords.cpp:    //TODO do the sin/cos outside of the GPU kernel?
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_ls, num_coords*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_ms, num_coords*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_ns, num_coords*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_ras, num_coords*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMemcpy(d_ras, ras,
src/fundamental_coords.cpp:                           num_coords*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_decs, num_coords*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMemcpy(d_decs, decs,
src/fundamental_coords.cpp:                           num_coords*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  gpuErrorCheckKernel("kern_calc_lmn",
src/fundamental_coords.cpp:  ( gpuMemcpy(ls, d_ls,
src/fundamental_coords.cpp:                             num_coords*sizeof(double),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(ms, d_ms,
src/fundamental_coords.cpp:                             num_coords*sizeof(double),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(ns, d_ns,
src/fundamental_coords.cpp:                             num_coords*sizeof(double),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuFree(d_ls) );
src/fundamental_coords.cpp:  ( gpuFree(d_ms) );
src/fundamental_coords.cpp:  ( gpuFree(d_ns) );
src/fundamental_coords.cpp:  ( gpuFree(d_ras) );
src/fundamental_coords.cpp:  ( gpuFree(d_decs) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_X_diff,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_X_diff, X_diff,
src/fundamental_coords.cpp:    num_times*num_baselines*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_Y_diff,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_Y_diff, Y_diff,
src/fundamental_coords.cpp:    num_times*num_baselines*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_Z_diff,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_Z_diff, Z_diff,
src/fundamental_coords.cpp:    num_times*num_baselines*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_sha0s,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_sha0s, sha0s,
src/fundamental_coords.cpp:                 num_cross*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_cha0s,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_cha0s, cha0s,
src/fundamental_coords.cpp:                 num_cross*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_wavelengths,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_wavelengths, wavelengths,
src/fundamental_coords.cpp:                 num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_u_metres, num_cross*sizeof(user_precision_t) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_v_metres, num_cross*sizeof(user_precision_t) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_w_metres, num_cross*sizeof(user_precision_t) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_us, num_cross*sizeof(user_precision_t) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_vs, num_cross*sizeof(user_precision_t) ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_ws, num_cross*sizeof(user_precision_t) ) );
src/fundamental_coords.cpp:  gpuErrorCheckKernel("kern_calc_uvw",
src/fundamental_coords.cpp:  ( gpuMemcpy(us, d_us,
src/fundamental_coords.cpp:                   num_cross*sizeof(user_precision_t),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(vs, d_vs,
src/fundamental_coords.cpp:                   num_cross*sizeof(user_precision_t),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(ws, d_ws,
src/fundamental_coords.cpp:                   num_cross*sizeof(user_precision_t),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(u_metres, d_u_metres,
src/fundamental_coords.cpp:                   num_cross*sizeof(user_precision_t),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(v_metres, d_v_metres,
src/fundamental_coords.cpp:                   num_cross*sizeof(user_precision_t),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(w_metres, d_w_metres,
src/fundamental_coords.cpp:                   num_cross*sizeof(user_precision_t),gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuFree(d_us) );
src/fundamental_coords.cpp:  ( gpuFree(d_vs) );
src/fundamental_coords.cpp:  ( gpuFree(d_ws) );
src/fundamental_coords.cpp:  ( gpuFree(d_u_metres) );
src/fundamental_coords.cpp:  ( gpuFree(d_v_metres) );
src/fundamental_coords.cpp:  ( gpuFree(d_w_metres) );
src/fundamental_coords.cpp:  ( gpuFree(d_sha0s) );
src/fundamental_coords.cpp:  ( gpuFree(d_cha0s) );
src/fundamental_coords.cpp:  ( gpuFree(d_X_diff) );
src/fundamental_coords.cpp:  ( gpuFree(d_Y_diff) );
src/fundamental_coords.cpp:  ( gpuFree(d_Z_diff) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_X_diff,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_X_diff, X_diff,
src/fundamental_coords.cpp:            num_times*num_baselines*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_Y_diff,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_Y_diff, Y_diff,
src/fundamental_coords.cpp:            num_times*num_baselines*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_Z_diff,
src/fundamental_coords.cpp:  ( gpuMemcpy( d_Z_diff, Z_diff,
src/fundamental_coords.cpp:            num_times*num_baselines*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_lsts, num_times*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMemcpy( d_lsts, lsts,
src/fundamental_coords.cpp:                      num_times*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_ras, num_shapes*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMemcpy( d_ras, ras,
src/fundamental_coords.cpp:                      num_shapes*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_decs, num_shapes*sizeof(double) ) );
src/fundamental_coords.cpp:  ( gpuMemcpy( d_decs, decs,
src/fundamental_coords.cpp:                      num_shapes*sizeof(double), gpuMemcpyHostToDevice ) );
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_u_shapes,
src/fundamental_coords.cpp:  ( gpuMalloc( (void**)&d_v_shapes,
src/fundamental_coords.cpp:  gpuErrorCheckKernel("kern_calc_uv_shapelet",
src/fundamental_coords.cpp:  ( gpuMemcpy(u_shapes, d_u_shapes,
src/fundamental_coords.cpp:                                                    gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuMemcpy(v_shapes, d_v_shapes,
src/fundamental_coords.cpp:                                                    gpuMemcpyDeviceToHost) );
src/fundamental_coords.cpp:  ( gpuFree(d_u_shapes) );
src/fundamental_coords.cpp:  ( gpuFree(d_v_shapes) );
src/fundamental_coords.cpp:  ( gpuFree(d_lsts) );
src/fundamental_coords.cpp:  ( gpuFree(d_ras) );
src/fundamental_coords.cpp:  ( gpuFree(d_decs) );
src/fundamental_coords.cpp:  ( gpuFree(d_X_diff) );
src/fundamental_coords.cpp:  ( gpuFree(d_Y_diff) );
src/fundamental_coords.cpp:  ( gpuFree(d_Z_diff) );
src/visibility_set.c:  //For easy indexing when running on GPUs, make 4 arrays that match
src/calculate_visibilities.cpp:#include "gpucomplex.h"
src/calculate_visibilities.cpp:#include "primary_beam_gpu.h"
src/calculate_visibilities.cpp:#include "gpu_macros.h"
src/calculate_visibilities.cpp:   gpuMalloc( (void**)&d_X_diff,
src/calculate_visibilities.cpp:   gpuMemcpy( d_X_diff, array_layout->X_diff_metres,
src/calculate_visibilities.cpp:        num_time_steps*num_baselines*sizeof(double), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:   gpuMalloc( (void**)&d_Y_diff,
src/calculate_visibilities.cpp:   gpuMemcpy( d_Y_diff, array_layout->Y_diff_metres,
src/calculate_visibilities.cpp:        num_time_steps*num_baselines*sizeof(double), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:   gpuMalloc( (void**)&d_Z_diff,
src/calculate_visibilities.cpp:   gpuMemcpy( d_Z_diff, array_layout->Z_diff_metres,
src/calculate_visibilities.cpp:        num_time_steps*num_baselines*sizeof(double), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:  gpuMalloc( (void**)&d_allsteps_sha0s, num_cross*sizeof(double) );
src/calculate_visibilities.cpp:  gpuMemcpy( d_allsteps_sha0s, visibility_set->allsteps_sha0s,
src/calculate_visibilities.cpp:                      num_cross*sizeof(double), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:  gpuMalloc( (void**)&d_allsteps_cha0s, num_cross*sizeof(double) );
src/calculate_visibilities.cpp:  gpuMemcpy( d_allsteps_cha0s, visibility_set->allsteps_cha0s,
src/calculate_visibilities.cpp:                      num_cross*sizeof(double), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:  gpuMalloc( (void**)&d_allsteps_wavelengths,
src/calculate_visibilities.cpp:  gpuMemcpy( d_allsteps_wavelengths, visibility_set->allsteps_wavelengths,
src/calculate_visibilities.cpp:                      num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_u_metres, num_visis*sizeof(user_precision_t) ) );
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_v_metres, num_visis*sizeof(user_precision_t) ) );
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_w_metres, num_visis*sizeof(user_precision_t) ) );
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_us, num_visis*sizeof(user_precision_t) ) );
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_vs, num_visis*sizeof(user_precision_t) ) );
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_ws, num_visis*sizeof(user_precision_t) ) );
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_XX_real,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_XX_imag,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_XY_real,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_XY_imag,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_YX_real,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_YX_imag,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_YY_real,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_visibility_set->sum_visi_YY_imag,
src/calculate_visibilities.cpp:  ( gpuMalloc( (void**)&d_freqs, num_freqs*sizeof(double) ) );
src/calculate_visibilities.cpp:  ( gpuMemcpy( d_freqs, visibility_set->channel_frequencies,
src/calculate_visibilities.cpp:                      num_freqs*sizeof(double), gpuMemcpyHostToDevice ) );
src/calculate_visibilities.cpp:  //into GPU memory
src/calculate_visibilities.cpp:    ( gpuMalloc( (void**)&(d_sbf), sbf_N*sbf_L*sizeof(user_precision_t) ));
src/calculate_visibilities.cpp:    ( gpuMemcpy( d_sbf, sbf, sbf_N*sbf_L*sizeof(user_precision_t),
src/calculate_visibilities.cpp:                        gpuMemcpyHostToDevice ));
src/calculate_visibilities.cpp:    int32_t status = new_gpu_fee_beam(beam_settings->fee_beam,
src/calculate_visibilities.cpp:                            &beam_settings->gpu_fee_beam);
src/calculate_visibilities.cpp:      // handle_hyperbeam_error(__FILE__, __LINE__, "new_gpu_fee_beam");
src/calculate_visibilities.cpp:      printf("Something went wrong launching new_gpu_fee_beam\n");
src/calculate_visibilities.cpp:    gpuMalloc( (void**)&d_ant1_to_baseline_map,
src/calculate_visibilities.cpp:    gpuMalloc( (void**)&d_ant2_to_baseline_map,
src/calculate_visibilities.cpp:    printf("About to copy the chunked source to the GPU\n");
src/calculate_visibilities.cpp:    source_t *d_chunked_source = copy_chunked_source_to_GPU(source);
src/calculate_visibilities.cpp:    printf("Have copied across the chunk to the GPU\n");
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_XX_real,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_XX_imag,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_XY_real,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_XY_imag,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_YX_real,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_YX_imag,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_YY_real,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuMemcpy(d_visibility_set->sum_visi_YY_imag,
src/calculate_visibilities.cpp:               num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/calculate_visibilities.cpp:    gpuErrorCheckKernel("kern_calc_uvw",
src/calculate_visibilities.cpp:      gpuErrorCheckKernel("set_auto_uvw_to_zero",
src/calculate_visibilities.cpp:      gpuErrorCheckKernel("set_auto_uvw_to_zero",
src/calculate_visibilities.cpp:      gpuErrorCheckKernel("kern_calc_visi_point_or_gauss",
src/calculate_visibilities.cpp:      gpuErrorCheckKernel("kern_calc_visi_point_or_gauss",
src/calculate_visibilities.cpp:      ( gpuMalloc( (void**)&(d_lsts),
src/calculate_visibilities.cpp:      ( gpuMemcpy( d_lsts, woden_settings->lsts,
src/calculate_visibilities.cpp:                             gpuMemcpyHostToDevice) );
src/calculate_visibilities.cpp:      ( gpuMalloc( (void**)&d_u_shapes,
src/calculate_visibilities.cpp:      ( gpuMalloc( (void**)&d_v_shapes,
src/calculate_visibilities.cpp:      gpuErrorCheckKernel("kern_calc_uv_shapelet",
src/calculate_visibilities.cpp:      gpuErrorCheckKernel("kern_calc_visi_shapelets",
src/calculate_visibilities.cpp:      gpuFree(d_v_shapes);
src/calculate_visibilities.cpp:      gpuFree(d_u_shapes);
src/calculate_visibilities.cpp:      gpuFree(d_lsts);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_XX_real,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_XX_imag,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_XY_real,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_XY_imag,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_YX_real,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_YX_imag,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_YY_real,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->sum_visi_YY_imag,
src/calculate_visibilities.cpp:                                                     gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->us_metres,
src/calculate_visibilities.cpp:                                                      gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->vs_metres,
src/calculate_visibilities.cpp:                                                      gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:    gpuMemcpy(chunk_visibility_set->ws_metres,
src/calculate_visibilities.cpp:                                                      gpuMemcpyDeviceToHost);
src/calculate_visibilities.cpp:  //Free up the GPU memory
src/calculate_visibilities.cpp:  ( gpuFree(d_freqs) );
src/calculate_visibilities.cpp:  ( gpuFree(d_ws) );
src/calculate_visibilities.cpp:  ( gpuFree(d_vs) );
src/calculate_visibilities.cpp:  ( gpuFree(d_us) );
src/calculate_visibilities.cpp:  ( gpuFree(d_w_metres) );
src/calculate_visibilities.cpp:  ( gpuFree(d_v_metres) );
src/calculate_visibilities.cpp:  ( gpuFree(d_u_metres) );
src/calculate_visibilities.cpp:  ( gpuFree(d_allsteps_wavelengths) );
src/calculate_visibilities.cpp:  ( gpuFree(d_allsteps_cha0s) );
src/calculate_visibilities.cpp:  ( gpuFree(d_allsteps_sha0s) );
src/calculate_visibilities.cpp:  ( gpuFree(d_Z_diff) );
src/calculate_visibilities.cpp:  ( gpuFree(d_Y_diff) );
src/calculate_visibilities.cpp:  ( gpuFree(d_X_diff) );
src/calculate_visibilities.cpp:    ( gpuFree( d_sbf ) );
src/calculate_visibilities.cpp:    ( gpuFree( d_ant1_to_baseline_map ) );
src/calculate_visibilities.cpp:    ( gpuFree( d_ant2_to_baseline_map ) );
src/calculate_visibilities.cpp:    free_gpu_fee_beam(beam_settings->gpu_fee_beam);
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_XX_imag) );
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_XX_real) );
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_XY_imag) );
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_XY_real) );
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_YX_imag) );
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_YX_real) );
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_YY_imag) );
src/calculate_visibilities.cpp:  ( gpuFree(d_visibility_set->sum_visi_YY_real) );
src/source_components.cpp:#include "gpucomplex.h"
src/source_components.cpp:#include "primary_beam_gpu.h"
src/source_components.cpp:#include "gpu_macros.h"
src/source_components.cpp:__device__  gpuUserComplex calc_measurement_equation(user_precision_t *d_us,
src/source_components.cpp:  gpuUserComplex visi;
src/source_components.cpp:__device__ void apply_beam_gains_stokesIQUV(gpuUserComplex g1x, gpuUserComplex D1x,
src/source_components.cpp:          gpuUserComplex D1y, gpuUserComplex g1y,
src/source_components.cpp:          gpuUserComplex g2x, gpuUserComplex D2x,
src/source_components.cpp:          gpuUserComplex D2y, gpuUserComplex g2y,
src/source_components.cpp:          gpuUserComplex visi_component,
src/source_components.cpp:          gpuUserComplex * visi_XX, gpuUserComplex * visi_XY,
src/source_components.cpp:          gpuUserComplex * visi_YX, gpuUserComplex * visi_YY) {
src/source_components.cpp:  gpuUserComplex g2x_conj = make_gpuUserComplex(g2x.x,-g2x.y);
src/source_components.cpp:  gpuUserComplex D2x_conj = make_gpuUserComplex(D2x.x,-D2x.y);
src/source_components.cpp:  gpuUserComplex D2y_conj = make_gpuUserComplex(D2y.x,-D2y.y);
src/source_components.cpp:  gpuUserComplex g2y_conj = make_gpuUserComplex(g2y.x,-g2y.y);
src/source_components.cpp:  gpuUserComplex visi_I = make_gpuUserComplex(flux_I, 0.0)*visi_component;
src/source_components.cpp:  gpuUserComplex visi_Q = make_gpuUserComplex(flux_Q, 0.0)*visi_component;
src/source_components.cpp:  gpuUserComplex visi_U = make_gpuUserComplex(flux_U, 0.0)*visi_component;
src/source_components.cpp:  gpuUserComplex visi_V = make_gpuUserComplex(flux_V, 0.0)*visi_component;
src/source_components.cpp:  gpuUserComplex this_XX;
src/source_components.cpp:  gpuUserComplex this_XY;
src/source_components.cpp:  gpuUserComplex this_YX;
src/source_components.cpp:  gpuUserComplex this_YY;
src/source_components.cpp:  this_XX += (make_gpuUserComplex(0.0,1.0)*visi_V)*(g1x*D2x_conj - D1x*g2x_conj);
src/source_components.cpp:  this_XY += (make_gpuUserComplex(0.0,1.0)*visi_V)* (g1x*g2y_conj - D1x*D2y_conj);
src/source_components.cpp:  this_YX += (make_gpuUserComplex(0.0,1.0)*visi_V)* (D1y*D2x_conj - g1y*g2x_conj);
src/source_components.cpp:  this_YY += (make_gpuUserComplex(0.0,1.0)*visi_V)* (D1y*g2y_conj - g1y*D2y_conj);
src/source_components.cpp:           gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
src/source_components.cpp:           gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
src/source_components.cpp:           gpuUserComplex * g1x, gpuUserComplex * D1x,
src/source_components.cpp:           gpuUserComplex * D1y, gpuUserComplex * g1y,
src/source_components.cpp:           gpuUserComplex * g2x, gpuUserComplex * D2x,
src/source_components.cpp:           gpuUserComplex * D2y, gpuUserComplex * g2y){
src/source_components.cpp:    * g1x = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * g2x = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * g1y = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * g2y = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * D1x = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:    * D2x = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:    * D1y = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:    * D2y = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:           gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
src/source_components.cpp:           gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
src/source_components.cpp:           gpuUserComplex * g1x, gpuUserComplex * D1x,
src/source_components.cpp:           gpuUserComplex * D1y, gpuUserComplex * g1y,
src/source_components.cpp:           gpuUserComplex * g2x, gpuUserComplex * D2x,
src/source_components.cpp:           gpuUserComplex * D2y, gpuUserComplex * g2y){
src/source_components.cpp:    * g1x = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * g2x = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * g1y = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * g2y = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:    * D1x = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:    * D2x = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:    * D1y = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:    * D2y = make_gpuUserComplex(0.0, 0.0);
src/source_components.cpp:__device__ void apply_beam_gains_stokesI(gpuUserComplex g1x, gpuUserComplex D1x,
src/source_components.cpp:          gpuUserComplex D1y, gpuUserComplex g1y,
src/source_components.cpp:          gpuUserComplex g2x, gpuUserComplex D2x,
src/source_components.cpp:          gpuUserComplex D2y, gpuUserComplex g2y,
src/source_components.cpp:          gpuUserComplex visi_component,
src/source_components.cpp:          gpuUserComplex * visi_XX, gpuUserComplex * visi_XY,
src/source_components.cpp:          gpuUserComplex * visi_YX, gpuUserComplex * visi_YY) {
src/source_components.cpp:  gpuUserComplex g2x_conj = make_gpuUserComplex(g2x.x,-g2x.y);
src/source_components.cpp:  gpuUserComplex D2x_conj = make_gpuUserComplex(D2x.x,-D2x.y);
src/source_components.cpp:  gpuUserComplex D2y_conj = make_gpuUserComplex(D2y.x,-D2y.y);
src/source_components.cpp:  gpuUserComplex g2y_conj = make_gpuUserComplex(g2y.x,-g2y.y);
src/source_components.cpp:  gpuUserComplex visi_I = make_gpuUserComplex(flux_I, 0.0)*visi_component;
src/source_components.cpp:  gpuUserComplex this_XX;
src/source_components.cpp:  gpuUserComplex this_XY;
src/source_components.cpp:  gpuUserComplex this_YX;
src/source_components.cpp:  gpuUserComplex this_YY;
src/source_components.cpp:    gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
src/source_components.cpp:    gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
src/source_components.cpp:    gpuUserComplex visi_component,
src/source_components.cpp:    gpuUserComplex g1x;
src/source_components.cpp:    gpuUserComplex D1x;
src/source_components.cpp:    gpuUserComplex D1y;
src/source_components.cpp:    gpuUserComplex g1y;
src/source_components.cpp:    gpuUserComplex g2x;
src/source_components.cpp:    gpuUserComplex D2x;
src/source_components.cpp:    gpuUserComplex D2y;
src/source_components.cpp:    gpuUserComplex g2y;
src/source_components.cpp:    gpuUserComplex visi_XX;
src/source_components.cpp:    gpuUserComplex visi_XY;
src/source_components.cpp:    gpuUserComplex visi_YX;
src/source_components.cpp:    gpuUserComplex visi_YY;
src/source_components.cpp:    gpuUserComplex *d_gxs, gpuUserComplex *d_Dxs,
src/source_components.cpp:    gpuUserComplex *d_Dys, gpuUserComplex *d_gys,
src/source_components.cpp:    gpuUserComplex visi_component,
src/source_components.cpp:    gpuUserComplex g1x;
src/source_components.cpp:    gpuUserComplex D1x;
src/source_components.cpp:    gpuUserComplex D1y;
src/source_components.cpp:    gpuUserComplex g1y;
src/source_components.cpp:    gpuUserComplex g2x;
src/source_components.cpp:    gpuUserComplex D2x;
src/source_components.cpp:    gpuUserComplex D2y;
src/source_components.cpp:    gpuUserComplex g2y;
src/source_components.cpp:    gpuUserComplex visi_XX;
src/source_components.cpp:    gpuUserComplex visi_XY;
src/source_components.cpp:    gpuUserComplex visi_YX;
src/source_components.cpp:    gpuUserComplex visi_YY;
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->extrap_stokesI,
src/source_components.cpp:      ( gpuMalloc( (void**)&d_components->extrap_stokesQ,
src/source_components.cpp:      ( gpuMalloc( (void**)&d_components->extrap_stokesU,
src/source_components.cpp:      ( gpuMalloc( (void**)&d_components->extrap_stokesV,
src/source_components.cpp:      gpuErrorCheckKernel("kern_make_zeros_user_precision",
src/source_components.cpp:      gpuErrorCheckKernel("kern_make_zeros_user_precision",
src/source_components.cpp:      gpuErrorCheckKernel("kern_make_zeros_user_precision",
src/source_components.cpp:    //GPU memory as we'll immediate overwrite it
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_power_laws_stokesI",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_curved_power_laws_stokesI",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_list_fluxes",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_power_laws_stokesV",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_curved_power_laws_stokesV",
src/source_components.cpp:    gpuErrorCheckKernel("kern_polarisation_fraction_stokesV",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_list_fluxes",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_power_laws_linpol",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_curved_power_laws_linpol",
src/source_components.cpp:    gpuErrorCheckKernel("kern_polarisation_fraction_linpol",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_list_fluxes",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_list_fluxes",
src/source_components.cpp:    gpuErrorCheckKernel("kern_extrap_list_fluxes",
src/source_components.cpp:  gpuErrorCheckKernel("kern_apply_rotation_measure",
src/source_components.cpp://   gpuErrorCheckKernel("kern_print_extrap_fluxes",
src/source_components.cpp:    gpuMalloc( (void**)&d_component_beam_gains->d_Dxs,
src/source_components.cpp:                    num_gains*sizeof(gpuUserComplex) );
src/source_components.cpp:    gpuMalloc( (void**)&d_component_beam_gains->d_Dys,
src/source_components.cpp:                    num_gains*sizeof(gpuUserComplex) );
src/source_components.cpp:  gpuMalloc( (void**)&d_component_beam_gains->d_gxs,
src/source_components.cpp:                    num_gains*sizeof(gpuUserComplex) );
src/source_components.cpp:  gpuMalloc( (void**)&d_component_beam_gains->d_gys,
src/source_components.cpp:                    num_gains*sizeof(gpuUserComplex) );
src/source_components.cpp:  gpuMalloc( (void**)&d_components->ls, num_components*sizeof(double));
src/source_components.cpp:  gpuMalloc( (void**)&d_components->ms, num_components*sizeof(double));
src/source_components.cpp:  gpuMalloc( (void**)&d_components->ns, num_components*sizeof(double));
src/source_components.cpp:  gpuErrorCheckKernel("kern_calc_lmn",
src/source_components.cpp:    run_hyperbeam_gpu(num_components,
src/source_components.cpp:           beam_settings->gpu_fee_beam,
src/source_components.cpp:      ( gpuMalloc( (void**)&d_ant_to_auto_map,
src/source_components.cpp:      ( gpuMemcpy(d_ant_to_auto_map, ant_to_auto_map,
src/source_components.cpp:                                      num_ants*sizeof(int), gpuMemcpyHostToDevice ));
src/source_components.cpp:    gpuErrorCheckKernel("kern_calc_autos",
src/source_components.cpp:    (  gpuFree( d_ant_to_auto_map ) );
src/source_components.cpp:    gpuUserComplex visi_comp;
src/source_components.cpp:    gpuUserComplex V_envelop;
src/source_components.cpp:        V_envelop = make_gpuUserComplex( 1.0, 0.0 );
src/source_components.cpp:        V_envelop = make_gpuUserComplex( exp( -0.5 * ( x*x*invsig_x*invsig_x*M_PI_2_2_LN_2 + y*y*invsig_y*invsig_y*M_PI_2_2_LN_2 ) ), 0.0 );
src/source_components.cpp:    gpuUserComplex visi_shape;
src/source_components.cpp:      gpuUserComplex Ipow_lookup[] = { make_gpuUserComplex(  1.0,  0.0 ),
src/source_components.cpp:                                       make_gpuUserComplex(  0.0,  1.0 ),
src/source_components.cpp:                                       make_gpuUserComplex( -1.0,  0.0 ),
src/source_components.cpp:                                       make_gpuUserComplex(  0.0, -1.0 ) };
src/source_components.cpp:      gpuUserComplex V_envelop = make_gpuUserComplex( 0.0, 0.0 );
src/source_components.cpp://Copy the sky model info from a set of components from the CPU to the GPU
src/source_components.cpp:void copy_components_to_GPU(source_t *chunked_source, source_t *d_chunked_source,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->ras,
src/source_components.cpp:  ( gpuMemcpy( d_components->ras, components->ras,
src/source_components.cpp:                      num_comps*sizeof(double), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->decs,
src/source_components.cpp:  ( gpuMemcpy( d_components->decs, components->decs,
src/source_components.cpp:                      num_comps*sizeof(double), gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components->pas,
src/source_components.cpp:    ( gpuMemcpy( d_components->pas, components->pas,
src/source_components.cpp:                        num_comps*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components->majors,
src/source_components.cpp:    ( gpuMemcpy( d_components->majors, components->majors,
src/source_components.cpp:                        num_comps*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components->minors,
src/source_components.cpp:    ( gpuMemcpy( d_components->minors, components->minors,
src/source_components.cpp:                        num_comps*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components->shape_coeffs,
src/source_components.cpp:    ( gpuMemcpy( d_components->shape_coeffs, components->shape_coeffs,
src/source_components.cpp:                        gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components->n1s,
src/source_components.cpp:    ( gpuMemcpy( d_components->n1s, components->n1s,
src/source_components.cpp:                        gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components->n2s,
src/source_components.cpp:    ( gpuMemcpy( d_components->n2s, components->n2s,
src/source_components.cpp:                        gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components->param_indexes,
src/source_components.cpp:    ( gpuMemcpy( d_components->param_indexes, components->param_indexes,
src/source_components.cpp:                        gpuMemcpyHostToDevice ) );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->power_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->power_comp_inds, components->power_comp_inds,
src/source_components.cpp:                        num_powers*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->power_ref_stokesI,
src/source_components.cpp:    gpuMemcpy( d_components->power_ref_stokesI, components->power_ref_stokesI,
src/source_components.cpp:                        num_powers*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->power_SIs,
src/source_components.cpp:    gpuMemcpy( d_components->power_SIs, components->power_SIs,
src/source_components.cpp:                        num_powers*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->curve_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->curve_comp_inds, components->curve_comp_inds,
src/source_components.cpp:                        num_curves*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->curve_ref_stokesI,
src/source_components.cpp:    gpuMemcpy( d_components->curve_ref_stokesI, components->curve_ref_stokesI,
src/source_components.cpp:                        num_curves*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->curve_SIs,
src/source_components.cpp:    gpuMemcpy( d_components->curve_SIs, components->curve_SIs,
src/source_components.cpp:                        num_curves*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->curve_qs,
src/source_components.cpp:    gpuMemcpy( d_components->curve_qs, components->curve_qs,
src/source_components.cpp:                        num_curves*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->list_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->list_comp_inds,
src/source_components.cpp:                        num_lists*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->num_list_values,
src/source_components.cpp:    gpuMemcpy( d_components->num_list_values,
src/source_components.cpp:                        num_lists*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->list_start_indexes,
src/source_components.cpp:    gpuMemcpy( d_components->list_start_indexes,
src/source_components.cpp:                        num_lists*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->list_freqs,
src/source_components.cpp:    gpuMemcpy( d_components->list_freqs, components->list_freqs,
src/source_components.cpp:                        num_list_values*sizeof(double), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->list_stokesI,
src/source_components.cpp:    gpuMemcpy( d_components->list_stokesI, components->list_stokesI,
src/source_components.cpp:                        num_list_values*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_pol_fracs,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_pol_fracs, components->stokesV_pol_fracs,
src/source_components.cpp:                n_stokesV_pol_frac*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_pol_frac_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_pol_frac_comp_inds, components->stokesV_pol_frac_comp_inds,
src/source_components.cpp:                n_stokesV_pol_frac*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_power_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_power_ref_flux, components->stokesV_power_ref_flux,
src/source_components.cpp:                n_stokesV_power*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_power_SIs,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_power_SIs, components->stokesV_power_SIs,
src/source_components.cpp:                n_stokesV_power*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_power_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_power_comp_inds, components->stokesV_power_comp_inds,
src/source_components.cpp:                n_stokesV_power*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_curve_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_curve_ref_flux, components->stokesV_curve_ref_flux,
src/source_components.cpp:                n_stokesV_curve*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_curve_SIs,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_curve_SIs, components->stokesV_curve_SIs,
src/source_components.cpp:                n_stokesV_curve*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_curve_qs,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_curve_qs, components->stokesV_curve_qs,
src/source_components.cpp:                n_stokesV_curve*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_curve_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_curve_comp_inds, components->stokesV_curve_comp_inds,
src/source_components.cpp:                n_stokesV_curve*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_num_list_values,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_num_list_values, components->stokesV_num_list_values,
src/source_components.cpp:                n_stokesV_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_list_start_indexes,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_list_start_indexes, components->stokesV_list_start_indexes,
src/source_components.cpp:                n_stokesV_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_list_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_list_comp_inds, components->stokesV_list_comp_inds,
src/source_components.cpp:                n_stokesV_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_list_ref_freqs,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_list_ref_freqs, components->stokesV_list_ref_freqs,
src/source_components.cpp:                n_stokesV_list_flux_entries*sizeof(double), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesV_list_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->stokesV_list_ref_flux, components->stokesV_list_ref_flux,
src/source_components.cpp:                n_stokesV_list_flux_entries*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_pol_fracs,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_pol_fracs, components->linpol_pol_fracs,
src/source_components.cpp:                n_linpol_pol_frac*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_pol_frac_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_pol_frac_comp_inds, components->linpol_pol_frac_comp_inds,
src/source_components.cpp:                n_linpol_pol_frac*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_power_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_power_ref_flux, components->linpol_power_ref_flux,
src/source_components.cpp:                n_linpol_power*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_power_SIs,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_power_SIs, components->linpol_power_SIs,
src/source_components.cpp:                n_linpol_power*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_power_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_power_comp_inds, components->linpol_power_comp_inds,
src/source_components.cpp:                n_linpol_power*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_curve_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_curve_ref_flux, components->linpol_curve_ref_flux,
src/source_components.cpp:                n_linpol_curve*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_curve_SIs,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_curve_SIs, components->linpol_curve_SIs,
src/source_components.cpp:                n_linpol_curve*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_curve_qs,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_curve_qs, components->linpol_curve_qs,
src/source_components.cpp:                n_linpol_curve*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_curve_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_curve_comp_inds, components->linpol_curve_comp_inds,
src/source_components.cpp:                n_linpol_curve*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesQ_num_list_values,
src/source_components.cpp:    gpuMemcpy( d_components->stokesQ_num_list_values, components->stokesQ_num_list_values,
src/source_components.cpp:                n_linpol_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesQ_list_start_indexes,
src/source_components.cpp:    gpuMemcpy( d_components->stokesQ_list_start_indexes, components->stokesQ_list_start_indexes,
src/source_components.cpp:                n_linpol_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesQ_list_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->stokesQ_list_comp_inds, components->stokesQ_list_comp_inds,
src/source_components.cpp:                n_linpol_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesQ_list_ref_freqs,
src/source_components.cpp:    gpuMemcpy( d_components->stokesQ_list_ref_freqs, components->stokesQ_list_ref_freqs,
src/source_components.cpp:                n_stokesQ_list_flux_entries*sizeof(double), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesQ_list_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->stokesQ_list_ref_flux, components->stokesQ_list_ref_flux,
src/source_components.cpp:                n_stokesQ_list_flux_entries*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesU_num_list_values,
src/source_components.cpp:    gpuMemcpy( d_components->stokesU_num_list_values, components->stokesU_num_list_values,
src/source_components.cpp:                n_linpol_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesU_list_start_indexes,
src/source_components.cpp:    gpuMemcpy( d_components->stokesU_list_start_indexes, components->stokesU_list_start_indexes,
src/source_components.cpp:                n_linpol_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesU_list_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->stokesU_list_comp_inds, components->stokesU_list_comp_inds,
src/source_components.cpp:                n_linpol_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesU_list_ref_freqs,
src/source_components.cpp:    gpuMemcpy( d_components->stokesU_list_ref_freqs, components->stokesU_list_ref_freqs,
src/source_components.cpp:                n_stokesU_list_flux_entries*sizeof(double), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->stokesU_list_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->stokesU_list_ref_flux, components->stokesU_list_ref_flux,
src/source_components.cpp:                n_stokesU_list_flux_entries*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_p_num_list_values,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_p_num_list_values, components->linpol_p_num_list_values,
src/source_components.cpp:                n_linpol_p_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_p_list_start_indexes,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_p_list_start_indexes, components->linpol_p_list_start_indexes,
src/source_components.cpp:                n_linpol_p_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_p_list_comp_inds,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_p_list_comp_inds, components->linpol_p_list_comp_inds,
src/source_components.cpp:                n_linpol_p_list*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_p_list_ref_freqs,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_p_list_ref_freqs, components->linpol_p_list_ref_freqs,
src/source_components.cpp:                n_linpol_p_list_flux_entries*sizeof(double), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_p_list_ref_flux,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_p_list_ref_flux, components->linpol_p_list_ref_flux,
src/source_components.cpp:                n_linpol_p_list_flux_entries*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->rm_values,
src/source_components.cpp:    gpuMemcpy( d_components->rm_values, components->rm_values,
src/source_components.cpp:                n_linpol_angles*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->intr_pol_angle,
src/source_components.cpp:    gpuMemcpy( d_components->intr_pol_angle, components->intr_pol_angle,
src/source_components.cpp:                n_linpol_angles*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    gpuMalloc( (void**)&d_components->linpol_angle_inds,
src/source_components.cpp:    gpuMemcpy( d_components->linpol_angle_inds, components->linpol_angle_inds,
src/source_components.cpp:                n_linpol_angles*sizeof(int), gpuMemcpyHostToDevice );
src/source_components.cpp:source_t * copy_chunked_source_to_GPU(source_t *chunked_source){
src/source_components.cpp:    copy_components_to_GPU(chunked_source, d_chunked_source, POINT);
src/source_components.cpp:    copy_components_to_GPU(chunked_source, d_chunked_source, GAUSSIAN);
src/source_components.cpp:    copy_components_to_GPU(chunked_source, d_chunked_source, SHAPELET);
src/source_components.cpp:  ( gpuFree( d_components->extrap_stokesI ) );
src/source_components.cpp:    ( gpuFree( d_components->extrap_stokesQ ) );
src/source_components.cpp:    ( gpuFree( d_components->extrap_stokesU ) );
src/source_components.cpp:    ( gpuFree( d_components->extrap_stokesV ) );
src/source_components.cpp:  gpuFree( d_components.decs);
src/source_components.cpp:  gpuFree( d_components.ras);
src/source_components.cpp:  gpuFree( d_components.ls);
src/source_components.cpp:  gpuFree( d_components.ms);
src/source_components.cpp:  gpuFree( d_components.ns);
src/source_components.cpp:    gpuFree( d_components.power_ref_stokesI );
src/source_components.cpp:    gpuFree( d_components.power_SIs );
src/source_components.cpp:    gpuFree( d_components.power_comp_inds );
src/source_components.cpp:    gpuFree( d_components.curve_ref_stokesI );
src/source_components.cpp:    gpuFree( d_components.curve_SIs );
src/source_components.cpp:    gpuFree( d_components.curve_qs );
src/source_components.cpp:    gpuFree( d_components.curve_comp_inds );
src/source_components.cpp:    gpuFree( d_components.list_comp_inds );
src/source_components.cpp:    gpuFree( d_components.list_freqs );
src/source_components.cpp:    gpuFree( d_components.list_stokesI );
src/source_components.cpp:    gpuFree( d_components.num_list_values );
src/source_components.cpp:    gpuFree( d_components.list_start_indexes );
src/source_components.cpp:    gpuFree( d_components.pas );
src/source_components.cpp:    gpuFree( d_components.majors );
src/source_components.cpp:    gpuFree( d_components.minors );
src/source_components.cpp:    gpuFree( d_components.shape_coeffs );
src/source_components.cpp:    gpuFree( d_components.n1s );
src/source_components.cpp:    gpuFree( d_components.n2s );
src/source_components.cpp:    gpuFree( d_components.param_indexes );
src/source_components.cpp:    gpuFree(d_components.stokesV_pol_fracs);
src/source_components.cpp:    gpuFree(d_components.stokesV_pol_frac_comp_inds);
src/source_components.cpp:    gpuFree(d_components.stokesV_power_ref_flux);
src/source_components.cpp:    gpuFree(d_components.stokesV_power_SIs);
src/source_components.cpp:    gpuFree(d_components.stokesV_power_comp_inds);
src/source_components.cpp:    gpuFree(d_components.stokesV_curve_ref_flux);
src/source_components.cpp:    gpuFree(d_components.stokesV_curve_SIs);
src/source_components.cpp:    gpuFree(d_components.stokesV_curve_qs);
src/source_components.cpp:    gpuFree(d_components.stokesV_curve_comp_inds);
src/source_components.cpp:    gpuFree(d_components.stokesV_num_list_values);
src/source_components.cpp:    gpuFree(d_components.stokesV_list_start_indexes);
src/source_components.cpp:    gpuFree(d_components.stokesV_list_comp_inds);
src/source_components.cpp:    gpuFree(d_components.stokesV_list_ref_freqs);
src/source_components.cpp:    gpuFree(d_components.stokesV_list_ref_flux);
src/source_components.cpp:    gpuFree(d_components.linpol_pol_fracs);
src/source_components.cpp:    gpuFree(d_components.linpol_pol_frac_comp_inds);
src/source_components.cpp:    gpuFree(d_components.linpol_power_ref_flux);
src/source_components.cpp:    gpuFree(d_components.linpol_power_SIs);
src/source_components.cpp:    gpuFree(d_components.linpol_power_comp_inds);
src/source_components.cpp:    gpuFree(d_components.linpol_curve_ref_flux);
src/source_components.cpp:    gpuFree(d_components.linpol_curve_SIs);
src/source_components.cpp:    gpuFree(d_components.linpol_curve_qs);
src/source_components.cpp:    gpuFree(d_components.linpol_curve_comp_inds);
src/source_components.cpp:    gpuFree(d_components.stokesQ_num_list_values);
src/source_components.cpp:    gpuFree(d_components.stokesQ_list_start_indexes);
src/source_components.cpp:    gpuFree(d_components.stokesQ_list_comp_inds);
src/source_components.cpp:    gpuFree(d_components.stokesQ_list_ref_freqs);
src/source_components.cpp:    gpuFree(d_components.stokesQ_list_ref_flux);
src/source_components.cpp:    gpuFree(d_components.stokesU_num_list_values);
src/source_components.cpp:    gpuFree(d_components.stokesU_list_start_indexes);
src/source_components.cpp:    gpuFree(d_components.stokesU_list_comp_inds);
src/source_components.cpp:    gpuFree(d_components.stokesU_list_ref_freqs);
src/source_components.cpp:    gpuFree(d_components.stokesU_list_ref_flux);
src/source_components.cpp:    gpuFree(d_components.linpol_p_num_list_values);
src/source_components.cpp:    gpuFree(d_components.linpol_p_list_start_indexes);
src/source_components.cpp:    gpuFree(d_components.linpol_p_list_comp_inds);
src/source_components.cpp:    gpuFree(d_components.linpol_p_list_ref_freqs);
src/source_components.cpp:    gpuFree(d_components.linpol_p_list_ref_flux);
src/source_components.cpp:    gpuFree(d_components.rm_values);
src/source_components.cpp:    gpuFree(d_components.intr_pol_angle);
src/source_components.cpp:    gpuFree(d_components.linpol_angle_inds);
src/source_components.cpp:  ( gpuFree( d_beam_gains.d_gxs) );
src/source_components.cpp:  ( gpuFree( d_beam_gains.d_gys) );
src/source_components.cpp:    ( gpuFree( d_beam_gains.d_Dxs ) );
src/source_components.cpp:    ( gpuFree( d_beam_gains.d_Dys ) );
src/source_components.cpp:    gpuUserComplex auto_XX, auto_XY, auto_YX, auto_YY;
src/source_components.cpp:    gpuUserComplex g1x, D1x, D1y, g1y, g2x, D2x, D2y, g2y;
src/source_components.cpp:      gpuUserComplex visi_component;
src/source_components.cpp:      visi_component = make_gpuUserComplex(1.0, 0.0);
src/source_components.cpp:  ( gpuMemcpy(d_ant1_to_baseline_map, ant1_to_baseline_map,
src/source_components.cpp:                                  num_baselines*sizeof(int), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_ant2_to_baseline_map, ant2_to_baseline_map,
src/source_components.cpp:                                  num_baselines*sizeof(int), gpuMemcpyHostToDevice ));
src/source_components.cpp:  source_t *d_chunked_source = copy_chunked_source_to_GPU(chunked_source);
src/source_components.cpp:  gpuMalloc( (void**)&d_extrap_freqs,
src/source_components.cpp:  gpuMemcpy(d_extrap_freqs, extrap_freqs,
src/source_components.cpp:             num_extrap_freqs*sizeof(double), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(extrap_flux_I, d_components.extrap_stokesI,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost );
src/source_components.cpp:  gpuMemcpy(extrap_flux_Q, d_components.extrap_stokesQ,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost );
src/source_components.cpp:  gpuMemcpy(extrap_flux_U, d_components.extrap_stokesU,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost );
src/source_components.cpp:  gpuMemcpy(extrap_flux_V, d_components.extrap_stokesV,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost );
src/source_components.cpp:  gpuFree( d_extrap_freqs );
src/source_components.cpp:          double *d_ls, double *d_ms, double *d_ns, gpuUserComplex *d_visis) {
src/source_components.cpp:    gpuUserComplex visi;
src/source_components.cpp:  ( gpuMalloc( (void**)&d_us, num_baselines*sizeof(user_precision_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_vs, num_baselines*sizeof(user_precision_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_ws, num_baselines*sizeof(user_precision_t) ));
src/source_components.cpp:  ( gpuMemcpy(d_us, us, num_baselines*sizeof(user_precision_t),
src/source_components.cpp:                                                        gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_vs, vs, num_baselines*sizeof(user_precision_t),
src/source_components.cpp:                                                        gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_ws, ws, num_baselines*sizeof(user_precision_t),
src/source_components.cpp:                                                        gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_ls, num_components*sizeof(double) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_ms, num_components*sizeof(double) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_ns, num_components*sizeof(double) ));
src/source_components.cpp:  ( gpuMemcpy(d_ls, ls, num_components*sizeof(double),
src/source_components.cpp:                                                      gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_ms, ms, num_components*sizeof(double),
src/source_components.cpp:                                                      gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_ns, ns, num_components*sizeof(double),
src/source_components.cpp:                                                      gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_visis, num_baselines*num_components*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  gpuErrorCheckKernel("kern_calc_measurement_equation",
src/source_components.cpp:                      (gpuUserComplex*)d_visis );
src/source_components.cpp:  ( gpuMemcpy(visis, (user_precision_complex_t*)d_visis, num_components*num_baselines*sizeof(user_precision_complex_t),gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuFree( d_us ) );
src/source_components.cpp:  ( gpuFree( d_vs ) );
src/source_components.cpp:  ( gpuFree( d_ws ) );
src/source_components.cpp:  ( gpuFree( d_ls ) );
src/source_components.cpp:  ( gpuFree( d_ms ) );
src/source_components.cpp:  ( gpuFree( d_ns ) );
src/source_components.cpp:  ( gpuFree(d_visis ) );
src/source_components.cpp:__global__ void kern_apply_beam_gains_stokesIQUV(int num_gains, gpuUserComplex *d_g1xs,
src/source_components.cpp:          gpuUserComplex *d_D1xs,
src/source_components.cpp:          gpuUserComplex *d_D1ys, gpuUserComplex *d_g1ys,
src/source_components.cpp:          gpuUserComplex *d_g2xs, gpuUserComplex *d_D2xs,
src/source_components.cpp:          gpuUserComplex *d_D2ys, gpuUserComplex *d_g2ys,
src/source_components.cpp:          gpuUserComplex *d_visi_components,
src/source_components.cpp:          gpuUserComplex *d_visi_XXs, gpuUserComplex *d_visi_XYs,
src/source_components.cpp:          gpuUserComplex *d_visi_YXs, gpuUserComplex *d_visi_YYs) {
src/source_components.cpp:    gpuUserComplex visi_XX;
src/source_components.cpp:    gpuUserComplex visi_XY;
src/source_components.cpp:    gpuUserComplex visi_YX;
src/source_components.cpp:    gpuUserComplex visi_YY;
src/source_components.cpp:  ( gpuMalloc( (void**)&d_g1xs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_D1xs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_D1ys,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_g1ys,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_g2xs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_D2xs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_D2ys,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_g2ys,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_Is,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_Qs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_Us,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_Vs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_visi_components,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_visi_XXs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_visi_XYs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_visi_YXs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_visi_YYs,
src/source_components.cpp:  ( gpuMemcpy(d_g1xs, g1xs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_D1xs, D1xs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_D1ys, D1ys,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_g1ys, g1ys,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_g2xs, g2xs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_D2xs, D2xs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_D2ys, D2ys,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_g2ys, g2ys,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_visi_components, visi_components,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_visi_XXs, visi_XXs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_visi_XYs, visi_XYs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_visi_YXs, visi_YXs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_visi_YYs, visi_YYs,
src/source_components.cpp:          num_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_Is, flux_Is,
src/source_components.cpp:                             num_gains*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_Qs, flux_Qs,
src/source_components.cpp:                             num_gains*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_Us, flux_Us,
src/source_components.cpp:                             num_gains*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_Vs, flux_Vs,
src/source_components.cpp:                             num_gains*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  gpuErrorCheckKernel("kern_apply_beam_gains_stokesIQUV",
src/source_components.cpp:                      (gpuUserComplex *)d_g1xs, (gpuUserComplex *)d_D1xs,
src/source_components.cpp:                      (gpuUserComplex *)d_D1ys, (gpuUserComplex *)d_g1ys,
src/source_components.cpp:                      (gpuUserComplex *)d_g2xs, (gpuUserComplex *)d_D2xs,
src/source_components.cpp:                      (gpuUserComplex *)d_D2ys, (gpuUserComplex *)d_g2ys,
src/source_components.cpp:                      (gpuUserComplex *)d_visi_components,
src/source_components.cpp:                      (gpuUserComplex *)d_visi_XXs, (gpuUserComplex *)d_visi_XYs,
src/source_components.cpp:                      (gpuUserComplex *)d_visi_YXs, (gpuUserComplex *)d_visi_YYs );
src/source_components.cpp:  ( gpuMemcpy(visi_XXs, d_visi_XXs,
src/source_components.cpp:           num_gains*sizeof(user_precision_complex_t),gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visi_XYs, d_visi_XYs,
src/source_components.cpp:           num_gains*sizeof(user_precision_complex_t),gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visi_YXs, d_visi_YXs,
src/source_components.cpp:           num_gains*sizeof(user_precision_complex_t),gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visi_YYs, d_visi_YYs,
src/source_components.cpp:           num_gains*sizeof(user_precision_complex_t),gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuFree( d_g1xs ) );
src/source_components.cpp:  ( gpuFree( d_D1xs ) );
src/source_components.cpp:  ( gpuFree( d_D1ys ) );
src/source_components.cpp:  ( gpuFree( d_g1ys ) );
src/source_components.cpp:  ( gpuFree( d_g2xs ) );
src/source_components.cpp:  ( gpuFree( d_D2xs ) );
src/source_components.cpp:  ( gpuFree( d_D2ys ) );
src/source_components.cpp:  ( gpuFree( d_g2ys ) );
src/source_components.cpp:  ( gpuFree( d_flux_Is ) );
src/source_components.cpp:  ( gpuFree( d_flux_Qs ) );
src/source_components.cpp:  ( gpuFree( d_flux_Us ) );
src/source_components.cpp:  ( gpuFree( d_flux_Vs ) );
src/source_components.cpp:  ( gpuFree( d_visi_components ) );
src/source_components.cpp:  ( gpuFree( d_visi_XXs ) );
src/source_components.cpp:  ( gpuFree( d_visi_XYs ) );
src/source_components.cpp:  ( gpuFree( d_visi_YXs ) );
src/source_components.cpp:  ( gpuFree( d_visi_YYs ) );
src/source_components.cpp:           gpuUserComplex *d_g1xs, gpuUserComplex *d_D1xs,
src/source_components.cpp:           gpuUserComplex *d_D1ys, gpuUserComplex *d_g1ys,
src/source_components.cpp:           gpuUserComplex *d_recov_g1x, gpuUserComplex *d_recov_D1x,
src/source_components.cpp:           gpuUserComplex *d_recov_D1y, gpuUserComplex *d_recov_g1y,
src/source_components.cpp:           gpuUserComplex *d_recov_g2x, gpuUserComplex *d_recov_D2x,
src/source_components.cpp:           gpuUserComplex *d_recov_D2y, gpuUserComplex *d_recov_g2y,
src/source_components.cpp:      gpuUserComplex g1x;
src/source_components.cpp:      gpuUserComplex D1x;
src/source_components.cpp:      gpuUserComplex D1y;
src/source_components.cpp:      gpuUserComplex g1y;
src/source_components.cpp:      gpuUserComplex g2x;
src/source_components.cpp:      gpuUserComplex D2x;
src/source_components.cpp:      gpuUserComplex D2y;
src/source_components.cpp:      gpuUserComplex g2y;
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_g1x, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_D1x, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_D1y, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_g1y, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_g2x, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_D2x, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_D2y, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_recover_g2y, num_recover_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_g1xs, num_input_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_D1xs, num_input_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_D1ys, num_input_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_g1ys, num_input_gains*sizeof(user_precision_complex_t) ));
src/source_components.cpp:  ( gpuMemcpy(d_g1xs, primay_beam_J00, num_input_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_D1xs, primay_beam_J01, num_input_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_D1ys, primay_beam_J10, num_input_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_g1ys, primay_beam_J11, num_input_gains*sizeof(user_precision_complex_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:    ( gpuMalloc( (void**)&d_ant1_to_baseline_map, num_baselines*sizeof(int) ));
src/source_components.cpp:    ( gpuMalloc( (void**)&d_ant2_to_baseline_map, num_baselines*sizeof(int) ));
src/source_components.cpp:  gpuErrorCheckKernel("kern_get_beam_gains",
src/source_components.cpp:                      (gpuUserComplex *)d_g1xs,
src/source_components.cpp:                      (gpuUserComplex *)d_D1xs,
src/source_components.cpp:                      (gpuUserComplex *)d_D1ys,
src/source_components.cpp:                      (gpuUserComplex *)d_g1ys,
src/source_components.cpp:                      (gpuUserComplex *)d_recover_g1x, (gpuUserComplex *)d_recover_D1x,
src/source_components.cpp:                      (gpuUserComplex *)d_recover_D1y, (gpuUserComplex *)d_recover_g1y,
src/source_components.cpp:                      (gpuUserComplex *)d_recover_g2x, (gpuUserComplex *)d_recover_D2x,
src/source_components.cpp:                      (gpuUserComplex *)d_recover_D2y, (gpuUserComplex *)d_recover_g2y,
src/source_components.cpp:  ( gpuMemcpy(recover_g1x, d_recover_g1x, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(recover_D1x, d_recover_D1x, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(recover_D1y, d_recover_D1y, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(recover_g1y, d_recover_g1y, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(recover_g2x, d_recover_g2x, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(recover_D2x, d_recover_D2x, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(recover_D2y, d_recover_D2y, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(recover_g2y, d_recover_g2y, num_recover_gains*sizeof(user_precision_complex_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuFree( d_recover_g1x ) );
src/source_components.cpp:  ( gpuFree( d_recover_D1x ) );
src/source_components.cpp:  ( gpuFree( d_recover_D1y ) );
src/source_components.cpp:  ( gpuFree( d_recover_g1y ) );
src/source_components.cpp:  ( gpuFree( d_recover_g2x ) );
src/source_components.cpp:  ( gpuFree( d_recover_D2x ) );
src/source_components.cpp:  ( gpuFree( d_recover_D2y ) );
src/source_components.cpp:  ( gpuFree( d_recover_g2y ) );
src/source_components.cpp:  ( gpuFree( d_g1xs ) );
src/source_components.cpp:  ( gpuFree( d_D1xs ) );
src/source_components.cpp:  ( gpuFree( d_D1ys ) );
src/source_components.cpp:  ( gpuFree( d_g1ys ) );
src/source_components.cpp:    ( gpuFree( d_ant1_to_baseline_map ) );
src/source_components.cpp:    ( gpuFree( d_ant2_to_baseline_map ) );
src/source_components.cpp:     gpuUserComplex *d_g1xs, gpuUserComplex *d_D1xs,
src/source_components.cpp:     gpuUserComplex *d_D1ys, gpuUserComplex *d_g1ys,
src/source_components.cpp:     gpuUserComplex *d_visi_components,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_gxs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_Dxs,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_Dys,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_gys,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_visi_components,
src/source_components.cpp:  ( gpuMemcpy(d_gxs, primay_beam_J00,
src/source_components.cpp:            gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_Dxs, primay_beam_J01,
src/source_components.cpp:            gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_Dys, primay_beam_J10,
src/source_components.cpp:            gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_gys, primay_beam_J11,
src/source_components.cpp:            gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_visi_components, visi_components,
src/source_components.cpp:                                     gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_I, num_components*num_times*num_freqs*sizeof(user_precision_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_Q, num_components*num_times*num_freqs*sizeof(user_precision_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_U, num_components*num_times*num_freqs*sizeof(user_precision_t) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_flux_V, num_components*num_times*num_freqs*sizeof(user_precision_t) ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_I, flux_I,
src/source_components.cpp:                    num_components*num_times*num_freqs*sizeof(user_precision_t),    gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_Q, flux_Q,
src/source_components.cpp:                    num_components*num_times*num_freqs*sizeof(user_precision_t),    gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_U, flux_U,
src/source_components.cpp:                    num_components*num_times*num_freqs*sizeof(user_precision_t),    gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_flux_V, flux_V,
src/source_components.cpp:                    num_components*num_times*num_freqs*sizeof(user_precision_t),    gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XX_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XY_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YX_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YY_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XX_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XY_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YX_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YY_imag,
src/source_components.cpp:  gpuMemcpy(d_sum_visi_XX_real, sum_visi_XX_real,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(d_sum_visi_XY_real, sum_visi_XY_real,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(d_sum_visi_YX_real, sum_visi_YX_real,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(d_sum_visi_YY_real, sum_visi_YY_real,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(d_sum_visi_XX_imag, sum_visi_XX_imag,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(d_sum_visi_XY_imag, sum_visi_XY_imag,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(d_sum_visi_YX_imag, sum_visi_YX_imag,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:  gpuMemcpy(d_sum_visi_YY_imag, sum_visi_YY_imag,
src/source_components.cpp:                    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_ant1_to_baseline_map, num_baselines*sizeof(int) ));
src/source_components.cpp:    ( gpuMalloc( (void**)&d_ant2_to_baseline_map, num_baselines*sizeof(int) ));
src/source_components.cpp:  gpuErrorCheckKernel("kern_update_sum_visis_stokesIQUV",
src/source_components.cpp:                      (gpuUserComplex *)d_gxs, (gpuUserComplex *)d_Dxs,
src/source_components.cpp:                      (gpuUserComplex *)d_Dys, (gpuUserComplex *)d_gys,
src/source_components.cpp:                      (gpuUserComplex *)d_visi_components,
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XX_real, d_sum_visi_XX_real,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XY_real, d_sum_visi_XY_real,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YX_real, d_sum_visi_YX_real,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YY_real, d_sum_visi_YY_real,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XX_imag, d_sum_visi_XX_imag,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XY_imag, d_sum_visi_XY_imag,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YX_imag, d_sum_visi_YX_imag,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YY_imag, d_sum_visi_YY_imag,
src/source_components.cpp:                  num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuFree( d_gxs ) );
src/source_components.cpp:  ( gpuFree( d_Dxs ) );
src/source_components.cpp:  ( gpuFree( d_Dys ) );
src/source_components.cpp:  ( gpuFree( d_gys ) );
src/source_components.cpp:  ( gpuFree( d_visi_components ) );
src/source_components.cpp:  ( gpuFree( d_flux_I ) );
src/source_components.cpp:  ( gpuFree( d_flux_Q ) );
src/source_components.cpp:  ( gpuFree( d_flux_U ) );
src/source_components.cpp:  ( gpuFree( d_flux_V ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_XX_real ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_XY_real ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_YX_real ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_YY_real ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_XX_imag ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_XY_imag ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_YX_imag ) );
src/source_components.cpp:  ( gpuFree( d_sum_visi_YY_imag ) );
src/source_components.cpp:    ( gpuFree( d_ant1_to_baseline_map ) );
src/source_components.cpp:    ( gpuFree( d_ant2_to_baseline_map ) );
src/source_components.cpp:  source_t *d_chunked_source = copy_chunked_source_to_GPU(chunked_source);
src/source_components.cpp:  gpuMalloc( (void**)&d_freqs, woden_settings->num_freqs*sizeof(double) );
src/source_components.cpp:  gpuMemcpy( d_freqs, freqs, woden_settings->num_freqs*sizeof(double), gpuMemcpyHostToDevice) ;
src/source_components.cpp:  //THIS IS THE GPU CALL ACTUALLY BEING TESTED JEEZUS--------------------------
src/source_components.cpp:  //THIS IS THE GPU CALL ACTUALLY BEING TESTED JEEZUS--------------------------
src/source_components.cpp:  ( gpuMemcpy(gxs, (user_precision_complex_t*)d_beam_gains.d_gxs,
src/source_components.cpp:              num_beam_values*sizeof(gpuUserComplex), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(gys, (user_precision_complex_t*)d_beam_gains.d_gys,
src/source_components.cpp:              num_beam_values*sizeof(gpuUserComplex), gpuMemcpyDeviceToHost ));
src/source_components.cpp:    ( gpuMemcpy(Dxs, (user_precision_complex_t*)d_beam_gains.d_Dxs,
src/source_components.cpp:                num_beam_values*sizeof(gpuUserComplex), gpuMemcpyDeviceToHost ));
src/source_components.cpp:    ( gpuMemcpy(Dys, (user_precision_complex_t*)d_beam_gains.d_Dys,
src/source_components.cpp:                num_beam_values*sizeof(gpuUserComplex), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(ls, d_components.ls,
src/source_components.cpp:                            gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(ms, d_components.ms,
src/source_components.cpp:                            gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(ns, d_components.ns,
src/source_components.cpp:                            gpuMemcpyDeviceToHost ));
src/source_components.cpp:    gpuMalloc( (void**)&d_components.extrap_stokesQ, num_things*sizeof(user_precision_t) );
src/source_components.cpp:    gpuMalloc( (void**)&d_components.extrap_stokesU, num_things*sizeof(user_precision_t) );
src/source_components.cpp:    gpuMalloc( (void**)&d_components.extrap_stokesV, num_things*sizeof(user_precision_t) );
src/source_components.cpp:    gpuErrorCheckKernel("kern_make_zeros_user_precision",
src/source_components.cpp:    gpuErrorCheckKernel("kern_make_zeros_user_precision",
src/source_components.cpp:    gpuErrorCheckKernel("kern_make_zeros_user_precision",
src/source_components.cpp:  ( gpuMemcpy(extrap_flux_I, d_components.extrap_stokesI,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(extrap_flux_Q, d_components.extrap_stokesQ,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(extrap_flux_U, d_components.extrap_stokesU,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(extrap_flux_V, d_components.extrap_stokesV,
src/source_components.cpp:                                                      gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuFree( d_freqs ) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->ls,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->ms,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->ns,
src/source_components.cpp:  ( gpuMemcpy(d_components->ls, components->ls, num_components*sizeof(double),
src/source_components.cpp:                                           gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_components->ms, components->ms, num_components*sizeof(double),
src/source_components.cpp:                                           gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_components->ns, components->ns, num_components*sizeof(double),
src/source_components.cpp:                                           gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_us, num_cross*sizeof(user_precision_t) ) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_vs, num_cross*sizeof(user_precision_t) ) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_ws, num_cross*sizeof(user_precision_t) ) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_allsteps_wavelengths, num_cross*sizeof(user_precision_t) ) );
src/source_components.cpp:  ( gpuMemcpy(d_us, us,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_vs, vs,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_ws, ws,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_allsteps_wavelengths, allsteps_wavelengths,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_extrap_freqs,
src/source_components.cpp:  ( gpuMemcpy(d_extrap_freqs, extrap_freqs,
src/source_components.cpp:             num_freqs*sizeof(double), gpuMemcpyHostToDevice ));
src/source_components.cpp:    d_chunked_source = copy_chunked_source_to_GPU(chunked_source);
src/source_components.cpp:    d_chunked_source = copy_chunked_source_to_GPU(chunked_source);
src/source_components.cpp:    d_chunked_source = copy_chunked_source_to_GPU(chunked_source);
src/source_components.cpp:  ( gpuMalloc( (void**)&d_beam_gains.d_gxs,
src/source_components.cpp:                                      num_beam_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_beam_gains.d_Dxs,
src/source_components.cpp:                                      num_beam_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_beam_gains.d_Dys,
src/source_components.cpp:                                      num_beam_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_beam_gains.d_gys,
src/source_components.cpp:                                      num_beam_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMemcpy(d_beam_gains.d_gxs, (gpuUserComplex *)gxs,
src/source_components.cpp:              num_beam_values*sizeof(gpuUserComplex), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_beam_gains.d_Dxs, (gpuUserComplex *)Dxs,
src/source_components.cpp:              num_beam_values*sizeof(gpuUserComplex), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_beam_gains.d_Dys, (gpuUserComplex *)Dys,
src/source_components.cpp:              num_beam_values*sizeof(gpuUserComplex), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMemcpy(d_beam_gains.d_gys, (gpuUserComplex *)gys,
src/source_components.cpp:              num_beam_values*sizeof(gpuUserComplex), gpuMemcpyHostToDevice ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XX_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XY_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YX_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YY_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XX_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XY_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YX_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YY_imag,
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_XX_real, sum_visi_XX_real,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_XY_real, sum_visi_XY_real,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_YX_real, sum_visi_YX_real,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_YY_real, sum_visi_YY_real,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_XX_imag, sum_visi_XX_imag,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_XY_imag, sum_visi_XY_imag,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_YX_imag, sum_visi_YX_imag,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy( d_sum_visi_YY_imag, sum_visi_YY_imag,
src/source_components.cpp:    num_cross*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_u_shapes,
src/source_components.cpp:    ( gpuMalloc( (void**)&d_v_shapes,
src/source_components.cpp:    ( gpuMemcpy(d_u_shapes, u_shapes,
src/source_components.cpp:                                                       gpuMemcpyHostToDevice ));
src/source_components.cpp:    ( gpuMemcpy(d_v_shapes, v_shapes,
src/source_components.cpp:                                                       gpuMemcpyHostToDevice ));
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components.shape_coeffs,
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components.n1s,
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components.n2s,
src/source_components.cpp:    ( gpuMalloc( (void**)&d_components.param_indexes,
src/source_components.cpp:    ( gpuMemcpy(d_components.shape_coeffs,
src/source_components.cpp:                          gpuMemcpyHostToDevice ));
src/source_components.cpp:    ( gpuMemcpy(d_components.n1s,
src/source_components.cpp:                          gpuMemcpyHostToDevice ));
src/source_components.cpp:    ( gpuMemcpy(d_components.n2s,
src/source_components.cpp:                          gpuMemcpyHostToDevice ));
src/source_components.cpp:    ( gpuMemcpy(d_components.param_indexes,
src/source_components.cpp:                          gpuMemcpyHostToDevice ));
src/source_components.cpp:    ( gpuMalloc( (void**)&(d_sbf), sbf_N*sbf_L*sizeof(user_precision_t) ));
src/source_components.cpp:    ( gpuMemcpy( d_sbf, sbf, sbf_N*sbf_L*sizeof(user_precision_t),
src/source_components.cpp:                        gpuMemcpyHostToDevice ));
src/source_components.cpp:    gpuErrorCheckKernel("kern_calc_visi_point_or_gauss",
src/source_components.cpp:    gpuErrorCheckKernel("kern_calc_visi_shapelets",
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XX_real, d_sum_visi_XX_real,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XY_real, d_sum_visi_XY_real,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YX_real, d_sum_visi_YX_real,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YY_real, d_sum_visi_YY_real,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XX_imag, d_sum_visi_XX_imag,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_XY_imag, d_sum_visi_XY_imag,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YX_imag, d_sum_visi_YX_imag,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(sum_visi_YY_imag, d_sum_visi_YY_imag,
src/source_components.cpp:                             num_cross*sizeof(user_precision_t), gpuMemcpyDeviceToHost ));
src/source_components.cpp:  (  gpuFree( d_sum_visi_XX_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_XX_imag ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_XY_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_XY_imag ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YX_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YX_imag ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YY_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YY_imag ) );
src/source_components.cpp:  (  gpuFree( d_allsteps_wavelengths ) );
src/source_components.cpp:  (  gpuFree( d_us ) );
src/source_components.cpp:  (  gpuFree( d_vs ) );
src/source_components.cpp:  (  gpuFree( d_ws ) );
src/source_components.cpp:  ( gpuFree( d_extrap_freqs ) );
src/source_components.cpp:    (  gpuFree( d_sbf) );
src/source_components.cpp:    (  gpuFree( d_u_shapes) );
src/source_components.cpp:    (  gpuFree( d_v_shapes) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->extrap_stokesI,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->extrap_stokesQ,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->extrap_stokesU,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_components->extrap_stokesV,
src/source_components.cpp:  ( gpuMemcpy(d_components->extrap_stokesI,
src/source_components.cpp:         components->extrap_stokesI, num_components*num_freqs*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_components->extrap_stokesQ,
src/source_components.cpp:         components->extrap_stokesQ, num_components*num_freqs*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_components->extrap_stokesU,
src/source_components.cpp:         components->extrap_stokesU, num_components*num_freqs*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_components->extrap_stokesV,
src/source_components.cpp:         components->extrap_stokesV, num_components*num_freqs*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_component_beam_gains->d_gxs,
src/source_components.cpp:                                        num_pb_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_component_beam_gains->d_Dxs,
src/source_components.cpp:                                        num_pb_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_component_beam_gains->d_Dys,
src/source_components.cpp:                                        num_pb_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMalloc( (void**)&d_component_beam_gains->d_gys,
src/source_components.cpp:                                        num_pb_values*sizeof(gpuUserComplex) ));
src/source_components.cpp:  ( gpuMemcpy(d_component_beam_gains->d_gxs,
src/source_components.cpp:          (gpuUserComplex* )components->gxs, num_pb_values*sizeof(gpuUserComplex),
src/source_components.cpp:                                                     gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_component_beam_gains->d_Dxs,
src/source_components.cpp:          (gpuUserComplex* )components->Dxs, num_pb_values*sizeof(gpuUserComplex),
src/source_components.cpp:                                                     gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_component_beam_gains->d_Dys,
src/source_components.cpp:          (gpuUserComplex* )components->Dys, num_pb_values*sizeof(gpuUserComplex),
src/source_components.cpp:                                                     gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_component_beam_gains->d_gys,
src/source_components.cpp:          (gpuUserComplex* )components->gys, num_pb_values*sizeof(gpuUserComplex),
src/source_components.cpp:                                                     gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XX_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XX_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XY_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_XY_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YX_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YX_imag,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YY_real,
src/source_components.cpp:  ( gpuMalloc( (void**)&d_sum_visi_YY_imag,
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_XX_real,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_XX_imag,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_XY_real,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_XY_imag,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_YX_real,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_YX_imag,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_YY_real,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:  ( gpuMemcpy(d_sum_visi_YY_imag,
src/source_components.cpp:             num_visis*sizeof(user_precision_t), gpuMemcpyHostToDevice ) );
src/source_components.cpp:    ( gpuMalloc( (void**)&d_ant_to_auto_map,
src/source_components.cpp:    ( gpuMemcpy(d_ant_to_auto_map, ant_to_auto_map,
src/source_components.cpp:                                    num_ants*sizeof(int), gpuMemcpyHostToDevice ));
src/source_components.cpp:  gpuErrorCheckKernel("kern_calc_autos",
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_XX_real,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_XY_real,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_YX_real,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_YY_real,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_XX_imag,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_XY_imag,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_YX_imag,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  ( gpuMemcpy(visibility_set->sum_visi_YY_imag,
src/source_components.cpp:                                                     gpuMemcpyDeviceToHost ));
src/source_components.cpp:  (  gpuFree( d_components->extrap_stokesI ) );
src/source_components.cpp:  (  gpuFree( d_components->extrap_stokesQ ) );
src/source_components.cpp:  (  gpuFree( d_components->extrap_stokesU ) );
src/source_components.cpp:  (  gpuFree( d_components->extrap_stokesV ) );
src/source_components.cpp:  (  gpuFree( d_component_beam_gains->d_gxs ) );
src/source_components.cpp:  (  gpuFree( d_component_beam_gains->d_Dxs ) );
src/source_components.cpp:  (  gpuFree( d_component_beam_gains->d_Dys ) );
src/source_components.cpp:  (  gpuFree( d_component_beam_gains->d_gys ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_XX_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_XX_imag ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_XY_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_XY_imag ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YX_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YX_imag ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YY_real ) );
src/source_components.cpp:  (  gpuFree( d_sum_visi_YY_imag ) );
src/source_components.cpp:    (  gpuFree( d_ant_to_auto_map ) );

```
