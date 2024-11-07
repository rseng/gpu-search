# https://github.com/ECP-WarpX/WarpX

```console
.zenodo.json:      "gpu",
.zenodo.json:      "Myers A, Almgren A, Amorim LD, Bell J, Fedeli L, Ge L, Gott K, Grote DP, Hogan M, Huebl A, Jambunathan R, Lehe R, Ng C, Rowan M, Shapoval O, Thevenet M, Vay JL, Vincenti H, Yang E, Zaim N, Zhang W, Zhao Y, Zoni E. Porting WarpX to GPU-accelerated platforms. Parallel Computing. 2021 Sep, 108:102833. https://doi.org/10.1016/j.parco.2021.102833",
.zenodo.json:      "Vay JL, Huebl A, Almgren A, Amorim LD, Bell J, Fedeli L, Ge L, Gott K, Grote DP, Hogan M, Jambunathan R, Lehe R, Myers A, Ng C, Rowan M, Shapoval O, Thevenet M, Vincenti H, Yang E, Zaim N, Zhang W, Zhao Y, Zoni E. Modeling of a chain of three plasma accelerator stages with the WarpX electromagnetic PIC code on GPUs. Physics of Plasmas. 2021 Feb 9, 28(2):023105. https://doi.org/10.1063/5.0028512",
.zenodo.json:      "Rowan ME, Gott KN, Deslippe J, Huebl A, Thevenet M, Lehe R, Vay JL. In-situ assessment of device-side compute work for dynamic load balancing in a GPU-accelerated PIC code. PASC '21: Proceedings of the Platform for Advanced Scientific Computing Conference. 2021 July, 10, pages 1-11. https://doi.org/10.1145/3468267.3470614",
Docs/Doxyfile:                         WARPX_USE_GPU=1 \
Docs/source/latex_theory/allbibs.bib:title = {{Accelerating a Spectral Algorithm for Plasma Physics with Python/Numba on GPU}},
Docs/source/latex_theory/allbibs.bib:url = {http://on-demand.gputechconf.com/gtc/2016/presentation/s6353-manuel-kirchen-spectral-algorithm-plasma-physics.pdf},
Docs/source/usage/faq.rst:When we start up WarpX, we report a couple of information on used MPI processes across parallel compute processes, CPU threads or GPUs and further capabilities.
Docs/source/usage/parameters.rst:    Note that when GPU threading is used,
Docs/source/usage/parameters.rst:* ``amrex.abort_on_out_of_gpu_memory``  (``0`` or ``1``; default is ``1`` for true)
Docs/source/usage/parameters.rst:    When running on GPUs, memory that does not fit on the device will be automatically swapped to host memory when this option is set to ``0``.
Docs/source/usage/parameters.rst:    `Please also see the documentation in AMReX <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__.
Docs/source/usage/parameters.rst:    When running on GPUs, device memory that is accessed from the host will automatically be transferred with managed memory.
Docs/source/usage/parameters.rst:    `Please also see the documentation in AMReX <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__.
Docs/source/usage/parameters.rst:    Particle weight factor used in `Heuristic` strategy for costs update; if running on GPU,
Docs/source/usage/parameters.rst:    the particle weight is set to a value determined from single-GPU tests on Summit,
Docs/source/usage/parameters.rst:    If running on CPU, the default value is `0.9`. If running on GPU, the default value is
Docs/source/usage/parameters.rst:    Cell weight factor used in `Heuristic` strategy for costs update; if running on GPU,
Docs/source/usage/parameters.rst:    the cell weight is set to a value determined from single-GPU tests on Summit,
Docs/source/usage/parameters.rst:    If running on CPU, the default value is `0.1`. If running on GPU, the default value is
Docs/source/usage/parameters.rst:* ``particles.do_tiling`` (`bool`) optional (default `false` if WarpX is compiled for GPUs, `true` otherwise)
Docs/source/usage/parameters.rst:    Tiling should be on when using OpenMP and off when using GPUs.
Docs/source/usage/parameters.rst:    When running in an accelerated platform, whether to call a ``amrex::Gpu::synchronize()`` around profiling regions.
Docs/source/usage/parameters.rst:* ``warpx.sort_intervals`` (`string`) optional (defaults: ``-1`` on CPU; ``4`` on GPU)
Docs/source/usage/parameters.rst:     It is turned on on GPUs for performance reasons (to improve memory locality).
Docs/source/usage/parameters.rst:* ``warpx.sort_particles_for_deposition`` (`bool`) optional (default: ``true`` for the CUDA backend, otherwise ``false``)
Docs/source/usage/parameters.rst:     ``true`` is recommend for best performance on NVIDIA GPUs, especially if there are many particles per cell.
Docs/source/usage/parameters.rst:     from particles. On GPUs these buffers will reside in ``__shared__``
Docs/source/usage/parameters.rst:     from particles. On GPUs these buffers will reside in ``__shared__``
Docs/source/usage/parameters.rst:     (e.g. for high particles per cell). This feature is only available for CUDA
Docs/source/usage/parameters.rst:    A typical example for `ADIOS2 output using lossless compression <https://openpmd-api.readthedocs.io/en/0.15.2/details/backendconfig.html#adios2>`__ with ``blosc`` using the ``zstd`` compressor and 6 CPU treads per MPI Rank (e.g. for a `GPU run with spare CPU resources <https://arxiv.org/abs/1706.00522>`__):
Docs/source/usage/parameters.rst:        <diag_name>.adios2_operator.parameters.nthreads = 6  # per MPI rank (and thus per GPU)
Docs/source/usage/parameters.rst:    diagnostics and visualization, the GPU may run out of memory with many large boxes with
Docs/source/usage/parameters.rst:    lab-frame snapshot data can be generated without running out of gpu memory.
Docs/source/usage/workflows/ml_materials/visualize.py:device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Docs/source/usage/workflows/ml_materials/train.py:device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Docs/source/usage/workflows/ml_materials/train.py:###### move data to device (GPU) if available ########
Docs/source/usage/workflows/ml_materials/run_warpx_training.py:    warpx_amrex_use_gpu_aware_mpi=True,
Docs/source/usage/workflows/debugging.rst:#. On Nvidia GPUs, if you suspect the problem might be a race condition due to a missing host / device synchronization, set the environment variable ``export CUDA_LAUNCH_BLOCKING=1`` and rerun.
Docs/source/usage/workflows/debugging.rst:   Particles shape does not fit within tile (CPU) or guard cells (GPU) used for charge deposition
Docs/source/usage/workflows/debugging.rst:   Particles shape does not fit within tile (CPU) or guard cells (GPU) used for current deposition
Docs/source/usage/workflows/python_extend.rst:               # guard/ghost region;     .to_cupy() for GPU!
Docs/source/usage/workflows/python_extend.rst:               # For GPUs use .to_cupy() above and compute with cupy or numba.
Docs/source/usage/workflows/python_extend.rst:For further details on how to `access GPU data <https://pyamrex.readthedocs.io/en/latest/usage/zerocopy.html>`__ or compute on ``E_x``, please see the `pyAMReX documentation <https://pyamrex.readthedocs.io/en/latest/usage/compute.html#fields>`__.
Docs/source/usage/workflows/python_extend.rst:               soa = pti.soa().to_cupy() if Config.have_gpu else \
Docs/source/usage/workflows/python_extend.rst:For further details on how to `access GPU data <https://pyamrex.readthedocs.io/en/latest/usage/zerocopy.html>`__ or compute on ``electrons``, please see the `pyAMReX documentation <https://pyamrex.readthedocs.io/en/latest/usage/compute.html#particles>`__.
Docs/source/usage/workflows/domain_decomposition.rst:In terms of performance, in general there is a trade off. Having many small boxes provides flexibility in terms of load balancing; however, the cost is increased time spent in communication due to surface-to-volume effects and increased kernel launch overhead when running on the GPUs. The ideal number of boxes per rank depends on how important dynamic load balancing is on your problem. If your problem is intrinsically well-balanced, like in a uniform plasma, then having a few, large boxes is best. But, if the problem is non-uniform and achieving a good load balance is critical for performance, having more, smaller `Boxes` can be worth it. In general, we find that running with something in the range of 4-8 `Boxes` per process is a good compromise for most problems.
Docs/source/usage/workflows/domain_decomposition.rst:* GPU or CPU
Docs/source/usage/examples/ohm_solver_magnetic_reconnection/README.rst:Running the full simulation should take about 4 hours if executed on 1 V100 GPU.
Docs/source/usage/examples/laser_ion/README.rst:   The following images for densities and electromagnetic fields were created with a run on 64 NVidia A100 GPUs featuring a total number of cells of ``nx = 8192`` and ``nz = 16384``, as well as 64 particles per cell per species.
Docs/source/usage/examples/laser_ion/inputs_test_2d_laser_ion_acc:#   Use larger values for GPUs, try to fill a GPU well with memory and place
Docs/source/usage/examples/laser_ion/inputs_test_2d_laser_ion_acc:# particle bin-sorting on GPU (ideal defaults not investigated in 2D)
Docs/source/usage/examples/laser_ion/inputs_test_2d_laser_ion_acc:#warpx.sort_intervals = 4    # default on CPU: -1 (off); on GPU: 4
Docs/source/usage/examples/laser_ion/inputs_test_2d_laser_ion_acc_picmi.py:# --> choose larger `max_grid_size` and `blocking_factor` for 1 to 8 grids per GPU accordingly
Docs/source/install/users.rst:   The ``warpx`` `conda package <https://anaconda.org/conda-forge/warpx>`__ does not yet provide GPU support.
Docs/source/install/users.rst:   # optional arguments:  -mpi ^warpx dims=2 compute=cuda
Docs/source/install/cmake.rst:For example, this builds WarpX in all geometries, enables Python bindings and Nvidia GPU (CUDA) support:
Docs/source/install/cmake.rst:   cmake -S . -B build -DWarpX_DIMS="1;2;RZ;3" -DWarpX_COMPUTE=CUDA
Docs/source/install/cmake.rst:``WarpX_COMPUTE``             NOACC/**OMP**/CUDA/SYCL/HIP                  On-node, accelerated computing backend
Docs/source/install/cmake.rst:* `GPU-specific options <https://amrex-codes.github.io/amrex/docs_html/GPU.html#building-gpu-support>`__.
Docs/source/install/cmake.rst:``AMReX_CUDA_PTX_VERBOSE``    ON/**OFF**                                     Print CUDA code generation statistics from ``ptxas``.
Docs/source/install/cmake.rst:If you also want to select a CUDA compiler:
Docs/source/install/cmake.rst:   export CUDACXX=$(which nvcc)
Docs/source/install/cmake.rst:   export CUDAHOSTCXX=$(which clang++)
Docs/source/install/cmake.rst:``WARPX_COMPUTE``             NOACC/**OMP**/CUDA/SYCL/HIP                  On-node, accelerated computing backend
Docs/source/install/hpc/lassen.rst:The `Lassen V100 GPU cluster <https://hpc.llnl.gov/hardware/platforms/lassen>`__ is located at LLNL.
Docs/source/install/hpc/lassen.rst:   source /usr/workspace/${USER}/lassen-toss3/gpu/venvs/warpx-lassen-toss3/bin/activate
Docs/source/install/hpc/lassen.rst:   cmake -S . -B build_lassen -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/lassen.rst:   cmake -S . -B build_lassen_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/lassen.rst:.. _running-cpp-lassen-V100-GPUs:
Docs/source/install/hpc/lassen.rst:V100 GPUs (16GB)
Docs/source/install/hpc/lassen.rst:Note that the only option so far is to run with one MPI rank per GPU.
Docs/source/install/hpc/lassen.rst:solver on V100 GPUs for a well load-balanced problem (in our case laser
Docs/source/install/hpc/lassen.rst:* **One MPI rank per GPU** (e.g., 4 MPI ranks for the 4 GPUs on each Lassen
Docs/source/install/hpc/lassen.rst:* **Two `128x128x128` grids per GPU**, or **one `128x128x256` grid per GPU**.
Docs/source/install/hpc/juwels.rst:   cmake -S . -B build -DWarpX_DIMS="1;2;3" -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_MPI_THREAD_MULTIPLE=OFF
Docs/source/install/hpc/juwels.rst:Queue: gpus (4 x Nvidia V100 GPUs)
Docs/source/install/hpc/juwels.rst:The `Juwels GPUs <https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html>`__ are V100 (16GB) and A100 (40GB).
Docs/source/install/hpc/leonardo.rst:* 4 x NVidia custom Ampere A100 GPU 64GB HBM2
Docs/source/install/hpc/leonardo.rst:* 2 x NVidia HDR 2×100 GB/s cards
Docs/source/install/hpc/leonardo.rst:We use system software modules, add environment hints and further dependencies via the file ``$HOME/leonardo_gpu_warpx.profile``.
Docs/source/install/hpc/leonardo.rst:   cp $HOME/src/warpx/Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example $HOME/leonardo_gpu_warpx.profile
Docs/source/install/hpc/leonardo.rst:   .. literalinclude:: ../../../../Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example
Docs/source/install/hpc/leonardo.rst:      source $HOME/leonardo_gpu_warpx.profile
Docs/source/install/hpc/leonardo.rst:   bash $HOME/src/warpx/Tools/machines/leonardo-cineca/install_gpu_dependencies.sh
Docs/source/install/hpc/leonardo.rst:   .. literalinclude:: ../../../../Tools/machines/leonardo-cineca/install_gpu_dependencies.sh
Docs/source/install/hpc/leonardo.rst:   rm -rf build_gpu
Docs/source/install/hpc/leonardo.rst:   cmake -S . -B build_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/leonardo.rst:   cmake --build build_gpu -j 16
Docs/source/install/hpc/leonardo.rst:The WarpX application executables are now in ``$HOME/src/warpx/build_gpu/bin/``.
Docs/source/install/hpc/leonardo.rst:   rm -rf build_gpu_py
Docs/source/install/hpc/leonardo.rst:   cmake -S . -B build_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_PYTHON=ON -DWarpX_APP=OFF -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/leonardo.rst:   cmake --build build_gpu_py -j 16 --target pip_install
Docs/source/install/hpc/leonardo.rst:- :ref:`update the leonardo_gpu_warpx.profile file <building-leonardo-preparation>`,
Docs/source/install/hpc/leonardo.rst:As a last step, clean the build directories ``rm -rf $HOME/src/warpx/build_gpu*`` and rebuild WarpX.
Docs/source/install/hpc/leonardo.rst:Note that we run one MPI rank per GPU.
Docs/source/install/hpc/leonardo.rst:  source $HOME/leonardo_gpu_warpx.profile
Docs/source/install/hpc/hpc3.rst:On HPC3, you recommend to run on the `fast GPU nodes with V100 GPUs <https://rcic.uci.edu/hpc3/slurm.html#memmap>`__.
Docs/source/install/hpc/hpc3.rst:We use system software modules, add environment hints and further dependencies via the file ``$HOME/hpc3_gpu_warpx.profile``.
Docs/source/install/hpc/hpc3.rst:  cp $HOME/src/warpx/Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example $HOME/hpc3_gpu_warpx.profile
Docs/source/install/hpc/hpc3.rst:  .. literalinclude:: ../../../../Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example
Docs/source/install/hpc/hpc3.rst:For example, if you are member of the project ``plasma``, then run ``vi $HOME/hpc3_gpu_warpx.profile``.
Docs/source/install/hpc/hpc3.rst:      source $HOME/hpc3_gpu_warpx.profile
Docs/source/install/hpc/hpc3.rst:   bash $HOME/src/warpx/Tools/machines/hpc3-uci/install_gpu_dependencies.sh
Docs/source/install/hpc/hpc3.rst:   source $HOME/sw/hpc3/gpu/venvs/warpx-gpu/bin/activate
Docs/source/install/hpc/hpc3.rst:   .. literalinclude:: ../../../../Tools/machines/hpc3-uci/install_gpu_dependencies.sh
Docs/source/install/hpc/hpc3.rst:   cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/hpc3.rst:   cmake -S . -B build_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/hpc3.rst:- :ref:`update the hpc3_gpu_warpx.profile file <building-hpc3-preparation>`,
Docs/source/install/hpc/hpc3.rst:This partition as up to `32 nodes <https://rcic.uci.edu/hpc3/slurm.html#memmap>`__ with four V100 GPUs (16 GB each) per node.
Docs/source/install/hpc/hpc3.rst:Note that we run one MPI rank per GPU.
Docs/source/install/hpc/hpc3.rst:.. literalinclude:: ../../../../Tools/machines/hpc3-uci/hpc3_gpu.sbatch
Docs/source/install/hpc/hpc3.rst:   :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/hpc3-uci/hpc3_gpu.sbatch``.
Docs/source/install/hpc/hpc3.rst:To run a simulation, copy the lines above to a file ``hpc3_gpu.sbatch`` and run
Docs/source/install/hpc/hpc3.rst:   sbatch hpc3_gpu.sbatch
Docs/source/install/hpc/adastra.rst:Each node contains 4 AMD MI250X GPUs, each with 2 Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.
Docs/source/install/hpc/adastra.rst:You can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).
Docs/source/install/hpc/adastra.rst:   source $SHAREDHOMEDIR/sw/adastra/gpu/venvs/warpx-adastra/bin/activate
Docs/source/install/hpc/adastra.rst:.. _running-cpp-adastra-MI250X-GPUs:
Docs/source/install/hpc/adastra.rst:MI250X GPUs (2x64 GB)
Docs/source/install/hpc/adastra.rst:   rocFFT in ROCm 5.1-5.3 tries to `write to a cache <https://rocfft.readthedocs.io/en/latest/#runtime-compilation>`__ in the home area by default.
Docs/source/install/hpc/adastra.rst:   We discovered a regression in AMD ROCm, leading to 2x slower current deposition (and other slowdowns) in ROCm 5.3 and 5.4.
Docs/source/install/hpc/adastra.rst:   Reported to AMD and fixed for the next release of ROCm.
Docs/source/install/hpc/adastra.rst:   Stay with the ROCm 5.2 module to avoid.
Docs/source/install/hpc/lonestar6.rst:   rm -rf build_pm_gpu
Docs/source/install/hpc/lonestar6.rst:   cmake -S . -B build_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/lonestar6.rst:   cmake --build build_gpu -j 16
Docs/source/install/hpc/lonestar6.rst:The WarpX application executables are now in ``$HOME/src/warpx/build_gpu/bin/``.
Docs/source/install/hpc/lonestar6.rst:   rm -rf build_pm_gpu_py
Docs/source/install/hpc/lonestar6.rst:   cmake -S . -B build_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/lonestar6.rst:   cmake --build build_gpu_py -j 16 --target pip_install
Docs/source/install/hpc/lonestar6.rst:.. _running-cpp-lonestar6-A100-GPUs:
Docs/source/install/hpc/lonestar6.rst:A100 GPUs (40 GB)
Docs/source/install/hpc/lonestar6.rst:`84 GPU nodes, each with 2 A100 GPUs (40 GB) <https://portal.tacc.utexas.edu/user-guides/lonestar6#system-gpu>`__.
Docs/source/install/hpc/lonestar6.rst:Note that we run one MPI rank per GPU.
Docs/source/install/hpc/taurus.rst:The cluster has multiple partitions, this section describes how to use the `AMD Rome CPUs + NVIDIA A100¶ <https://doc.zih.tu-dresden.de/jobs_and_resources/hardware_overview/#amd-rome-cpus-nvidia-a100>`__.
Docs/source/install/hpc/taurus.rst:   cmake -S . -B build -DWarpX_DIMS="1;2;3" -DWarpX_COMPUTE=CUDA
Docs/source/install/hpc/taurus.rst:.. _running-cpp-taurus-A100-GPUs:
Docs/source/install/hpc/taurus.rst:A100 GPUs (40 GB)
Docs/source/install/hpc/taurus.rst:The `alpha` partition has 34 nodes with 8 x NVIDIA A100-SXM4 Tensor Core-GPUs and 2 x AMD EPYC CPU 7352 (24 cores) @ 2.3 GHz (multithreading disabled) per node.
Docs/source/install/hpc/taurus.rst:Note that we run one MPI rank per GPU.
Docs/source/install/hpc/summit.rst:On Summit, each compute node provides six V100 GPUs (16GB) and two Power9 CPUs.
Docs/source/install/hpc/summit.rst:   bash $HOME/src/warpx/Tools/machines/summit-olcf/install_gpu_dependencies.sh
Docs/source/install/hpc/summit.rst:   source /ccs/proj/$proj/${USER}/sw/summit/gpu/venvs/warpx-summit/bin/activate
Docs/source/install/hpc/summit.rst:   .. literalinclude:: ../../../../Tools/machines/summit-olcf/install_gpu_dependencies.sh
Docs/source/install/hpc/summit.rst:      runNode bash $HOME/src/warpx/Tools/machines/summit-olcf/install_gpu_ml.sh
Docs/source/install/hpc/summit.rst:      .. literalinclude:: ../../../../Tools/machines/summit-olcf/install_gpu_ml.sh
Docs/source/install/hpc/summit.rst:   cmake -S . -B build_summit -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/summit.rst:   cmake -S . -B build_summit_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/summit.rst:.. _running-cpp-summit-V100-GPUs:
Docs/source/install/hpc/summit.rst:V100 GPUs (16GB)
Docs/source/install/hpc/summit.rst:Note that WarpX runs with one MPI rank per GPU and there are 6 GPUs per node:
Docs/source/install/hpc/summit.rst:* **One MPI rank per GPU** (e.g., 6 MPI ranks for the 6 GPUs on each Summit
Docs/source/install/hpc/summit.rst:* **Two `128x128x128` grids per GPU**, or **one `128x128x256` grid per GPU**.
Docs/source/install/hpc/summit.rst:1 node on the supercomputer Summit at OLCF, on Power9 CPUs (i.e., the GPUs are
Docs/source/install/hpc/lxplus.rst:Through LXPLUS we have access to CPU and GPU nodes (the latter equipped with NVIDIA V100 and T4 GPUs).
Docs/source/install/hpc/lxplus.rst:If the GPU support or the Python bindings are not needed, it's possible to skip the installation by respectively setting
Docs/source/install/hpc/lxplus.rst:the following environment variables export ``SPACK_STACK_USE_PYTHON=0`` and ``export SPACK_STACK_USE_CUDA = 0`` before
Docs/source/install/hpc/lxplus.rst:The environment ``warpx-lxplus`` (or ``-cuda`` or ``-cuda-py``) must be reactivated everytime that we log in so it could
Docs/source/install/hpc/lxplus.rst:Or if we need to compile with CUDA:
Docs/source/install/hpc/lxplus.rst:    cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/lxplus.rst:Then we compile WarpX as in the previous section (with or without CUDA) adding ``-DWarpX_PYTHON=ON`` and then we install it into our Python:
Docs/source/install/hpc/lxplus.rst:   cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_DIMS="1;2;RZ;3" -DWarpX_APP=OFF -DWarpX_PYTHON=ON
Docs/source/install/hpc/pitzer.rst:The `Pitzer cluster <https://www.osc.edu/supercomputing/computing/pitzer>`__ is located at the Ohio Supercomputer Center (OSC). It is currently the main CPU/GPU cluster at OSC. However, the `Cardinal cluster <https://www.osc.edu/resources/technical_support/supercomputers/cardinal>`__ is soon going to take over Pitzer to become the next major CPU/GPU cluster at OSC in the second half of 2024. A list of all OSC clusters can be found `here <https://www.osc.edu/services/cluster_computing>`__.
Docs/source/install/hpc/pitzer.rst:The Pitzer cluster offers a variety of partitions suitable for different computational needs, including GPU nodes, CPU nodes, and nodes with large memory capacities. For more information on the specifications and capabilities of these partitions, visit the `Ohio Supercomputer Center's Pitzer page <https://www.osc.edu/supercomputing/computing/pitzer>`__.
Docs/source/install/hpc/pitzer.rst:On Pitzer, you can run either on GPU nodes with V100 GPUs or CPU nodes.
Docs/source/install/hpc/pitzer.rst:   .. tab-item:: V100 GPUs
Docs/source/install/hpc/pitzer.rst:   .. tab-item:: V100 GPUs
Docs/source/install/hpc/pitzer.rst:         cmake -S . -B build_v100 -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/pitzer.rst:         cmake -S . -B build_v100_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/pitzer.rst:   .. tab-item:: V100 GPUs
Docs/source/install/hpc/pitzer.rst:      Pitzer's GPU partition includes:
Docs/source/install/hpc/pitzer.rst:      - 32 nodes, each equipped with two V100 (16GB) GPUs.
Docs/source/install/hpc/pitzer.rst:      - 42 nodes, each with two V100 (32GB) GPUs.
Docs/source/install/hpc/pitzer.rst:      - 4 large memory nodes, each with quad V100 (32GB) GPUs.
Docs/source/install/hpc/pitzer.rst:      To run a WarpX simulation on the GPU nodes, use the batch script provided below. Adjust the ``-N`` parameter in the script to match the number of nodes you intend to use. Each node in this partition supports running one MPI rank per GPU.
Docs/source/install/hpc/polaris.rst:On Polaris, you can run either on GPU nodes with fast A100 GPUs (recommended) or CPU nodes.
Docs/source/install/hpc/polaris.rst:   .. tab-item:: A100 GPUs
Docs/source/install/hpc/polaris.rst:      We use system software modules, add environment hints and further dependencies via the file ``$HOME/polaris_gpu_warpx.profile``.
Docs/source/install/hpc/polaris.rst:         cp $HOME/src/warpx/Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example $HOME/polaris_gpu_warpx.profile
Docs/source/install/hpc/polaris.rst:         .. literalinclude:: ../../../../Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example
Docs/source/install/hpc/polaris.rst:      For example, if you are member of the project ``proj_name``, then run ``nano $HOME/polaris_gpu_warpx.profile`` and edit line 2 to read:
Docs/source/install/hpc/polaris.rst:            source $HOME/polaris_gpu_warpx.profile
Docs/source/install/hpc/polaris.rst:         bash $HOME/src/warpx/Tools/machines/polaris-alcf/install_gpu_dependencies.sh
Docs/source/install/hpc/polaris.rst:         source ${CFS}/${proj%_g}/${USER}/sw/polaris/gpu/venvs/warpx/bin/activate
Docs/source/install/hpc/polaris.rst:         .. literalinclude:: ../../../../Tools/machines/polaris-alcf/install_gpu_dependencies.sh
Docs/source/install/hpc/polaris.rst:   .. tab-item:: A100 GPUs
Docs/source/install/hpc/polaris.rst:         rm -rf build_pm_gpu
Docs/source/install/hpc/polaris.rst:         cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/polaris.rst:         cmake --build build_pm_gpu -j 16
Docs/source/install/hpc/polaris.rst:      The WarpX application executables are now in ``$HOME/src/warpx/build_pm_gpu/bin/``.
Docs/source/install/hpc/polaris.rst:         rm -rf build_pm_gpu_py
Docs/source/install/hpc/polaris.rst:         cmake -S . -B build_pm_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/polaris.rst:         cmake --build build_pm_gpu_py -j 16 --target pip_install
Docs/source/install/hpc/polaris.rst:- :ref:`update the polaris_gpu_warpx.profile or polaris_cpu_warpx files <building-polaris-preparation>`,
Docs/source/install/hpc/polaris.rst:   .. tab-item:: A100 (40GB) GPUs
Docs/source/install/hpc/polaris.rst:      Note that we run one MPI rank per GPU.
Docs/source/install/hpc/polaris.rst:      .. literalinclude:: ../../../../Tools/machines/polaris-alcf/polaris_gpu.pbs
Docs/source/install/hpc/polaris.rst:         :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/polaris-alcf/polaris_gpu.pbs``.
Docs/source/install/hpc/polaris.rst:      To run a simulation, copy the lines above to a file ``polaris_gpu.pbs`` and run
Docs/source/install/hpc/polaris.rst:         qsub polaris_gpu.pbs
Docs/source/install/hpc/tioga.rst:The `Tioga AMD GPU cluster <https://hpc.llnl.gov/hardware/compute-platforms/tioga>`__ is located at LLNL.
Docs/source/install/hpc/tioga.rst:There are also "conventional" MI250X GPUs on Tioga nodes, which we did not yet document.
Docs/source/install/hpc/tioga.rst:El Capitan will use MI300A GPUs.
Docs/source/install/hpc/tioga.rst:     source /p/lustre1/${USER}/tioga/warpx/mi300a/gpu/venvs/warpx-trioga-mi300a/bin/activate
Docs/source/install/hpc/tioga.rst:WarpX runs with one MPI rank per GPU.
Docs/source/install/hpc/tioga.rst:* ``amrex.use_gpu_aware_mpi=1``: make use of fast APU to APU MPI communications
Docs/source/install/hpc/tioga.rst:* ``amrex.the_arena_init_size=1``: avoid overallocating memory that is *shared* on APUs between CPU & GPU
Docs/source/install/hpc/frontier.rst:On Frontier, each compute node provides four AMD MI250X GPUs, each with two Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.
Docs/source/install/hpc/frontier.rst:You can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).
Docs/source/install/hpc/frontier.rst:   source $HOME/sw/frontier/gpu/venvs/warpx-frontier/bin/activate
Docs/source/install/hpc/frontier.rst:.. _running-cpp-frontier-MI250X-GPUs:
Docs/source/install/hpc/frontier.rst:MI250X GPUs (2x64 GB)
Docs/source/install/hpc/frontier.rst:   rocFFT in ROCm 5.1-5.3 tries to `write to a cache <https://rocfft.readthedocs.io/en/latest/#runtime-compilation>`__ in the home area by default.
Docs/source/install/hpc/frontier.rst:   We discovered a regression in AMD ROCm, leading to 2x slower current deposition (and other slowdowns) in ROCm 5.3 and 5.4.
Docs/source/install/hpc/frontier.rst:   Although a fix was planned for ROCm 5.5, we still see the same issue in this release and continue to exchange with AMD and HPE on the issue.
Docs/source/install/hpc/frontier.rst:   Stay with the ROCm 5.2 module to avoid a 2x slowdown.
Docs/source/install/hpc/frontier.rst:      amrex.async_out_nfiles = 4096  # set to number of GPUs used
Docs/source/install/hpc/karolina.rst:On Karolina, you can run either on GPU nodes with fast A100 GPUs (recommended) or CPU nodes.
Docs/source/install/hpc/karolina.rst:   rm -rf build_gpu
Docs/source/install/hpc/karolina.rst:   cmake -S . -B build_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/karolina.rst:   cmake --build build_gpu -j 48
Docs/source/install/hpc/karolina.rst:The WarpX application executables are now in ``$WORK/src/warpx/build_gpu/bin/``.
Docs/source/install/hpc/karolina.rst:   rm -rf build_gpu_py
Docs/source/install/hpc/karolina.rst:   cmake -S . -B build_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/karolina.rst:   cmake --build build_gpu_py -j 48 --target pip_install
Docs/source/install/hpc/karolina.rst:The batch script below can be used to run a WarpX simulation on multiple GPU nodes (change ``#SBATCH --nodes=`` accordingly) on the supercomputer Karolina at IT4I.
Docs/source/install/hpc/karolina.rst:Every node has 8x A100 (40GB) GPUs and 2x AMD EPYC 7763, 64-core, 2.45 GHz processors.
Docs/source/install/hpc/karolina.rst:Note that we run one MPI rank per GPU.
Docs/source/install/hpc/karolina.rst:.. literalinclude:: ../../../../Tools/machines/karolina-it4i/karolina_gpu.sbatch
Docs/source/install/hpc/karolina.rst:   :caption: You can copy this file from ``$WORK/src/warpx/Tools/machines/karolina-it4i/karolina_gpu.sbatch``.
Docs/source/install/hpc/karolina.rst:To run a simulation, copy the lines above to a file ``karolina_gpu.sbatch`` and run
Docs/source/install/hpc/karolina.rst:   sbatch karolina_gpu.sbatch
Docs/source/install/hpc/crusher.rst:On Crusher, each compute node provides four AMD MI250X GPUs, each with two Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.
Docs/source/install/hpc/crusher.rst:You can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).
Docs/source/install/hpc/crusher.rst:   source $HOME/sw/crusher/gpu/venvs/warpx-crusher/bin/activate
Docs/source/install/hpc/crusher.rst:.. _running-cpp-crusher-MI250X-GPUs:
Docs/source/install/hpc/crusher.rst:MI250X GPUs (2x64 GB)
Docs/source/install/hpc/perlmutter.rst:On Perlmutter, you can run either on GPU nodes with fast A100 GPUs (recommended) or CPU nodes.
Docs/source/install/hpc/perlmutter.rst:   .. tab-item:: A100 GPUs
Docs/source/install/hpc/perlmutter.rst:      We use system software modules, add environment hints and further dependencies via the file ``$HOME/perlmutter_gpu_warpx.profile``.
Docs/source/install/hpc/perlmutter.rst:         cp $HOME/src/warpx/Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example $HOME/perlmutter_gpu_warpx.profile
Docs/source/install/hpc/perlmutter.rst:         .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example
Docs/source/install/hpc/perlmutter.rst:      Perlmutter GPU projects must end in ``..._g``.
Docs/source/install/hpc/perlmutter.rst:      For example, if you are member of the project ``m3239``, then run ``nano $HOME/perlmutter_gpu_warpx.profile`` and edit line 2 to read:
Docs/source/install/hpc/perlmutter.rst:            source $HOME/perlmutter_gpu_warpx.profile
Docs/source/install/hpc/perlmutter.rst:         bash $HOME/src/warpx/Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh
Docs/source/install/hpc/perlmutter.rst:         source ${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/venvs/warpx-gpu/bin/activate
Docs/source/install/hpc/perlmutter.rst:         .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh
Docs/source/install/hpc/perlmutter.rst:   .. tab-item:: A100 GPUs
Docs/source/install/hpc/perlmutter.rst:         rm -rf build_pm_gpu
Docs/source/install/hpc/perlmutter.rst:         cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/perlmutter.rst:         cmake --build build_pm_gpu -j 16
Docs/source/install/hpc/perlmutter.rst:      The WarpX application executables are now in ``$HOME/src/warpx/build_pm_gpu/bin/``.
Docs/source/install/hpc/perlmutter.rst:         rm -rf build_pm_gpu_py
Docs/source/install/hpc/perlmutter.rst:         cmake -S . -B build_pm_gpu_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/perlmutter.rst:         cmake --build build_pm_gpu_py -j 16 --target pip_install
Docs/source/install/hpc/perlmutter.rst:- :ref:`update the perlmutter_gpu_warpx.profile or perlmutter_cpu_warpx files <building-perlmutter-preparation>`,
Docs/source/install/hpc/perlmutter.rst:   .. tab-item:: A100 (40GB) GPUs
Docs/source/install/hpc/perlmutter.rst:      Note that we run one MPI rank per GPU.
Docs/source/install/hpc/perlmutter.rst:      .. literalinclude:: ../../../../Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch
Docs/source/install/hpc/perlmutter.rst:         :caption: You can copy this file from ``$HOME/src/warpx/Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch``.
Docs/source/install/hpc/perlmutter.rst:      To run a simulation, copy the lines above to a file ``perlmutter_gpu.sbatch`` and run
Docs/source/install/hpc/perlmutter.rst:         sbatch perlmutter_gpu.sbatch
Docs/source/install/hpc/perlmutter.rst:   .. tab-item:: A100 (80GB) GPUs
Docs/source/install/hpc/perlmutter.rst:      Perlmutter has `256 nodes <https://docs.nersc.gov/systems/perlmutter/architecture/>`__ that provide 80 GB HBM per A100 GPU.
Docs/source/install/hpc/perlmutter.rst:      In the A100 (40GB) batch script, replace ``-C gpu`` with ``-C gpu&hbm80g`` to use these large-memory GPUs.
Docs/source/install/hpc/greatlakes.rst:The cluster has various partitions, including `GPU nodes and CPU nodes <https://arc.umich.edu/greatlakes/configuration/>`__.
Docs/source/install/hpc/greatlakes.rst:On Great Lakes, you can run either on GPU nodes with `fast V100 GPUs (recommended), the even faster A100 GPUs (only a few available) or CPU nodes <https://arc.umich.edu/greatlakes/configuration/>`__.
Docs/source/install/hpc/greatlakes.rst:   .. tab-item:: V100 GPUs
Docs/source/install/hpc/greatlakes.rst:   .. tab-item:: V100 GPUs
Docs/source/install/hpc/greatlakes.rst:         cmake -S . -B build_v100 -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/greatlakes.rst:         cmake -S . -B build_v100_py -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS="1;2;RZ;3"
Docs/source/install/hpc/greatlakes.rst:   .. tab-item:: V100 (16GB) GPUs
Docs/source/install/hpc/greatlakes.rst:      This partition has `20 nodes, each with two V100 GPUs <https://arc.umich.edu/greatlakes/configuration/>`__.
Docs/source/install/hpc/greatlakes.rst:      Note that we run one MPI rank per GPU.
Docs/source/install/hpc/greatlakes.rst:   .. tab-item:: A100 (80GB) GPUs
Docs/source/install/hpc/greatlakes.rst:      This partition has `2 nodes, each with four A100 GPUs <https://arc.umich.edu/greatlakes/configuration/>`__ that provide 80 GB HBM per A100 GPU.
Docs/source/install/hpc/greatlakes.rst:      To the user, each node will appear as if it has 8 A100 GPUs with 40 GB memory each.
Docs/source/install/hpc/lawrencium.rst:   cmake -S src/blaspp -B src/blaspp-v100-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=$HOME/sw/v100/blaspp-master
Docs/source/install/hpc/lawrencium.rst:   cmake -S src/lapackpp -B src/lapackpp-v100-build -DCMAKE_CXX_STANDARD=17 -Dgpu_backend=cuda -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=$HOME/sw/v100/lapackpp-master -Duse_cmake_find_lapack=ON -DBLAS_LIBRARIES=${LAPACK_DIR}/lib/libblas.a -DLAPACK_LIBRARIES=${LAPACK_DIR}/lib/liblapack.a
Docs/source/install/hpc/lawrencium.rst:   cmake -S . -B build -DWarpX_DIMS="1;2;RZ;3" -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_QED_TABLE_GEN=ON
Docs/source/install/hpc/lawrencium.rst:   WARPX_MPI=ON WARPX_COMPUTE=CUDA WARPX_FFT=ON BUILD_PARALLEL=12 python3 -m pip install --force-reinstall --no-deps -v .
Docs/source/install/hpc/lawrencium.rst:   cmake -S . -B build -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_APP=OFF -DWarpX_PYTHON=ON -DWarpX_DIMS=RZ
Docs/source/install/hpc/lawrencium.rst:.. _running-cpp-lawrencium-V100-GPUs:
Docs/source/install/hpc/lawrencium.rst:V100 GPUs (16 GB)
Docs/source/install/hpc/lawrencium.rst:12 nodes with each two NVIDIA V100 GPUs.
Docs/source/install/hpc/lawrencium.rst:.. _running-cpp-lawrencium-2080Ti-GPUs:
Docs/source/install/hpc/lawrencium.rst:2080 Ti GPUs (10 GB)
Docs/source/install/hpc/lawrencium.rst:18 nodes with each four NVIDIA 2080 TI GPUs.
Docs/source/install/hpc/lumi.rst:Each node contains 4 AMD MI250X GPUs, each with 2 Graphics Compute Dies (GCDs) for a total of 8 GCDs per node.
Docs/source/install/hpc/lumi.rst:You can think of the 8 GCDs as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).
Docs/source/install/hpc/lumi.rst:   source $HOME/sw/lumi/gpu/venvs/warpx-lumi/bin/activate
Docs/source/install/hpc/lumi.rst:.. _running-cpp-lumi-MI250X-GPUs:
Docs/source/install/hpc/lumi.rst:MI250X GPUs (2x64 GB)
Docs/source/install/hpc/lumi.rst:The GPU partition on the supercomputer LUMI at CSC has up to `2978 nodes <https://docs.lumi-supercomputer.eu/hardware/lumig/>`__, each with 8 Graphics Compute Dies (GCDs).
Docs/source/install/hpc/lumi.rst:   We discovered a regression in AMD ROCm, leading to 2x slower current deposition (and other slowdowns) in ROCm 5.3 and 5.4.
Docs/source/install/hpc/lumi.rst:   Although a fix was planned for ROCm 5.5, we still see the same issue in this release and continue to exchange with AMD and HPE on the issue.
Docs/source/install/hpc/lumi.rst:   Stay with the ROCm 5.2 module to avoid a 2x slowdown.
Docs/source/install/hpc/lumi.rst:   rocFFT in ROCm 5.1-5.3 tries to `write to a cache <https://rocfft.readthedocs.io/en/latest/#runtime-compilation>`__ in the home area by default.
Docs/source/install/batch/slurm.rst:  * GPU allocation on most machines require additional flags, e.g. ``--gpus-per-task=1`` or ``--gres=...``
Docs/source/install/dependencies.rst:- `MPI 3.0+ <https://www.mpi-forum.org/docs/>`__: for multi-node and/or multi-GPU execution
Docs/source/install/dependencies.rst:  - `CUDA Toolkit 11.7+ <https://developer.nvidia.com/cuda-downloads>`__: for Nvidia GPU support (see `matching host-compilers <https://gist.github.com/ax3l/9489132>`__) or
Docs/source/install/dependencies.rst:  - `ROCm 5.2+ (5.5+ recommended) <https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-rocm-installation-readme/>`__: for AMD GPU support
Docs/source/install/dependencies.rst:- `CCache <https://ccache.dev>`__: to speed up rebuilds (For CUDA support, needs version 3.7.9+ and 4.2+ is recommended)
Docs/source/install/dependencies.rst:For Nvidia CUDA GPU support, you will need to have `a recent CUDA driver installed <https://developer.nvidia.com/cuda-downloads>`__ or you can lower the CUDA version of `the Nvidia cuda package <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation>`__ and `conda-forge to match your drivers <https://docs.cupy.dev/en/stable/install.html#install-cupy-from-conda-forge>`__ and then add these packages:
Docs/source/install/dependencies.rst:   conda install -c nvidia -c conda-forge cuda cuda-nvtx-dev cupy
Docs/source/install/dependencies.rst:More info for `CUDA-enabled ML packages <https://twitter.com/jeremyphoward/status/1697435241152127369>`__.
Docs/source/install/dependencies.rst:For most desktop developments, pick the OpenMP environment for CPUs unless you have a supported GPU.
Docs/source/install/dependencies.rst:  * CUDA: ``system=ubuntu; compute=cuda`` (Nvidia GPUs)
Docs/source/install/dependencies.rst:  * ROCm: ``system=ubuntu; compute=rocm`` (AMD GPUs)
Docs/source/install/dependencies.rst:  * SYCL: *todo* (Intel GPUs)
Docs/source/install/dependencies.rst:         # for CUDA, either install
Docs/source/install/dependencies.rst:         #   https://developer.nvidia.com/cuda-downloads (preferred)
Docs/source/install/dependencies.rst:         #   sudo apt install nvidia-cuda-dev libcub-dev
Docs/source/install/dependencies.rst:         # for CUDA, either install
Docs/source/install/dependencies.rst:         #   https://developer.nvidia.com/cuda-downloads (preferred)
Docs/source/install/dependencies.rst:         #   sudo apt install nvidia-cuda-dev libcub-dev
Docs/source/refs.bib:title = {{Synthesizing Particle-In-Cell Simulations through Learning and GPU Computing for Hybrid Particle Accelerator Beamlines}},
Docs/source/index.rst:WarpX is a *highly-parallel and highly-optimized code*, which can run on GPUs and multi-core CPUs, and includes load balancing capabilities.
Docs/source/dataanalysis/ascent.rst:It provides rendering runtimes that can leverage many-core CPUs and GPUs to render images of simulation meshes.
Docs/source/dataanalysis/ez2d.session:                        <Field name="GPUsPerNode" type="int">1</Field>
Docs/source/developers/gnumake.rst:    * ``USE_GPU=TRUE`` or ``FALSE``: Whether to compile for Nvidia GPUs (requires CUDA).
Docs/source/developers/gnumake.rst:   gnumake/gpu_local
Docs/source/developers/faq.rst:- want to write to CPU memory from a GPU kernel
Docs/source/developers/faq.rst:Read more on this here: `How to Optimize Data Transfers in CUDA C/C++ <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`__ (note that pinned memory is a host memory feature and works with all GPU vendors we support)
Docs/source/developers/faq.rst:One of the benefits of GPU-aware MPI implementations is, besides the possibility to use direct device-device transfers, that MPI and GPU API calls `are aware of each others' pinning ambitions <https://www.open-mpi.org/community/lists/users/2012/11/20659.php>`__ and do not create `data races to unpin the same memory <https://github.com/ComputationalRadiationPhysics/picongpu/pull/438>`__.
Docs/source/developers/contributing.rst:  This is particularly useful to avoid capturing member variables by value in a lambda function, which causes the whole object to be copied to GPU when running on a GPU-accelerated architecture.
Docs/source/developers/amrex_basics.rst:* ``amrex::ParticleContainer``: A collection of particles, typically for particles of a physical species. Particles in a ``ParticleContainer`` are organized per ``Box``. Particles in a ``Box`` are organized per tile (this feature is off when running on GPU). Particles within a tile are stored in several structures, each being contiguous in memory: (i) a Struct-of-Array (SoA) for ``amrex::ParticleReal`` data such as positions, weight, momentum, etc., (ii) a Struct-of-Array (SoA) for ``int`` data, such as ionization levels, and (iii) a Struct-of-Array (SoA) for a ``uint64_t`` unique identifier index per particle (containing a 40bit id and 24bit cpu sub-identifier as assigned at particle creation time). This id is also used to check if a particle is active/valid or marked for removal.
Docs/source/developers/particles.rst:On a loop over boxes in a ``MultiFab`` (``MFIter``), it can be useful to access particle data on a GPU-friendly way. This can be done by:
Docs/source/developers/particles.rst:  // Get GPU-friendly arrays of particle data
Docs/source/developers/local_compile.rst:This workflow is the best and fastest to compile WarpX, when you just want to change code in WarpX and have the above central dependencies already made available *in the right configurations* (e.g., w/ or w/o MPI or GPU support) from a :ref:`module system <install-hpc>` or :ref:`package manager <install-dependencies>`.
Docs/source/developers/local_compile.rst:   You can contribute to `this pyAMReX pull request <https://github.com/AMReX-Codes/pyamrex/pull/127>`__ to help exploring this library (and if it works for the HPC/GPU compilers that we need to support).
Docs/source/developers/local_compile.rst:For power developers that switch a lot between fundamentally different WarpX configurations (e.g., 1D to 3D, GPU and CPU builds, many branches with different bases, developing AMReX and WarpX at the same time), also consider increasing the `CCache cache size <https://ccache.dev/manual/4.9.html#_cache_size_management>`__ and changing the `cache directory <https://ccache.dev/manual/4.9.html#config_cache_dir>`__ if needed, e.g., due to storage quota constraints or to choose a fast(er) filesystem for the cache files.
Docs/source/developers/profiling.rst:Nvidia Nsight-Systems
Docs/source/developers/profiling.rst:`Vendor homepage <https://developer.nvidia.com/nsight-systems>`__ and `product manual <https://docs.nvidia.com/nsight-systems/>`__.
Docs/source/developers/profiling.rst:Nsight-Systems provides system level profiling data, including CPU and GPU
Docs/source/developers/profiling.rst:Example on how to create traces on a multi-GPU system that uses the Slurm scheduler (e.g., NERSC's Perlmutter system).
Docs/source/developers/profiling.rst:   # GPU-aware MPI
Docs/source/developers/profiling.rst:   export MPICH_GPU_SUPPORT_ENABLED=1
Docs/source/developers/profiling.rst:   srun --ntasks=4 --gpus=4 --cpu-bind=cores \
Docs/source/developers/profiling.rst:         -t mpi,cuda,nvtx,osrt,openmp        \
Docs/source/developers/profiling.rst:       ./warpx.3d.MPI.CUDA.DP.QED            \
Docs/source/developers/profiling.rst:    In WarpX, every MPI rank is associated with one GPU, which each creates one trace file.
Docs/source/developers/profiling.rst: Example on how to create traces on a multi-GPU system that uses the ``jsrun`` scheduler (e.g., `OLCF's Summit system <https://docs.olcf.ornl.gov/systems/summit_user_guide.html#optimizing-and-profiling>`__):
Docs/source/developers/profiling.rst:          -t mpi,cuda,nvtx,osrt,openmp   \
Docs/source/developers/profiling.rst:        ./warpx.3d.MPI.CUDA.DP.QED inputs_3d \
Docs/source/developers/profiling.rst:   The Nsight-Compute (``nsys``) version installed on Summit does not record details of GPU kernels.
Docs/source/developers/profiling.rst:   This is reported to Nvidia and OLCF.
Docs/source/developers/profiling.rst:* ``srun``: execute multi-GPU runs with ``srun`` (Slurm's ``mpiexec`` wrapper), here for four GPUs
Docs/source/developers/profiling.rst:* ``-o``: record one profile file per MPI rank (per GPU); if you run ``mpiexec``/``mpirun`` with OpenMPI directly, replace ``SLURM_TASK_PID`` with ``OMPI_COMM_WORLD_RANK``
Docs/source/developers/profiling.rst:Nvidia Nsight-Compute
Docs/source/developers/profiling.rst:`Vendor homepage <https://developer.nvidia.com/nsight-compute>`__ and `product manual <https://docs.nvidia.com/nsight-compute/>`__.
Docs/source/developers/profiling.rst:Example of how to create traces on a single-GPU system. A jobscript for
Docs/source/developers/profiling.rst:   #SBATCH -C gpu
Docs/source/developers/profiling.rst:   #SBATCH --gpus-per-task=1
Docs/source/developers/profiling.rst:   #SBATCH --gpu-bind=map_gpu:0
Docs/source/developers/profiling.rst:  `see this Q&A <https://forums.developer.nvidia.com/t/profiling-failed-because-a-driver-resource-was-unavailable/205435>`__.
Docs/source/developers/profiling.rst:    significant. For full information, see the Nvidia's documentation on `NVTX filtering <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering>`__ .
Docs/source/developers/gnumake/gpu_local.rst:Building WarpX with GPU support (Linux only)
Docs/source/developers/gnumake/gpu_local.rst:  In order to build WarpX on a specific GPU cluster (e.g. Summit),
Docs/source/developers/gnumake/gpu_local.rst:In order to build WarpX with GPU support, make sure that you have `cuda`
Docs/source/developers/gnumake/gpu_local.rst:fails.) Then compile WarpX with the option `USE_GPU=TRUE`, e.g.
Docs/source/developers/gnumake/gpu_local.rst:  make -j 4 USE_GPU=TRUE
Docs/source/developers/repo_organization.rst:sometimes the case for AMReX headers. For instance ``AMReX_GpuLaunch.H`` is a façade header for ``AMReX_GpuLaunchFunctsC.H`` and ``AMReX_GpuLaunchFunctsG.H``, which
Docs/source/developers/repo_organization.rst:contain respectively the CPU and the GPU implementation of some methods, and which should not be included directly.
Docs/source/developers/fields.rst:      // Loop over boxes (or tiles if not on GPU)
Docs/source/developers/fields.rst:      for ( MFIter mfi(*Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Docs/source/developers/fields.rst:The innermost step ``// Apply field solver on the FAB`` could be done with 3 nested ``for`` loops for the 3 dimensions (in 3D). However, for portability reasons (see section :ref:`Developers: Portability <developers-portability>`), this is done in two steps: (i) extract AMReX data structures into plain-old-data simple structures, and (ii) call a general ``ParallelFor`` function (translated into nested loops on CPU or a kernel launch on GPU, for instance):
Docs/source/developers/fields.rst:      [=] AMREX_GPU_DEVICE (int j, int k, int l)
Docs/source/acknowledge_us.rst:  **Synthesizing Particle-in-Cell Simulations Through Learning and GPU Computing for Hybrid Particle Accelerator Beamlines**.
Docs/source/acknowledge_us.rst:  **Porting WarpX to GPU-accelerated platforms**.
Docs/source/acknowledge_us.rst:  **Modeling of a chain of three plasma accelerator stages with the WarpX electromagnetic PIC code on GPUs**. *Physics of Plasmas*. 2021 Feb 9, 28(2):023105.
Docs/source/acknowledge_us.rst:  **In-situ assessment of device-side compute work for dynamic load balancing in a GPU-accelerated PIC code**. *PASC '21: Proceedings of the Platform for Advanced Scientific Computing Conference*. 2021 July, 10, pages 1-11.
Docs/source/glossary.rst:* **CPU:** `central processing unit <https://en.wikipedia.org/wiki/Central_processing_unit>`__, we usual mean a socket or generally the host-side of a computer (compared to the accelerator, e.g. GPU)
Docs/source/glossary.rst:* **GPU:** originally graphics processing unit, now used for fast `general purpose computing (GPGPU) <https://en.wikipedia.org/wiki/Graphics_processing_unit#Stream_processing_and_general_purpose_GPUs_(GPGPU)>`__; also called (hardware) accelerator
Docs/source/glossary.rst:* **accelerator:** depending on context, either a *particle accelerator* in physics or a *hardware accelerator* (e.g. GPU) in computing
Docs/source/highlights.rst:   **Synthesizing Particle-in-Cell Simulations Through Learning and GPU Computing for Hybrid Particle Accelerator Beamlines**.
Docs/source/highlights.rst:   **Synthesizing Particle-in-Cell Simulations Through Learning and GPU Computing for Hybrid Particle Accelerator Beamlines**.
Docs/source/highlights.rst:   **FerroX: A GPU-accelerated, 3D Phase-Field Simulation Framework for Modeling Ferroelectric Devices**.
.azure-pipelines.yml:          -DHeffte_ENABLE_CUDA=OFF -DHeffte_ENABLE_ROCM=OFF      \
README.md:WarpX is a *highly-parallel and highly-optimized code*, which can run on GPUs and multi-core CPUs, and includes load balancing capabilities.
Examples/Tests/ohm_solver_magnetic_reconnection/README.rst:Running the full simulation should take about 4 hours if executed on 1 V100 GPU.
Examples/Tests/reduced_diags/analysis_reduced_diags_load_balance_costs.py:#      cost_box_0, proc_box_0, lev_box_0, i_low_box_0, j_low_box_0, k_low_box_0(, gpu_ID_box_0 if GPU run), hostname_box_0,
Examples/Tests/reduced_diags/analysis_reduced_diags_load_balance_costs.py:#      cost_box_1, proc_box_1, lev_box_1, i_low_box_1, j_low_box_1, k_low_box_1(, gpu_ID_box_1 if GPU run), hostname_box_1,
Examples/Tests/reduced_diags/analysis_reduced_diags_load_balance_costs.py:#      cost_box_n, proc_box_n, lev_box_n, i_low_box_n, j_low_box_n, k_low_box_n(, gpu_ID_box_n if GPU run), hostname_box_n]
Examples/Physics_applications/laser_ion/README.rst:   The following images for densities and electromagnetic fields were created with a run on 64 NVidia A100 GPUs featuring a total number of cells of ``nx = 8192`` and ``nz = 16384``, as well as 64 particles per cell per species.
Examples/Physics_applications/laser_ion/inputs_test_2d_laser_ion_acc:#   Use larger values for GPUs, try to fill a GPU well with memory and place
Examples/Physics_applications/laser_ion/inputs_test_2d_laser_ion_acc:# particle bin-sorting on GPU (ideal defaults not investigated in 2D)
Examples/Physics_applications/laser_ion/inputs_test_2d_laser_ion_acc:#warpx.sort_intervals = 4    # default on CPU: -1 (off); on GPU: 4
Examples/Physics_applications/laser_ion/inputs_test_2d_laser_ion_acc_picmi.py:# --> choose larger `max_grid_size` and `blocking_factor` for 1 to 8 grids per GPU accordingly
CMakeLists.txt:# AMReX 21.06+ supports CUDA_ARCHITECTURES with CMake 3.20+
CMakeLists.txt:# CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
CMakeLists.txt:set(WarpX_COMPUTE_VALUES NOACC OMP CUDA SYCL HIP)
CMakeLists.txt:set(WarpX_COMPUTE OMP CACHE STRING "On-node, accelerated computing backend (NOACC/OMP/CUDA/SYCL/HIP)")
CMakeLists.txt:    if(WarpX_COMPUTE STREQUAL CUDA)
CMakeLists.txt:        set(_heFFTe_COMPS CUDA)
CMakeLists.txt:        set(_heFFTe_COMPS ROCM)
CMakeLists.txt:    # note: we could also enforce GPUAWARE for CUDA and HIP, which can still be
CMakeLists.txt:            CUDA_VISIBILITY_PRESET "hidden"
CMakeLists.txt:            # BLAS++ forgets to declare cuBLAS and cudaRT dependencies
CMakeLists.txt:            if(WarpX_COMPUTE STREQUAL CUDA)
CMakeLists.txt:                find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:                target_link_libraries(ablastr_${SD} PUBLIC CUDA::cudart CUDA::cublas)
CMakeLists.txt:if(WarpX_COMPUTE STREQUAL CUDA)
CMakeLists.txt:    # AMReX helper function: propagate CUDA specific target & source properties
CMakeLists.txt:        setup_target_for_cuda_compilation(${warpx_tgt})
CMakeLists.txt:        target_compile_features(${warpx_tgt} PUBLIC cuda_std_17)
CMakeLists.txt:        CUDA_EXTENSIONS OFF
CMakeLists.txt:        CUDA_STANDARD_REQUIRED ON
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/particle_containers.py:            For GPU-enabled runs, one can either return the GPU
Python/pywarpx/picmi.py:    warpx_amrex_use_gpu_aware_mpi: bool, optional
Python/pywarpx/picmi.py:        Whether to use GPU-aware MPI communications
Python/pywarpx/picmi.py:    warpx_sort_intervals: string, optional (defaults: -1 on CPU; 4 on GPU)
Python/pywarpx/picmi.py:        It is turned on on GPUs for performance reasons (to improve memory locality).
Python/pywarpx/picmi.py:    warpx_sort_particles_for_deposition: bool, optional (default: true for the CUDA backend, otherwise false)
Python/pywarpx/picmi.py:        `true` is recommended for best performance on NVIDIA GPUs, especially if there are many particles per cell.
Python/pywarpx/picmi.py:        self.amrex_use_gpu_aware_mpi = kw.pop("warpx_amrex_use_gpu_aware_mpi", None)
Python/pywarpx/picmi.py:        if self.amrex_use_gpu_aware_mpi is not None:
Python/pywarpx/picmi.py:            pywarpx.amrex.use_gpu_aware_mpi = self.amrex_use_gpu_aware_mpi
Python/pywarpx/fields.py:        if libwarpx.libwarpx_so.Config.have_gpu:
Python/pywarpx/fields.py:            if libwarpx.libwarpx_so.Config.have_gpu:
Python/pywarpx/fields.py:                    if libwarpx.libwarpx_so.Config.have_gpu:
Python/pywarpx/LoadThirdParty.py:    if amr.Config.have_gpu:
Python/pywarpx/LoadThirdParty.py:            status = "Warning: GPU found but cupy not available! Trying managed memory in numpy..."
Python/pywarpx/LoadThirdParty.py:        if amr.Config.gpu_backend == "SYCL":
Python/pywarpx/LoadThirdParty.py:            status = "Warning: SYCL GPU backend not yet implemented for Python"
Source/Parallelization/WarpXComm_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Parallelization/WarpXComm_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Parallelization/WarpXComm_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Parallelization/WarpXComm_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Parallelization/WarpXComm.cpp:#include <AMReX_GpuContainers.H>
Source/Parallelization/WarpXComm.cpp:#include <AMReX_GpuControl.H>
Source/Parallelization/WarpXComm.cpp:#include <AMReX_GpuQualifiers.H>
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:    for (MFIter mfi(*Bfield_aux[0][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Parallelization/WarpXComm.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:                for (MFIter mfi(*Bfield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:                for (MFIter mfi(*Bfield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:                for (MFIter mfi(*Efield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:                for (MFIter mfi(*Efield_aux[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:                    [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Parallelization/WarpXComm.cpp:    for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Parallelization/WarpXComm.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Parallelization/WarpXComm.cpp:                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k, int n)
Source/Parallelization/WarpXComm.cpp:            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k, int n)
Source/EmbeddedBoundary/WarpXInitEB.cpp:#  include <AMReX_GpuControl.H>
Source/EmbeddedBoundary/WarpXInitEB.cpp:#  include <AMReX_GpuDevice.H>
Source/EmbeddedBoundary/WarpXInitEB.cpp:#  include <AMReX_GpuQualifiers.H>
Source/EmbeddedBoundary/WarpXInitEB.cpp:        : public amrex::GPUable
Source/EmbeddedBoundary/WarpXInitEB.cpp:        AMREX_GPU_HOST_DEVICE inline
Source/EmbeddedBoundary/WarpXInitEB.cpp:                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/WarpXInitEB.cpp:            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/EmbeddedBoundary/ParticleBoundaryProcess.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/ParticleBoundaryProcess.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/DistanceToEB.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/EmbeddedBoundary/DistanceToEB.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/EmbeddedBoundary/DistanceToEB.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/EmbeddedBoundary/DistanceToEB.H:                               amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi) noexcept
Source/EmbeddedBoundary/ParticleScraper.H: *        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/ParticleScraper.H: *        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/ParticleScraper.H: *        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/ParticleScraper.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/EmbeddedBoundary/ParticleScraper.H:            [=] AMREX_GPU_DEVICE (const int ip, amrex::RandomEngine const& engine) noexcept
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<int> {
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:                                                    [=] AMREX_GPU_DEVICE (int icell) {
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:                                                [=] AMREX_GPU_DEVICE (int icell, int ps){
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:                                                     [=] AMREX_GPU_DEVICE (int icell){
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:            [=] AMREX_GPU_DEVICE (int icell, int ps) {
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:    for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel()), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/EmbeddedBoundary/WarpXFaceExtensions.cpp:        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/EmbeddedBoundary/WarpXFaceInfoBox.H:#include <AMReX_Gpu.H>
Source/EmbeddedBoundary/WarpXFaceInfoBox.H:    amrex::Gpu::DeviceVector<Neighbours> neigh_faces;
Source/EmbeddedBoundary/WarpXFaceInfoBox.H:    amrex::Gpu::DeviceVector<amrex::Real> area;
Source/EmbeddedBoundary/WarpXFaceInfoBox.H:    amrex::Gpu::DeviceVector<int> inds;
Source/EmbeddedBoundary/WarpXFaceInfoBox.H:    AMREX_GPU_HOST_DEVICE
Source/EmbeddedBoundary/WarpXFaceInfoBox.H:    AMREX_GPU_HOST_DEVICE
Source/EmbeddedBoundary/WarpXFaceExtensions.H:AMREX_GPU_DEVICE
Source/EmbeddedBoundary/WarpXFaceExtensions.H:AMREX_GPU_DEVICE
Source/AcceleratorLattice/AcceleratorLattice.H:     * on the device (such as a GPU)
Source/AcceleratorLattice/LatticeElementFinder.cpp:LatticeElementFinder::setup_lattice_indices (amrex::Gpu::DeviceVector<amrex::ParticleReal> const & zs,
Source/AcceleratorLattice/LatticeElementFinder.cpp:                       amrex::Gpu::DeviceVector<amrex::ParticleReal> const & ze,
Source/AcceleratorLattice/LatticeElementFinder.cpp:                       amrex::Gpu::DeviceVector<int> & indices) const
Source/AcceleratorLattice/LatticeElementFinder.cpp:        [=] AMREX_GPU_DEVICE (int iz) {
Source/AcceleratorLattice/LatticeElementFinder.H:#include <AMReX_GpuContainers.H>
Source/AcceleratorLattice/LatticeElementFinder.H:    amrex::Gpu::DeviceVector<int> d_quad_indices;
Source/AcceleratorLattice/LatticeElementFinder.H:    amrex::Gpu::DeviceVector<int> d_plasmalens_indices;
Source/AcceleratorLattice/LatticeElementFinder.H:    void setup_lattice_indices (amrex::Gpu::DeviceVector<amrex::ParticleReal> const & zs,
Source/AcceleratorLattice/LatticeElementFinder.H:                                amrex::Gpu::DeviceVector<amrex::ParticleReal> const & ze,
Source/AcceleratorLattice/LatticeElementFinder.H:                                amrex::Gpu::DeviceVector<int> & indices) const;
Source/AcceleratorLattice/LatticeElementFinder.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/AcceleratorLattice/LatticeElements/HardEdged_K.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/AcceleratorLattice/LatticeElements/HardEdgedQuadrupole.H:#include <AMReX_GpuContainers.H>
Source/AcceleratorLattice/LatticeElements/HardEdgedQuadrupole.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_dEdx;
Source/AcceleratorLattice/LatticeElements/HardEdgedQuadrupole.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_dBdx;
Source/AcceleratorLattice/LatticeElements/HardEdgedQuadrupole.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/AcceleratorLattice/LatticeElements/HardEdgedQuadrupole.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dEdx.begin(), h_dEdx.end(), d_dEdx.begin());
Source/AcceleratorLattice/LatticeElements/HardEdgedQuadrupole.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dBdx.begin(), h_dBdx.end(), d_dBdx.begin());
Source/AcceleratorLattice/LatticeElements/HardEdgedPlasmaLens.H:#include <AMReX_GpuContainers.H>
Source/AcceleratorLattice/LatticeElements/HardEdgedPlasmaLens.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_dEdx;
Source/AcceleratorLattice/LatticeElements/HardEdgedPlasmaLens.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_dBdx;
Source/AcceleratorLattice/LatticeElements/HardEdgedPlasmaLens.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/AcceleratorLattice/LatticeElements/LatticeElementBase.H:#include <AMReX_GpuContainers.H>
Source/AcceleratorLattice/LatticeElements/LatticeElementBase.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_zs;
Source/AcceleratorLattice/LatticeElements/LatticeElementBase.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_ze;
Source/AcceleratorLattice/LatticeElements/LatticeElementBase.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_zs.begin(), h_zs.end(), d_zs.begin());
Source/AcceleratorLattice/LatticeElements/LatticeElementBase.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_ze.begin(), h_ze.end(), d_ze.begin());
Source/AcceleratorLattice/LatticeElements/HardEdgedPlasmaLens.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dEdx.begin(), h_dEdx.end(), d_dEdx.begin());
Source/AcceleratorLattice/LatticeElements/HardEdgedPlasmaLens.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dBdx.begin(), h_dBdx.end(), d_dBdx.begin());
Source/ablastr/math/fft/Make.package:  ifeq ($(USE_CUDA),TRUE)
Source/ablastr/math/fft/WrapCuFFT.cpp:        // make sure that this is done on the same GPU stream as the above copy
Source/ablastr/math/fft/WrapCuFFT.cpp:        cudaStream_t stream = amrex::Gpu::Device::cudaStream();
Source/ablastr/math/fft/CMakeLists.txt:    if(WarpX_COMPUTE STREQUAL CUDA)
Source/ablastr/math/fft/AnyFFT.H:#   include <AMReX_GpuComplex.H>
Source/ablastr/math/fft/AnyFFT.H:#   if defined(AMREX_USE_CUDA)
Source/ablastr/math/fft/AnyFFT.H:#       if __has_include(<rocfft/rocfft.h>)  // ROCm 5.3+
Source/ablastr/math/fft/AnyFFT.H:#   if defined(AMREX_USE_CUDA)
Source/ablastr/math/fft/AnyFFT.H:        using Complex = amrex::GpuComplex<amrex::Real>;
Source/ablastr/math/fft/AnyFFT.H:#   if defined(AMREX_USE_CUDA)
Source/ablastr/math/fft/AnyFFT.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = cuCmulf(a, b); }
Source/ablastr/math/fft/AnyFFT.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = cuCmul(a, b); }
Source/ablastr/math/fft/AnyFFT.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = hipCmulf(a, b); }
Source/ablastr/math/fft/AnyFFT.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = hipCmul(a, b); }
Source/ablastr/math/fft/AnyFFT.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) { c = a * b; }
Source/ablastr/math/fft/AnyFFT.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void multiply (Complex & c, Complex const & a, Complex const & b) {
Source/ablastr/math/fft/AnyFFT.H:#   if defined(AMREX_USE_CUDA)
Source/ablastr/math/fft/AnyFFT.H:        amrex::gpuStream_t m_stream;
Source/ablastr/math/fft/WrapMklFFT.cpp:        fft_plan.m_plan->commit(amrex::Gpu::Device::streamQueue());
Source/ablastr/math/fft/WrapMklFFT.cpp:        fft_plan.m_stream = amrex::Gpu::gpuStream();
Source/ablastr/math/fft/WrapMklFFT.cpp:        if (!(fft_plan.m_stream == amrex::Gpu::gpuStream())) {
Source/ablastr/math/fft/WrapMklFFT.cpp:            amrex::Gpu::streamSynchronize();
Source/ablastr/math/fft/WrapRocFFT.cpp:        result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());
Source/ablastr/math/fft/WrapRocFFT.cpp:        amrex::Gpu::streamSynchronize();
Source/ablastr/coarsen/sample.cpp:#include <AMReX_GpuControl.H>
Source/ablastr/coarsen/sample.cpp:#include <AMReX_GpuLaunch.H>
Source/ablastr/coarsen/sample.cpp:        auto sf = amrex::GpuArray<int,3>{0,0,0}; // staggering of source fine MultiFab
Source/ablastr/coarsen/sample.cpp:        auto sc = amrex::GpuArray<int,3>{0,0,0}; // staggering of destination coarse MultiFab
Source/ablastr/coarsen/sample.cpp:        auto cr = amrex::GpuArray<int,3>{1,1,1}; // coarsening ratio
Source/ablastr/coarsen/sample.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/ablastr/coarsen/sample.cpp:        // Loop over boxes (or tiles if not on GPU)
Source/ablastr/coarsen/sample.cpp:        for (amrex::MFIter mfi( mf_dst, amrex::TilingIfNotGPU() ); mfi.isValid(); ++mfi)
Source/ablastr/coarsen/sample.cpp:                         [=] AMREX_GPU_DEVICE( int i, int j, int k, int n )
Source/ablastr/coarsen/average.cpp:#include <AMReX_GpuControl.H>
Source/ablastr/coarsen/average.cpp:#include <AMReX_GpuLaunch.H>
Source/ablastr/coarsen/average.cpp:        auto sf = amrex::GpuArray<int,3>{0,0,0}; // staggering of source fine MultiFab
Source/ablastr/coarsen/average.cpp:        auto sc = amrex::GpuArray<int,3>{0,0,0}; // staggering of destination coarse MultiFab
Source/ablastr/coarsen/average.cpp:        auto cr = amrex::GpuArray<int,3>{1,1,1}; // coarsening ratio
Source/ablastr/coarsen/average.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/ablastr/coarsen/average.cpp:        // Loop over boxes (or tiles if not on GPU)
Source/ablastr/coarsen/average.cpp:        for (amrex::MFIter mfi(mf_dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ablastr/coarsen/average.cpp:                        [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/ablastr/coarsen/average.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/coarsen/average.H:    AMREX_GPU_DEVICE
Source/ablastr/coarsen/average.H:        amrex::GpuArray<int, 3> const &sf,
Source/ablastr/coarsen/average.H:        amrex::GpuArray<int, 3> const &sc,
Source/ablastr/coarsen/average.H:        amrex::GpuArray<int, 3> const &cr,
Source/ablastr/coarsen/average.H:                AMREX_GPU_DEVICE(int const ix, int const iy, int const iz, int const n) noexcept {
Source/ablastr/coarsen/sample.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/coarsen/sample.H:    AMREX_GPU_DEVICE
Source/ablastr/coarsen/sample.H:        amrex::GpuArray<int,3> const& sf,
Source/ablastr/coarsen/sample.H:        amrex::GpuArray<int,3> const& sc,
Source/ablastr/coarsen/sample.H:        amrex::GpuArray<int,3> const& cr,
Source/ablastr/fields/IntegratedGreenFunctionSolver.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/fields/IntegratedGreenFunctionSolver.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/ablastr/fields/IntegratedGreenFunctionSolver.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/ablastr/fields/Interpolate.H:        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:#include <AMReX_GpuControl.H>
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:#include <AMReX_GpuLaunch.H>
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:#include <AMReX_GpuQualifiers.H>
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:    for (amrex::MFIter mfi(igf_compute_box, dm_global_fft, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:        using SpectralField = amrex::BaseFab< amrex::GpuComplex< amrex::Real > > ;
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:#if     defined(AMREX_USE_CUDA)
Source/ablastr/fields/IntegratedGreenFunctionSolver.cpp:        amrex::Gpu::streamSynchronize();
Source/ablastr/fields/PoissonSolver.H:#include <AMReX_GpuControl.H>
Source/ablastr/fields/PoissonSolver.H:#include <AMReX_GpuLaunch.H>
Source/ablastr/fields/PoissonSolver.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/fields/PoissonSolver.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/ablastr/fields/PoissonSolver.H:    for (amrex::MFIter mfi(*phi_lev_plus_one, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/ablastr/fields/VectorPoissonSolver.H:#include <AMReX_GpuControl.H>
Source/ablastr/fields/VectorPoissonSolver.H:#include <AMReX_GpuLaunch.H>
Source/ablastr/fields/VectorPoissonSolver.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/fields/VectorPoissonSolver.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/ablastr/fields/VectorPoissonSolver.H:                for (amrex::MFIter mfi(*A[lev+1][adim],amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/ablastr/particles/NodalFieldGather.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/particles/NodalFieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/ablastr/particles/NodalFieldGather.H:                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
Source/ablastr/particles/NodalFieldGather.H:                            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
Source/ablastr/particles/NodalFieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/ablastr/particles/NodalFieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/ablastr/particles/NodalFieldGather.H:                                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
Source/ablastr/particles/NodalFieldGather.H:                                      amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& lo) noexcept
Source/ablastr/particles/NodalFieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/ablastr/particles/NodalFieldGather.H:amrex::GpuArray<amrex::Real, 3>
Source/ablastr/particles/NodalFieldGather.H:                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
Source/ablastr/particles/NodalFieldGather.H:                          amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& lo) noexcept
Source/ablastr/particles/NodalFieldGather.H:    amrex::GpuArray<amrex::Real, 3> const field_interp = {
Source/ablastr/particles/ParticleMoments.H:            [=] AMREX_GPU_DEVICE(const ConstParticleTileDataType& ptd, const int i) noexcept
Source/ablastr/particles/ParticleMoments.H:            [=] AMREX_GPU_DEVICE(const ConstParticleTileDataType& ptd, const int i) noexcept
Source/ablastr/particles/DepositCharge.H:    // On GPU: particles deposit directly on the rho array, which usually have a larger number of guard cells
Source/ablastr/particles/DepositCharge.H:#ifndef AMREX_USE_GPU
Source/ablastr/particles/DepositCharge.H:        "Particles shape does not fit within tile (CPU) or guard cells (GPU) used for charge deposition");
Source/ablastr/particles/DepositCharge.H:#ifndef AMREX_USE_GPU
Source/ablastr/particles/DepositCharge.H:#ifdef AMREX_USE_GPU
Source/ablastr/particles/DepositCharge.H:    // GPU, no tiling: rho_fab points to the full rho array
Source/ablastr/particles/DepositCharge.H:#ifndef AMREX_USE_GPU
Source/ablastr/utils/SignalHandling.cpp:    // Due to a bug in Cray's MPICH 8.1.13 implementation (CUDA builds on Perlmutter@NERSC in 2022),
Source/ablastr/utils/Communication.H:#include <AMReX_GpuDevice.H>
Source/ablastr/utils/Communication.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/utils/Communication.H:    [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k, int n) noexcept
Source/ablastr/utils/Communication.H:    amrex::Gpu::synchronize();
Source/ablastr/parallelization/KernelTimer.H:#include <AMReX_GpuAtomic.H>
Source/ablastr/parallelization/KernelTimer.H:#include <AMReX_GpuQualifiers.H>
Source/ablastr/parallelization/KernelTimer.H: * \brief Defines a timer object to be used on GPU; measures summed thread cycles.
Source/ablastr/parallelization/KernelTimer.H:    AMREX_GPU_DEVICE
Source/ablastr/parallelization/KernelTimer.H:#if (defined AMREX_USE_GPU)
Source/ablastr/parallelization/KernelTimer.H:#if (defined AMREX_USE_GPU)
Source/ablastr/parallelization/KernelTimer.H:#   if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/ablastr/parallelization/KernelTimer.H:#else  // AMREX_USE_GPU
Source/ablastr/parallelization/KernelTimer.H:#endif // AMREX_USE_GPU
Source/ablastr/parallelization/KernelTimer.H:#if (defined AMREX_USE_GPU)
Source/ablastr/parallelization/KernelTimer.H:#   if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/ablastr/parallelization/KernelTimer.H:        AMREX_GPU_DEVICE
Source/ablastr/parallelization/KernelTimer.H:                amrex::Gpu::Atomic::Add( m_cost, amrex::Real(m_wt));
Source/ablastr/parallelization/KernelTimer.H:        AMREX_GPU_DEVICE
Source/ablastr/parallelization/KernelTimer.H:#endif //AMREX_USE_GPU
Source/ablastr/parallelization/KernelTimer.H:#if (defined AMREX_USE_GPU)
Source/ablastr/parallelization/KernelTimer.H:#endif //AMREX_USE_GPU
Source/Particles/Resampling/VelocityCoincidenceThinning.cpp:    // create a GPU vector to hold the momentum cluster index for each particle
Source/Particles/Resampling/VelocityCoincidenceThinning.cpp:    amrex::Gpu::DeviceVector<int> momentum_bin_number(n_parts_in_tile);
Source/Particles/Resampling/VelocityCoincidenceThinning.cpp:    // create a GPU vector to hold the index sorting for the momentum bins
Source/Particles/Resampling/VelocityCoincidenceThinning.cpp:    amrex::Gpu::DeviceVector<int> sorted_indices(n_parts_in_tile);
Source/Particles/Resampling/VelocityCoincidenceThinning.cpp:            reduce_op.eval(n_parts_in_tile, reduce_data, [=] AMREX_GPU_DEVICE(int i) -> ReduceTuple {
Source/Particles/Resampling/VelocityCoincidenceThinning.cpp:        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
Source/Particles/Resampling/LevelingThinning.cpp:#include <AMReX_GpuLaunch.H>
Source/Particles/Resampling/LevelingThinning.cpp:#include <AMReX_GpuQualifiers.H>
Source/Particles/Resampling/LevelingThinning.cpp:        [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
Source/Particles/Resampling/VelocityCoincidenceThinning.H:     * \brief This merging routine requires functionality to sort a GPU vector
Source/Particles/Resampling/VelocityCoincidenceThinning.H:     * based on another GPU vector's values. The heap-sort functions below were
Source/Particles/Resampling/VelocityCoincidenceThinning.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Resampling/VelocityCoincidenceThinning.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Resampling/VelocityCoincidenceThinning.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Resampling/VelocityCoincidenceThinning.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Resampling/VelocityCoincidenceThinning.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleBoundaries_K.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleBoundaries_K.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleBoundaries_K.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/PhotonParticleContainer.cpp:#include <AMReX_GpuLaunch.H>
Source/Particles/PhotonParticleContainer.cpp:#include <AMReX_GpuQualifiers.H>
Source/Particles/PhotonParticleContainer.cpp:                       [=] AMREX_GPU_DEVICE (long i, auto exteb_control,
Source/Particles/Gather/GetExternalFields.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/Gather/GetExternalFields.H:    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Gather/GetExternalFields.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Gather/FieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Gather/FieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Gather/FieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Gather/FieldGather.H:        [=] AMREX_GPU_DEVICE (long ip) {
Source/Particles/Gather/FieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Gather/FieldGather.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Gather/ScaleFields.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Sorting/SortingUtils.H:#include <AMReX_Gpu.H>
Source/Particles/Sorting/SortingUtils.H:void fillWithConsecutiveIntegers( amrex::Gpu::DeviceVector<int>& v );
Source/Particles/Sorting/SortingUtils.H:                                amrex::Gpu::DeviceVector<int> const& predicate)
Source/Particles/Sorting/SortingUtils.H:#ifdef AMREX_USE_GPU
Source/Particles/Sorting/SortingUtils.H:    // On GPU: Use amrex
Source/Particles/Sorting/SortingUtils.H:        [predicate_ptr] AMREX_GPU_DEVICE (int i) { return predicate_ptr[i]; });
Source/Particles/Sorting/SortingUtils.H:                        amrex::Gpu::DeviceVector<int>& inexflag,
Source/Particles/Sorting/SortingUtils.H:            // Extract simple structure that can be used directly on the GPU
Source/Particles/Sorting/SortingUtils.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Sorting/SortingUtils.H:        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_prob_lo;
Source/Particles/Sorting/SortingUtils.H:        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_inv_cell_size;
Source/Particles/Sorting/SortingUtils.H:                        amrex::Gpu::DeviceVector<int>& inexflag,
Source/Particles/Sorting/SortingUtils.H:                        amrex::Gpu::DeviceVector<int> const& particle_indices,
Source/Particles/Sorting/SortingUtils.H:            // Extract simple structure that can be used directly on the GPU
Source/Particles/Sorting/SortingUtils.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Sorting/SortingUtils.H:        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_prob_lo;
Source/Particles/Sorting/SortingUtils.H:        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> m_inv_cell_size;
Source/Particles/Sorting/SortingUtils.H:            amrex::Gpu::DeviceVector<T> const& src,
Source/Particles/Sorting/SortingUtils.H:            amrex::Gpu::DeviceVector<T>& dst,
Source/Particles/Sorting/SortingUtils.H:            amrex::Gpu::DeviceVector<int> const& indices ):
Source/Particles/Sorting/SortingUtils.H:                // Extract simple structure that can be used directly on the GPU
Source/Particles/Sorting/SortingUtils.H:        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Particles/Sorting/SortingUtils.cpp:void fillWithConsecutiveIntegers( amrex::Gpu::DeviceVector<int>& v )
Source/Particles/Sorting/SortingUtils.cpp:#ifdef AMREX_USE_GPU
Source/Particles/Sorting/SortingUtils.cpp:    // On GPU: Use amrex
Source/Particles/Sorting/Partition.cpp:#include <AMReX_GpuContainers.H>
Source/Particles/Sorting/Partition.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/Sorting/Partition.cpp:#include <AMReX_GpuLaunch.H>
Source/Particles/Sorting/Partition.cpp:    Gpu::DeviceVector<int> inexflag;
Source/Particles/Sorting/Partition.cpp:    Gpu::DeviceVector<int> pid;
Source/Particles/Sorting/Partition.cpp:        // the GPU kernels finish running
Source/Particles/Sorting/Partition.cpp:        Gpu::streamSynchronize();
Source/Particles/Sorting/Partition.cpp:    // the GPU kernels finish running
Source/Particles/Sorting/Partition.cpp:    Gpu::streamSynchronize();
Source/Particles/WarpXParticleContainer.H:#include <AMReX_GpuAllocators.H>
Source/Particles/WarpXParticleContainer.H:#include <AMReX_GpuContainers.H>
Source/Particles/WarpXParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::Real> ionization_energies;
Source/Particles/WarpXParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::Real> adk_power;
Source/Particles/WarpXParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::Real> adk_prefactor;
Source/Particles/WarpXParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::Real> adk_exp_prefactor;
Source/Particles/WarpXParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::Real> adk_correction_factors;
Source/Particles/WarpXParticleContainer.H:    using TmpParticleTile = std::array<amrex::Gpu::DeviceVector<amrex::ParticleReal>,
Source/Particles/ShapeFactors.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ShapeFactors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ShapeFactors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ShapeFactors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleIO.H:#include <AMReX_Gpu.H>
Source/Particles/ParticleIO.H:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particles/ParticleIO.H:                         [=] AMREX_GPU_DEVICE (long i) {
Source/Particles/Pusher/PushSelector.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Particles/Pusher/CopyParticleAttribs.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Pusher/UpdateMomentumHigueraCary.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Pusher/UpdateMomentumBoris.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Pusher/UpdateMomentumBorisWithRadiationReaction.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Pusher/GetAndSetPosition.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Pusher/GetAndSetPosition.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Pusher/GetAndSetPosition.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Pusher/GetAndSetPosition.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Pusher/GetAndSetPosition.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Pusher/UpdateMomentumVay.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Pusher/UpdatePositionPhoton.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Pusher/UpdatePosition.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Pusher/UpdatePosition.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Pusher/UpdatePosition.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/AddPlasmaUtilities.H:#include <AMReX_GpuContainers.H>
Source/Particles/AddPlasmaUtilities.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/AddPlasmaUtilities.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/AddPlasmaUtilities.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/AddPlasmaUtilities.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/AddPlasmaUtilities.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/AddPlasmaUtilities.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/AddPlasmaUtilities.H:                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
Source/Particles/AddPlasmaUtilities.H:                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
Source/Particles/AddPlasmaUtilities.H:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
Source/Particles/AddPlasmaUtilities.H:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
Source/Particles/AddPlasmaUtilities.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/AddPlasmaUtilities.H:amrex::Real compute_scale_fac_volume (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
Source/Particles/AddPlasmaUtilities.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/AddPlasmaUtilities.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/AddPlasmaUtilities.H:    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
Source/Particles/AddPlasmaUtilities.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/AddPlasmaUtilities.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/AddPlasmaUtilities.H:amrex::Real compute_scale_fac_area_plane (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > m_user_int_attrib_parserexec_pinned;
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::PinnedVector< amrex::ParserExecutor<7> > m_user_real_attrib_parserexec_pinned;
Source/Particles/AddPlasmaUtilities.H:#ifdef AMREX_USE_GPU
Source/Particles/AddPlasmaUtilities.H:#ifdef AMREX_USE_GPU
Source/Particles/AddPlasmaUtilities.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_pa_user_int_pinned.begin(),
Source/Particles/AddPlasmaUtilities.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_pa_user_real_pinned.begin(),
Source/Particles/AddPlasmaUtilities.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, wrapper.m_user_int_attrib_parserexec_pinned.begin(),
Source/Particles/AddPlasmaUtilities.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, wrapper.m_user_real_attrib_parserexec_pinned.begin(),
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::PinnedVector<int*> m_pa_user_int_pinned;
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::PinnedVector<amrex::ParticleReal*> m_pa_user_real_pinned;
Source/Particles/AddPlasmaUtilities.H:#ifdef AMREX_USE_GPU
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::DeviceVector<int*> m_d_pa_user_int;
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal*> m_d_pa_user_real;
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > m_d_user_int_attrib_parserexec;
Source/Particles/AddPlasmaUtilities.H:    amrex::Gpu::DeviceVector< amrex::ParserExecutor<7> > m_d_user_real_attrib_parserexec;
Source/Particles/PhysicalParticleContainer.cpp:#include <AMReX_GpuAtomic.H>
Source/Particles/PhysicalParticleContainer.cpp:#include <AMReX_GpuBuffer.H>
Source/Particles/PhysicalParticleContainer.cpp:#include <AMReX_GpuControl.H>
Source/Particles/PhysicalParticleContainer.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/PhysicalParticleContainer.cpp:#include <AMReX_GpuElixir.H>
Source/Particles/PhysicalParticleContainer.cpp:#include <AMReX_GpuLaunch.H>
Source/Particles/PhysicalParticleContainer.cpp:#include <AMReX_GpuQualifiers.H>
Source/Particles/PhysicalParticleContainer.cpp:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/PhysicalParticleContainer.cpp:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/PhysicalParticleContainer.cpp:    XDim3 getCellCoords (const GpuArray<Real, AMREX_SPACEDIM>& lo_corner,
Source/Particles/PhysicalParticleContainer.cpp:                         const GpuArray<Real, AMREX_SPACEDIM>& dx,
Source/Particles/PhysicalParticleContainer.cpp:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/PhysicalParticleContainer.cpp:        const GpuArray<ParticleReal*,PIdx::nattribs>& pa, long& ip,
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_x;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_y;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_z;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_ux;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_uy;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_uz;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_w;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_x;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_z;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_ux;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_uz;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_w;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_y;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal> particle_uy;
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal>& particle_x,
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal>& particle_y,
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal>& particle_z,
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal>& particle_ux,
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal>& particle_uy,
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal>& particle_uz,
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::HostVector<ParticleReal>& particle_w,
Source/Particles/PhysicalParticleContainer.cpp:    if (do_tiling && Gpu::notInLaunchRegion()) {
Source/Particles/PhysicalParticleContainer.cpp:            amrex::Gpu::synchronize();
Source/Particles/PhysicalParticleContainer.cpp:        const GpuArray<Real,AMREX_SPACEDIM> overlap_corner
Source/Particles/PhysicalParticleContainer.cpp:        Gpu::DeviceVector<amrex::Long> counts(overlap_box.numPts(), 0);
Source/Particles/PhysicalParticleContainer.cpp:        Gpu::DeviceVector<amrex::Long> offset(overlap_box.numPts());
Source/Particles/PhysicalParticleContainer.cpp:        amrex::ParallelFor(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Particles/PhysicalParticleContainer.cpp:                const auto xlim = GpuArray<Real, 3>{lo.x,(lo.x+hi.x)/2._rt,hi.x};
Source/Particles/PhysicalParticleContainer.cpp:                const auto ylim = GpuArray<Real, 3>{lo.y,(lo.y+hi.y)/2._rt,hi.y};
Source/Particles/PhysicalParticleContainer.cpp:                const auto zlim = GpuArray<Real, 3>{lo.z,(lo.z+hi.z)/2._rt,hi.z};
Source/Particles/PhysicalParticleContainer.cpp:        GpuArray<ParticleReal*,PIdx::nattribs> pa;
Source/Particles/PhysicalParticleContainer.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
Source/Particles/PhysicalParticleContainer.cpp:        amrex::Gpu::synchronize();
Source/Particles/PhysicalParticleContainer.cpp:    if (do_tiling && Gpu::notInLaunchRegion()) {
Source/Particles/PhysicalParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/PhysicalParticleContainer.cpp:            amrex::Gpu::synchronize();
Source/Particles/PhysicalParticleContainer.cpp:        const GpuArray<Real,AMREX_SPACEDIM> overlap_corner
Source/Particles/PhysicalParticleContainer.cpp:        Gpu::DeviceVector<int> counts(overlap_box.numPts(), 0);
Source/Particles/PhysicalParticleContainer.cpp:        Gpu::DeviceVector<int> offset(overlap_box.numPts());
Source/Particles/PhysicalParticleContainer.cpp:        amrex::ParallelForRNG(overlap_box, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
Source/Particles/PhysicalParticleContainer.cpp:        GpuArray<ParticleReal*,PIdx::nattribs> pa;
Source/Particles/PhysicalParticleContainer.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
Source/Particles/PhysicalParticleContainer.cpp:        amrex::Gpu::synchronize();
Source/Particles/PhysicalParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/PhysicalParticleContainer.cpp:            amrex::Gpu::synchronize();
Source/Particles/PhysicalParticleContainer.cpp:    // Safeguard for GPU
Source/Particles/PhysicalParticleContainer.cpp:                               np, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control)
Source/Particles/PhysicalParticleContainer.cpp:        [=] AMREX_GPU_DEVICE (long ip, auto exteb_control, auto qed_control)
Source/Particles/PhysicalParticleContainer.cpp:    amrex::Gpu::Buffer<amrex::Long> unconverged_particles({0});
Source/Particles/PhysicalParticleContainer.cpp:                       np_to_push, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control,
Source/Particles/PhysicalParticleContainer.cpp:#if !defined(AMREX_USE_GPU)
Source/Particles/PhysicalParticleContainer.cpp:                amrex::Gpu::Atomic::Add(unconverged_particles_ptr, amrex::Long(1));
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::copyAsync(Gpu::hostToDevice,
Source/Particles/PhysicalParticleContainer.cpp:        Gpu::copyAsync(Gpu::hostToDevice,
Source/Particles/PhysicalParticleContainer.cpp:    amrex::ParallelFor(ion_atomic_number, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/Particles/PhysicalParticleContainer.cpp:    Gpu::synchronize();
Source/Particles/Filter/FilterFunctors.H:#include <AMReX_Gpu.H>
Source/Particles/Filter/FilterFunctors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Filter/FilterFunctors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Filter/FilterFunctors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Filter/FilterFunctors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Deposition/SharedDepositionUtils.H:#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
Source/Particles/Deposition/SharedDepositionUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Deposition/SharedDepositionUtils.H:            amrex::Gpu::Atomic::AddNoRet( &global(i, j, k), local(i, j, k));
Source/Particles/Deposition/SharedDepositionUtils.H:#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
Source/Particles/Deposition/SharedDepositionUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Deposition/SharedDepositionUtils.H:        amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/SharedDepositionUtils.H:            amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/SharedDepositionUtils.H:                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+j_j+ix, lo.y+l_j+iz, 0, 2*imode-1), 2._rt*sx_j[ix]*sz_j[iz]*wqx*xy.real());
Source/Particles/Deposition/SharedDepositionUtils.H:                amrex::Gpu::Atomic::AddNoRet( &j_buff(lo.x+j_j+ix, lo.y+l_j+iz, 0, 2*imode  ), 2._rt*sx_j[ix]*sz_j[iz]*wqx*xy.imag());
Source/Particles/Deposition/SharedDepositionUtils.H:                amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Deposition/CurrentDeposition.H:        amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:        amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:        amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:            amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:            amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:            amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode-1), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.real());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &jx_arr(lo.x+j_jx+ix, lo.y+l_jx+iz, 0, 2*imode  ), 2._rt*sx_jx[ix]*sz_jx[iz]*wqx*xy.imag());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode-1), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.real());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &jy_arr(lo.x+j_jy+ix, lo.y+l_jy+iz, 0, 2*imode  ), 2._rt*sx_jy[ix]*sz_jy[iz]*wqy*xy.imag());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode-1), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.real());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &jz_arr(lo.x+j_jz+ix, lo.y+l_jz+iz, 0, 2*imode  ), 2._rt*sx_jz[ix]*sz_jz[iz]*wqz*xy.imag());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/CurrentDeposition.H:        [=] AMREX_GPU_DEVICE (long ip) {
Source/Particles/Deposition/CurrentDeposition.H:            [=] AMREX_GPU_DEVICE (long ip) {
Source/Particles/Deposition/CurrentDeposition.H:#if defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA)
Source/Particles/Deposition/CurrentDeposition.H:    const std::size_t max_shared_mem_bytes = amrex::Gpu::Device::sharedMemPerBlock();
Source/Particles/Deposition/CurrentDeposition.H:                                     "Tile size too big for GPU shared memory current deposition");
Source/Particles/Deposition/CurrentDeposition.H:            nblocks, threads_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
Source/Particles/Deposition/CurrentDeposition.H:            [=] AMREX_GPU_DEVICE () noexcept {
Source/Particles/Deposition/CurrentDeposition.H:        Gpu::SharedMemory<amrex::Real> gsm;
Source/Particles/Deposition/CurrentDeposition.H:#else // not using hip/cuda
Source/Particles/Deposition/CurrentDeposition.H:    // using HIP/CUDA, and those things are checked prior
Source/Particles/Deposition/CurrentDeposition.H:    WARPX_ABORT_WITH_MESSAGE("Shared memory only implemented for HIP/CUDA");
Source/Particles/Deposition/CurrentDeposition.H:        [=] AMREX_GPU_DEVICE (long const ip) {
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
Source/Particles/Deposition/CurrentDeposition.H:        [=] AMREX_GPU_DEVICE (long const ip) {
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdxi);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdyj);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+j_new-1+j, lo.z+k_new-1+k), sdzk);
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdxi);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djr_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djr_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdyj);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djt_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djt_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 0), sdzk);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode-1), djz_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i_new-1+i, lo.y+k_new-1+k, 0, 2*imode), djz_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k_new-1+k, 0, 0, 0), sdxi);
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k_new-1+k, 0, 0, 0), sdyj);
Source/Particles/Deposition/CurrentDeposition.H:                amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k_new-1+k, 0, 0, 0), sdzk);
Source/Particles/Deposition/CurrentDeposition.H:        [=] AMREX_GPU_DEVICE (long const ip) {
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+j0_node+j, lo.z+k0_node+k), this_Jx);
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+j0_cell+j, lo.z+k0_node+k), this_Jy);
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+j0_node+j, lo.z+k0_cell+k), this_Jz);
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 0), this_Jx);
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode-1), djr_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+i0_cell+i, lo.y+k0_node+k, 0, 2*imode), djr_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 0), this_Jy);
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode-1), djy_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+i0_node+i, lo.y+k0_node+k, 0, 2*imode), djy_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 0), this_Jz);
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode-1), djz_cmplx.real());
Source/Particles/Deposition/CurrentDeposition.H:                            amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+i0_node+i, lo.y+k0_cell+k, 0, 2*imode), djz_cmplx.imag());
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jx_arr(lo.x+k0_node+k, 0, 0), wqx*weight);
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jy_arr(lo.x+k0_node+k, 0, 0), wqy*weight);
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet( &Jz_arr(lo.x+k0_cell+k, 0, 0), this_Jz);
Source/Particles/Deposition/CurrentDeposition.H:    amrex::ParallelFor(np_to_deposit, [=] AMREX_GPU_DEVICE (long ip)
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 1),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 1),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 1),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_new + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_new + i, lo.y + k_old + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_new + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(&Dy_arr(lo.x + i_old + i, lo.y + k_old + k, 0, 0),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 1),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 2),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 3),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_new + k, 0),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_old + k, 0),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_new + j, lo.z + k_old + k, 1),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_old + j, lo.z + k_new + k, 1),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_new + k, 2),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_old + k, 2),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_old + i, lo.y + j_new + j, lo.z + k_new + k, 3),
Source/Particles/Deposition/CurrentDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(&temp_arr(lo.x + i_new + i, lo.y + j_old + j, lo.z + k_old + k, 3),
Source/Particles/Deposition/CurrentDeposition.H:    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Particles/Deposition/CurrentDeposition.H:    amrex::ParallelFor(Dx_fab.box(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
Source/Particles/Deposition/CurrentDeposition.H:    amrex::Gpu::streamSynchronize();
Source/Particles/Deposition/ChargeDeposition.H:            [=] AMREX_GPU_DEVICE (long ip) {
Source/Particles/Deposition/ChargeDeposition.H:                amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/ChargeDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/ChargeDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &rho_arr(lo.x+i+ix, lo.y+k+iz, 0, 2*imode-1), 2._rt*sx[ix]*sz[iz]*wq*xy.real());
Source/Particles/Deposition/ChargeDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &rho_arr(lo.x+i+ix, lo.y+k+iz, 0, 2*imode  ), 2._rt*sx[ix]*sz[iz]*wq*xy.imag());
Source/Particles/Deposition/ChargeDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/ChargeDeposition.H:#if !defined(AMREX_USE_GPU)
Source/Particles/Deposition/ChargeDeposition.H:#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/Deposition/ChargeDeposition.H:    const std::size_t max_shared_mem_bytes = amrex::Gpu::Device::sharedMemPerBlock();
Source/Particles/Deposition/ChargeDeposition.H:                                     "Tile size too big for GPU shared memory charge deposition");
Source/Particles/Deposition/ChargeDeposition.H:#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/Deposition/ChargeDeposition.H:                  nblocks, threads_per_block, shared_mem_bytes, amrex::Gpu::gpuStream(),
Source/Particles/Deposition/ChargeDeposition.H:                  [=] AMREX_GPU_DEVICE () noexcept
Source/Particles/Deposition/ChargeDeposition.H:#else // defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/Deposition/ChargeDeposition.H:    amrex::ParallelFor(np_to_deposit, [=] AMREX_GPU_DEVICE (long ip_orig) noexcept
Source/Particles/Deposition/ChargeDeposition.H:#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/Deposition/ChargeDeposition.H:        Gpu::SharedMemory<amrex::Real> gsm;
Source/Particles/Deposition/ChargeDeposition.H:#endif // defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/Deposition/ChargeDeposition.H:#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/Deposition/ChargeDeposition.H:                amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/ChargeDeposition.H:                    amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/ChargeDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &buf(lo.x+i+ix, lo.y+k+iz, 0, 2*imode-1), 2._rt*sx[ix]*sz[iz]*wq*xy.real());
Source/Particles/Deposition/ChargeDeposition.H:                        amrex::Gpu::Atomic::AddNoRet( &buf(lo.x+i+ix, lo.y+k+iz, 0, 2*imode  ), 2._rt*sx[ix]*sz[iz]*wq*xy.imag());
Source/Particles/Deposition/ChargeDeposition.H:                        amrex::Gpu::Atomic::AddNoRet(
Source/Particles/Deposition/ChargeDeposition.H:#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/Deposition/ChargeDeposition.H:#endif // defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
Source/Particles/PhysicalParticleContainer.H:#include <AMReX_GpuContainers.H>
Source/Particles/PhysicalParticleContainer.H:        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_x,
Source/Particles/PhysicalParticleContainer.H:        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_y,
Source/Particles/PhysicalParticleContainer.H:        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_z,
Source/Particles/PhysicalParticleContainer.H:        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_ux,
Source/Particles/PhysicalParticleContainer.H:        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_uy,
Source/Particles/PhysicalParticleContainer.H:        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_uz,
Source/Particles/PhysicalParticleContainer.H:        amrex::Gpu::HostVector<amrex::ParticleReal>& particle_w,
Source/Particles/PhysicalParticleContainer.H:            --between ParIter iterations-- on GPU) for field Ex
Source/Particles/PhysicalParticleContainer.H:            --between ParIter iterations-- on GPU) for field Ey
Source/Particles/PhysicalParticleContainer.H:            --between ParIter iterations-- on GPU) for field Ez
Source/Particles/PhysicalParticleContainer.H:            --between ParIter iterations-- on GPU) for field Bx
Source/Particles/PhysicalParticleContainer.H:            --between ParIter iterations-- on GPU) for field By
Source/Particles/PhysicalParticleContainer.H:            --between ParIter iterations-- on GPU) for field Bz
Source/Particles/Algorithms/KineticEnergy.H:#include "AMReX_GpuQualifiers.H"
Source/Particles/Algorithms/KineticEnergy.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Algorithms/KineticEnergy.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/WarpXParticleContainer_fwd.H:#include <AMReX_GpuAllocators.H>
Source/Particles/MultiParticleContainer.H:#include <AMReX_GpuControl.H>
Source/Particles/MultiParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_starts;
Source/Particles/MultiParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_lengths;
Source/Particles/MultiParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_strengths_E;
Source/Particles/MultiParticleContainer.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_repeated_plasma_lens_strengths_B;
Source/Particles/MultiParticleContainer.H:        if (WarpXParticleContainer::do_tiling && amrex::Gpu::notInLaunchRegion()) {
Source/Particles/MultiParticleContainer.H:        if (WarpXParticleContainer::do_tiling && amrex::Gpu::notInLaunchRegion()) {
Source/Particles/WarpXParticleContainer.cpp:#include <AMReX_GpuAllocators.H>
Source/Particles/WarpXParticleContainer.cpp:#include <AMReX_GpuAtomic.H>
Source/Particles/WarpXParticleContainer.cpp:#include <AMReX_GpuControl.H>
Source/Particles/WarpXParticleContainer.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/WarpXParticleContainer.cpp:#include <AMReX_GpuLaunch.H>
Source/Particles/WarpXParticleContainer.cpp:#include <AMReX_GpuQualifiers.H>
Source/Particles/WarpXParticleContainer.cpp:    // On GPU: particles deposit directly on the J arrays, which usually have a larger number of guard cells
Source/Particles/WarpXParticleContainer.cpp:#ifndef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:        "Particles shape does not fit within tile (CPU) or guard cells (GPU) used for current deposition");
Source/Particles/WarpXParticleContainer.cpp:#ifndef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:#ifdef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:    // GPU, no tiling: j<xyz>_arr point to the full j<xyz> arrays
Source/Particles/WarpXParticleContainer.cpp:                    [=] AMREX_GPU_HOST_DEVICE (const ParticleType& p) -> unsigned int
Source/Particles/WarpXParticleContainer.cpp:#ifndef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/WarpXParticleContainer.cpp:        // On GPU: particles deposit directly on the rho array, which usually have a larger number of guard cells
Source/Particles/WarpXParticleContainer.cpp:#ifndef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:                                  "Particles shape does not fit within tile (CPU) or guard cells (GPU) used for charge deposition");
Source/Particles/WarpXParticleContainer.cpp:#ifndef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:#ifdef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:        // GPU, no tiling: rho_fab points to the full rho array
Source/Particles/WarpXParticleContainer.cpp:                       [=] AMREX_GPU_HOST_DEVICE (ParticleType const & p) -> unsigned int
Source/Particles/WarpXParticleContainer.cpp:        amrex::Gpu::DeviceVector<Box> tboxes(bins.numBins(), amrex::Box());
Source/Particles/WarpXParticleContainer.cpp:                               [=] AMREX_GPU_DEVICE (int ibin) {
Source/Particles/WarpXParticleContainer.cpp:                           [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
Source/Particles/WarpXParticleContainer.cpp:#ifndef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/WarpXParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/WarpXParticleContainer.cpp:                            [=] AMREX_GPU_DEVICE (int ip)
Source/Particles/WarpXParticleContainer.cpp:#ifdef AMREX_USE_GPU
Source/Particles/WarpXParticleContainer.cpp:    if (Gpu::inLaunchRegion())
Source/Particles/WarpXParticleContainer.cpp:                               [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
Source/Particles/WarpXParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/WarpXParticleContainer.cpp:                [=] AMREX_GPU_DEVICE (int ip)
Source/Particles/WarpXParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/WarpXParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/WarpXParticleContainer.cpp:                [=] AMREX_GPU_DEVICE (long i) {
Source/Particles/WarpXParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/WarpXParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/WarpXParticleContainer.cpp:                [=] AMREX_GPU_DEVICE (long i, amrex::RandomEngine const& engine) {
Source/Particles/ParticleBoundaryBuffer.cpp:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_plo;
Source/Particles/ParticleBoundaryBuffer.cpp:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_phi;
Source/Particles/ParticleBoundaryBuffer.cpp:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleBoundaryBuffer.cpp:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxi;
Source/Particles/ParticleBoundaryBuffer.cpp:    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_plo;
Source/Particles/ParticleBoundaryBuffer.cpp:    AMREX_GPU_HOST_DEVICE
Source/Particles/ParticleBoundaryBuffer.cpp:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dxi = m_dxi;
Source/Particles/ParticleBoundaryBuffer.cpp:        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const plo = m_plo;
Source/Particles/ParticleBoundaryBuffer.cpp:    AMREX_GPU_HOST_DEVICE
Source/Particles/ParticleBoundaryBuffer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/ParticleBoundaryBuffer.cpp:                          reduce_op.eval(np, reduce_data, [=] AMREX_GPU_HOST_DEVICE (int ip)
Source/Particles/ParticleBoundaryBuffer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/ParticleBoundaryBuffer.cpp:                    auto predicate = [=] AMREX_GPU_HOST_DEVICE(const SrcData & /*src*/, const int ip)
Source/Particles/ParticleBoundaryBuffer.cpp:                                       [=] AMREX_GPU_HOST_DEVICE(int ip) { return predicate(ptile_data, ip) ? 1 : 0; });
Source/Particles/ElementaryProcess/QEDPhotonEmission.H:#include <AMReX_GpuLaunch.H>
Source/Particles/ElementaryProcess/QEDPhotonEmission.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ElementaryProcess/QEDPhotonEmission.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/QEDPhotonEmission.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/QEDPhotonEmission.H:    amrex::ParallelFor(num_added, [=] AMREX_GPU_DEVICE (int ip) noexcept
Source/Particles/ElementaryProcess/Ionization.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ElementaryProcess/Ionization.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/Ionization.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/QEDPairGeneration.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ElementaryProcess/QEDPairGeneration.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/QEDPairGeneration.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.cpp:    amrex::Gpu::synchronize();
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.cpp:    amrex::Gpu::synchronize();
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.cpp:    amrex::Gpu::synchronize();
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.cpp:    amrex::Gpu::synchronize();
Source/Particles/ElementaryProcess/QEDInternals/QedChiFunctions.H:    * Suitable for GPU kernels.
Source/Particles/ElementaryProcess/QEDInternals/QedChiFunctions.H:    AMREX_GPU_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/QedChiFunctions.H:    * Suitable for GPU kernels.
Source/Particles/ElementaryProcess/QEDInternals/QedChiFunctions.H:    AMREX_GPU_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:#include <AMReX_GpuDevice.H>
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H: * PICSAR uses PXRMP_GPU to decorate methods which should be
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H: * compiled for GPU. The user has to set it to the right value
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H: * (AMREX_GPU_DEVICE in this case).
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:#define PXRMP_WITH_GPU
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:#define PXRMP_GPU_QUALIFIER AMREX_GPU_HOST_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:#ifdef AMREX_USE_GPU
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:* an amrex::Gpu::DeviceVector. It provides a pointer to the Device data via the data() function.
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:    using AGDV = amrex::Gpu::DeviceVector<Real>;
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:    * It forces a copy of the CPU data to the GPU
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:        amrex::Gpu::Device::streamSynchronize();
Source/Particles/ElementaryProcess/QEDInternals/QedWrapperCommons.H:    * amrex::Gpu::DeviceVector<Real> m_device_data.
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:// They can be included in GPU kernels.
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:     * generate the optical depth. It can be used on GPU.
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:     * Evolves the optical depth. It can be used on GPU.
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:    AMREX_GPU_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:     * allocations should be triggered on GPU
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:     * It can be used on GPU.
Source/Particles/ElementaryProcess/QEDInternals/QuantumSyncEngineWrapper.H:    AMREX_GPU_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:// They can be included in GPU kernels.
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:     * generate the optical depth. It can be used on GPU.
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:    AMREX_GPU_HOST_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:     * Evolves the optical depth. It can be used on GPU.
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:    AMREX_GPU_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:     * allocations should be triggered on GPU
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:     * It can be used on GPU.
Source/Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H:    AMREX_GPU_DEVICE
Source/Particles/ElementaryProcess/QEDInternals/SchwingerProcessWrapper.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/QEDSchwingerProcess.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ElementaryProcess/QEDSchwingerProcess.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/RigidInjectedParticleContainer.cpp:#include <AMReX_GpuContainers.H>
Source/Particles/RigidInjectedParticleContainer.cpp:#include <AMReX_GpuControl.H>
Source/Particles/RigidInjectedParticleContainer.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/RigidInjectedParticleContainer.cpp:#include <AMReX_GpuLaunch.H>
Source/Particles/RigidInjectedParticleContainer.cpp:#include <AMReX_GpuQualifiers.H>
Source/Particles/RigidInjectedParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particles/RigidInjectedParticleContainer.cpp:                    amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long i)
Source/Particles/RigidInjectedParticleContainer.cpp:    amrex::Gpu::DeviceVector<ParticleReal> xp_save, yp_save, zp_save;
Source/Particles/RigidInjectedParticleContainer.cpp:                            [=] AMREX_GPU_DEVICE (long i) {
Source/Particles/RigidInjectedParticleContainer.cpp:                            [=] AMREX_GPU_DEVICE (long i) {
Source/Particles/RigidInjectedParticleContainer.cpp:            amrex::Gpu::DeviceVector<ParticleReal> uxp_save(np);
Source/Particles/RigidInjectedParticleContainer.cpp:            amrex::Gpu::DeviceVector<ParticleReal> uyp_save(np);
Source/Particles/RigidInjectedParticleContainer.cpp:            amrex::Gpu::DeviceVector<ParticleReal> uzp_save(np);
Source/Particles/RigidInjectedParticleContainer.cpp:                               np, [=] AMREX_GPU_DEVICE (long ip, auto exteb_control)
Source/Particles/RigidInjectedParticleContainer.cpp:            amrex::ParallelFor( pti.numParticles(), [=] AMREX_GPU_DEVICE (long i)
Source/Particles/RigidInjectedParticleContainer.cpp:            amrex::Gpu::synchronize();
Source/Particles/LaserParticleContainer.cpp:#include <AMReX_GpuAtomic.H>
Source/Particles/LaserParticleContainer.cpp:#include <AMReX_GpuContainers.H>
Source/Particles/LaserParticleContainer.cpp:#include <AMReX_GpuControl.H>
Source/Particles/LaserParticleContainer.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/LaserParticleContainer.cpp:#include <AMReX_GpuLaunch.H>
Source/Particles/LaserParticleContainer.cpp:#include <AMReX_GpuQualifiers.H>
Source/Particles/LaserParticleContainer.cpp:    const Vector<int>& procmap = plane_dm.ProcessorMap();
Source/Particles/LaserParticleContainer.cpp:        if (procmap[i] == myproc)
Source/Particles/LaserParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/LaserParticleContainer.cpp:        Gpu::DeviceVector<Real> plane_Xp, plane_Yp, amplitude_E;
Source/Particles/LaserParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/LaserParticleContainer.cpp:            amrex::Gpu::synchronize();
Source/Particles/LaserParticleContainer.cpp:        [=] AMREX_GPU_DEVICE (int i) {
Source/Particles/LaserParticleContainer.cpp:    // Copy member variables to tmp copies for GPU runs.
Source/Particles/LaserParticleContainer.cpp:        [=] AMREX_GPU_DEVICE (int i) {
Source/Particles/ParticleCreation/FilterCreateTransformFromFAB.H:    Gpu::DeviceVector<Index> offsets(ncells);
Source/Particles/ParticleCreation/FilterCreateTransformFromFAB.H:    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
Source/Particles/ParticleCreation/FilterCreateTransformFromFAB.H:    Gpu::synchronize();
Source/Particles/ParticleCreation/FilterCreateTransformFromFAB.H:    // This may be unnecessary because of the Gpu::streamSynchronize() in
Source/Particles/ParticleCreation/FilterCreateTransformFromFAB.H:    Gpu::DeviceVector<Index> mask(ncells);
Source/Particles/ParticleCreation/FilterCreateTransformFromFAB.H:    [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine){
Source/Particles/ParticleCreation/SmartCreate.H:#include <AMReX_GpuContainers.H>
Source/Particles/ParticleCreation/SmartCreate.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/DefaultInitialization.H:#include <AMReX_GpuContainers.H>
Source/Particles/ParticleCreation/DefaultInitialization.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/DefaultInitialization.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/DefaultInitialization.H:                // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
Source/Particles/ParticleCreation/DefaultInitialization.H:                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
Source/Particles/ParticleCreation/DefaultInitialization.H:                                              [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
Source/Particles/ParticleCreation/DefaultInitialization.H:                // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
Source/Particles/ParticleCreation/DefaultInitialization.H:                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
Source/Particles/ParticleCreation/DefaultInitialization.H:                                              [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
Source/Particles/ParticleCreation/DefaultInitialization.H:                    // If the particle tile was allocated in a memory pool that can run on GPU, launch GPU kernel
Source/Particles/ParticleCreation/DefaultInitialization.H:                    if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<amrex::Real>>::value) {
Source/Particles/ParticleCreation/DefaultInitialization.H:                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
Source/Particles/ParticleCreation/DefaultInitialization.H:                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<int>>::value) {
Source/Particles/ParticleCreation/DefaultInitialization.H:                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
Source/Particles/ParticleCreation/DefaultInitialization.H:                if constexpr (amrex::RunOnGpu<typename PTile::template AllocatorType<int>>::value) {
Source/Particles/ParticleCreation/DefaultInitialization.H:                                           [=] AMREX_GPU_DEVICE (int i) noexcept {
Source/Particles/ParticleCreation/FilterCopyTransform.H:#include <AMReX_GpuContainers.H>
Source/Particles/ParticleCreation/FilterCopyTransform.H: *            AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/FilterCopyTransform.H:    Gpu::DeviceVector<Index> offsets(np);
Source/Particles/ParticleCreation/FilterCopyTransform.H:    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
Source/Particles/ParticleCreation/FilterCopyTransform.H:    Gpu::synchronize();
Source/Particles/ParticleCreation/FilterCopyTransform.H: *            AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/FilterCopyTransform.H:    Gpu::DeviceVector<Index> mask(np);
Source/Particles/ParticleCreation/FilterCopyTransform.H:    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
Source/Particles/ParticleCreation/FilterCopyTransform.H: *            AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/FilterCopyTransform.H:    Gpu::DeviceVector<Index> offsets(np);
Source/Particles/ParticleCreation/FilterCopyTransform.H:    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
Source/Particles/ParticleCreation/FilterCopyTransform.H:    Gpu::synchronize();
Source/Particles/ParticleCreation/FilterCopyTransform.H: *            AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/FilterCopyTransform.H:    Gpu::DeviceVector<Index> mask(np);
Source/Particles/ParticleCreation/FilterCopyTransform.H:    [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine)
Source/Particles/ParticleCreation/SmartUtils.cpp:#include <AMReX_GpuContainers.H>
Source/Particles/ParticleCreation/SmartUtils.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_policies.begin(), h_policies.end(), policies.begin());
Source/Particles/ParticleCreation/SmartUtils.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Particles/ParticleCreation/SmartUtils.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_src_comps.begin(), h_src_comps.end(), tag.src_comps.begin());
Source/Particles/ParticleCreation/SmartUtils.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_dst_comps.begin(), h_dst_comps.end(), tag.dst_comps.begin());
Source/Particles/ParticleCreation/SmartUtils.cpp:    amrex::Gpu::Device::streamSynchronize();
Source/Particles/ParticleCreation/SmartCopy.H:#include <AMReX_GpuContainers.H>
Source/Particles/ParticleCreation/SmartCopy.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/ParticleCreation/SmartUtils.H:#include <AMReX_GpuContainers.H>
Source/Particles/ParticleCreation/SmartUtils.H:#include <AMReX_GpuLaunch.H>
Source/Particles/ParticleCreation/SmartUtils.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/ParticleCreation/SmartUtils.H:using PolicyVec = amrex::Gpu::DeviceVector<InitializationPolicy>;
Source/Particles/ParticleCreation/SmartUtils.H:    amrex::Gpu::DeviceVector<int> src_comps;
Source/Particles/ParticleCreation/SmartUtils.H:    amrex::Gpu::DeviceVector<int> dst_comps;
Source/Particles/ParticleCreation/SmartUtils.H:    amrex::ParallelFor(num_added, [=] AMREX_GPU_DEVICE (int ip) noexcept
Source/Particles/AddPlasmaUtilities.cpp:                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
Source/Particles/AddPlasmaUtilities.cpp:                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
Source/Particles/AddPlasmaUtilities.cpp:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
Source/Particles/AddPlasmaUtilities.cpp:                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
Source/Particles/AddPlasmaUtilities.cpp:#ifdef AMREX_USE_GPU
Source/Particles/AddPlasmaUtilities.cpp:#ifdef AMREX_USE_GPU
Source/Particles/AddPlasmaUtilities.cpp:#ifdef AMREX_USE_GPU
Source/Particles/AddPlasmaUtilities.cpp:#ifdef AMREX_USE_GPU
Source/Particles/Collision/ScatteringProcess.H:#include <AMReX_GpuContainers.H>
Source/Particles/Collision/ScatteringProcess.H:                               amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas
Source/Particles/Collision/ScatteringProcess.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Collision/ScatteringProcess.H:#ifdef AMREX_USE_GPU
Source/Particles/Collision/ScatteringProcess.H:#ifdef AMREX_USE_GPU
Source/Particles/Collision/ScatteringProcess.H:    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_sigmas_d;
Source/Particles/Collision/ScatteringProcess.H:    amrex::Gpu::HostVector<amrex::ParticleReal> m_sigmas_h;
Source/Particles/Collision/ScatteringProcess.cpp:#ifdef AMREX_USE_GPU
Source/Particles/Collision/ScatteringProcess.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_sigmas_h.begin(), m_sigmas_h.end(),
Source/Particles/Collision/ScatteringProcess.cpp:    amrex::Gpu::streamSynchronize();
Source/Particles/Collision/ScatteringProcess.cpp:                                  amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas )
Source/Particles/Collision/BackgroundStopping/BackgroundStopping.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/Collision/BackgroundStopping/BackgroundStopping.cpp:                amrex::Gpu::synchronize();
Source/Particles/Collision/BackgroundStopping/BackgroundStopping.cpp:                amrex::Gpu::synchronize();
Source/Particles/Collision/BackgroundStopping/BackgroundStopping.cpp:    // So that CUDA code gets its intrinsic, not the host-only C++ library version
Source/Particles/Collision/BackgroundStopping/BackgroundStopping.cpp:        [=] AMREX_GPU_HOST_DEVICE (long ip)
Source/Particles/Collision/BackgroundStopping/BackgroundStopping.cpp:    // So that CUDA code gets its intrinsic, not the host-only C++ library version
Source/Particles/Collision/BackgroundStopping/BackgroundStopping.cpp:        [=] AMREX_GPU_HOST_DEVICE (long ip)
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:#ifdef AMREX_USE_GPU
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_ionization_processes_exe;
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_ionization_processes_exe.begin(),
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:    amrex::Gpu::streamSynchronize();
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:                amrex::Gpu::synchronize();
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:                amrex::Gpu::synchronize();
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:    // So that CUDA code gets its intrinsic, not the host-only C++ library version
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:                          [=] AMREX_GPU_HOST_DEVICE (long ip, amrex::RandomEngine const& engine)
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:            amrex::Gpu::synchronize();
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.cpp:            amrex::Gpu::synchronize();
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.H:#include <AMReX_GpuContainers.H>
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.H:    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_scattering_processes_exe;
Source/Particles/Collision/BackgroundMCC/BackgroundMCCCollision.H:    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_ionization_processes_exe;
Source/Particles/Collision/BackgroundMCC/ImpactIonization.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Collision/BackgroundMCC/ImpactIonization.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Particles/Collision/BinaryCollision/ShuffleFisherYates.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.cpp:#include <AMReX_GpuContainers.H>
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.cpp:#ifndef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.cpp:#ifndef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.cpp:#ifdef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.cpp:     amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_num_products_host.begin(),
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.cpp:     amrex::Gpu::streamSynchronize();
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#include <AMReX_GpuAtomic.H>
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#include <AMReX_GpuContainers.H>
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#include <AMReX_GpuControl.H>
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#include <AMReX_GpuDevice.H>
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#include <AMReX_GpuLaunch.H>
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#include <AMReX_GpuQualifiers.H>
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#ifdef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:        amrex::Gpu::DeviceVector<SmartCopy> device_copy_species1(n_product_species);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:        amrex::Gpu::DeviceVector<SmartCopy> device_copy_species2(n_product_species);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, copy_species1.begin(),
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, copy_species2.begin(),
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:        amrex::Gpu::streamSynchronize();
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:        if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(species1.tile_size); }
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                    amrex::Gpu::synchronize();
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                    amrex::Gpu::synchronize();
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:        auto volume_factor = [=] AMREX_GPU_DEVICE(int i_cell) noexcept {
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> n_pairs_in_each_cell(n_cells_products);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> pair_offsets(n_cells_products);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> n_ind_pairs_in_each_cell(n_cells+1);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> coll_offsets(n_cells+1);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> mask(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> pair_indices_1(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> pair_indices_2(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<amrex::ParticleReal> pair_reaction_weight(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<amrex::ParticleReal> n1_vec;
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<amrex::ParticleReal> T1_vec;
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            // To speed up binary collisions on GPU, we try to expose as much parallelism
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            // as possible (while avoiding race conditions): Instead of looping with one GPU
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            // thread per cell, we loop with one GPU thread per "independent pairs" (i.e. pairs
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_coll, amrex::RandomEngine const& engine) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> n_pairs_in_each_cell(n_cells_products);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> pair_offsets(n_cells_products);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> n_ind_pairs_in_each_cell(n_cells+1);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> coll_offsets(n_cells+1);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> mask(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> pair_indices_1(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<index_type> pair_indices_2(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<amrex::ParticleReal> pair_reaction_weight(n_total_pairs);
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<amrex::ParticleReal> n1_vec, n2_vec;
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            amrex::Gpu::DeviceVector<amrex::ParticleReal> T1_vec, T2_vec;
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            // To speed up binary collisions on GPU, we try to expose as much parallelism
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            // as possible (while avoiding race conditions): Instead of looping with one GPU
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:            // thread per cell, we loop with one GPU thread per "independent pairs" (i.e. pairs
Source/Particles/Collision/BinaryCollision/BinaryCollision.H:                [=] AMREX_GPU_DEVICE (int i_coll, amrex::RandomEngine const& engine) noexcept
Source/Particles/Collision/BinaryCollision/BinaryCollisionUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/BinaryCollisionUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/BinaryCollisionUtils.H:        amrex::Gpu::Atomic::AddNoRet(&weight, -reaction_weight);
Source/Particles/Collision/BinaryCollision/BinaryCollisionUtils.H:            amrex::Gpu::Atomic::Exch(
Source/Particles/Collision/BinaryCollision/DSMC/DSMCFunc.H:        AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/DSMC/DSMCFunc.H:    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_scattering_processes_exe;
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.cpp:#ifndef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.cpp:#ifdef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.cpp:     amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_num_products_host.begin(),
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.cpp:     amrex::Gpu::streamSynchronize();
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:            [=] AMREX_GPU_DEVICE (index_type i) -> index_type { return mask[i] ? 1 : 0; },
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:            [=] AMREX_GPU_DEVICE (index_type i, index_type s) { offsets_data[i] = s; },
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        // Create necessary GPU vectors, that will be used in the kernel below
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:#ifdef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        amrex::Gpu::DeviceVector<SoaData_type> device_soa_products(m_num_product_species);
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        amrex::Gpu::DeviceVector<index_type> device_products_np(m_num_product_species);
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, soa_products.begin(),
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, products_np.begin(),
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        amrex::Gpu::streamSynchronize();
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:        amrex::Gpu::synchronize();
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:    // for device) which is necessary with GPUs but redundant on CPU.
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:    amrex::Gpu::DeviceVector<int> m_num_products_device;
Source/Particles/Collision/BinaryCollision/DSMC/SplitAndScatterFunc.H:    amrex::Gpu::HostVector<int> m_num_products_host;
Source/Particles/Collision/BinaryCollision/DSMC/DSMCFunc.cpp:#ifdef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/DSMC/DSMCFunc.cpp:    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
Source/Particles/Collision/BinaryCollision/DSMC/DSMCFunc.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
Source/Particles/Collision/BinaryCollision/DSMC/DSMCFunc.cpp:    amrex::Gpu::streamSynchronize();
Source/Particles/Collision/BinaryCollision/DSMC/CollisionFilterFunc.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/Coulomb/ComputeTemperature.H:AMREX_GPU_HOST_DEVICE
Source/Particles/Collision/BinaryCollision/Coulomb/UpdateMomentumPerezElastic.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/Coulomb/ElasticCollisionPerez.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/Coulomb/PairWiseCoulombCollisionFunc.H:        AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:#include <AMReX_GpuAtomic.H>
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:#include <AMReX_GpuDevice.H>
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:#include <AMReX_GpuContainers.H>
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::DeviceVector<index_type> offsets(n_total_pairs);
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        // Create necessary GPU vectors, that will be used in the kernel below
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:#ifdef AMREX_USE_GPU
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::DeviceVector<SoaData_type> device_soa_products(m_num_product_species);
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::DeviceVector<index_type> device_products_np(m_num_product_species);
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::DeviceVector<amrex::ParticleReal> device_products_mass(m_num_product_species);
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, soa_products.begin(),
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, products_np.begin(),
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, products_mass.begin(),
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::streamSynchronize();
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:        amrex::Gpu::synchronize();
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:    // for device) which is necessary with GPUs but redundant on CPU.
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:    amrex::Gpu::DeviceVector<int> m_num_products_device;
Source/Particles/Collision/BinaryCollision/ParticleCreationFunc.H:    amrex::Gpu::HostVector<int> m_num_products_host;
Source/Particles/Collision/BinaryCollision/NuclearFusion/TwoProductFusionUtil.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/ProtonBoronFusionCrossSection.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/ProtonBoronFusionCrossSection.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/ProtonBoronFusionCrossSection.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/TwoProductFusionInitializeMomentum.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/BoschHaleFusionCrossSection.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/NuclearFusionFunc.H:        AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/SingleNuclearFusionEvent.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/Collision/BinaryCollision/NuclearFusion/ProtonBoronFusionInitializeMomentum.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Particles/MultiParticleContainer.cpp:#include <AMReX_GpuAtomic.H>
Source/Particles/MultiParticleContainer.cpp:#include <AMReX_GpuDevice.H>
Source/Particles/MultiParticleContainer.cpp:            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Particles/MultiParticleContainer.cpp:            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Particles/MultiParticleContainer.cpp:            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Particles/MultiParticleContainer.cpp:            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Particles/MultiParticleContainer.cpp:            amrex::Gpu::synchronize();
Source/Particles/MultiParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particles/MultiParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/MultiParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/MultiParticleContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Particles/MultiParticleContainer.cpp:    for (MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Particles/MultiParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particles/MultiParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/MultiParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/MultiParticleContainer.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Particles/MultiParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Particles/MultiParticleContainer.cpp:                amrex::Gpu::synchronize();
Source/Diagnostics/FlushFormats/FlushFormatPlotfile.cpp:#include <AMReX_GpuAllocators.H>
Source/Diagnostics/FlushFormats/FlushFormatPlotfile.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/FlushFormats/FlushFormatPlotfile.cpp:                              AMREX_GPU_HOST_DEVICE
Source/Diagnostics/WarpXOpenPMD.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/WarpXOpenPMD.cpp:            AMREX_GPU_HOST_DEVICE
Source/Diagnostics/WarpXOpenPMD.cpp:            AMREX_GPU_HOST_DEVICE
Source/Diagnostics/WarpXOpenPMD.cpp:                //   GPU pointers to the I/O library
Source/Diagnostics/WarpXOpenPMD.cpp:#ifdef AMREX_USE_GPU
Source/Diagnostics/WarpXOpenPMD.cpp:                    amrex::Gpu::dtoh_memcpy_async(data_pinned.get(), fab.dataPtr(icomp), local_box.numPts()*sizeof(amrex::Real));
Source/Diagnostics/WarpXOpenPMD.cpp:                    // intentionally delayed until before we .flush(): amrex::Gpu::streamSynchronize();
Source/Diagnostics/WarpXOpenPMD.cpp:#ifdef AMREX_USE_GPU
Source/Diagnostics/WarpXOpenPMD.cpp:        amrex::Gpu::streamSynchronize();
Source/Diagnostics/ComputeDiagFunctors/PartPerGridFunctor.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:            [=] AMREX_GPU_DEVICE (const WarpXParticleContainer::SuperParticleType& p,
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 0), (amrex::Real)(p.rdata(PIdx::w) * value));
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                [=] AMREX_GPU_DEVICE (const WarpXParticleContainer::SuperParticleType& p,
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                    amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 0), (amrex::Real)(p.rdata(PIdx::w) * filter));
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:        for (amrex::MFIter mfi(red_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diagnostics/ComputeDiagFunctors/ParticleReductionFunctor.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:#include <AMReX_GpuLaunch.H>
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:#ifdef AMREX_USE_GPU
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:        Gpu::DeviceVector<int> d_map_varnames(m_map_varnames.size());
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:        Gpu::copyAsync(Gpu::hostToDevice,
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:        Gpu::synchronize();
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:        for (amrex::MFIter mfi(tmp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:    for (amrex::MFIter mfi(data, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diagnostics/ComputeDiagFunctors/BackTransformFunctor.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Diagnostics/ComputeDiagFunctors/BackTransformParticleFunctor.cpp:            amrex::Gpu::DeviceVector<int> FlagForPartCopy;
Source/Diagnostics/ComputeDiagFunctors/BackTransformParticleFunctor.cpp:            amrex::Gpu::DeviceVector<int> IndexForPartCopy;
Source/Diagnostics/ComputeDiagFunctors/BackTransformParticleFunctor.cpp:                [=] AMREX_GPU_DEVICE(int i)
Source/Diagnostics/ComputeDiagFunctors/BackTransformParticleFunctor.cpp:                [=] AMREX_GPU_DEVICE(int i)
Source/Diagnostics/ComputeDiagFunctors/BackTransformParticleFunctor.cpp:                amrex::Gpu::synchronize();
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:            [=] AMREX_GPU_DEVICE (const WarpXParticleContainer::SuperParticleType& p,
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi)
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 0), (amrex::Real)(w));
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 1), (amrex::Real)(w*ux));
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 2), (amrex::Real)(w*uy));
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 3), (amrex::Real)(w*uz));
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:    for (amrex::MFIter mfi(sum_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:            [=] AMREX_GPU_DEVICE (long ip) {
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 4), (amrex::Real)(w*ux*ux));
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 5), (amrex::Real)(w*uy*uy));
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                amrex::Gpu::Atomic::AddNoRet(&out_array(ii, jj, kk, 6), (amrex::Real)(w*uz*uz));
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:    for (amrex::MFIter mfi(sum_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diagnostics/ComputeDiagFunctors/TemperatureFunctor.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Diagnostics/ComputeDiagFunctors/BackTransformParticleFunctor.H:    AMREX_GPU_HOST_DEVICE
Source/Diagnostics/ComputeDiagFunctors/BackTransformParticleFunctor.H:    AMREX_GPU_HOST_DEVICE
Source/Diagnostics/ParticleIO.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ParticleIO.cpp:#include <AMReX_GpuLaunch.H>
Source/Diagnostics/ParticleIO.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ParticleIO.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ParticleIO.cpp:                [=] AMREX_GPU_DEVICE (long ip) {
Source/Diagnostics/ReducedDiags/FieldReduction.H:#include <AMReX_GpuControl.H>
Source/Diagnostics/ReducedDiags/FieldReduction.H:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/FieldReduction.H:        const amrex::GpuArray<int,3> cellCenteredtype{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        const amrex::GpuArray<int,3> reduction_coarsening_ratio{1,1,1};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto Extype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto Eytype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto Eztype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto Bxtype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto Bytype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto Bztype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto jxtype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto jytype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:        auto jztype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldReduction.H:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/FieldReduction.H:        for ( amrex::MFIter mfi(Ex, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Diagnostics/ReducedDiags/FieldReduction.H:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:#include <AMReX_GpuAtomic.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:#include <AMReX_GpuContainers.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:#include <AMReX_GpuLaunch.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:    amrex::Gpu::DeviceVector< amrex::Real > d_data( m_data.size(), 0.0 );
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:                   [=] AMREX_GPU_DEVICE(int i)
Source/Diagnostics/ReducedDiags/ParticleHistogram.cpp:    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
Source/Diagnostics/ReducedDiags/ColliderRelevant.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/ColliderRelevant.cpp:            [=] AMREX_GPU_HOST_DEVICE (const PType& p)
Source/Diagnostics/ReducedDiags/ColliderRelevant.cpp:            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real,
Source/Diagnostics/ReducedDiags/ColliderRelevant.cpp:                [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real>
Source/Diagnostics/ReducedDiags/ColliderRelevant.cpp:            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real,
Source/Diagnostics/ReducedDiags/ColliderRelevant.cpp:                [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real, Real, Real>
Source/Diagnostics/ReducedDiags/ColliderRelevant.cpp:                [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
Source/Diagnostics/ReducedDiags/FieldEnergy.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/FieldEnergy.cpp:    for ( amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Diagnostics/ReducedDiags/FieldEnergy.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) -> ReduceTuple
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.H:     *  (cost, processor, level, i_low, j_low, k_low, gpu_ID [if GPU run], num_cells, num_macro_particles
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.H:#ifdef AMREX_USE_GPU
Source/Diagnostics/ReducedDiags/FieldProbe.cpp:                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
Source/Diagnostics/ReducedDiags/FieldProbe.cpp:                // Temporarily defining modes and interp outside ParallelFor to avoid GPU compilation errors.
Source/Diagnostics/ReducedDiags/FieldProbe.cpp:                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
Source/Diagnostics/ReducedDiags/FieldProbe.cpp:                    amrex::Gpu::DeviceVector<amrex::Real> dv(np*noutputs);
Source/Diagnostics/ReducedDiags/FieldProbe.cpp:                    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip)
Source/Diagnostics/ReducedDiags/FieldProbe.cpp:                    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
Source/Diagnostics/ReducedDiags/FieldProbe.cpp:                    Gpu::streamSynchronize();
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:#ifdef AMREX_USE_GPU
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:            m_data[shift_m_data + mfi.index()*m_nDataFields + 8] = amrex::Gpu::Device::deviceId();
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:     *  [[cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 0 at level 0,
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 1 at level 0,
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 2 at level 0,
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:     *   [cost, proc, lev, i_low, j_low, k_low num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 0 at level 1,
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 1 at level 1,
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 2 at level 1,
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:        //   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID_box), hostname]
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:#ifdef AMREX_USE_GPU
Source/Diagnostics/ReducedDiags/LoadBalanceCosts.cpp:            ofstmp << "[" << c++ << "]gpu_ID_box_" + std::to_string(boxNumber) + "()";
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:#include <AMReX_GpuAtomic.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:#include <AMReX_GpuContainers.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:#include <AMReX_GpuLaunch.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:                                   [=] AMREX_GPU_DEVICE(int i)
Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp:    // Copy data from GPU memory
Source/Diagnostics/ReducedDiags/ParticleNumber.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        const GpuArray<int,3> cellCenteredtype{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        const GpuArray<int,3> reduction_coarsening_ratio{1,1,1};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        auto Extype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        auto Eytype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        auto Eztype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        auto Bxtype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        auto Bytype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        auto Bztype = amrex::GpuArray<int,3>{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:        for ( MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Diagnostics/ReducedDiags/FieldMaximum.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
Source/Diagnostics/ReducedDiags/ParticleMomentum.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/ParticleMomentum.cpp:            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real, Real, Real>
Source/Diagnostics/ReducedDiags/ChargeOnEB.cpp:#include <AMReX_GpuAtomic.H>
Source/Diagnostics/ReducedDiags/ChargeOnEB.cpp:    const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();
Source/Diagnostics/ReducedDiags/ChargeOnEB.cpp:    amrex::Gpu::Buffer<amrex::Real> surface_integral({0.0_rt});
Source/Diagnostics/ReducedDiags/ChargeOnEB.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/ChargeOnEB.cpp:    for (amrex::MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diagnostics/ReducedDiags/ChargeOnEB.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Diagnostics/ReducedDiags/ParticleEnergy.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/ParticleEnergy.cpp:                [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real>
Source/Diagnostics/ReducedDiags/ParticleEnergy.cpp:                [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real>
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        const amrex::GpuArray<int,3> cc{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        const amrex::GpuArray<int,3> cr{1,1,1};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        amrex::GpuArray<int,3> Ex_stag{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        amrex::GpuArray<int,3> Ey_stag{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        amrex::GpuArray<int,3> Ez_stag{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        amrex::GpuArray<int,3> Bx_stag{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        amrex::GpuArray<int,3> By_stag{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        amrex::GpuArray<int,3> Bz_stag{0,0,0};
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:        for (amrex::MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Diagnostics/ReducedDiags/FieldMomentum.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<Real, Real, Real>
Source/Diagnostics/ReducedDiags/ParticleExtrema.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/ParticleExtrema.cpp:            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
Source/Diagnostics/ReducedDiags/ParticleExtrema.cpp:            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real,
Source/Diagnostics/ReducedDiags/ParticleExtrema.cpp:                    [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
Source/Diagnostics/ReducedDiags/FieldProbeParticleContainer.cpp:#include <AMReX_GpuAllocators.H>
Source/Diagnostics/ReducedDiags/FieldProbeParticleContainer.cpp:#include <AMReX_GpuAtomic.H>
Source/Diagnostics/ReducedDiags/FieldProbeParticleContainer.cpp:#include <AMReX_GpuControl.H>
Source/Diagnostics/ReducedDiags/FieldProbeParticleContainer.cpp:#include <AMReX_GpuDevice.H>
Source/Diagnostics/ReducedDiags/FieldProbeParticleContainer.cpp:#include <AMReX_GpuLaunch.H>
Source/Diagnostics/ReducedDiags/FieldProbeParticleContainer.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.cpp:    if (amrex::Gpu::notInLaunchRegion()) { info.EnableTiling(WarpXParticleContainer::tile_size); }
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.cpp:                [=] AMREX_GPU_DEVICE (int i_cell) noexcept
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.cpp:    // so we copy it from the GPU to the CPU and reduce across MPI ranks.
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.cpp:        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.H:#include <AMReX_GpuContainers.H>
Source/Diagnostics/ReducedDiags/DifferentialLuminosity.H:    amrex::Gpu::DeviceVector< amrex::Real > d_data;
Source/Diagnostics/ReducedDiags/BeamRelevant.cpp:#include <AMReX_GpuQualifiers.H>
Source/Diagnostics/ReducedDiags/BeamRelevant.cpp:            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> ReducedDataT1::Type
Source/Diagnostics/ReducedDiags/BeamRelevant.cpp:            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> ReducedDataT2::Type
Source/Diagnostics/WarpXOpenPMD.H:#include <AMReX_GpuAllocators.H>
Source/Filter/NCIGodfreyFilter.cpp:#include <AMReX_GpuContainers.H>
Source/Filter/NCIGodfreyFilter.cpp:#include <AMReX_GpuDevice.H>
Source/Filter/NCIGodfreyFilter.cpp:    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_x.begin(),h_stencil_x.end(),m_stencil_0.begin());
Source/Filter/NCIGodfreyFilter.cpp:    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_y.begin(),h_stencil_y.end(),m_stencil_1.begin());
Source/Filter/NCIGodfreyFilter.cpp:    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_z.begin(),h_stencil_z.end(),m_stencil_2.begin());
Source/Filter/NCIGodfreyFilter.cpp:    Gpu::copyAsync(Gpu::hostToDevice,h_stencil_z.begin(),h_stencil_z.end(),m_stencil_1.begin());
Source/Filter/NCIGodfreyFilter.cpp:    Gpu::synchronize();
Source/Filter/Filter.H:#include <AMReX_GpuContainers.H>
Source/Filter/Filter.H:    // public for cuda
Source/Filter/Filter.H:    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_0, m_stencil_1, m_stencil_2;
Source/Filter/Filter.cpp:#ifdef AMREX_USE_GPU
Source/Filter/Filter.cpp:/* \brief Apply stencil on MultiFab (GPU version, 2D/3D).
Source/Filter/Filter.cpp:            amrex::Gpu::synchronize();
Source/Filter/Filter.cpp:            amrex::Gpu::synchronize();
Source/Filter/Filter.cpp:/* \brief Apply stencil on FArrayBox (GPU version, 2D/3D).
Source/Filter/Filter.cpp:/* \brief Apply stencil (CPU/GPU)
Source/Filter/Filter.cpp:// never runs on GPU since in the else branch of AMREX_USE_GPU
Source/Filter/Filter.cpp:                amrex::Gpu::synchronize();
Source/Filter/Filter.cpp:                amrex::Gpu::synchronize();
Source/Filter/Filter.cpp:#endif // #ifdef AMREX_USE_CUDA
Source/Filter/BilinearFilter.cpp:#include <AMReX_GpuContainers.H>
Source/Filter/BilinearFilter.cpp:#include <AMReX_GpuDevice.H>
Source/Filter/BilinearFilter.cpp:    void compute_stencil(Gpu::DeviceVector<Real> &stencil, unsigned int npass)
Source/Filter/BilinearFilter.cpp:        Gpu::copyAsync(Gpu::hostToDevice,old_s.begin(),old_s.end(),stencil.begin());
Source/Filter/BilinearFilter.cpp:        amrex::Gpu::synchronize();
Source/Laser/LaserProfilesImpl/LaserProfileFieldFunction.cpp:#include <AMReX_GpuLaunch.H>
Source/Laser/LaserProfilesImpl/LaserProfileFieldFunction.cpp:#include <AMReX_GpuQualifiers.H>
Source/Laser/LaserProfilesImpl/LaserProfileFieldFunction.cpp:    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/Laser/LaserProfilesImpl/LaserProfileGaussian.cpp:#include <AMReX_GpuComplex.H>
Source/Laser/LaserProfilesImpl/LaserProfileGaussian.cpp:#include <AMReX_GpuLaunch.H>
Source/Laser/LaserProfilesImpl/LaserProfileGaussian.cpp:#include <AMReX_GpuQualifiers.H>
Source/Laser/LaserProfilesImpl/LaserProfileGaussian.cpp:    // Copy member variables to tmp copies for GPU runs.
Source/Laser/LaserProfilesImpl/LaserProfileGaussian.cpp:        [=] AMREX_GPU_DEVICE (int i) {
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:#include <AMReX_GpuContainers.H>
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:#include <AMReX_GpuDevice.H>
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:#include <AMReX_GpuLaunch.H>
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:#include <AMReX_GpuQualifiers.H>
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:            [=] AMREX_GPU_DEVICE (int i) {
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    Gpu::copyAsync(Gpu::hostToDevice,h_E_lasy_data.begin(),h_E_lasy_data.end(),m_params.E_lasy_data.begin());
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    Gpu::synchronize();
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    Gpu::copyAsync(Gpu::hostToDevice,h_E_binary_data.begin(),h_E_binary_data.end(),m_params.E_binary_data.begin());
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    Gpu::synchronize();
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    // and get pointers to underlying data for GPU.
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    [=] AMREX_GPU_DEVICE (int i) {
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    // and get pointers to underlying data for GPU.
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    [=] AMREX_GPU_DEVICE (int i) {
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    // and get pointers to underlying data for GPU.
Source/Laser/LaserProfilesImpl/LaserProfileFromFile.cpp:    [=] AMREX_GPU_DEVICE (int i) {
Source/Laser/LaserProfiles.H:#include <AMReX_Gpu.H>
Source/Laser/LaserProfiles.H:        amrex::Gpu::DeviceVector<Complex> E_lasy_data;
Source/Laser/LaserProfiles.H:        amrex::Gpu::DeviceVector<amrex::Real> E_binary_data;
Source/Make.WarpX:USE_GPU_RDC   = FALSE
Source/Make.WarpX:ifeq ($(USE_GPU),TRUE)
Source/Make.WarpX:  USE_CUDA = TRUE
Source/Make.WarpX:  USE_GPU_RDC = TRUE
Source/Make.WarpX:  USE_GPU_RDC = TRUE
Source/Make.WarpX:  ifeq ($(USE_CUDA),TRUE)
Source/Make.WarpX:ifeq ($(USE_CUDA),TRUE)
Source/Python/Particles/MultiParticleContainer.cpp:#include <AMReX_GpuContainers.H>
Source/Python/Particles/MultiParticleContainer.cpp:                 amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Python/Particles/MultiParticleContainer.cpp:                 amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Python/Particles/MultiParticleContainer.cpp:                 amrex::Gpu::synchronize();
Source/Python/WarpX.cpp:            "have_gpu",
Source/Python/WarpX.cpp:#ifdef AMREX_USE_GPU
Source/Python/WarpX.cpp:            "gpu_backend",
Source/Python/WarpX.cpp:#ifdef AMREX_USE_CUDA
Source/Python/WarpX.cpp:                return "CUDA";
Source/WarpX.cpp:#include <AMReX_GpuControl.H>
Source/WarpX.cpp:#include <AMReX_GpuDevice.H>
Source/WarpX.cpp:#include <AMReX_GpuLaunch.H>
Source/WarpX.cpp:#include <AMReX_GpuQualifiers.H>
Source/WarpX.cpp:#if defined(AMREX_USE_CUDA)
Source/WarpX.cpp:    // Default values listed here for the case AMREX_USE_GPU are determined
Source/WarpX.cpp:    // from single-GPU tests on Summit.
Source/WarpX.cpp:#ifdef AMREX_USE_GPU
Source/WarpX.cpp:#endif // AMREX_USE_GPU
Source/WarpX.cpp:                const unsigned long gpu_seed = myproc_1 * dist(rd);
Source/WarpX.cpp:                ResetRandomSeed(cpu_seed, gpu_seed);
Source/WarpX.cpp:                const unsigned long gpu_seed = (myproc_1 + nprocs) * seed_long;
Source/WarpX.cpp:                ResetRandomSeed(cpu_seed, gpu_seed);
Source/WarpX.cpp:#if !(defined(AMREX_USE_HIP) || defined(AMREX_USE_CUDA))
Source/WarpX.cpp:                "requested shared memory for current deposition, but shared memory is only available for CUDA or HIP");
Source/WarpX.cpp:#ifdef AMREX_USE_GPU
Source/WarpX.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/WarpX.cpp:    for (MFIter mfi(divB, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/WarpX.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/WarpX.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/WarpX.cpp:    amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/WarpX.cpp:void WarpX::AllocateCenteringCoefficients (amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_x,
Source/WarpX.cpp:                                           amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_y,
Source/WarpX.cpp:                                           amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_z,
Source/WarpX.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/WarpX.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/WarpX.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/WarpX.cpp:    amrex::Gpu::synchronize();
Source/NonlinearSolvers/NonlinearSolver.H:#include <AMReX_GpuContainers.H>
Source/Fluids/MusclHancockUtils.H:#include <AMReX_Gpu.H>
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/MusclHancockUtils.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Fluids/WarpXFluidContainer.cpp:#ifdef AMREX_USE_GPU
Source/Fluids/WarpXFluidContainer.cpp:        amrex::Gpu::htod_memcpy_async(d_inj_rho, h_inj_rho.get(), sizeof(InjectorDensity));
Source/Fluids/WarpXFluidContainer.cpp:#ifdef AMREX_USE_GPU
Source/Fluids/WarpXFluidContainer.cpp:        amrex::Gpu::htod_memcpy_async(d_inj_mom, h_inj_mom.get(), sizeof(InjectorMomentum));
Source/Fluids/WarpXFluidContainer.cpp:    amrex::Gpu::synchronize();
Source/Fluids/WarpXFluidContainer.cpp:    // Create local copies of pointers for GPU kernels
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:    auto Nodal_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto Ex_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto Ey_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto Ez_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto Bx_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto By_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto Bz_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:        const amrex::GpuArray<int, 3U> coarsening_ratio = {1, 1, 1};
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k,
Source/Fluids/WarpXFluidContainer.cpp:#ifdef AMREX_USE_CUDA
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:    auto j_nodal_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto jx_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto jy_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:    auto jz_type = amrex::GpuArray<int, 3>{0, 0, 0};
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Fluids/WarpXFluidContainer.cpp:    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Fluids/WarpXFluidContainer.cpp:        const amrex::GpuArray<int, 3U> coarsening_ratio = {1, 1, 1};
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Fluids/WarpXFluidContainer.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/Initialization/WarpXInitData.cpp:#include <AMReX_GpuLaunch.H>
Source/Initialization/WarpXInitData.cpp:#include <AMReX_GpuQualifiers.H>
Source/Initialization/WarpXInitData.cpp:    #ifdef AMREX_USE_GPU
Source/Initialization/WarpXInitData.cpp:                << "  On GPUs, consider using 1-8 boxes per GPU that together fill "
Source/Initialization/WarpXInitData.cpp:                << "each GPU's memory sufficiently. If you do not rely on dynamic "
Source/Initialization/WarpXInitData.cpp:                << "load-balancing, then one large box per GPU is ideal.\n"
Source/Initialization/WarpXInitData.cpp:    #ifdef AMREX_USE_GPU
Source/Initialization/WarpXInitData.cpp:        // Check: Are there more than 12 boxes per GPU?
Source/Initialization/WarpXInitData.cpp:            warnMsg << "Too many boxes per GPU!\n"
Source/Initialization/WarpXInitData.cpp:                << amrex::Long(total_nboxes/nprocs) << ") per GPU. "
Source/Initialization/WarpXInitData.cpp:                << "  On GPUs, consider using 1-8 boxes per GPU that together fill "
Source/Initialization/WarpXInitData.cpp:                << "each GPU's memory sufficiently. If you do not rely on dynamic "
Source/Initialization/WarpXInitData.cpp:                << "load-balancing, then one large box per GPU is ideal.\n"
Source/Initialization/WarpXInitData.cpp:        // TODO: check MPI-rank to GPU ratio (should be 1:1)
Source/Initialization/WarpXInitData.cpp:        // TODO: check memory per MPI rank, especially if GPUs are underutilized
Source/Initialization/WarpXInitData.cpp:    for ( MFIter mfi(*mfx, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/Initialization/WarpXInitData.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Initialization/WarpXInitData.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Initialization/WarpXInitData.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Initialization/WarpXInitData.cpp:#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ == 6)
Source/Initialization/WarpXInitData.cpp:            "CUDA 11.6 does not work with the Yee Maxwell "
Source/Initialization/WarpXInitData.cpp:    // Load data to GPU
Source/Initialization/WarpXInitData.cpp:    amrex::Gpu::DeviceVector<double> FC_data_gpu(total_extent);
Source/Initialization/WarpXInitData.cpp:    auto *FC_data = FC_data_gpu.data();
Source/Initialization/WarpXInitData.cpp:    amrex::Gpu::copy(amrex::Gpu::hostToDevice, FC_data_host, FC_data_host + total_extent, FC_data);
Source/Initialization/WarpXInitData.cpp:    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Initialization/WarpXInitData.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/Initialization/SampleGaussianFluxDistribution.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorFlux.H:#include <AMReX_GpuQualifiers.H>
Source/Initialization/InjectorFlux.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorFlux.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorFlux.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/GetVelocity.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/GetVelocity.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:#include <AMReX_GpuQualifiers.H>
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorMomentum.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorDensity.H:#include <AMReX_GpuQualifiers.H>
Source/Initialization/InjectorDensity.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorDensity.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorDensity.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorDensity.H:    amrex::GpuArray<amrex::Real,6> p;
Source/Initialization/InjectorDensity.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/ExternalField.H:    amrex::GpuArray<amrex::Real,3> E_external_grid = {0,0,0};
Source/Initialization/ExternalField.H:    amrex::GpuArray<amrex::Real,3> B_external_grid = {0,0,0};
Source/Initialization/PlasmaInjector.cpp:#include <AMReX_GpuDevice.H>
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:        amrex::Gpu::htod_memcpy_async(d_inj_rho, h_inj_rho.get(), sizeof(InjectorDensity));
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:        amrex::Gpu::htod_memcpy_async(d_inj_mom, h_inj_mom.get(), sizeof(InjectorMomentum));
Source/Initialization/PlasmaInjector.cpp:    amrex::Gpu::synchronize();
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:    amrex::Gpu::htod_memcpy_async(d_inj_pos, h_inj_pos.get(), sizeof(InjectorPosition));
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:    amrex::Gpu::htod_memcpy_async(d_flux_pos, h_flux_pos.get(), sizeof(InjectorPosition));
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:    amrex::Gpu::htod_memcpy_async(d_inj_pos, h_inj_pos.get(), sizeof(InjectorPosition));
Source/Initialization/PlasmaInjector.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/PlasmaInjector.cpp:        amrex::Gpu::htod_memcpy_async(d_inj_flux, h_inj_flux.get(), sizeof(InjectorFlux));
Source/Initialization/InjectorPosition.H:#include <AMReX_Gpu.H>
Source/Initialization/InjectorPosition.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorPosition.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorPosition.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorPosition.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorPosition.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorPosition.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/InjectorPosition.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Initialization/WarpXAMReXInit.cpp:#ifdef AMREX_USE_GPU
Source/Initialization/WarpXAMReXInit.cpp:        constexpr auto amrex_use_gpu = true;
Source/Initialization/WarpXAMReXInit.cpp:        constexpr auto amrex_use_gpu = false;
Source/Initialization/WarpXAMReXInit.cpp:    void override_default_abort_on_out_of_gpu_memory ()
Source/Initialization/WarpXAMReXInit.cpp:        // https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters
Source/Initialization/WarpXAMReXInit.cpp:        bool abort_on_out_of_gpu_memory = true; // AMReX's default: false
Source/Initialization/WarpXAMReXInit.cpp:        pp_amrex.queryAdd("abort_on_out_of_gpu_memory", abort_on_out_of_gpu_memory);
Source/Initialization/WarpXAMReXInit.cpp:        auto warpx_do_device_synchronize = amrex_use_gpu;
Source/Initialization/WarpXAMReXInit.cpp:        // "false" in AMReX, to "false" if compiling for GPU execution and "true"
Source/Initialization/WarpXAMReXInit.cpp:        auto do_tiling = !amrex_use_gpu; // By default, tiling is off on GPU
Source/Initialization/WarpXAMReXInit.cpp:        override_default_abort_on_out_of_gpu_memory();
Source/Initialization/GetTemperature.H:    AMREX_GPU_HOST_DEVICE
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:    amrex::Gpu::synchronize();
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:        for (MFIter mfi(*m_solution[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/Initialization/DivCleaner/ProjectionDivCleaner.cpp:        amrex::Gpu::synchronize();
Source/Initialization/DivCleaner/ProjectionDivCleaner.H:#include <AMReX_GpuControl.H>
Source/Initialization/DivCleaner/ProjectionDivCleaner.H:#include <AMReX_GpuLaunch.H>
Source/Initialization/DivCleaner/ProjectionDivCleaner.H:#include <AMReX_GpuQualifiers.H>
Source/Initialization/DivCleaner/ProjectionDivCleaner.H:    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_x;
Source/Initialization/DivCleaner/ProjectionDivCleaner.H:    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_y;
Source/Initialization/DivCleaner/ProjectionDivCleaner.H:    amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_z;
Source/FieldSolver/WarpXPushFieldsEM.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/WarpXPushFieldsEM.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/WarpXPushFieldsEM.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/WarpXPushFieldsEM.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/FieldSolver/WarpXPushFieldsEM.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/FieldSolver/WarpXPushFieldsEM.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/FieldSolver/WarpXPushFieldsEM.cpp:            for ( amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/WarpXPushFieldsEM.cpp:                    tex_guard, Efield[0]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/FieldSolver/WarpXPushFieldsEM.cpp:                    tey_guard, Efield[1]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/FieldSolver/WarpXPushFieldsEM.cpp:                    tez_guard, Efield[2]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/FieldSolver/WarpXPushFieldsEM.cpp:                    tbx_guard, Bfield[0]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/FieldSolver/WarpXPushFieldsEM.cpp:                    tby_guard, Bfield[1]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/FieldSolver/WarpXPushFieldsEM.cpp:                    tbz_guard, Bfield[2]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/FieldSolver/WarpXPushFieldsEM.cpp:            for (amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/FieldSolver/WarpXPushFieldsEM.cpp:                    tx_guard, mf->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/FieldSolver/WarpXPushFieldsEM.cpp:    for ( MFIter mfi(*Jx, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/WarpXPushFieldsEM.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/FieldSolver/WarpXPushFieldsEM.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/FieldSolver/WarpXPushFieldsEM.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/FieldSolver/WarpXPushFieldsEM.cpp:    for ( MFIter mfi(*Rho, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/WarpXPushFieldsEM.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
Source/FieldSolver/ImplicitSolvers/WarpXImplicitOps.cpp:                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
Source/FieldSolver/ImplicitSolvers/WarpXImplicitOps.cpp:                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
Source/FieldSolver/ImplicitSolvers/WarpXImplicitOps.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/ImplicitSolvers/WarpXImplicitOps.cpp:       for ( amrex::MFIter mfi(*Field_fp[lev][0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/ImplicitSolvers/WarpXImplicitOps.cpp:            tbx, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/FieldSolver/ImplicitSolvers/WarpXImplicitOps.cpp:            tby, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/FieldSolver/ImplicitSolvers/WarpXImplicitOps.cpp:            tbz, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:#include <AMReX_GpuAtomic.H>
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:#include <AMReX_GpuDevice.H>
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:#include <AMReX_GpuElixir.H>
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:    for ( MFIter mfi(*Bx, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:        // Temporary arrays for electric field, protected by Elixir on GPU
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:            [=] AMREX_GPU_DEVICE (int j, int k, int l)
Source/FieldSolver/WarpX_QED_Field_Pushers.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/WarpX_QED_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/FieldSolver/WarpX_QED_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/FieldSolver/WarpXPushFieldsEM_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/FieldSolver/WarpXPushFieldsEM_K.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.cpp:                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx_lev,
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.cpp:    for ( amrex::MFIter mfi(*macro_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:                                  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx_lev,
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    /** Gpu Vector with index type of the conductivity multifab */
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    amrex::GpuArray<int, 3> sigma_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    /** Gpu Vector with index type of the permittivity multifab */
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    amrex::GpuArray<int, 3> epsilon_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    /** Gpu Vector with index type of the permeability multifab */
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    amrex::GpuArray<int, 3> mu_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    /** Gpu Vector with index type of the Ex multifab */
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    amrex::GpuArray<int, 3> Ex_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    /** Gpu Vector with index type of the Ey multifab */
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    amrex::GpuArray<int, 3> Ey_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    /** Gpu Vector with index type of the Ez multifab */
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    amrex::GpuArray<int, 3> Ez_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    /** Gpu Vector with index type of coarsening ratio with default value (1,1,1) */
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    amrex::GpuArray<int, 3> macro_cr_ratio;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#include <AMReX_GpuAtomic.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#include <AMReX_GpuDevice.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveB.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveG.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveG.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveG.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveG.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveG.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveG.cpp:    for (amrex::MFIter mfi(*Gfield, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/FieldSolver/FiniteDifferenceSolver/EvolveG.cpp:        amrex::ParallelFor(tf, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:    for ( MFIter mfi(divEfield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:    for ( MFIter mfi(divEfield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/ComputeDivE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveFPML.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveFPML.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveFPML.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveFPML.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveFPML.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveFPML.cpp:    for ( MFIter mfi(*Ffield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveFPML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H:        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_r;
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H:        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_z;
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H:        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_x;
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H:        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_y;
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H:        amrex::Gpu::DeviceVector<amrex::Real> m_stencil_coefs_z;
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:    for ( MFIter mfi(*Bfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveBPML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Jx multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Jx_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Jy multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Jy_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Jz multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Jz_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Bx multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Bx_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the By multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> By_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Bz multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Bz_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Ex multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Ex_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Ey multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Ey_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    /** Gpu Vector with index type of the Ez multifab */
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    amrex::GpuArray<int, 3> Ez_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.cpp:    for ( MFIter mfi(Pe_field, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.cpp:        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:#include <AMReX_GpuDevice.H>
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:        amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.cpp:    amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:    amrex::GpuArray<int, 3> const& sigma_stag = macroscopic_properties->sigma_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:    amrex::GpuArray<int, 3> const& epsilon_stag = macroscopic_properties->epsilon_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:    amrex::GpuArray<int, 3> const& macro_cr     = macroscopic_properties->macro_cr_ratio;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:    amrex::GpuArray<int, 3> const& Ex_stag = macroscopic_properties->Ex_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:    amrex::GpuArray<int, 3> const& Ey_stag = macroscopic_properties->Ey_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:    amrex::GpuArray<int, 3> const& Ez_stag = macroscopic_properties->Ez_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/MacroscopicEvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:#include <AMReX_GpuAtomic.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:#include <AMReX_GpuDevice.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:    for (amrex::MFIter mfi(*ECTRhofield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveECTRho.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:    // tiling is usually set by TilingIfNotGPU()
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:    // tiling is usually set by TilingIfNotGPU()
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/ApplySilverMuellerBoundary.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    for ( MFIter mfi(*Jfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Er_stag = hybrid_model->Ex_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Et_stag = hybrid_model->Ey_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Jr_stag = hybrid_model->Jx_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Jt_stag = hybrid_model->Jy_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Br_stag = hybrid_model->Bx_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Bt_stag = hybrid_model->By_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Ex_stag = hybrid_model->Ex_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Ey_stag = hybrid_model->Ey_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Ez_stag = hybrid_model->Ez_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Jx_stag = hybrid_model->Jx_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Jy_stag = hybrid_model->Jy_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Jz_stag = hybrid_model->Jz_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Bx_stag = hybrid_model->Bx_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& By_stag = hybrid_model->By_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& Bz_stag = hybrid_model->Bz_IndexType;
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& nodal = {1, 1, 1};
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    amrex::GpuArray<int, 3> const& coarsen = {1, 1, 1};
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    for ( MFIter mfi(enE_nodal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:    for ( MFIter mfi(*Ffield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:    for ( MFIter mfi(*Ffield, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveF.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:#include <AMReX_Gpu.H>
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:   AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/FieldAccessorFunctors.H:#include <AMReX_Gpu.H>
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/FieldAccessorFunctors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/FieldAccessorFunctors.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H:#include <AMReX_Gpu.H>
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianCKCAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:#include <AMReX_Gpu.H>
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H:#include <AMReX_Gpu.H>
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianNodalAlgorithm.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#include <AMReX_GpuAtomic.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#include <AMReX_GpuContainers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#include <AMReX_GpuDevice.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:    for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/){
Source/FieldSolver/FiniteDifferenceSolver/EvolveE.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:        for ( MFIter mfi(*phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:#    pragma omp parallel if (Gpu::notInLaunchRegion())
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:        for ( MFIter mfi(*phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:#    pragma omp parallel if (Gpu::notInLaunchRegion())
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:        for ( MFIter mfi(*phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.cpp:                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/ElectrostaticSolvers/PoissonBoundaryHandler.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/ElectrostaticSolvers/PoissonBoundaryHandler.H:        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.cpp:    // the data readily accessible from the GPU.
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:#include <AMReX_GpuAtomic.H>
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:#include <AMReX_GpuDevice.H>
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            ParallelFor(mf_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralFieldData.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:#ifndef AMREX_USE_CUDA
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // The backward plan is not needed with CUDA since it would be the same
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:#if defined(AMREX_USE_CUDA)
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:#if defined(AMREX_USE_CUDA)
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:#if defined(AMREX_USE_CUDA)
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // Perform Fast Fourier Transform on GPU using cuFFT.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // GPU stream as the above copy.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    cudaStream_t stream = amrex::Gpu::Device::cudaStream();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:#if defined(AMREX_USE_CUDA)
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // Perform Fast Fourier Transform on GPU using cuFFT.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // GPU stream as the above copy.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    cudaStream_t stream = amrex::Gpu::Device::cudaStream();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    result = rocfft_execution_info_set_stream(execinfo, amrex::Gpu::gpuStream());
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // A full multifab is created so that each GPU stream has its own temp space.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // Full multifabs are created for the temps so that each GPU stream has its own temp space.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // A full multifab is created so that each GPU stream has its own temp space.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int icomp) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:    // Full multifabs are created for the temps so that each GPU stream has its own temp space.
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int icomp) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept {
Source/FieldSolver/SpectralSolver/SpectralFieldDataRZ.cpp:            amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralKSpace.H:#include <AMReX_GpuContainers.H>
Source/FieldSolver/SpectralSolver/SpectralKSpace.H:using RealKVector = amrex::Gpu::DeviceVector<amrex::Real>;
Source/FieldSolver/SpectralSolver/SpectralKSpace.H:                           amrex::Gpu::DeviceVector<Complex> >;
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmPml.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmPml.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmPml.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmPml.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmPml.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmFirstOrder.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmFirstOrder.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmFirstOrder.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmFirstOrder.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJLinearInTime.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJLinearInTime.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJLinearInTime.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJLinearInTime.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJLinearInTime.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJLinearInTime.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmComoving.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmComoving.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmComoving.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmComoving.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmComoving.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmComoving.cpp:        // Local copy of member variables before GPU loop
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmComoving.cpp:        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmRZ.cpp:        // Local copy of member variables before GPU loop
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:        // Local copy of member variables before GPU loop
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmJConstantInTime.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmPmlRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmPmlRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmGalileanRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmGalileanRZ.cpp:        // Extract real (for portability on GPU)
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmGalileanRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmGalileanRZ.cpp:        // Local copy of member variables before GPU loop
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/PsatdAlgorithmGalileanRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/SpectralBaseAlgorithm.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/SpectralBaseAlgorithm.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/SpectralBaseAlgorithm.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/SpectralBaseAlgorithm.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
Source/FieldSolver/SpectralSolver/SpectralAlgorithms/SpectralBaseAlgorithmRZ.cpp:        [=] AMREX_GPU_DEVICE(int i, int j, int /*k*/, int mode) noexcept
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:        [=] AMREX_GPU_DEVICE (int ir)
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:    amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:        amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:    amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:        amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:        amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/SpectralHankelTransformer.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.H:#include <AMReX_GpuContainers.H>
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.H:#ifdef AMREX_USE_GPU
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.H:        using RealVector = amrex::Gpu::DeviceVector<amrex::Real>;
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.H:#ifdef AMREX_USE_GPU
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:#ifdef AMREX_USE_GPU
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    int const device_id = amrex::Gpu::Device::deviceId();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    blas::Queue::stream_t stream_id = amrex::Gpu::gpuStream();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, kr.begin(), kr.end(), m_kr.begin());
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, invM.begin(), invM.end(), m_invM.begin());
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, M.begin(), M.end(), m_M.begin());
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:#ifdef AMREX_USE_GPU
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:               , *m_queue // Calls the GPU version of blas::gemm
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:#ifdef AMREX_USE_GPU
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:               , *m_queue // Calls the GPU version of blas::gemm
Source/FieldSolver/SpectralSolver/SpectralHankelTransform/HankelTransform.cpp:    amrex::Gpu::streamSynchronize();
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:#include <AMReX_GpuComplex.H>
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:#include <AMReX_GpuDevice.H>
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:        Gpu::DeviceVector<Real>& k = k_comp[mfi];
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:        const Gpu::DeviceVector<Real>& k = k_vec[i_dim][mfi];
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:        Gpu::DeviceVector<Complex>& shift = shift_factor[mfi];
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            const Gpu::DeviceVector<Real>& k = k_vec[i_dim][mfi];
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            Gpu::DeviceVector<Real>& modified_k = modified_k_comp[mfi];
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            Gpu::copyAsync(Gpu::deviceToDevice, k.begin(), k.end(), modified_k.begin());
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:        Gpu::DeviceVector<Real> d_stencil_coef(h_stencil_coef.size());
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:        Gpu::copyAsync(Gpu::hostToDevice, h_stencil_coef.begin(), h_stencil_coef.end(),
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:        Gpu::synchronize();
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            const Gpu::DeviceVector<Real>& k = k_vec[i_dim][mfi];
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            Gpu::DeviceVector<Real>& modified_k = modified_k_comp[mfi];
Source/FieldSolver/SpectralSolver/SpectralKSpace.cpp:            amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/FieldSolver/SpectralSolver/SpectralBinomialFilter.H:        using KFilterArray = amrex::Gpu::DeviceVector<amrex::Real>;
Source/FieldSolver/SpectralSolver/SpectralBinomialFilter.cpp:    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/FieldSolver/WarpX_FDTD.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:#include <AMReX_GpuControl.H>
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:#include <AMReX_GpuLaunch.H>
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:#include <AMReX_GpuQualifiers.H>
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:            for ( MFIter mfi(*A[lev][Direction{adim}], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:                            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:    for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/FieldSolver/MagnetostaticSolver/MagnetostaticSolver.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Utils/WarpXUtil.H:#include <AMReX_GpuQualifiers.H>
Source/Utils/WarpXUtil.H: * \param[in] mf_type GpuArray containing the staggering type to convert (i,j,k) to (x,y,z)
Source/Utils/WarpXUtil.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/Utils/WarpXUtil.H:                         amrex::GpuArray<int, 3> const mf_type,
Source/Utils/WarpXUtil.H:                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const domain_lo,
Source/Utils/WarpXUtil.H:                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
Source/Utils/WarpXMovingWindow.cpp:#include <AMReX_GpuControl.H>
Source/Utils/WarpXMovingWindow.cpp:#include <AMReX_GpuLaunch.H>
Source/Utils/WarpXMovingWindow.cpp:#include <AMReX_GpuQualifiers.H>
Source/Utils/WarpXMovingWindow.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/WarpXMovingWindow.cpp:    for (amrex::MFIter mfi(tmpmf, TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/Utils/WarpXMovingWindow.cpp:            amrex::Gpu::synchronize();
Source/Utils/WarpXMovingWindow.cpp:                      [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
Source/Utils/WarpXMovingWindow.cpp:            amrex::Gpu::synchronize();
Source/Utils/WarpXTagging.cpp:#include <AMReX_GpuControl.H>
Source/Utils/WarpXTagging.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Utils/WarpXTagging.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/Utils/Interpolate.cpp:#include <AMReX_GpuControl.H>
Source/Utils/Interpolate.cpp:#include <AMReX_GpuLaunch.H>
Source/Utils/Interpolate.cpp:#include <AMReX_GpuQualifiers.H>
Source/Utils/Interpolate.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Utils/Interpolate.cpp:                amrex::Gpu::streamSynchronize();
Source/Utils/Interpolate.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/Utils/Interpolate.cpp:        for (MFIter mfi(*interpolated_F[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
Source/Utils/Interpolate.cpp:                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Utils/Interpolate.cpp:                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Utils/Interpolate.cpp:                [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
Source/Utils/Algorithms/LinearInterpolation.H:#include <AMReX_GpuQualifiers.H>
Source/Utils/Algorithms/LinearInterpolation.H:    template<typename TCoord, typename TVal> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Utils/Algorithms/LinearInterpolation.H:    template<typename TCoord, typename TVal> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Utils/Algorithms/LinearInterpolation.H:    template<typename TCoord, typename TVal> AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Utils/ParticleUtils.cpp:#include <AMReX_GpuLaunch.H>
Source/Utils/ParticleUtils.cpp:#include <AMReX_GpuQualifiers.H>
Source/Utils/ParticleUtils.cpp:            [=] AMREX_GPU_DEVICE (ParticleType const & p) noexcept -> amrex::IntVect
Source/Utils/ParticleUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Utils/ParticleUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Utils/ParticleUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Utils/ParticleUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Utils/ParticleUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Utils/ParticleUtils.H:    AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/Utils/WarpX_Complex.H:#include <AMReX_Gpu.H>
Source/Utils/WarpX_Complex.H:#include <AMReX_GpuComplex.H>
Source/Utils/WarpX_Complex.H:// Defines a complex type on GPU & CPU
Source/Utils/WarpX_Complex.H:using Complex = amrex::GpuComplex<amrex::Real>;
Source/Utils/Interpolate_K.H:AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/Utils/WarpXUtil.cpp:#include <AMReX_GpuControl.H>
Source/Utils/WarpXUtil.cpp:#include <AMReX_GpuLaunch.H>
Source/Utils/WarpXUtil.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/Utils/WarpXUtil.cpp:    for(amrex::MFIter mfi(*mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
Source/Utils/WarpXUtil.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept{
Source/BoundaryConditions/WarpX_PEC.cpp:#include <AMReX_GpuControl.H>
Source/BoundaryConditions/WarpX_PEC.cpp:#include <AMReX_GpuLaunch.H>
Source/BoundaryConditions/WarpX_PEC.cpp:#include <AMReX_GpuQualifiers.H>
Source/BoundaryConditions/WarpX_PEC.cpp:    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PEC.cpp:    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PEC.cpp:                                amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_lo,
Source/BoundaryConditions/WarpX_PEC.cpp:                                amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_hi )
Source/BoundaryConditions/WarpX_PEC.cpp:    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PEC.cpp:                           amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_lo,
Source/BoundaryConditions/WarpX_PEC.cpp:                           amrex::GpuArray<FieldBoundaryType, 3> const& fbndry_hi )
Source/BoundaryConditions/WarpX_PEC.cpp:    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PEC.cpp:                                amrex::GpuArray<GpuArray<int, 2>, AMREX_SPACEDIM> const& mirrorfac,
Source/BoundaryConditions/WarpX_PEC.cpp:                                amrex::GpuArray<GpuArray<amrex::Real, 2>, AMREX_SPACEDIM> const& psign,
Source/BoundaryConditions/WarpX_PEC.cpp:                                amrex::GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> const& is_reflective,
Source/BoundaryConditions/WarpX_PEC.cpp:                                amrex::GpuArray<bool, AMREX_SPACEDIM> const& tangent_to_bndy,
Source/BoundaryConditions/WarpX_PEC.cpp:    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PEC.cpp:                          amrex::GpuArray<GpuArray<int, 2>, AMREX_SPACEDIM> const& mirrorfac,
Source/BoundaryConditions/WarpX_PEC.cpp:                          amrex::GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> const& is_pec,
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<FieldBoundaryType, 3> fbndry_lo;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<FieldBoundaryType, 3> fbndry_hi;
Source/BoundaryConditions/WarpX_PEC.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpX_PEC.cpp:    for (amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/BoundaryConditions/WarpX_PEC.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<FieldBoundaryType, 3> fbndry_lo;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<FieldBoundaryType, 3> fbndry_hi;
Source/BoundaryConditions/WarpX_PEC.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpX_PEC.cpp:    for (amrex::MFIter mfi(*Bfield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/BoundaryConditions/WarpX_PEC.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<bool,2>, AMREX_SPACEDIM> is_reflective;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<bool, AMREX_SPACEDIM> is_tangent_to_bndy;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<amrex::Real,2>, AMREX_SPACEDIM> psign;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<int,2>, AMREX_SPACEDIM> mirrorfac;
Source/BoundaryConditions/WarpX_PEC.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpX_PEC.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<bool, 2>, AMREX_SPACEDIM> is_reflective;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<bool, AMREX_SPACEDIM>, 3> is_tangent_to_bndy;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<GpuArray<amrex::Real, 2>, AMREX_SPACEDIM>, 3> psign;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<GpuArray<int, 2>, AMREX_SPACEDIM>, 3> mirrorfac;
Source/BoundaryConditions/WarpX_PEC.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpX_PEC.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpX_PEC.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpX_PEC.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<bool,2>, AMREX_SPACEDIM> is_pec;
Source/BoundaryConditions/WarpX_PEC.cpp:    amrex::GpuArray<GpuArray<int,2>, AMREX_SPACEDIM> mirrorfac;
Source/BoundaryConditions/WarpX_PEC.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpX_PEC.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/WarpX_PML_kernels.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PML_kernels.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PML_kernels.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PML_kernels.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PML_kernels.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PML_kernels.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/WarpX_PML_kernels.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/PEC_Insulator.cpp:#include <AMReX_GpuControl.H>
Source/BoundaryConditions/PEC_Insulator.cpp:#include <AMReX_GpuLaunch.H>
Source/BoundaryConditions/PEC_Insulator.cpp:#include <AMReX_GpuQualifiers.H>
Source/BoundaryConditions/PEC_Insulator.cpp:    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Source/BoundaryConditions/PEC_Insulator.cpp:                                  amrex::GpuArray<FieldBoundaryType, 3> const fbndry_lo,
Source/BoundaryConditions/PEC_Insulator.cpp:                                  amrex::GpuArray<FieldBoundaryType, 3> const fbndry_hi)
Source/BoundaryConditions/PEC_Insulator.cpp:    amrex::GpuArray<FieldBoundaryType, 3> fbndry_lo;
Source/BoundaryConditions/PEC_Insulator.cpp:    amrex::GpuArray<FieldBoundaryType, 3> fbndry_hi;
Source/BoundaryConditions/PEC_Insulator.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/PEC_Insulator.cpp:    for (amrex::MFIter mfi(*field[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/BoundaryConditions/PEC_Insulator.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/PEC_Insulator.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/PEC_Insulator.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
Source/BoundaryConditions/PML.H:#include <AMReX_GpuContainers.H>
Source/BoundaryConditions/PML.H:struct Sigma : amrex::Gpu::DeviceVector<amrex::Real>
Source/BoundaryConditions/PML_current.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/BoundaryConditions/PML_current.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/BoundaryConditions/PML_current.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/BoundaryConditions/PML_current.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/BoundaryConditions/PML_current.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/BoundaryConditions/PML_current.H:AMREX_GPU_HOST_DEVICE AMREX_INLINE
Source/BoundaryConditions/WarpXFieldBoundaries.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpXFieldBoundaries.cpp:    for ( amrex::MFIter mfi(*Er, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/BoundaryConditions/WarpXFieldBoundaries.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/BoundaryConditions/WarpXFieldBoundaries.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/BoundaryConditions/WarpXFieldBoundaries.cpp:        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
Source/BoundaryConditions/WarpXEvolvePML.cpp:#include <AMReX_GpuControl.H>
Source/BoundaryConditions/WarpXEvolvePML.cpp:#include <AMReX_GpuLaunch.H>
Source/BoundaryConditions/WarpXEvolvePML.cpp:#include <AMReX_GpuQualifiers.H>
Source/BoundaryConditions/WarpXEvolvePML.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpXEvolvePML.cpp:        for ( MFIter mfi(*pml_E[0], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/BoundaryConditions/WarpXEvolvePML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:                amrex::ParallelFor(tnd, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/WarpXEvolvePML.cpp:                amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Source/BoundaryConditions/WarpXEvolvePML.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/WarpXEvolvePML.cpp:        for ( MFIter mfi(*pml_j[0], TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/BoundaryConditions/WarpXEvolvePML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/WarpXEvolvePML.cpp:                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Source/BoundaryConditions/PML.cpp:#include <AMReX_GpuControl.H>
Source/BoundaryConditions/PML.cpp:#include <AMReX_GpuDevice.H>
Source/BoundaryConditions/PML.cpp:#include <AMReX_GpuLaunch.H>
Source/BoundaryConditions/PML.cpp:#include <AMReX_GpuQualifiers.H>
Source/BoundaryConditions/PML.cpp:        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/BoundaryConditions/PML.cpp:        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/BoundaryConditions/PML.cpp:        amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (int i) noexcept
Source/BoundaryConditions/PML.cpp:    amrex::Gpu::streamSynchronize();
Source/BoundaryConditions/PML.cpp:    amrex::Gpu::streamSynchronize();
Source/BoundaryConditions/PML.cpp:    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_star_fac;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_star_cumsum_fac;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma_star;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma_star_cumsum;
Source/BoundaryConditions/PML.cpp:    GpuArray<int, AMREX_SPACEDIM> N;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real, AMREX_SPACEDIM> dx;
Source/BoundaryConditions/PML.cpp:    [=] AMREX_GPU_DEVICE (int i) noexcept
Source/BoundaryConditions/PML.cpp:    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_fac;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real*,AMREX_SPACEDIM> p_sigma_cumsum_fac;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real const*,AMREX_SPACEDIM> p_sigma_cumsum;
Source/BoundaryConditions/PML.cpp:    GpuArray<int, AMREX_SPACEDIM> N;
Source/BoundaryConditions/PML.cpp:    GpuArray<Real, AMREX_SPACEDIM> dx;
Source/BoundaryConditions/PML.cpp:    [=] AMREX_GPU_DEVICE (int i) noexcept
Source/BoundaryConditions/PML.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/PML.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/PML.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/BoundaryConditions/PML.cpp:                                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
Source/BoundaryConditions/PML_RZ.cpp:#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
Source/BoundaryConditions/PML_RZ.cpp:    for ( amrex::MFIter mfi(*Et_fp, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
Source/BoundaryConditions/PML_RZ.cpp:            [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
Source/WarpX.H:#include <AMReX_GpuContainers.H>
Source/WarpX.H:    // these should be private, but can't due to Cuda limitations
Source/WarpX.H:    amrex::Gpu::DeviceVector<amrex::Real> device_field_centering_stencil_coeffs_x;
Source/WarpX.H:    amrex::Gpu::DeviceVector<amrex::Real> device_field_centering_stencil_coeffs_y;
Source/WarpX.H:    amrex::Gpu::DeviceVector<amrex::Real> device_field_centering_stencil_coeffs_z;
Source/WarpX.H:    amrex::Gpu::DeviceVector<amrex::Real> device_current_centering_stencil_coeffs_x;
Source/WarpX.H:    amrex::Gpu::DeviceVector<amrex::Real> device_current_centering_stencil_coeffs_y;
Source/WarpX.H:    amrex::Gpu::DeviceVector<amrex::Real> device_current_centering_stencil_coeffs_z;
Source/WarpX.H:    // This needs to be public for CUDA.
Source/WarpX.H:    // for cuda
Source/WarpX.H:    void AllocateCenteringCoefficients (amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_x,
Source/WarpX.H:                                        amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_y,
Source/WarpX.H:                                        amrex::Gpu::DeviceVector<amrex::Real>& device_centering_stencil_coeffs_z,
Source/WarpX.H:     * Default values on GPU are determined from single-GPU tests on Summit.
Source/WarpX.H:     * Default values on GPU are determined from single-GPU tests on Summit.
CONTRIBUTING.rst:  This is particularly useful to avoid capturing member variables by value in a lambda function, which causes the whole object to be copied to GPU when running on a GPU-accelerated architecture.
cmake/dependencies/FFT.cmake:    # cuFFT  (CUDA)
cmake/dependencies/FFT.cmake:    if(WarpX_COMPUTE STREQUAL CUDA)
cmake/dependencies/FFT.cmake:        # nothing to do (cuFFT is part of the CUDA SDK)
cmake/dependencies/FFT.cmake:    if(WarpX_COMPUTE STREQUAL CUDA)
cmake/dependencies/FFT.cmake:        # CUDA_ADD_CUFFT_TO_TARGET(WarpX::thirdparty::FFT)
cmake/dependencies/AMReX.cmake:            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
cmake/dependencies/AMReX.cmake:            set(AMReX_GPU_BACKEND  "NONE" CACHE INTERNAL "")
cmake/dependencies/AMReX.cmake:            set(AMReX_GPU_BACKEND  "${WarpX_COMPUTE}" CACHE INTERNAL "")
cmake/dependencies/AMReX.cmake:            set(AMReX_GPU_RDC ON CACHE BOOL "")
cmake/dependencies/AMReX.cmake:            set(AMReX_GPU_RDC OFF CACHE BOOL "")
cmake/dependencies/AMReX.cmake:            if(WarpX_COMPUTE STREQUAL CUDA)
cmake/dependencies/AMReX.cmake:                set(AMReX_CUDA_LTO ON CACHE BOOL "")
cmake/dependencies/AMReX.cmake:            if(WarpX_COMPUTE STREQUAL CUDA)
cmake/dependencies/AMReX.cmake:                enable_language(CUDA)
cmake/dependencies/AMReX.cmake:                # AMReX 21.06+ supports CUDA_ARCHITECTURES
cmake/dependencies/AMReX.cmake:            if(WarpX_COMPUTE STREQUAL CUDA)
cmake/dependencies/AMReX.cmake:                enable_language(CUDA)
cmake/dependencies/AMReX.cmake:                # AMReX 21.06+ supports CUDA_ARCHITECTURES
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_COMPILATION_TIMER)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_ERROR_CAPTURE_THIS)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_ERROR_CROSS_EXECUTION_SPACE_CALL)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_FASTMATH)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_KEEP_FILES)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_LTO)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_MAXREGCOUNT)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_MAX_THREADS)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_PTX_VERBOSE)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_SHOW_CODELINES)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_SHOW_LINENUMBERS)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_CUDA_WARN_CAPTURE_THIS)
cmake/dependencies/AMReX.cmake:        mark_as_advanced(AMReX_GPU_RDC)
cmake/dependencies/AMReX.cmake:        if(WarpX_COMPUTE STREQUAL CUDA)
cmake/dependencies/AMReX.cmake:            enable_language(CUDA)
cmake/dependencies/openPMD.cmake:        mark_as_advanced(openPMD_USE_CUDA_EXAMPLES)
cmake/WarpXFunctions.cmake:    if(NOT DEFINED CMAKE_CUDA_STANDARD)
cmake/WarpXFunctions.cmake:        set(CMAKE_CUDA_STANDARD 17)
cmake/WarpXFunctions.cmake:    if(NOT DEFINED CMAKE_CUDA_EXTENSIONS)
cmake/WarpXFunctions.cmake:        set(CMAKE_CUDA_EXTENSIONS OFF)
cmake/WarpXFunctions.cmake:    if(NOT DEFINED CMAKE_CUDA_STANDARD_REQUIRED)
cmake/WarpXFunctions.cmake:        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
cmake/WarpXFunctions.cmake:        if(WarpX_COMPUTE STREQUAL CUDA)
cmake/WarpXFunctions.cmake:            set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
Tools/Release/updateAMReX.py:# CI: legacy build check in .github/workflows/cuda.yml
Tools/Release/updateAMReX.py:ci_gnumake_path = str(REPO_DIR.joinpath(".github/workflows/cuda.yml"))
Tools/machines/lassen-llnl/lassen_v100.bsub:jsrun -r 4 -a 1 -g 1 -c 7 -l GPU-CPU -d packed -b rs -e prepended -M "-gpu" <path/to/executable> <input file> > output.txt
Tools/machines/lassen-llnl/install_v100_dependencies_toss3.sh:SW_DIR="/usr/workspace/${USER}/lassen-toss3/gpu"
Tools/machines/lassen-llnl/install_v100_dependencies_toss3.sh:cmake -S ${SRC_DIR}/blaspp -B ${build_dir}/blaspp-lassen-build -Duse_openmp=ON -Dgpu_backend=cuda -Duse_cmake_find_blas=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example:module load cuda/12.0.0
Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example:SW_DIR="/usr/workspace/${USER}/lassen-toss3/gpu"
Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example:# optimize CUDA compilation for V100
Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example:export AMREX_CUDA_ARCH=7.0
Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example:export CUDAARCHS=70
Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/lassen-llnl/lassen_v100_warpx_toss3.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/lassen-llnl/install_v100_ml.sh:LDFLAGS="-L${CUDA_HOME}/nvidia/targets/ppc64le-linux/lib/" \
Tools/machines/lassen-llnl/install_v100_ml.sh:USE_CUDA=1 BLAS=OpenBLAS MAX_JOBS=64 ATEN_AVX512_256=OFF BUILD_TEST=0 python3 setup.py develop
Tools/machines/summit-olcf/install_gpu_ml.sh:#   Was perlmutter_gpu_warpx.profile sourced and configured correctly?
Tools/machines/summit-olcf/install_gpu_ml.sh:# for basic python dependencies, see install_gpu_dependencies.sh
Tools/machines/summit-olcf/install_gpu_ml.sh:USE_CUDA=1 BLAS=OpenBLAS MAX_JOBS=64 ATEN_AVX512_256=OFF BUILD_TEST=0 python3 setup.py develop
Tools/machines/summit-olcf/install_gpu_dependencies.sh:#   Was perlmutter_gpu_warpx.profile sourced and configured correctly?
Tools/machines/summit-olcf/install_gpu_dependencies.sh:SW_DIR="/ccs/proj/${proj}/${USER}/sw/summit/gpu/"
Tools/machines/summit-olcf/install_gpu_dependencies.sh:cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-summit-build -Duse_openmp=ON -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/summit-olcf/install_gpu_dependencies.sh:# for ML dependencies, see install_gpu_ml.sh
Tools/machines/summit-olcf/summit_v100.bsub:#   https://docs.olcf.ornl.gov/systems/summit_user_guide.html#cuda-aware-mpi
Tools/machines/summit-olcf/summit_v100.bsub:jsrun -r 6 -a 1 -g 1 -c 7 -l GPU-CPU -d packed -b rs --smpiargs="-gpu" <path/to/executable> <input file> > output.txt
Tools/machines/summit-olcf/summit_profiling.bsub:#module load cuda/9.1.85
Tools/machines/summit-olcf/summit_profiling.bsub:#EXE="../main3d.pgi.DEBUG.TPROF.MPI.ACC.CUDA.ex"
Tools/machines/summit-olcf/summit_profiling.bsub:EXE="../main3d.pgi.TPROF.MPI.ACC.CUDA.ex"
Tools/machines/summit-olcf/summit_profiling.bsub:${JSRUN} --smpiargs="-gpu" ${EXE} inputs
Tools/machines/summit-olcf/summit_profiling.bsub:# 2. Run under cuda-memcheck
Tools/machines/summit-olcf/summit_profiling.bsub:# ${JSRUN} --smpiargs="-gpu" cuda-memcheck ${EXE} inputs &> memcheck.txt
Tools/machines/summit-olcf/summit_profiling.bsub:#${JSRUN} --smpiargs="-gpu" nvprof --profile-child-processes ${EXE} inputs &> nvprof.txt
Tools/machines/summit-olcf/summit_profiling.bsub:#${JSRUN} --smpiargs="-gpu" nvprof --profile-child-processes -o nvprof-timeline-%p.nvvp ${EXE} inputs
Tools/machines/summit-olcf/summit_profiling.bsub:#${JSRUN} --smpiargs="-gpu" nvprof --profile-child-processes --kernels '(deposit_current|gather_\w+_field|push_\w+_boris)' --analysis-metrics -o nvprof-metrics-kernel-%p.nvvp ${EXE} inputs
Tools/machines/summit-olcf/summit_profiling.bsub:#${JSRUN} --smpiargs="-gpu" nvprof --profile-child-processes --analysis-metrics -o nvprof-metrics-%p.nvvp ${EXE} inputs
Tools/machines/summit-olcf/summit_warpx.profile.example:module load cuda/11.7.1
Tools/machines/summit-olcf/summit_warpx.profile.example:export CMAKE_PREFIX_PATH=/ccs/proj/$proj/${USER}/sw/summit/gpu/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/summit-olcf/summit_warpx.profile.example:export CMAKE_PREFIX_PATH=/ccs/proj/$proj/${USER}/sw/summit/gpu/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/summit-olcf/summit_warpx.profile.example:export LD_LIBRARY_PATH=/ccs/proj/$proj/${USER}/sw/summit/gpu/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/summit-olcf/summit_warpx.profile.example:export LD_LIBRARY_PATH=/ccs/proj/$proj/${USER}/sw/summit/gpu/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/summit-olcf/summit_warpx.profile.example:export Ascent_DIR=/sw/summit/ums/ums010/ascent/0.8.0_warpx/summit/cuda/gnu/ascent-install/
Tools/machines/summit-olcf/summit_warpx.profile.example:if [ -d "/ccs/proj/$proj/${USER}/sw/summit/gpu/venvs/warpx-summit" ]
Tools/machines/summit-olcf/summit_warpx.profile.example:  source /ccs/proj/$proj/${USER}/sw/summit/gpu/venvs/warpx-summit/bin/activate
Tools/machines/summit-olcf/summit_warpx.profile.example:# optimize CUDA compilation for V100
Tools/machines/summit-olcf/summit_warpx.profile.example:export AMREX_CUDA_ARCH=7.0
Tools/machines/summit-olcf/summit_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/summit-olcf/summit_warpx.profile.example:export CUDAHOSTCXX=$(which g++)
Tools/machines/frontier-olcf/frontier_warpx.profile.example:module load rocm/5.7.1
Tools/machines/frontier-olcf/frontier_warpx.profile.example:module load cce/17.0.0  # must be loaded after rocm
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/frontier/gpu/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/frontier/gpu/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/frontier/gpu/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/frontier/gpu/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/frontier-olcf/frontier_warpx.profile.example:if [ -d "${HOME}/sw/frontier/gpu/venvs/warpx-frontier" ]
Tools/machines/frontier-olcf/frontier_warpx.profile.example:  source ${HOME}/sw/frontier/gpu/venvs/warpx-frontier/bin/activate
Tools/machines/frontier-olcf/frontier_warpx.profile.example:# GPU-aware MPI
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/frontier-olcf/frontier_warpx.profile.example:# optimize ROCm/HIP compilation for MI250X
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export CFLAGS="-I${ROCM_PATH}/include"
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
Tools/machines/frontier-olcf/frontier_warpx.profile.example:export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64 ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi_gtl_hsa"
Tools/machines/frontier-olcf/install_dependencies.sh:SW_DIR="${HOME}/sw/frontier/gpu"
Tools/machines/frontier-olcf/install_dependencies.sh:rm -rf $HOME/src/blaspp-frontier-gpu-build
Tools/machines/frontier-olcf/install_dependencies.sh:CXX=$(which CC) cmake -S $HOME/src/blaspp -B $HOME/src/blaspp-frontier-gpu-build -Duse_openmp=OFF -Dgpu_backend=hip -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/frontier-olcf/install_dependencies.sh:cmake --build $HOME/src/blaspp-frontier-gpu-build --target install --parallel 16
Tools/machines/frontier-olcf/install_dependencies.sh:rm -rf $HOME/src/blaspp-frontier-gpu-build
Tools/machines/frontier-olcf/install_dependencies.sh:rm -rf $HOME/src/lapackpp-frontier-gpu-build
Tools/machines/frontier-olcf/install_dependencies.sh:CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B $HOME/src/lapackpp-frontier-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
Tools/machines/frontier-olcf/install_dependencies.sh:cmake --build $HOME/src/lapackpp-frontier-gpu-build --target install --parallel 16
Tools/machines/frontier-olcf/install_dependencies.sh:rm -rf $HOME/src/lapackpp-frontier-gpu-build
Tools/machines/frontier-olcf/install_dependencies.sh:# cupy for ROCm
Tools/machines/frontier-olcf/install_dependencies.sh:#   https://docs.cupy.dev/en/stable/install.html#building-cupy-for-rocm-from-source
Tools/machines/frontier-olcf/install_dependencies.sh:ROCM_HOME=${ROCM_PATH}  \
Tools/machines/frontier-olcf/install_dependencies.sh:HCC_AMDGPU_TARGET=${AMREX_AMD_ARCH}  \
Tools/machines/frontier-olcf/install_dependencies.sh:#python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.4.2
Tools/machines/frontier-olcf/submit.sh:#SBATCH --gpus-per-task=1
Tools/machines/frontier-olcf/submit.sh:#SBATCH --gpu-bind=closest
Tools/machines/frontier-olcf/submit.sh:# load cray libs and ROCm libs
Tools/machines/frontier-olcf/submit.sh:# as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).
Tools/machines/lawrencium-lbnl/lawrencium_v100.sbatch:#SBATCH --gres=gpu:1
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:module load cuda/12.2.1
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:alias getNode="salloc -N 1 -t 1:00:00 --qos=es_debug --partition=es1 --constraint=es1_v100 --gres=gpu:1 --cpus-per-task=4 -A $proj"
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:alias runNode="srun -N 1 -t 1:00:00 --qos=es_debug --partition=es1 --constraint=es1_v100 --gres=gpu:1 --cpus-per-task=4 -A $proj"
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:# optimize CUDA compilation for 1080 Ti (deprecated)
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:#export AMREX_CUDA_ARCH=6.1
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:# optimize CUDA compilation for V100
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:export AMREX_CUDA_ARCH=7.0
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:# optimize CUDA compilation for 2080 Ti
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:#export AMREX_CUDA_ARCH=7.5
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/lawrencium-lbnl/lawrencium_warpx.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:#   Was perlmutter_gpu_warpx.profile sourced and configured correctly?
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your perlmutter_gpu_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:    echo "Please edit line 2 of your perlmutter_gpu_warpx.profile file to continue!"
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:SW_DIR="${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu"
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf $HOME/src/c-blosc-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:cmake -S $HOME/src/c-blosc -B ${build_dir}/c-blosc-pm-gpu-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:cmake --build ${build_dir}/c-blosc-pm-gpu-build --target install --parallel 16
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf ${build_dir}/c-blosc-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf $HOME/src/adios2-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:cmake -S $HOME/src/adios2 -B ${build_dir}/adios2-pm-gpu-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:cmake --build ${build_dir}/adios2-pm-gpu-build --target install -j 16
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf ${build_dir}/adios2-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf $HOME/src/blaspp-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:CXX=$(which CC) cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-pm-gpu-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:cmake --build ${build_dir}/blaspp-pm-gpu-build --target install --parallel 16
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf ${build_dir}/blaspp-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf $HOME/src/lapackpp-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B ${build_dir}/lapackpp-pm-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:cmake --build ${build_dir}/lapackpp-pm-gpu-build --target install --parallel 16
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf ${build_dir}/lapackpp-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf ${HOME}/src/heffte-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:    -B ${build_dir}/heffte-pm-gpu-build \
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:    -DHeffte_DISABLE_GPU_AWARE_MPI=OFF  \
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:    -DHeffte_ENABLE_CUDA=ON             \
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:    -DHeffte_ENABLE_ROCM=OFF            \
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:cmake --build ${build_dir}/heffte-pm-gpu-build --target install --parallel 16
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf ${build_dir}/heffte-pm-gpu-build
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:rm -rf ${SW_DIR}/venvs/warpx-gpu
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:python3 -m venv ${SW_DIR}/venvs/warpx-gpu
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:source ${SW_DIR}/venvs/warpx-gpu/bin/activate
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:MPICC="cc -target-accel=nvidia80 -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:python3 -m pip install --upgrade cupy-cuda12x  # CUDA 12 compatible wheel
Tools/machines/perlmutter-nersc/install_gpu_dependencies.sh:python3 -m pip install --upgrade torch  # CUDA 12 compatible wheel
Tools/machines/perlmutter-nersc/install_cpu_dependencies.sh:cmake -S $HOME/src/adios2 -B ${build_dir}/adios2-pm-cpu-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_CUDA=OFF -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
Tools/machines/perlmutter-nersc/install_cpu_dependencies.sh:CXX=$(which CC) cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-pm-cpu-build -Duse_openmp=ON -Dgpu_backend=OFF -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/perlmutter-nersc/install_cpu_dependencies.sh:    -DHeffte_DISABLE_GPU_AWARE_MPI=ON   \
Tools/machines/perlmutter-nersc/install_cpu_dependencies.sh:    -DHeffte_ENABLE_CUDA=OFF            \
Tools/machines/perlmutter-nersc/install_cpu_dependencies.sh:    -DHeffte_ENABLE_ROCM=OFF            \
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export proj=""  # change me! GPU projects must end in "..._g"
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:module load gpu
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:module load craype-accel-nvidia80
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:module load cudatoolkit
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/c-blosc-1.21.1:$CMAKE_PREFIX_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/adios2-2.8.3:$CMAKE_PREFIX_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/heffte-2.4.0:$CMAKE_PREFIX_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/c-blosc-1.21.1/lib64:$LD_LIBRARY_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/adios2-2.8.3/lib64:$LD_LIBRARY_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/heffte-2.4.0/lib64:$LD_LIBRARY_PATH
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export PATH=${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/adios2-2.8.3/bin:${PATH}
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:if [ -d "${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/venvs/warpx-gpu" ]
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:  source ${CFS}/${proj%_g}/${USER}/sw/perlmutter/gpu/venvs/warpx-gpu/bin/activate
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:alias getNode="salloc -N 1 --ntasks-per-node=4 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 4 -A $proj"
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:alias runNode="srun -N 1 --ntasks-per-node=4 -t 0:30:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 4 -A $proj"
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:# necessary to use CUDA-Aware MPI and run a job
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CRAY_ACCEL_TARGET=nvidia80
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:# optimize CUDA compilation for A100
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export AMREX_CUDA_ARCH=8.0
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/perlmutter-nersc/perlmutter_gpu_warpx.profile.example:export CUDAHOSTCXX=CC
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:#SBATCH -C gpu
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:#S BATCH -C gpu&hbm80g
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:#SBATCH --gpu-bind=none
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:#SBATCH --gpus-per-node=4
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:# pin to closest NIC to GPU
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:export MPICH_OFI_NIC_POLICY=GPU
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:# GPU-aware MPI optimizations
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:# CUDA visible devices are ordered inverse to local task IDs
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:#   Reference: nvidia-smi topo -m
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch:    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" \
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:#   Was leonardo_gpu_warpx.profile sourced and configured correctly?
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:rm -rf $HOME/src/adios2-gpu-build
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:cmake -S $HOME/src/adios2 -B $HOME/src/adios2-gpu-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-master
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:cmake --build $HOME/src/adios2-gpu-build --target install -j 16
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:rm -rf $HOME/src/adios2-gpu-build
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:rm -rf $HOME/src/blaspp-gpu-build
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:CXX=$(which g++) cmake -S $HOME/src/blaspp -B $HOME/src/blaspp-gpu-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:cmake --build $HOME/src/blaspp-gpu-build --target install --parallel 16
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:rm -rf $HOME/src/blaspp-gpu-build
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:rm -rf $HOME/src/lapackpp-gpu-build
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B $HOME/src/lapackpp-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:cmake --build $HOME/src/lapackpp-gpu-build --target install --parallel 16
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:rm -rf $HOME/src/lapackpp-gpu-build
Tools/machines/leonardo-cineca/install_gpu_dependencies.sh:python3 -m pip install --upgrade torch  # CUDA 11.8 compatible wheel
Tools/machines/leonardo-cineca/job.sh:#SBATCH --gpus-per-node=4
Tools/machines/leonardo-cineca/job.sh:#SBATCH --gpus-per-task=1
Tools/machines/leonardo-cineca/job.sh:#SBATCH --gres=gpu:4
Tools/machines/leonardo-cineca/job.sh:srun /leonardo/home/userexternal/<username>/src/warpx/build_gpu/bin/warpx.2d <input file> > output.txt
Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example:module load cuda/11.8
Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example:module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example:# optimize CUDA compilation for A100
Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example:export AMREX_CUDA_ARCH=8.0
Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/leonardo-cineca/leonardo_gpu_warpx.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:  - cuda@11.7.0
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:    cuda:
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:      - spec: cuda@11.7.0
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:        - CUDA/11.7.0
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:      - spec: openmpi@4.1.4 +atomics +cuda %gcc@11.3.0
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:        - OpenMPI/4.1.4-GCC-11.3.0-CUDA-11.7.0
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:      variants: +mpi ~fortran +cuda cuda_arch=80
Tools/machines/karolina-it4i/spack-karolina-cuda.yaml:        cuda: [cuda@11.7.0]
Tools/machines/karolina-it4i/karolina_gpu.sbatch:#SBATCH --partition=qgpu
Tools/machines/karolina-it4i/karolina_gpu.sbatch:#SBATCH --gpus-per-node=8
Tools/machines/karolina-it4i/karolina_gpu.sbatch:#SBATCH --gpu-bind=single:1
Tools/machines/karolina-it4i/install_dependencies.sh:    spack env rm -y warpx-karolina-cuda || true
Tools/machines/karolina-it4i/install_dependencies.sh:spack env create warpx-karolina-cuda $WORK/src/warpx/Tools/machines/karolina-it4i/spack-karolina-cuda.yaml
Tools/machines/karolina-it4i/install_dependencies.sh:spack env activate warpx-karolina-cuda
Tools/machines/karolina-it4i/karolina_warpx.profile.example:module load OpenMPI/4.1.4-GCC-11.3.0-CUDA-11.7.0
Tools/machines/karolina-it4i/karolina_warpx.profile.example:source $WORK/spack/share/spack/setup-env.sh && spack env activate warpx-karolina-cuda && {
Tools/machines/karolina-it4i/karolina_warpx.profile.example:    echo "Spack environment 'warpx-karolina-cuda' activated successfully."
Tools/machines/karolina-it4i/karolina_warpx.profile.example:    echo "Failed to activate Spack environment 'warpx-karolina-cuda'. Please run install_dependencies.sh."
Tools/machines/karolina-it4i/karolina_warpx.profile.example:    srun --time=1:00:00 --nodes=$numNodes --ntasks=$((8 * $numNodes)) --ntasks-per-node=8 --cpus-per-task=16 --exclusive --gpus-per-node=8 -p qgpu -A $proj --pty bash
Tools/machines/karolina-it4i/karolina_warpx.profile.example:# optimize CUDA compilation for A100
Tools/machines/karolina-it4i/karolina_warpx.profile.example:export AMREX_CUDA_ARCH="8.0"
Tools/machines/karolina-it4i/karolina_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/karolina-it4i/karolina_warpx.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/crusher-olcf/install_dependencies.sh:#   Was crusher_gpu_warpx.profile sourced and configured correctly?
Tools/machines/crusher-olcf/install_dependencies.sh:SW_DIR="${HOME}/sw/crusher/gpu"
Tools/machines/crusher-olcf/install_dependencies.sh:rm -rf $HOME/src/blaspp-crusher-gpu-build
Tools/machines/crusher-olcf/install_dependencies.sh:CXX=$(which CC) cmake -S $HOME/src/blaspp -B $HOME/src/blaspp-crusher-gpu-build -Duse_openmp=OFF -Dgpu_backend=hip -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/crusher-olcf/install_dependencies.sh:cmake --build $HOME/src/blaspp-crusher-gpu-build --target install --parallel 16
Tools/machines/crusher-olcf/install_dependencies.sh:rm -rf $HOME/src/blaspp-crusher-gpu-build
Tools/machines/crusher-olcf/install_dependencies.sh:rm -rf $HOME/src/lapackpp-crusher-gpu-build
Tools/machines/crusher-olcf/install_dependencies.sh:CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B $HOME/src/lapackpp-crusher-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
Tools/machines/crusher-olcf/install_dependencies.sh:cmake --build $HOME/src/lapackpp-crusher-gpu-build --target install --parallel 16
Tools/machines/crusher-olcf/install_dependencies.sh:rm -rf $HOME/src/lapackpp-crusher-gpu-build
Tools/machines/crusher-olcf/install_dependencies.sh:#python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.4.2
Tools/machines/crusher-olcf/crusher_warpx.profile.example:module load rocm/5.2.0  # waiting for 5.6 for next bump
Tools/machines/crusher-olcf/crusher_warpx.profile.example:module load cce/15.0.0  # must be loaded after rocm
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/crusher/gpu/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/crusher/gpu/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/crusher/gpu/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/crusher/gpu/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/crusher-olcf/crusher_warpx.profile.example:if [ -d "${HOME}/sw/crusher/gpu/venvs/warpx-crusher" ]
Tools/machines/crusher-olcf/crusher_warpx.profile.example:  source ${HOME}/sw/crusher/gpu/venvs/warpx-crusher/bin/activate
Tools/machines/crusher-olcf/crusher_warpx.profile.example:# GPU-aware MPI
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/crusher-olcf/crusher_warpx.profile.example:# optimize ROCm/HIP compilation for MI250X
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export CFLAGS="-I${ROCM_PATH}/include"
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
Tools/machines/crusher-olcf/crusher_warpx.profile.example:export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64 ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi_gtl_hsa"
Tools/machines/crusher-olcf/submit.sh:#SBATCH --gpus-per-task=1
Tools/machines/crusher-olcf/submit.sh:#SBATCH --gpu-bind=closest
Tools/machines/crusher-olcf/submit.sh:# as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:module load cuda/12.1.1
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:module load openmpi/4.1.6-cuda
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:alias getNode="salloc -N 1 --partition=gpu --ntasks-per-node=2 --cpus-per-task=20 --gpus-per-task=v100:1 -t 1:00:00 -A $proj"
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:alias runNode="srun -N 1 --partition=gpu --ntasks-per-node=2 --cpus-per-task=20 --gpus-per-task=v100:1 -t 1:00:00 -A $proj"
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:# optimize CUDA compilation for V100
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:export AMREX_CUDA_ARCH=7.0
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/greatlakes-umich/greatlakes_v100_warpx.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/greatlakes-umich/greatlakes_v100.sbatch:#SBATCH --partition=gpu
Tools/machines/greatlakes-umich/greatlakes_v100.sbatch:#SBATCH --gpus-per-task=v100:1
Tools/machines/greatlakes-umich/greatlakes_v100.sbatch:#SBATCH --gpu-bind=single:1
Tools/machines/greatlakes-umich/greatlakes_v100.sbatch:# GPU-aware MPI optimizations
Tools/machines/greatlakes-umich/greatlakes_v100.sbatch:GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"
Tools/machines/greatlakes-umich/greatlakes_v100.sbatch:  ${EXE} ${INPUTS} ${GPU_AWARE_MPI} \
Tools/machines/greatlakes-umich/install_v100_dependencies.sh:cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-v100-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/greatlakes-umich/install_v100_dependencies.sh:python3 -m pip install --upgrade cupy-cuda12x  # CUDA 12 compatible wheel
Tools/machines/greatlakes-umich/install_v100_dependencies.sh:python3 -m pip install --upgrade torch  # CUDA 12 compatible wheel
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:#   Was polaris_gpu_warpx.profile sourced and configured correctly?
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your polaris_gpu_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:SW_DIR="/home/${USER}/sw/polaris/gpu"
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/c-blosc-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:cmake -S $HOME/src/c-blosc -B $HOME/src/c-blosc-pm-gpu-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:cmake --build $HOME/src/c-blosc-pm-gpu-build --target install --parallel 16
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/c-blosc-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/adios2-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:cmake -S $HOME/src/adios2 -B $HOME/src/adios2-pm-gpu-build -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:cmake --build $HOME/src/adios2-pm-gpu-build --target install -j 16
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/adios2-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/blaspp-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:CXX=$(which CC) cmake -S $HOME/src/blaspp -B $HOME/src/blaspp-pm-gpu-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:cmake --build $HOME/src/blaspp-pm-gpu-build --target install --parallel 16
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/blaspp-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/lapackpp-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B $HOME/src/lapackpp-pm-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:cmake --build $HOME/src/lapackpp-pm-gpu-build --target install --parallel 16
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:rm -rf $HOME/src/lapackpp-pm-gpu-build
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:MPICC="CC -target-accel=nvidia80 -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:python3 -m pip install cupy-cuda11x  # CUDA 11.8 compatible wheel
Tools/machines/polaris-alcf/install_gpu_dependencies.sh:python3 -m pip install --upgrade torch  # CUDA 11.8 compatible wheel
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=/home/${USER}/sw/polaris/gpu/c-blosc-1.21.1:$CMAKE_PREFIX_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=/home/${USER}/sw/polaris/gpu/adios2-2.8.3:$CMAKE_PREFIX_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=/home/${USER}/sw/polaris/gpu/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=/home/${USER}/sw/polaris/gpu/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export LD_LIBRARY_PATH=/home/${USER}/sw/polaris/gpu/c-blosc-1.21.1/lib64:$LD_LIBRARY_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export LD_LIBRARY_PATH=/home/${USER}/sw/polaris/gpu/adios2-2.8.3/lib64:$LD_LIBRARY_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export LD_LIBRARY_PATH=/home/${USER}/sw/polaris/gpu/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export LD_LIBRARY_PATH=/home/${USER}/sw/polaris/gpu/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export PATH=/home/${USER}/sw/polaris/gpu/adios2-2.8.3/bin:${PATH}
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:if [ -d "/home/${USER}/sw/polaris/gpu/venvs/warpx" ]
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:  source /home/${USER}/sw/polaris/gpu/venvs/warpx/bin/activate
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:# necessary to use CUDA-Aware MPI and run a job
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export CRAY_ACCEL_TARGET=nvidia80
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:# optimize CUDA compilation for A100
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export AMREX_CUDA_ARCH=8.0
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/polaris-alcf/polaris_gpu_warpx.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/polaris-alcf/polaris_gpu.pbs:# support gpu-aware-mpi
Tools/machines/polaris-alcf/polaris_gpu.pbs:# export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/adastra-cines/adastra_warpx.profile.example:module load CCE-GPU-3.0.0
Tools/machines/adastra-cines/adastra_warpx.profile.example:export CMAKE_PREFIX_PATH=${SHAREDHOMEDIR}/sw/adastra/gpu/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/adastra-cines/adastra_warpx.profile.example:export CMAKE_PREFIX_PATH=${SHAREDHOMEDIR}/sw/adastra/gpu/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/adastra-cines/adastra_warpx.profile.example:export LD_LIBRARY_PATH=${SHAREDHOMEDIR}/sw/adastra/gpu/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/adastra-cines/adastra_warpx.profile.example:export LD_LIBRARY_PATH=${SHAREDHOMEDIR}/sw/adastra/gpu/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/adastra-cines/adastra_warpx.profile.example:export CMAKE_PREFIX_PATH=${SHAREDHOMEDIR}/sw/adastra/gpu/c-blosc-1.21.1:$CMAKE_PREFIX_PATH
Tools/machines/adastra-cines/adastra_warpx.profile.example:export CMAKE_PREFIX_PATH=${SHAREDHOMEDIR}/sw/adastra/gpu/adios2-2.8.3:$CMAKE_PREFIX_PATH
Tools/machines/adastra-cines/adastra_warpx.profile.example:export PATH=${HOME}/sw/adastra/gpu/adios2-2.8.3/bin:${PATH}
Tools/machines/adastra-cines/adastra_warpx.profile.example:alias getNode="salloc --account=$proj --job-name=warpx --constraint=MI250 --nodes=1 --ntasks-per-node=8 --cpus-per-task=8 --gpus-per-node=8 --threads-per-core=1 --exclusive --time=01:00:00"
Tools/machines/adastra-cines/adastra_warpx.profile.example:# GPU-aware MPI
Tools/machines/adastra-cines/adastra_warpx.profile.example:export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/adastra-cines/adastra_warpx.profile.example:# optimize ROCm/HIP compilation for MI250X
Tools/machines/adastra-cines/install_dependencies.sh:#   Was perlmutter_gpu_warpx.profile sourced and configured correctly?
Tools/machines/adastra-cines/install_dependencies.sh:SW_DIR="${SHAREDHOMEDIR}/sw/adastra/gpu"
Tools/machines/adastra-cines/install_dependencies.sh:rm -rf $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build
Tools/machines/adastra-cines/install_dependencies.sh:CXX=$(which CC) cmake -S $SHAREDHOMEDIR/src/blaspp -B $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build -Duse_openmp=OFF -Dgpu_backend=hip -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/adastra-cines/install_dependencies.sh:cmake --build $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build --target install --parallel 16
Tools/machines/adastra-cines/install_dependencies.sh:rm -rf $SHAREDHOMEDIR/src/blaspp-adastra-gpu-build
Tools/machines/adastra-cines/install_dependencies.sh:rm -rf $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build
Tools/machines/adastra-cines/install_dependencies.sh:CXX=$(which CC) CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $SHAREDHOMEDIR/src/lapackpp -B $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
Tools/machines/adastra-cines/install_dependencies.sh:cmake --build $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build --target install --parallel 16
Tools/machines/adastra-cines/install_dependencies.sh:rm -rf $SHAREDHOMEDIR/src/lapackpp-adastra-gpu-build
Tools/machines/adastra-cines/install_dependencies.sh:#python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.4.2
Tools/machines/adastra-cines/submit.sh:module load CCE-GPU-3.0.0
Tools/machines/adastra-cines/submit.sh:export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/adastra-cines/submit.sh:     --cpus-per-task=8 --threads-per-core=1 --gpu-bind=closest \
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:#SBATCH -C gpu
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:#SBATCH --gpu-bind=none
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:#SBATCH --gpus-per-node=4
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:# pin to closest NIC to GPU
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:export MPICH_OFI_NIC_POLICY=GPU
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:#GPU_AWARE_MPI="amrex.the_arena_is_managed=0 amrex.use_gpu_aware_mpi=1"
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:GPU_AWARE_MPI=""
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:# CUDA visible devices are ordered inverse to local task IDs
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:#   Reference: nvidia-smi topo -m
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
Tools/machines/lonestar6-tacc/lonestar6_a100.sbatch:    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" \
Tools/machines/lonestar6-tacc/install_a100_dependencies.sh:cmake -S $HOME/src/blaspp -B ${build_dir}/blaspp-a100-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/lonestar6-tacc/install_a100_dependencies.sh:    -DHeffte_DISABLE_GPU_AWARE_MPI=OFF  \
Tools/machines/lonestar6-tacc/install_a100_dependencies.sh:    -DHeffte_ENABLE_CUDA=ON             \
Tools/machines/lonestar6-tacc/install_a100_dependencies.sh:    -DHeffte_ENABLE_ROCM=OFF            \
Tools/machines/lonestar6-tacc/install_a100_dependencies.sh:#python3 -m pip install --upgrade cupy-cuda12x  # CUDA 12 compatible wheel
Tools/machines/lonestar6-tacc/install_a100_dependencies.sh:#python3 -m pip install --upgrade torch  # CUDA 12 compatible wheel
Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example:module load cuda/12.2
Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example:alias getNode="salloc -N 1 --ntasks-per-node=2 -t 1:00:00 -p gpu-100 --gpu-bind=single:1 -c 32 -G 2 -A $proj"
Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example:alias runNode="srun -N 1 --ntasks-per-node=2 -t 0:30:00 -p gpu-100 --gpu-bind=single:1 -c 32 -G 2 -A $proj"
Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example:# optimize CUDA compilation for A100
Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example:export AMREX_CUDA_ARCH=8.0
Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/lonestar6-tacc/lonestar6_warpx_a100.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/tioga-llnl/install_mi300a_ml.sh:python3 -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:    -Dgpu_backend=hip                         \
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:    -Dgpu_backend=hip                           \
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:    -DHeffte_DISABLE_GPU_AWARE_MPI=OFF  \
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:    -DHeffte_ENABLE_CUDA=OFF            \
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:    -DHeffte_ENABLE_ROCM=ON             \
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:# cupy for ROCm
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:#   https://docs.cupy.dev/en/stable/install.html#building-cupy-for-rocm-from-source
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:#   https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:#CXXFLAGS="-I${ROCM_PATH}/include/hipblas -I${ROCM_PATH}/include/hipsparse -I${ROCM_PATH}/include/hipfft -I${ROCM_PATH}/include/rocsolver -I${ROCM_PATH}/include/rccl" \
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:#ROCM_HOME=${ROCM_PATH}  \
Tools/machines/tioga-llnl/install_mi300a_dependencies.sh:#HCC_AMDGPU_TARGET=${AMREX_AMD_ARCH}  \
Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example:# module load rocmcc/6.1.2-cce-18.0.0-magic
Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example:module load craype-accel-amd-gfx942  # GPU
Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example:module load rocm/6.1.2
Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example:# GPU-aware MPI
Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example:export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/tioga-llnl/tioga_mi300a_warpx.profile.example:# optimize ROCm/HIP compilation for MI300A
Tools/machines/tioga-llnl/tioga_mi300a.sbatch:#SBATCH --gpu-bind=none
Tools/machines/tioga-llnl/tioga_mi300a.sbatch:#SBATCH --gpus-per-node=4
Tools/machines/tioga-llnl/tioga_mi300a.sbatch:# pin to closest NIC to GPU
Tools/machines/tioga-llnl/tioga_mi300a.sbatch:export MPICH_OFI_NIC_POLICY=GPU
Tools/machines/tioga-llnl/tioga_mi300a.sbatch:# GPU-aware MPI optimizations
Tools/machines/tioga-llnl/tioga_mi300a.sbatch:GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"
Tools/machines/tioga-llnl/tioga_mi300a.sbatch:  ${GPU_AWARE_MPI} ${APU_SHARED_MEMORY} \
Tools/machines/desktop/spack-debian-rocm.yaml:#   spack env create warpx-rocm-dev spack-ubuntu-rocm.yaml
Tools/machines/desktop/spack-debian-rocm.yaml:#   spack env activate warpx-rocm-dev
Tools/machines/desktop/spack-debian-rocm.yaml:  - llvm-amdgpu
Tools/machines/desktop/spack-debian-rocm.yaml:      variants: +mpi ~fortran +rocm
Tools/machines/desktop/spack-ubuntu-openmp.yaml:  - blaspp ~cuda +openmp ~rocm
Tools/machines/desktop/spack-ubuntu-openmp.yaml:  - heffte ~cuda +fftw
Tools/machines/desktop/spack-ubuntu-openmp.yaml:  - lapackpp ~cuda ~rocm ^blaspp ~cuda +openmp ~rocm
Tools/machines/desktop/spack-ubuntu-openmp.yaml:      variants: +mpi ~fortran ~cuda ~rocm
Tools/machines/desktop/spack-ubuntu-cuda.yaml:#   spack env create warpx-cuda-dev spack-ubuntu-cuda.yaml
Tools/machines/desktop/spack-ubuntu-cuda.yaml:#   spack env activate warpx-cuda-dev
Tools/machines/desktop/spack-ubuntu-cuda.yaml:  - adios2 ~cuda
Tools/machines/desktop/spack-ubuntu-cuda.yaml:  - cuda
Tools/machines/desktop/spack-ubuntu-cuda.yaml:# This always enables DevilRay, which builds too long on CUDA and we mainly use VTK-m
Tools/machines/desktop/spack-ubuntu-cuda.yaml:      # note: add +cuda cuda_arch=70
Tools/machines/desktop/spack-ubuntu-cuda.yaml:      #       or respective CUDA capability instead of 70 to variants below
Tools/machines/desktop/spack-ubuntu-cuda.yaml:      variants: +mpi ~fortran +cuda cuda_arch=70
Tools/machines/desktop/spack-ubuntu-cuda.yaml:      # g++-10 gcc-10 gfortran-10 (or 11) for CUDA 11.7.0
Tools/machines/desktop/spack-macos-openmp.yaml:  - blaspp ~cuda +openmp ~rocm
Tools/machines/desktop/spack-macos-openmp.yaml:  - heffte ~cuda +fftw
Tools/machines/desktop/spack-macos-openmp.yaml:  - lapackpp ~cuda ~rocm ^blaspp ~cuda +openmp ~rocm
Tools/machines/desktop/spack-macos-openmp.yaml:      variants: +mpi ~fortran ~cuda ~rocm
Tools/machines/desktop/spack-debian-cuda.yaml:#   spack env create warpx-cuda-dev spack-ubuntu-cuda.yaml
Tools/machines/desktop/spack-debian-cuda.yaml:#   spack env activate warpx-cuda-dev
Tools/machines/desktop/spack-debian-cuda.yaml:  - adios2 ~cuda
Tools/machines/desktop/spack-debian-cuda.yaml:  - cuda
Tools/machines/desktop/spack-debian-cuda.yaml:# This always enables DevilRay, which builds too long on CUDA and we mainly use VTK-m
Tools/machines/desktop/spack-debian-cuda.yaml:      # note: add +cuda cuda_arch=70
Tools/machines/desktop/spack-debian-cuda.yaml:      #       or respective CUDA capability instead of 70 to variants below
Tools/machines/desktop/spack-debian-cuda.yaml:      variants: +mpi ~fortran +cuda cuda_arch=70
Tools/machines/desktop/spack-debian-cuda.yaml:      # g++-10 gcc-10 gfortran-10 (or 11) for CUDA 11.7.0
Tools/machines/desktop/spack-ubuntu-rocm.yaml:#   spack env create warpx-rocm-dev spack-ubuntu-rocm.yaml
Tools/machines/desktop/spack-ubuntu-rocm.yaml:#   spack env activate warpx-rocm-dev
Tools/machines/desktop/spack-ubuntu-rocm.yaml:  - llvm-amdgpu
Tools/machines/desktop/spack-ubuntu-rocm.yaml:      variants: +mpi ~fortran +rocm
Tools/machines/desktop/spack-debian-openmp.yaml:  - blaspp ~cuda +openmp ~rocm
Tools/machines/desktop/spack-debian-openmp.yaml:  - heffte ~cuda +fftw
Tools/machines/desktop/spack-debian-openmp.yaml:  - lapackpp ~cuda ~rocm ^blaspp ~cuda +openmp ~rocm
Tools/machines/desktop/spack-debian-openmp.yaml:      variants: +mpi ~fortran ~cuda ~rocm
Tools/machines/lumi-csc/lumi_warpx.profile.example:module load rocm/6.0.3
Tools/machines/lumi-csc/lumi_warpx.profile.example:SW_DIR="${HOME}/sw/lumi/gpu"
Tools/machines/lumi-csc/lumi_warpx.profile.example:alias getNode="salloc -A $proj -J warpx -t 01:00:00 -p dev-g -N 1 --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=closest"
Tools/machines/lumi-csc/lumi_warpx.profile.example:alias runNode="srun -A $proj -J warpx -t 00:30:00 -p dev-g -N 1 --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=closest"
Tools/machines/lumi-csc/lumi_warpx.profile.example:# GPU-aware MPI
Tools/machines/lumi-csc/lumi_warpx.profile.example:export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/lumi-csc/lumi_warpx.profile.example:# optimize ROCm/HIP compilation for MI250X
Tools/machines/lumi-csc/lumi_warpx.profile.example:export CFLAGS="-I${ROCM_PATH}/include"
Tools/machines/lumi-csc/lumi_warpx.profile.example:export CXXFLAGS="-I${ROCM_PATH}/include -Wno-pass-failed"
Tools/machines/lumi-csc/lumi_warpx.profile.example:export LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64 ${PE_MPICH_GTL_DIR_amd_gfx90a} -lmpi_gtl_hsa"
Tools/machines/lumi-csc/install_dependencies.sh:SW_DIR="${HOME}/sw/lumi/gpu"
Tools/machines/lumi-csc/install_dependencies.sh:rm -rf ${build_dir}/blaspp-lumi-gpu-build
Tools/machines/lumi-csc/install_dependencies.sh:      -B ${build_dir}/blaspp-lumi-gpu-build  \
Tools/machines/lumi-csc/install_dependencies.sh:      -Dgpu_backend=hip                      \
Tools/machines/lumi-csc/install_dependencies.sh:cmake --build ${build_dir}/blaspp-lumi-gpu-build --target install --parallel 16
Tools/machines/lumi-csc/install_dependencies.sh:rm -rf ${build_dir}/blaspp-lumi-gpu-build
Tools/machines/lumi-csc/install_dependencies.sh:rm -rf ${build_dir}/lapackpp-lumi-gpu-build
Tools/machines/lumi-csc/install_dependencies.sh:      -B ${build_dir}/lapackpp-lumi-gpu-build    \
Tools/machines/lumi-csc/install_dependencies.sh:cmake --build ${build_dir}/lapackpp-lumi-gpu-build --target install --parallel 16
Tools/machines/lumi-csc/install_dependencies.sh:rm -rf ${build_dir}/lapackpp-lumi-gpu-build
Tools/machines/lumi-csc/install_dependencies.sh:      -DCMAKE_INSTALL_PREFIX=${HOME}/sw/lumi/gpu/c-blosc-1.21.1
Tools/machines/lumi-csc/install_dependencies.sh:      -DCMAKE_INSTALL_PREFIX=${HOME}/sw/lumi/gpu/adios2-2.8.3
Tools/machines/lumi-csc/install_dependencies.sh:#python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/rocm5.4.2
Tools/machines/lumi-csc/lumi.sbatch:#SBATCH --gpus-per-node=8
Tools/machines/lumi-csc/lumi.sbatch:cat << EOF > select_gpu
Tools/machines/lumi-csc/lumi.sbatch:chmod +x ./select_gpu
Tools/machines/lumi-csc/lumi.sbatch:# in order to have 6 threads per GPU (blosc compression in adios2 uses threads)
Tools/machines/lumi-csc/lumi.sbatch:export MPICH_GPU_SUPPORT_ENABLED=1
Tools/machines/lumi-csc/lumi.sbatch:srun --cpu-bind=${CPU_BIND} ./select_gpu ./warpx inputs | tee outputs.txt
Tools/machines/lumi-csc/lumi.sbatch:rm -rf ./select_gpu
Tools/machines/juwels-jsc/juwels_warpx.profile.example:module load CUDA/11.3
Tools/machines/juwels-jsc/juwels_warpx.profile.example:# JUWELS' job scheduler may not map ranks to GPUs,
Tools/machines/juwels-jsc/juwels_warpx.profile.example:export GPUS_PER_SOCKET=2
Tools/machines/juwels-jsc/juwels_warpx.profile.example:export GPUS_PER_NODE=4
Tools/machines/juwels-jsc/juwels_warpx.profile.example:# optimize CUDA compilation for V100 (7.0) or for A100 (8.0)
Tools/machines/juwels-jsc/juwels_warpx.profile.example:export AMREX_CUDA_ARCH=8.0
Tools/machines/juwels-jsc/juwels.sbatch:#SBATCH --gres=gpu:4
Tools/machines/juwels-jsc/juwels.sbatch:module load CUDA/11.3
Tools/machines/juwels-jsc/juwels.sbatch:srun -n 8 --cpu_bind=sockets $HOME/src/warpx/build/bin/warpx.3d.MPI.CUDA.DP.OPMD.QED inputs
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:#   Was hpc3_gpu_warpx.profile sourced and configured correctly?
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your hpc3_gpu_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:#    echo "Please edit line 2 of your hpc3_gpu_warpx.profile file to continue!"
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:SW_DIR="${HOME}/sw/hpc3/gpu"
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/c-blosc-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:cmake -S $HOME/src/c-blosc -B $HOME/src/c-blosc-pm-gpu-build -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DDEACTIVATE_AVX2=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-1.21.1
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:cmake --build $HOME/src/c-blosc-pm-gpu-build --target install --parallel 8
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/c-blosc-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/adios2-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:cmake -S $HOME/src/adios2 -B $HOME/src/adios2-pm-gpu-build -DBUILD_TESTING=OFF -DADIOS2_BUILD_EXAMPLES=OFF -DADIOS2_USE_Blosc=ON -DADIOS2_USE_Fortran=OFF -DADIOS2_USE_HDF5=OFF -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.8.3
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:cmake --build $HOME/src/adios2-pm-gpu-build --target install --parallel 8
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/adios2-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/blaspp-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:cmake -S $HOME/src/blaspp -B $HOME/src/blaspp-pm-gpu-build -Duse_openmp=OFF -Dgpu_backend=cuda -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:cmake --build $HOME/src/blaspp-pm-gpu-build --target install --parallel 8
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/blaspp-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/lapackpp-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:CXXFLAGS="-DLAPACK_FORTRAN_ADD_" cmake -S $HOME/src/lapackpp -B $HOME/src/lapackpp-pm-gpu-build -DCMAKE_CXX_STANDARD=17 -Dbuild_tests=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:cmake --build $HOME/src/lapackpp-pm-gpu-build --target install --parallel 8
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf $HOME/src/lapackpp-pm-gpu-build
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:rm -rf ${SW_DIR}/venvs/warpx-gpu
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:python3 -m venv ${SW_DIR}/venvs/warpx-gpu
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:source ${SW_DIR}/venvs/warpx-gpu/bin/activate
Tools/machines/hpc3-uci/install_gpu_dependencies.sh:python3 -m pip install --upgrade torch  # CUDA 11.7 compatible wheel
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export proj=""  # change me! GPU projects must end in "..._g"
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:module load cuda/11.7.1
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/hpc3/gpu/c-blosc-1.21.1:$CMAKE_PREFIX_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/hpc3/gpu/adios2-2.8.3:$CMAKE_PREFIX_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/hpc3/gpu/blaspp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export CMAKE_PREFIX_PATH=${HOME}/sw/hpc3/gpu/lapackpp-2024.05.31:$CMAKE_PREFIX_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/hpc3/gpu/c-blosc-1.21.1/lib64:$LD_LIBRARY_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/hpc3/gpu/adios2-2.8.3/lib64:$LD_LIBRARY_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/hpc3/gpu/blaspp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export LD_LIBRARY_PATH=${HOME}/sw/hpc3/gpu/lapackpp-2024.05.31/lib64:$LD_LIBRARY_PATH
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export PATH=${HOME}/sw/hpc3/gpu/adios2-2.8.3/bin:${PATH}
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:if [ -d "${HOME}/sw/hpc3/gpu/venvs/warpx-gpu" ]
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:  source ${HOME}/sw/hpc3/gpu/venvs/warpx-gpu/bin/activate
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:alias getNode="salloc -N 1 -t 0:30:00 --gres=gpu:V100:1 -p free-gpu"
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:alias runNode="srun -N 1 -t 0:30:00 --gres=gpu:V100:1 -p free-gpu"
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:# optimize CUDA compilation for V100
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export AMREX_CUDA_ARCH=7.0
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/hpc3-uci/hpc3_gpu_warpx.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/hpc3-uci/hpc3_gpu.sbatch:# V100 GPU options: gpu, free-gpu, debug-gpu
Tools/machines/hpc3-uci/hpc3_gpu.sbatch:#SBATCH -p free-gpu
Tools/machines/hpc3-uci/hpc3_gpu.sbatch:# use all four GPUs per node
Tools/machines/hpc3-uci/hpc3_gpu.sbatch:#SBATCH --gres=gpu:V100:4
Tools/machines/hpc3-uci/hpc3_gpu.sbatch:    export CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID};
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:module load cuda/11.8.0
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:module load openmpi-cuda/4.1.5-hpcx
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:alias getNode="salloc -N 1 --ntasks-per-node=2 --cpus-per-task=20 --gpus-per-task=v100:1 -t 1:00:00 -A $proj"
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:alias runNode="srun -N 1 --ntasks-per-node=2 --cpus-per-task=20 --gpus-per-task=v100:1 -t 1:00:00 -A $proj"
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:export CUDAFLAGS="--host-linker-script=use-lcs" # https://github.com/ECP-WarpX/WarpX/pull/3673
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:export AMREX_CUDA_ARCH=7.0 # 7.0: V100, 8.0: V100, 9.0: H100 https://github.com/ECP-WarpX/WarpX/issues/3214
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/pitzer-osc/pitzer_v100_warpx.profile.example:export CUDAHOSTCXX=${CXX}
Tools/machines/pitzer-osc/pitzer_v100.sbatch:#SBATCH --gpus-per-task=1
Tools/machines/pitzer-osc/pitzer_v100.sbatch:#SBATCH --gpu-bind=closest
Tools/machines/pitzer-osc/pitzer_v100.sbatch:# Pitzer cluster has 32 GPU nodes with dual Intel Xeon 6148 and dual V100 (16GB) GPUs and 42 nodes with dual Intel Xeon 8268 and dual V100 (32GB) GPUs. https://www.osc.edu/resources/technical_support/supercomputers/pitzer
Tools/machines/pitzer-osc/pitzer_v100.sbatch:echo "GPU Information:"
Tools/machines/pitzer-osc/pitzer_v100.sbatch:nvidia-smi
Tools/machines/pitzer-osc/pitzer_v100.sbatch:GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"
Tools/machines/pitzer-osc/pitzer_v100.sbatch:srun --cpu-bind=cores ${EXE} ${INPUTS} ${GPU_AWARE_MPI} >./logs/${SLURM_JOB_NAME}_${SLURM_JOBID}.log 2>&1
Tools/machines/pitzer-osc/install_v100_dependencies.sh:  -Dgpu_backend=cuda \
Tools/machines/pitzer-osc/install_cpu_dependencies.sh:  -Dgpu_backend=OFF \
Tools/machines/taurus-zih/taurus_warpx.profile.example:module load CUDA/11.8.0
Tools/machines/taurus-zih/taurus_warpx.profile.example:alias getNode="salloc --time=2:00:00 -N1 -n1 --cpus-per-task=6 --mem-per-cpu=2048 --gres=gpu:1 --gpu-bind=single:1 -p alpha-interactive --pty bash"
Tools/machines/taurus-zih/taurus_warpx.profile.example:alias runNode="srun --time=2:00:00 -N1 -n1 --cpus-per-task=6 --mem-per-cpu=2048 --gres=gpu:1 --gpu-bind=single:1 -p alpha-interactive --pty bash"
Tools/machines/taurus-zih/taurus_warpx.profile.example:# optimize CUDA compilation for A100
Tools/machines/taurus-zih/taurus_warpx.profile.example:export AMREX_CUDA_ARCH=8.0
Tools/machines/taurus-zih/taurus_warpx.profile.example:#export CUDACXX=$(which nvcc)
Tools/machines/taurus-zih/taurus_warpx.profile.example:#export CUDAHOSTCXX=${CXX}
Tools/machines/taurus-zih/taurus.sbatch:#SBATCH --gres=gpu:1
Tools/machines/taurus-zih/taurus.sbatch:#SBATCH --gpu-bind=single:1
Tools/machines/lxplus-cern/spack.yaml:  # CUDA
Tools/machines/lxplus-cern/spack.yaml:  - cuda: []
Tools/machines/lxplus-cern/spack.yaml:  - cuda: [cuda, blaspp +cuda, lapackpp ^blaspp +cuda]
Tools/machines/lxplus-cern/spack.yaml:    when: env.get("SPACK_STACK_USE_CUDA", "1") == "1"
Tools/machines/lxplus-cern/spack.yaml:  - cuda: [cuda, blaspp, lapackpp]
Tools/machines/lxplus-cern/spack.yaml:    when: env.get("SPACK_STACK_USE_CUDA", "1") != "1"
Tools/machines/lxplus-cern/spack.yaml:  - $cuda
Tools/machines/lxplus-cern/lxplus_warpx.profile.example:    export SPACK_STACK_USE_CUDA=1
Tools/machines/lxplus-cern/lxplus_warpx.profile.example:    spack env create warpx-lxplus-cuda-py $WORK/warpx/Tools/machines/lxplus-cern/spack.yaml
Tools/machines/lxplus-cern/lxplus_warpx.profile.example:    spack env activate warpx-lxplus-cuda-py
Tools/machines/lxplus-cern/lxplus_warpx.profile.example:    spack env activate warpx-lxplus-cuda-py
Tools/machines/lxplus-cern/lxplus_warpx.profile.example:export AMREX_CUDA_ARCH="7.0;7.5"
Tools/machines/lxplus-cern/lxplus_warpx.profile.example:export CUDACXX=$(which nvcc)
Tools/machines/lxplus-cern/lxplus_warpx.profile.example:export CUDAHOSTCXX=$(which g++)
Tools/PostProcessing/plot_distribution_mapping.py:        # Either 9 or 10 depending if GPU
Tools/PostProcessing/plot_distribution_mapping.py:        #           (, gpu_ID_box_0 if GPU run), hostname_box_0,
Tools/PostProcessing/plot_distribution_mapping.py:        #           (, gpu_ID_box_1 if GPU run), hostname_box_1
Tools/PostProcessing/plot_distribution_mapping.py:        #           (, gpu_ID_box_n if GPU run), hostname_box_n
GNUmakefile:USE_GPU   = FALSE

```
