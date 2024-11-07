# https://github.com/jacobblum/simDRIFT

```console
README.rst:This library, ``simDRIFT``, provides for rapid and flexible Monte Carlo simulations of Pulsed Gradient Spin Echo (PGSE) diffusion-weighted magnetic resonance imaging (dMRI) experiments, which we expect to be useful for dMRI signal processing model development and validation purposes. The primary focus of this library is forward simulations of molecular self-diffusion processes within an ensemble of nuclear magnetic resonance (NMR) active nuclei ("spins") residing in complex, biophysical tissue systems. ``simDRIFT`` is written in Python and supported by a Numba backend. Thus, ``simDRIFT`` benefits from Numba's CUDA API, allowing individual spin trajectories to be simulated in parallel on single Graphics Processing Unit (GPU) threads. The resulting performance gains support ``simDRIFT``'s aim to provide a customizable tool for rapidly prototyping diffusion models, ground-truth model validation, and in silico phantom production.
README.rst:``simDRIFT`` is compatible with Python 3.8 or later, and requires a CUDA device with a compute capability of 3 or higher. We find that in typical use-case simulations on isotropic imaging voxels on the micometer size scale, ``simDRIFT`` will use less than 1.5 Gb of VRAM. For much larger simulations of imaging voxels on the millimeter size scale, typical GPU memory consumption doesn't exceed 2.0 Gb. Thus, we don't anticipate any memory issues given the available memory of compatible GPUs. 
README.rst:After numba has been installed, please download and install the appropriate `NVIDIA Drivers <https://www.nvidia.com/Download/index.aspx>`_ . Afer the driver installation is complete, install ``cudatoolkit``:
README.rst:  (simDRIFT) >conda install cudatoolkit
README.rst:Also, please install the appropriate version of `pytorch <https://pytorch.org>`_ (shown below using the syntax for our lab's GPUs; however, the specific syntax will vary based on your CUDA Toolkit version) :
README.rst:  (simDRIFT) >conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
docs/source/test_suite/index.rst:As a remark, the DWI forward simulations run here are relatively small-scale, and thus you may not notice a significant increase in GPU utilization while the test suite runs.
docs/source/installation/index.rst:``simDRIFT`` is compatible with Python 3.8 or later and requires a CUDA device with a compute capability of 3 or higher. We find that in typical use-case simulations on isotropic imaging voxels on the micrometer size scale, ``simDRIFT`` will use less than 1.5 Gb of VRAM. For much larger simulations of image voxels on the millimeter size scale, typical GPU memory consumption doesn't exceed 2.0 Gb. Thus, we don't anticipate any memory issues given the available memory of compatible GPUs. 
docs/source/installation/index.rst:After numba has been installed, please download and install the appropriate `NVIDIA Drivers <https://www.nvidia.com/Download/index.aspx>`_ . Afer the driver installation is complete, install ``cudatoolkit``:
docs/source/installation/index.rst:  (simDRIFT) >conda install cudatoolkit
docs/source/installation/index.rst:Also, please install the appropriate version of `pytorch <https://pytorch.org>`_ (shown below using the syntax for our lab's GPUs; however, the specific syntax will vary based on your CUDA Toolkit version) :
docs/source/installation/index.rst:  (simDRIFT) >conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
docs/source/theory/index.rst:graphical processing unit (GPU), thus allowing for a non-linear relationship between the number of spins populated in the simulated voxel and overall runtime of the simulation. Typical experiements feature :math:`{[0.25,\, 1] \times 10^6}` spins and are completed within :math:`\sim 15` and :math:`60` minutes, depending on the complexity of the simulated microstructure. 
docs/source/quickstart/index.rst:If you have an NVIDA-Cuda capable GPU, install ``simDRIFT`` by following these `instructions <https://simdrift.readthedocs.io/en/latest/installation/index.html>`_ . If not, scroll to the bottom of the page and we will breifly go through an example using `Google Colab <https://colab.research.google.com/?utm_source=scs-index>`_. 
docs/source/quickstart/index.rst:First, open a new Google Colab notebook. Then, nagivate to Edit> Notebook Settings and change the ``Hardware Accelorator`` to GPU. 
docs/source/troubleshooting/index.rst:After numba has been installed, please download and install the appropriate `NVIDIA Drivers <https://www.nvidia.com/Download/index.aspx>`_ . Afer the driver installation is complete, we will test the numba install to confirm everything is working. Launch a Python session
docs/source/troubleshooting/index.rst:Now, type the following commands. If the installation is correct (in the sense that ``numba`` can send data to the GPU), then the output should look something like this:
docs/source/troubleshooting/index.rst:  >>> from numba import cuda
docs/source/troubleshooting/index.rst:  >>> print(cuda.to_device([1]) 
docs/source/troubleshooting/index.rst:  <numba.cuda.cudadrv.devicearray.DeviceNDArray object at ....>
docs/source/troubleshooting/index.rst:After this step, installation proceeds as usual. In particular, please install the appropriate version of `pytorch <https://pytorch.org>`_ (shown below using the syntax for our lab's GPUs; however, the specific syntax will vary based on your CUDA Toolkit version) :
docs/source/troubleshooting/index.rst:  (simDRIFT) >conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
joss/paper.md:  - CUDA
joss/paper.md:This library, `simDRIFT`, provides for rapid and flexible Monte-Carlo simulations of Pulsed Gradient Spin Echo (PGSE) Diffusion-Weighted Magnetic Resonance Imaging (DWI) experiments, which we expect to be useful for DWI signal processing model development and validation purposes. The primary focus of this library is forward simulations of molecular self-diffusion processes within an ensemble of nuclear magnetic resonance (NMR) active nuclei ("spins") residing in complex, biophysical tissue systems. To achieve a large variety of tissue configurations, `simDRIFT` provides support for $n$ fiber bundles (with user-defined radii, intrinsic diffusivities, orientation angles, and densities) and $m$ cells (with user-defined radii and volume fractions). `simDrift` is written in Python (Python Software Foundation [@VanRossum2010]) and supported by a Numba [@Lam2015] backend. Thus, `simDRIFT` benefits from Numba's CUDA API, allowing the simulation of individual spin trajectories to be performed in parallel on single Graphics Processing Unit (GPU) threads. The resulting performance gains support `simDRIFT`'s aim to provide a customizable tool for the rapid prototyping of diffusion models, ground-truth model validation, and in silico phantom production.
joss/paper.md:``simDRIFT``'s superior computational performance represents an important advantage relative to other available open-source software. Comparisons between simDRIFT and Disimpy, a mesh-based, CUDA-enabled DWI forward simulator known to be faster than CAMINO [@Kerkelae2020], on identical image voxel geometries featuring a 5 $\mu$m radius cell, reveals orders of magnitude improved performance, particularly for large resident spin ensemble sizes. Thus, ``simDRIFT`` is especially useful in a regime where the computational cost of existing software may be cumbersome or even prohibitive. In particular, ``simDRIFT`` is able to perform large-scale simulations very quickly, therefore benefiting from favorable convergence properties with respect to the diffusion weighted signal. In this context, using just desktop- or even laptop-level GPU hardware, ``simDRIFT`` users are able to quickly and easily generate large amounts of synthetic DWI data from a wide variety of voxel geometries.
joss/paper.md:![Runtime comparison between simDRIFT and Disimpy, another DWI simulator that runs on the GPU. These simulations were performed on a Windows 10 desktop with an Nvidia RTX 3090 GPU.\label{fig:performance}](figs/simDRIFT_vs_Disimpy.png){ width=70% }
CONTRIBUTING.md:- Version information for Python, CUDA, and PyTorch
src/simulation.py:from numba import jit, njit, cuda, int32, float32
src/cli.py:        ## Check that GPU is available 
src/cli.py:            assert numba.cuda.is_available(), "Trying to use Cuda device, " \
src/cli.py:                                            "but Cuda device is not available."
src/save.py:    """Calculates the PGSE signal from the forward simulated spin trajectories [3]_. Note that this computation is executed on the GPU using PyTorch.
src/jp/curvature.py:from numba import jit, cuda
src/jp/curvature.py:from numba.cuda import random 
src/jp/curvature.py:from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
src/jp/random.py:from numba import jit, cuda
src/jp/random.py:from numba.cuda import random 
src/jp/random.py:from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
src/jp/random.py:@cuda.jit(device = True,nopython = False)
src/jp/linalg.py:from numba import jit, cuda
src/jp/linalg.py:from numba.cuda import random 
src/jp/linalg.py:from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
src/jp/linalg.py:@cuda.jit(device = True,nopython = False)
src/jp/linalg.py:@cuda.jit(device = True, nopython = False)
src/jp/linalg.py:@cuda.jit(device = True,nopython = False)
src/jp/linalg.py:@cuda.jit(device = True,nopython = False)
src/jp/linalg.py:    c = cuda.local.array(shape = A.shape[0], dtype = numba.float32)
src/jp/linalg.py:@cuda.jit(device = True,nopython = False)
src/jp/linalg.py:    :type  a: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :type  b: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :rtype: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:@cuda.jit(device = True,nopython = False)
src/jp/linalg.py:    :type  a: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :type  b: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/jp/linalg.py:    :rtype: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/diffusion.py:from numba import jit, njit, cuda
src/physics/diffusion.py:from numba.cuda import random 
src/physics/diffusion.py:from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
src/physics/diffusion.py:def _package_data(self) -> Dict[str, Dict[str, Type[numba.cuda.cudadrv.devicearray.DeviceNDArray]]]:
src/physics/diffusion.py:    #                              Send Data to GPU                                     #
src/physics/diffusion.py:        outputArgs[k]['data'] = cuda.to_device(
src/physics/diffusion.py:        At each iteration, the updated spin position is written to ``spin_positions_cuda``
src/physics/diffusion.py:    random_states_cuda = cuda.to_device(create_xoroshiro128p_states(len(self.spins), seed = 42))  
src/physics/diffusion.py:            _diffusion_context_manager[blocks_per_grid,threads_per_block](random_states_cuda, 
src/physics/diffusion.py:            cuda.synchronize()
src/physics/diffusion.py:            cuda.synchronize()
src/physics/diffusion.py:            _diffusion_context_manager[blocks_per_grid,threads_per_block](random_states_cuda, 
src/physics/diffusion.py:            cuda.synchronize()
src/physics/diffusion.py:            cuda.synchronize()
src/physics/diffusion.py:@numba.cuda.jit
src/physics/diffusion.py:                               theta_cuda,
src/physics/diffusion.py:    i = cuda.grid(1)
src/physics/diffusion.py:                                            theta_cuda[spin_in_fiber_at_index[i]],
src/physics/diffusion.py:                                            theta_cuda, 
src/physics/diffusion.py:                                          theta_cuda,
src/physics/diffusion.py:@numba.cuda.jit
src/physics/diffusion.py:    i = cuda.grid(1)
src/physics/walk_in_fiber.py:from numba import jit, cuda
src/physics/walk_in_fiber.py:@numba.cuda.jit(nopython=True,parallel=True)
src/physics/walk_in_fiber.py:    :type random_states: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_fiber.py:    :type fiber_center: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_fiber.py:    :type fiber_radius: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_fiber.py:    :type fiber_direction: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_fiber.py:    :type spin_positions: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_fiber.py:    previous_position       = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_fiber.py:    proposed_new_position   = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_fiber.py:    dynamic_fiber_center    = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_fiber.py:    dynamic_fiber_direction = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_water.py:from numba import jit, cuda
src/physics/walk_in_water.py:@numba.cuda.jit(nopython=True, parallel=True)
src/physics/walk_in_water.py:    :type random_states: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_water.py:    :type fiber_centers: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_water.py:    :type fiber_directions: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_water.py:    :type fiber_radii: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_water.py:    :type cell_centers: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_water.py:    :type spin_positions: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_water.py:    previous_position = cuda.local.array(shape=3, dtype=numba.float32)
src/physics/walk_in_water.py:    proposed_new_position = cuda.local.array(shape=3, dtype=numba.float32)
src/physics/walk_in_water.py:    dynamic_fiber_center    = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_water.py:    dynamic_fiber_direction = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_water.py:    u3 = cuda.local.array(shape=3, dtype=numba.float32)
src/physics/walk_in_cell.py:from numba import jit, cuda
src/physics/walk_in_cell.py:@numba.cuda.jit(nopython=True)
src/physics/walk_in_cell.py:    :type random_states: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_cell.py:    :type cell_center: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_cell.py:    :type fiber_centers: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_cell.py:    :type fiber_radii: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_cell.py:    :type fiber_directions: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_cell.py:    :type spin_positions: numba.cuda.cudadrv.devicearray.DeviceNDArray
src/physics/walk_in_cell.py:    previous_position = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_cell.py:    proposed_new_position = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_cell.py:    u3 = cuda.local.array(shape = 3, dtype= numba.float32)
src/physics/walk_in_cell.py:    dynamic_fiber_center    = cuda.local.array(shape = 3, dtype = numba.float32)
src/physics/walk_in_cell.py:    dynamic_fiber_direction = cuda.local.array(shape = 3, dtype = numba.float32)
src/setup/spin_init_positions.py:from numba import jit, njit, cuda
src/setup/spin_init_positions.py:from numba.cuda import random 
src/setup/spin_init_positions.py:from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
src/setup/spin_init_positions.py:@numba.cuda.jit
src/setup/spin_init_positions.py:def _find_spin_locations_kernel(resident_fiber_indxs_cuda: numba.cuda.cudadrv.devicearray.DeviceNDArray, 
src/setup/spin_init_positions.py:                                resident_cell_indxs_cuda:  numba.cuda.cudadrv.devicearray.DeviceNDArray , 
src/setup/spin_init_positions.py:                                fiber_centers_cuda:        numba.cuda.cudadrv.devicearray.DeviceNDArray,
src/setup/spin_init_positions.py:                                fiber_directions_cuda:     numba.cuda.cudadrv.devicearray.DeviceNDArray,
src/setup/spin_init_positions.py:                                fiber_radii_cuda:          numba.cuda.cudadrv.devicearray.DeviceNDArray,
src/setup/spin_init_positions.py:                                cell_centers_cuda:         numba.cuda.cudadrv.devicearray.DeviceNDArray,
src/setup/spin_init_positions.py:                                cell_radii_cuda:           numba.cuda.cudadrv.devicearray.DeviceNDArray,
src/setup/spin_init_positions.py:                                spin_positions_cuda:       numba.cuda.cudadrv.devicearray.DeviceNDArray,
src/setup/spin_init_positions.py:                                theta_cuda,
src/setup/spin_init_positions.py:    :param resident_fiber_indxs_cuda: Array to write resident fiber indices to
src/setup/spin_init_positions.py:    :type resident_fiber_indxs_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    :param resident_cell_indxs_cuda: Array to write resident cell indices to
src/setup/spin_init_positions.py:    :type resident_cell_indxs_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    :param fiber_centers_cuda:  Coordinates for the centers of each fiber
src/setup/spin_init_positions.py:    :type fiber_centers_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    :param fiber_directions_cuda: Direction of each fiber
src/setup/spin_init_positions.py:    :type fiber_directions_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    :param fiber_radii_cuda: Radii of each fiber
src/setup/spin_init_positions.py:    :type fiber_radii_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    :param cell_centers_cuda: Coordinates for the centers of each cell
src/setup/spin_init_positions.py:    :type cell_centers_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    :param cell_radii_cuda: Radii of each cell
src/setup/spin_init_positions.py:    :type cell_radii_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    :param spin_positions_cuda: Initial spin positions
src/setup/spin_init_positions.py:    :type spin_positions_cuda: CUDA Device Array
src/setup/spin_init_positions.py:    i = cuda.grid(1)
src/setup/spin_init_positions.py:    if i > spin_positions_cuda.shape[0]:
src/setup/spin_init_positions.py:    u3                      = cuda.local.array(shape = 3, dtype= numba.float32)
src/setup/spin_init_positions.py:    dynamic_fiber_center    = cuda.local.array(shape = 3, dtype = numba.float32)
src/setup/spin_init_positions.py:    dynamic_fiber_direction = cuda.local.array(shape = 3, dtype = numba.float32)
src/setup/spin_init_positions.py:    for j in range(fiber_centers_cuda.shape[0]):
src/setup/spin_init_positions.py:        dynamic_fiber_center = linalg.gamma(spin_positions_cuda[i, :], 
src/setup/spin_init_positions.py:                                            fiber_directions_cuda[j,:], 
src/setup/spin_init_positions.py:                                            theta_cuda[j],
src/setup/spin_init_positions.py:        dynamic_fiber_direction = linalg.d_gamma__d_t(spin_positions_cuda[i, :],
src/setup/spin_init_positions.py:                                                      fiber_directions_cuda[j,:], 
src/setup/spin_init_positions.py:                                                      theta_cuda[j],
src/setup/spin_init_positions.py:            dynamic_fiber_center[k] = dynamic_fiber_center[k] + fiber_centers_cuda[j, k]
src/setup/spin_init_positions.py:        dFv = linalg.dL2(spin_positions_cuda[i, :], 
src/setup/spin_init_positions.py:        if dFv < fiber_radii_cuda[j]:
src/setup/spin_init_positions.py:            resident_fiber_indxs_cuda[i] = j
src/setup/spin_init_positions.py:    for j in range(cell_centers_cuda.shape[0]):
src/setup/spin_init_positions.py:       dC = linalg.dL2(spin_positions_cuda[i,:], cell_centers_cuda[j,:], u3, False)
src/setup/spin_init_positions.py:       if dC < cell_radii_cuda[j]:
src/setup/spin_init_positions.py:           resident_cell_indxs_cuda[i] = j
src/setup/spin_init_positions.py:    resident_fiber_indxs_cuda = cuda.to_device( -1 * np.ones(shape = (len(self.spins),), dtype= np.int32))
src/setup/spin_init_positions.py:    resident_cell_indxs_cuda  = cuda.to_device( -1 * np.ones(shape = (len(self.spins),), dtype= np.int32))
src/setup/spin_init_positions.py:    fiber_centers_cuda        = cuda.to_device(np.array([fiber._get_center() for fiber in self.fibers], dtype= np.float32))
src/setup/spin_init_positions.py:    fiber_directions_cuda     = cuda.to_device(np.array([fiber._get_direction() for fiber in self.fibers], dtype= np.float32))
src/setup/spin_init_positions.py:    fiber_radii_cuda          = cuda.to_device(np.array([fiber._get_radius() for fiber in self.fibers], dtype= np.float32))
src/setup/spin_init_positions.py:    cell_centers_cuda         = cuda.to_device(np.array([cell._get_center() for cell in self.cells], dtype= np.float32))
src/setup/spin_init_positions.py:    cell_radii_cuda           = cuda.to_device(np.array([cell._get_radius() for cell in self.cells], dtype=np.float32))
src/setup/spin_init_positions.py:    spin_positions_cuda       = cuda.to_device(np.array([spin._get_position_t1m() for spin in self.spins], dtype= np.float32))
src/setup/spin_init_positions.py:    theta_cuda                = cuda.to_device(np.array([fiber.theta for fiber in self.fibers], dtype= np.float32))
src/setup/spin_init_positions.py:    curvature_params          = cuda.to_device(np.array([[fiber.__dict__['kappa'], fiber.__dict__['L'], fiber.__dict__['A'], fiber.__dict__['P']] for fiber in self.fibers], dtype = np.float32))
src/setup/spin_init_positions.py:    _find_spin_locations_kernel[blocks_per_grid,threads_per_block](resident_fiber_indxs_cuda,
src/setup/spin_init_positions.py:                                                                   resident_cell_indxs_cuda,
src/setup/spin_init_positions.py:                                                                   fiber_centers_cuda,
src/setup/spin_init_positions.py:                                                                   fiber_directions_cuda,
src/setup/spin_init_positions.py:                                                                   fiber_radii_cuda,
src/setup/spin_init_positions.py:                                                                   cell_centers_cuda,
src/setup/spin_init_positions.py:                                                                   cell_radii_cuda,
src/setup/spin_init_positions.py:                                                                   spin_positions_cuda,
src/setup/spin_init_positions.py:                                                                   theta_cuda,
src/setup/spin_init_positions.py:    resident_fiber_indxs = resident_fiber_indxs_cuda.copy_to_host()
src/setup/spin_init_positions.py:    resident_cell_indxs  = resident_cell_indxs_cuda.copy_to_host()

```
