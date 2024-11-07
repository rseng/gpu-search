# https://github.com/tberlok/paicos

```console
.pytest.ini:    tests/cuda-gpu
docs/source/examples.md:## Main GPU classes
docs/source/examples.md:The main GPU classes are
docs/source/examples.md:- GpuSphProjector (gpu_sph_projector.py)
docs/source/examples.md:- GpuRayProjector (gpu_ray_projector.py)
docs/source/tests.md:pytest cuda-gpu
docs/source/installation.md:## GPU/CUDA requirements
docs/source/installation.md:The visualization routines that run on GPU require installing CuPy (a drop-in replacement
docs/source/installation.md:for NumPy that runs on the GPU) and Numba CUDA (just-in-time compilation of kernel
docs/source/installation.md:and device functions on the GPU). These packages only work on CUDA-enabled GPUs,
docs/source/installation.md:which means that you need a recent Nvidia GPU. An Nvidia GPU with good FP64 performance
docs/source/installation.md:- Numba: https://numba.readthedocs.io/en/stable/cuda/overview.html#supported-gpus
docs/source/installation.md:At the time of writing, we have had success installing for CUDA version
docs/source/installation.md:pip install cupy-cuda112
docs/source/installation.md:and then setting the path to the CUDA installation in .bashrc as e.g.
docs/source/installation.md:(substitute with the path to the CUDA installation on your system)
docs/source/installation.md:export CUDA_HOME=/software/astro/cuda/11.2 # numba
docs/source/installation.md:export CUDA_PATH=/software/astro/cuda/11.2 # cupy
docs/source/installation.md:# Whether to load GPU/cuda functionality on startup
docs/source/installation.md:pa.load_cuda_functionality_on_startup(True)
docs/source/configuration.md:# Whether to load GPU/cuda functionality on startup
docs/source/configuration.md:pa.load_cuda_functionality_on_startup(False)
docs/source/index.md:/notebooks/notebook5a_sph_projection_on_the_gpu.ipynb
docs/source/index.md:/notebooks/notebook5b_ray_tracing_on_the_gpu.ipynb
docs/source/index.md:/notebooks/notebook6_interactive_visualization_on_the_gpu.ipynb
tests/cuda-gpu/test_gpu_binary_tree.py:def test_gpu_binary_tree():
tests/cuda-gpu/test_gpu_binary_tree.py:               + " and GPU-tests can't run.")
tests/cuda-gpu/test_gpu_binary_tree.py:    from paicos.trees.bvh_gpu import GpuBinaryTree
tests/cuda-gpu/test_gpu_binary_tree.py:    bvh_tree = GpuBinaryTree(pos, 1.2 * sizes)
tests/cuda-gpu/test_gpu_binary_tree.py:    test_gpu_binary_tree()
tests/cuda-gpu/test_gpu_sph_projector.py:def test_gpu_sph_projector(show=False):
tests/cuda-gpu/test_gpu_sph_projector.py:    We compare the CPU and GPU implementations of SPH-projection.
tests/cuda-gpu/test_gpu_sph_projector.py:               + " and GPU-tests can't run.")
tests/cuda-gpu/test_gpu_sph_projector.py:    pa.gpu_init()
tests/cuda-gpu/test_gpu_sph_projector.py:    gpu_projector = pa.GpuSphProjector(snap, center, widths, orientation,
tests/cuda-gpu/test_gpu_sph_projector.py:    gpu_dens = gpu_projector.project_variable(
tests/cuda-gpu/test_gpu_sph_projector.py:        '0_Masses') / gpu_projector.project_variable('0_Volume')
tests/cuda-gpu/test_gpu_sph_projector.py:    max_rel_err = np.max(np.abs(dens.value - gpu_dens.value) / dens.value)
tests/cuda-gpu/test_gpu_sph_projector.py:            gpu_dens.value, extent=projector.centered_extent.value, norm=LogNorm())
tests/cuda-gpu/test_gpu_sph_projector.py:        axes[2].imshow(np.abs(dens.value - gpu_dens.value) / dens.value,
tests/cuda-gpu/test_gpu_sph_projector.py:    test_gpu_sph_projector(True)
tests/cuda-gpu/test_gpu_ray_projector.py:def test_gpu_ray_projector(show=False):
tests/cuda-gpu/test_gpu_ray_projector.py:    We compare the CPU and GPU implementations of ray-tracing.
tests/cuda-gpu/test_gpu_ray_projector.py:               + " and GPU-tests can't run.")
tests/cuda-gpu/test_gpu_ray_projector.py:    pa.gpu_init()
tests/cuda-gpu/test_gpu_ray_projector.py:    gpu_projector = pa.GpuRayProjector(snap, center, widths, orientation, npix=npix,
tests/cuda-gpu/test_gpu_ray_projector.py:    gpu_dens = gpu_projector.project_variable('0_Density', additive=False)
tests/cuda-gpu/test_gpu_ray_projector.py:    max_rel_err = np.max(np.abs(tree_dens.value - gpu_dens.value) / tree_dens.value)
tests/cuda-gpu/test_gpu_ray_projector.py:        axes[1].imshow(gpu_dens.value, extent=tree_projector.centered_extent.value,
tests/cuda-gpu/test_gpu_ray_projector.py:        axes[2].imshow(np.abs(tree_dens.value - gpu_dens.value) / tree_dens.value,
tests/cuda-gpu/test_gpu_ray_projector.py:        # plt.savefig('gpu_ray_tracer_test.png')
tests/cuda-gpu/test_gpu_ray_projector.py:    test_gpu_ray_projector(True)
Makefile:	cd tests; pytest cuda-gpu
paicos/settings.py:# Whether to load GPU/cuda functionality on startup
paicos/settings.py:load_cuda_functionality_on_startup = False
paicos/paicos_user_settings_template.py:# Whether to load GPU/cuda functionality on startup
paicos/paicos_user_settings_template.py:pa.load_cuda_functionality_on_startup(False)
paicos/trees/bvh_cpu.py:    https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
paicos/trees/bvh_cpu.py:    This code has has been adapted from the CornerStone CUDA/C++ code,
paicos/trees/bvh_gpu.py:from numba import cuda
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:@cuda.jit
paicos/trees/bvh_gpu.py:    ip = cuda.grid(1)
paicos/trees/bvh_gpu.py:@cuda.jit
paicos/trees/bvh_gpu.py:    ip = cuda.grid(1)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:    # https://docs.nvidia.com/cuda/'libdevice-users-guide/__nv_clzll.html#__nv_clzll
paicos/trees/bvh_gpu.py:    return cuda.libdevice.clzll(tmp)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:    commonPrefix = cuda.libdevice.clzll(firstCode ^ lastCode)
paicos/trees/bvh_gpu.py:            splitPrefix = cuda.libdevice.clzll(firstCode ^ splitCode)
paicos/trees/bvh_gpu.py:@cuda.jit()
paicos/trees/bvh_gpu.py:    https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
paicos/trees/bvh_gpu.py:    idx = cuda.grid(1)
paicos/trees/bvh_gpu.py:    cuda.syncthreads()
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:    This code has has been adapted from the CornerStone CUDA/C++ code,
paicos/trees/bvh_gpu.py:@cuda.jit()
paicos/trees/bvh_gpu.py:    ip = cuda.grid(1)
paicos/trees/bvh_gpu.py:@cuda.jit
paicos/trees/bvh_gpu.py:    ip = cuda.grid(1)
paicos/trees/bvh_gpu.py:            cuda.atomic.min(tree_bounds, (next_parent, 0, 0), x_min)
paicos/trees/bvh_gpu.py:            cuda.atomic.min(tree_bounds, (next_parent, 1, 0), y_min)
paicos/trees/bvh_gpu.py:            cuda.atomic.min(tree_bounds, (next_parent, 2, 0), z_min)
paicos/trees/bvh_gpu.py:            cuda.atomic.max(tree_bounds, (next_parent, 0, 1), x_max)
paicos/trees/bvh_gpu.py:            cuda.atomic.max(tree_bounds, (next_parent, 1, 1), y_max)
paicos/trees/bvh_gpu.py:            cuda.atomic.max(tree_bounds, (next_parent, 2, 1), z_max)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:@cuda.jit(device=True, inline=True)
paicos/trees/bvh_gpu.py:    queue = cuda.local.array(128, numba.int64)
paicos/trees/bvh_gpu.py:@cuda.jit
paicos/trees/bvh_gpu.py:    ip = cuda.grid(1)
paicos/trees/bvh_gpu.py:class GpuBinaryTree:
paicos/trees/bvh_gpu.py:        # TODO: check whether query points are already on GPU
paicos/trees/bvh_gpu.py:        self.release_gpu_memory()
paicos/trees/bvh_gpu.py:    def release_gpu_memory(self):
paicos/trees/bvh_gpu.py:    bvh_tree = GpuBinaryTree(pos, 1.2 * sizes)
paicos/image_creators/gpu_ray_projector.py:from numba import cuda
paicos/image_creators/gpu_ray_projector.py:from ..trees.bvh_gpu import GpuBinaryTree
paicos/image_creators/gpu_ray_projector.py:from ..trees.bvh_gpu import nearest_neighbor_device
paicos/image_creators/gpu_ray_projector.py:from ..trees.bvh_gpu import is_point_in_box, distance
paicos/image_creators/gpu_ray_projector.py:@cuda.jit(device=True, inline=True)
paicos/image_creators/gpu_ray_projector.py:@cuda.jit
paicos/image_creators/gpu_ray_projector.py:    ix, iy = cuda.grid(2)
paicos/image_creators/gpu_ray_projector.py:        query_point = numba.cuda.local.array(3, numba.float64)
paicos/image_creators/gpu_ray_projector.py:        tmp_point = numba.cuda.local.array(3, numba.float64)
paicos/image_creators/gpu_ray_projector.py:class GpuRayProjector(ImageCreator):
paicos/image_creators/gpu_ray_projector.py:    It only works on cuda-enabled GPUs.
paicos/image_creators/gpu_ray_projector.py:                           + "you have turned on. If your GPU has enough memory "
paicos/image_creators/gpu_ray_projector.py:        # Send subset of snapshot to GPU
paicos/image_creators/gpu_ray_projector.py:            self._send_data_to_gpu()
paicos/image_creators/gpu_ray_projector.py:            self.tree = GpuBinaryTree(self.gpu_variables['pos'],
paicos/image_creators/gpu_ray_projector.py:                                      self.gpu_variables['hsml'])
paicos/image_creators/gpu_ray_projector.py:            del self.gpu_variables['pos']
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['hsml'] = self.gpu_variables['hsml'][self.tree.sort_index]
paicos/image_creators/gpu_ray_projector.py:        # Send entirety of snapshot to GPU (if we have not already
paicos/image_creators/gpu_ray_projector.py:                self._send_data_to_gpu()
paicos/image_creators/gpu_ray_projector.py:                self.tree = GpuBinaryTree(self.gpu_variables['pos'],
paicos/image_creators/gpu_ray_projector.py:                                          self.gpu_variables['hsml'])
paicos/image_creators/gpu_ray_projector.py:                del self.gpu_variables['pos']
paicos/image_creators/gpu_ray_projector.py:                self.gpu_variables['hsml'] = self.gpu_variables['hsml'][
paicos/image_creators/gpu_ray_projector.py:        self._send_small_data_to_gpu()
paicos/image_creators/gpu_ray_projector.py:    def _send_data_to_gpu(self):
paicos/image_creators/gpu_ray_projector.py:        self.gpu_variables = {}
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['pos'] = cp.array(self.pos.value)
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['hsml'] = cp.array(self.hsml.value)
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['pos'] = cp.array(self.pos)
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['hsml'] = cp.array(self.hsml)
paicos/image_creators/gpu_ray_projector.py:        self._send_small_data_to_gpu()
paicos/image_creators/gpu_ray_projector.py:    def _send_small_data_to_gpu(self):
paicos/image_creators/gpu_ray_projector.py:        self.gpu_variables['rotation_matrix'] = cp.array(
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['widths'] = cp.array(self.widths.value)
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['center'] = cp.array(self.center.value)
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['widths'] = cp.array(self.widths)
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables['center'] = cp.array(self.center)
paicos/image_creators/gpu_ray_projector.py:    def _gpu_project(self, variable_str, additive):
paicos/image_creators/gpu_ray_projector.py:        Private method for projecting using cuda code
paicos/image_creators/gpu_ray_projector.py:        gpu_vars = self.gpu_variables
paicos/image_creators/gpu_ray_projector.py:        rotation_matrix = gpu_vars['rotation_matrix']
paicos/image_creators/gpu_ray_projector.py:        widths = gpu_vars['widths']
paicos/image_creators/gpu_ray_projector.py:        center = gpu_vars['center']
paicos/image_creators/gpu_ray_projector.py:        hsml = gpu_vars['hsml']
paicos/image_creators/gpu_ray_projector.py:            variable = gpu_vars[variable_str] / gpu_vars[f'{self.parttype}_Volume']
paicos/image_creators/gpu_ray_projector.py:            variable = gpu_vars[variable_str]
paicos/image_creators/gpu_ray_projector.py:    def _send_variable_to_gpu(self, variable, gpu_key='projection_variable'):
paicos/image_creators/gpu_ray_projector.py:        if variable_str in self.gpu_variables and variable_str != gpu_key:
paicos/image_creators/gpu_ray_projector.py:            # Send variable to gpu
paicos/image_creators/gpu_ray_projector.py:                self.gpu_variables[variable_str] = cp.array(variable.value)
paicos/image_creators/gpu_ray_projector.py:                self.gpu_variables[variable_str] = cp.array(variable)
paicos/image_creators/gpu_ray_projector.py:            self.gpu_variables[variable_str] = self.gpu_variables[variable_str][
paicos/image_creators/gpu_ray_projector.py:        variable_str, unit_quantity = self._send_variable_to_gpu(variable)
paicos/image_creators/gpu_ray_projector.py:            _, vol_unit_quantity = self._send_variable_to_gpu(f'{self.parttype}_Volume')
paicos/image_creators/gpu_ray_projector.py:        projection = self._gpu_project(variable_str, additive)
paicos/image_creators/gpu_ray_projector.py:        self.release_gpu_memory()
paicos/image_creators/gpu_ray_projector.py:    def release_gpu_memory(self):
paicos/image_creators/gpu_ray_projector.py:        if hasattr(self, 'gpu_variables'):
paicos/image_creators/gpu_ray_projector.py:            for key in list(self.gpu_variables):
paicos/image_creators/gpu_ray_projector.py:                del self.gpu_variables[key]
paicos/image_creators/gpu_ray_projector.py:            del self.gpu_variables
paicos/image_creators/gpu_ray_projector.py:            self.tree.release_gpu_memory()
paicos/image_creators/gpu_sph_projector.py:from numba import cuda
paicos/image_creators/gpu_sph_projector.py:@cuda.jit(device=True, inline=True)
paicos/image_creators/gpu_sph_projector.py:@cuda.jit(device=True, inline=True)
paicos/image_creators/gpu_sph_projector.py:@cuda.jit(max_registers=64)
paicos/image_creators/gpu_sph_projector.py:    ip = cuda.grid(1)
paicos/image_creators/gpu_sph_projector.py:                        cuda.atomic.add(image1, (ix, iy), weight / norm)
paicos/image_creators/gpu_sph_projector.py:                        cuda.atomic.add(image2, (ix, iy), weight / norm)
paicos/image_creators/gpu_sph_projector.py:                        cuda.atomic.add(image4, (ix, iy), weight / norm)
paicos/image_creators/gpu_sph_projector.py:                        cuda.atomic.add(image8, (ix, iy), weight / norm)
paicos/image_creators/gpu_sph_projector.py:                        cuda.atomic.add(image16, (ix, iy), weight / norm)
paicos/image_creators/gpu_sph_projector.py:class GpuSphProjector(ImageCreator):
paicos/image_creators/gpu_sph_projector.py:    This GPU implementation of SPH-like projection splits the particles
paicos/image_creators/gpu_sph_projector.py:        # TODO: add split into GPU and CPU based on cell sizes here.
paicos/image_creators/gpu_sph_projector.py:                           + "you have turned on. If your GPU has enough memory "
paicos/image_creators/gpu_sph_projector.py:        # Send subset of snapshot to GPU
paicos/image_creators/gpu_sph_projector.py:            self._send_data_to_gpu()
paicos/image_creators/gpu_sph_projector.py:            self._send_small_data_to_gpu()
paicos/image_creators/gpu_sph_projector.py:        # Send entirety of snapshot to GPU (if we have not already
paicos/image_creators/gpu_sph_projector.py:                self._send_data_to_gpu()
paicos/image_creators/gpu_sph_projector.py:            self._send_small_data_to_gpu()
paicos/image_creators/gpu_sph_projector.py:    def _send_data_to_gpu(self):
paicos/image_creators/gpu_sph_projector.py:        self.gpu_variables = {}
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['x'] = cp.array(self.pos[:, 0].value)
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['y'] = cp.array(self.pos[:, 1].value)
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['z'] = cp.array(self.pos[:, 2].value)
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['hsml'] = cp.array(self.hsml.value)
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['x'] = cp.array(self.pos[:, 0])
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['y'] = cp.array(self.pos[:, 1])
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['z'] = cp.array(self.pos[:, 2])
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['hsml'] = cp.array(self.hsml)
paicos/image_creators/gpu_sph_projector.py:        self._send_small_data_to_gpu()
paicos/image_creators/gpu_sph_projector.py:    def _send_small_data_to_gpu(self):
paicos/image_creators/gpu_sph_projector.py:        self.gpu_variables['unit_vector_x'] = cp.array(unit_vectors['x'])
paicos/image_creators/gpu_sph_projector.py:        self.gpu_variables['unit_vector_y'] = cp.array(unit_vectors['y'])
paicos/image_creators/gpu_sph_projector.py:        self.gpu_variables['unit_vector_z'] = cp.array(unit_vectors['z'])
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['widths'] = cp.array(self.widths.value)
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['center'] = cp.array(self.center.value)
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['widths'] = cp.array(self.widths)
paicos/image_creators/gpu_sph_projector.py:            self.gpu_variables['center'] = cp.array(self.center)
paicos/image_creators/gpu_sph_projector.py:    def _gpu_project(self, variable_str):
paicos/image_creators/gpu_sph_projector.py:        Private method for projecting using cuda code
paicos/image_creators/gpu_sph_projector.py:        unit_vector_x = self.gpu_variables['unit_vector_x']
paicos/image_creators/gpu_sph_projector.py:        unit_vector_y = self.gpu_variables['unit_vector_y']
paicos/image_creators/gpu_sph_projector.py:        unit_vector_z = self.gpu_variables['unit_vector_z']
paicos/image_creators/gpu_sph_projector.py:        x = self.gpu_variables['x']
paicos/image_creators/gpu_sph_projector.py:        y = self.gpu_variables['y']
paicos/image_creators/gpu_sph_projector.py:        z = self.gpu_variables['z']
paicos/image_creators/gpu_sph_projector.py:        widths = self.gpu_variables['widths']
paicos/image_creators/gpu_sph_projector.py:        center = self.gpu_variables['center']
paicos/image_creators/gpu_sph_projector.py:        hsml = self.gpu_variables['hsml']
paicos/image_creators/gpu_sph_projector.py:        variable = self.gpu_variables[variable_str]
paicos/image_creators/gpu_sph_projector.py:        if variable_str in self.gpu_variables and variable_str != 'projection_variable':
paicos/image_creators/gpu_sph_projector.py:            # Send variable to gpu
paicos/image_creators/gpu_sph_projector.py:                self.gpu_variables[variable_str] = cp.array(variable.value)
paicos/image_creators/gpu_sph_projector.py:                self.gpu_variables[variable_str] = cp.array(variable)
paicos/image_creators/gpu_sph_projector.py:        projection = self._gpu_project(variable_str)
paicos/image_creators/gpu_sph_projector.py:        self.release_gpu_memory()
paicos/image_creators/gpu_sph_projector.py:    def release_gpu_memory(self):
paicos/image_creators/gpu_sph_projector.py:        if hasattr(self, 'gpu_variables'):
paicos/image_creators/gpu_sph_projector.py:            for key in list(self.gpu_variables):
paicos/image_creators/gpu_sph_projector.py:                del self.gpu_variables[key]
paicos/image_creators/gpu_sph_projector.py:            del self.gpu_variables
paicos/__init__.py:and a CUDA GPU implementation for visualization.
paicos/__init__.py:def load_cuda_functionality_on_startup(option):
paicos/__init__.py:    Turns on/off whether to import GPU/cuda on startup.
paicos/__init__.py:    settings.load_cuda_functionality_on_startup = option
paicos/__init__.py:# Import of GPU functionality only if
paicos/__init__.py:def gpu_init(gpu_num=0):
paicos/__init__.py:    Calling this function initializes the GPU parts of the code.
paicos/__init__.py:    You can set settings.load_cuda_functionality_on_startup = True,
paicos/__init__.py:        gpu_num (int): The GPU that you want to use for computations,
paicos/__init__.py:                       i.e., we call cupy.cuda.Device(gpu_num).use()
paicos/__init__.py:        from numba import cuda
paicos/__init__.py:        if gpu_num != 0:
paicos/__init__.py:            cp.cuda.Device(gpu_num).use()
paicos/__init__.py:        @cuda.jit
paicos/__init__.py:            pos = cuda.grid(1)
paicos/__init__.py:        from .image_creators.gpu_sph_projector import GpuSphProjector
paicos/__init__.py:        from .image_creators.gpu_ray_projector import GpuRayProjector
paicos/__init__.py:        paicos.GpuSphProjector = GpuSphProjector
paicos/__init__.py:        paicos.GpuRayProjector = GpuRayProjector
paicos/__init__.py:        err_msg = ('\nPaicos: The simple cuda example using cupy and numba failed '
paicos/__init__.py:                   'a GPU that supports CUDA.\n')
paicos/__init__.py:if settings.load_cuda_functionality_on_startup:
paicos/__init__.py:    gpu_init()
examples/gpu_sph_speed_test_example.py:timing_gpu = np.empty(20)
examples/gpu_sph_speed_test_example.py:def print_timing(timing_gpu):
examples/gpu_sph_speed_test_example.py:    timing_gpu *= 1e6  # convert to us
examples/gpu_sph_speed_test_example.py:    print(f"Elapsed time GPU: {timing_gpu.mean():.0f} ± {timing_gpu.std():.0f} us")
examples/gpu_sph_speed_test_example.py:    timing_gpu /= 1e3  # convert to ms
examples/gpu_sph_speed_test_example.py:    print(f"Elapsed time GPU: {timing_gpu.mean():.1f} ± {timing_gpu.std():.2f} ms")
examples/gpu_sph_speed_test_example.py:projector = pa.GpuSphProjector(snap, center, widths, orientation, npix=nx, threadsperblock=64, do_pre_selection=True)
examples/gpu_sph_speed_test_example.py:for i in range(timing_gpu.size):
examples/gpu_sph_speed_test_example.py:    timing_gpu[i] = toc - tic
examples/gpu_sph_speed_test_example.py:print_timing(timing_gpu)
examples/gpu_sph_speed_test_example.py:    plt.savefig('gpu_dens.png', dpi=1400)

```
