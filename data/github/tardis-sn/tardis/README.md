# https://github.com/tardis-sn/tardis

```console
docs/contributing/CHANGELOG.md:- [1919](https://github.com/tardis-sn/tardis/pull/1919) GPU Options (1919) (@KevinCawley)
docs/contributing/CHANGELOG.md:- [1907](https://github.com/tardis-sn/tardis/pull/1907) Add note on GPU selection to formal integral (1907) (@KevinCawley)
docs/contributing/CHANGELOG.md:- [1837](https://github.com/tardis-sn/tardis/pull/1837) CUDA Version of the Formal Integral (1837) (@KevinCawley)
tardis/spectrum/tests/test_cuda_formal_integral.py:from numba import cuda
tardis/spectrum/tests/test_cuda_formal_integral.py:import tardis.spectrum.formal_integral_cuda as formal_integral_cuda
tardis/spectrum/tests/test_cuda_formal_integral.py:# Test cases must also take into account use of a GPU to run. If there is no GPU then the test cases will fail.
tardis/spectrum/tests/test_cuda_formal_integral.py:GPUs_available = cuda.is_available()
tardis/spectrum/tests/test_cuda_formal_integral.py:    not GPUs_available, reason="No GPU is available to test CUDA function"
tardis/spectrum/tests/test_cuda_formal_integral.py:def test_intensity_black_body_cuda(nu, temperature):
tardis/spectrum/tests/test_cuda_formal_integral.py:    Initializes the test of the cuda version
tardis/spectrum/tests/test_cuda_formal_integral.py:@cuda.jit
tardis/spectrum/tests/test_cuda_formal_integral.py:    This calls the CUDA function and fills out
tardis/spectrum/tests/test_cuda_formal_integral.py:    x = cuda.grid(1)
tardis/spectrum/tests/test_cuda_formal_integral.py:    actual[x] = formal_integral_cuda.intensity_black_body_cuda(nu, temperature)
tardis/spectrum/tests/test_cuda_formal_integral.py:    not GPUs_available, reason="No GPU is available to test CUDA function"
tardis/spectrum/tests/test_cuda_formal_integral.py:def test_trapezoid_integration_cuda(N):
tardis/spectrum/tests/test_cuda_formal_integral.py:    Initializes the test of the cuda version
tardis/spectrum/tests/test_cuda_formal_integral.py:@cuda.jit
tardis/spectrum/tests/test_cuda_formal_integral.py:    This calls the CUDA function and fills out
tardis/spectrum/tests/test_cuda_formal_integral.py:    x = cuda.grid(1)
tardis/spectrum/tests/test_cuda_formal_integral.py:    actual[x] = formal_integral_cuda.trapezoid_integration_cuda(data, h)
tardis/spectrum/tests/test_cuda_formal_integral.py:    not GPUs_available, reason="No GPU is available to test CUDA function"
tardis/spectrum/tests/test_cuda_formal_integral.py:def test_calculate_z_cuda(formal_integral_geometry, time_explosion, p, p_loc):
tardis/spectrum/tests/test_cuda_formal_integral.py:    Initializes the test of the cuda version
tardis/spectrum/tests/test_cuda_formal_integral.py:@cuda.jit
tardis/spectrum/tests/test_cuda_formal_integral.py:    This calls the CUDA function and fills out
tardis/spectrum/tests/test_cuda_formal_integral.py:    x = cuda.grid(1)
tardis/spectrum/tests/test_cuda_formal_integral.py:    actual[x] = formal_integral_cuda.calculate_z_cuda(r, p, inv_t)
tardis/spectrum/tests/test_cuda_formal_integral.py:    not GPUs_available, reason="No GPU is available to test CUDA function"
tardis/spectrum/tests/test_cuda_formal_integral.py:    Initializes the test of the cuda version
tardis/spectrum/tests/test_cuda_formal_integral.py:@cuda.jit
tardis/spectrum/tests/test_cuda_formal_integral.py:    This calls the CUDA function and fills out
tardis/spectrum/tests/test_cuda_formal_integral.py:    x = cuda.grid(1)
tardis/spectrum/tests/test_cuda_formal_integral.py:    actual[x] = formal_integral_cuda.populate_z_cuda(
tardis/spectrum/tests/test_cuda_formal_integral.py:    Initializes the test of the cuda version
tardis/spectrum/tests/test_cuda_formal_integral.py:    actual[::] = formal_integral_cuda.calculate_p_values(r, N)
tardis/spectrum/tests/test_cuda_formal_integral.py:    not GPUs_available, reason="No GPU is available to test CUDA function"
tardis/spectrum/tests/test_cuda_formal_integral.py:def test_line_search_cuda(nu_insert, simulation_verysimple_opacity_state):
tardis/spectrum/tests/test_cuda_formal_integral.py:    Initializes the test of the cuda version
tardis/spectrum/tests/test_cuda_formal_integral.py:    line_search_cuda_caller[1, 1](line_list_nu, nu_insert, actual)
tardis/spectrum/tests/test_cuda_formal_integral.py:@cuda.jit
tardis/spectrum/tests/test_cuda_formal_integral.py:def line_search_cuda_caller(line_list_nu, nu_insert, actual):
tardis/spectrum/tests/test_cuda_formal_integral.py:    This calls the CUDA function and fills out
tardis/spectrum/tests/test_cuda_formal_integral.py:    x = cuda.grid(1)
tardis/spectrum/tests/test_cuda_formal_integral.py:    actual[x] = formal_integral_cuda.line_search_cuda(
tardis/spectrum/tests/test_cuda_formal_integral.py:    not GPUs_available, reason="No GPU is available to test CUDA function"
tardis/spectrum/tests/test_cuda_formal_integral.py:    Initializes the test of the cuda version
tardis/spectrum/tests/test_cuda_formal_integral.py:    reverse_binary_search_cuda_caller[1, 1](
tardis/spectrum/tests/test_cuda_formal_integral.py:@cuda.jit
tardis/spectrum/tests/test_cuda_formal_integral.py:def reverse_binary_search_cuda_caller(
tardis/spectrum/tests/test_cuda_formal_integral.py:    This calls the CUDA function and fills out
tardis/spectrum/tests/test_cuda_formal_integral.py:    x = cuda.grid(1)
tardis/spectrum/tests/test_cuda_formal_integral.py:    actual[x] = formal_integral_cuda.reverse_binary_search_cuda(
tardis/spectrum/tests/test_cuda_formal_integral.py:    not GPUs_available, reason="No GPU is available to test CUDA function"
tardis/spectrum/tests/test_cuda_formal_integral.py:    This function initializes both the cuda and numba formal_integrator,
tardis/spectrum/tests/test_cuda_formal_integral.py:    formal_integrator_cuda = FormalIntegrator(
tardis/spectrum/tests/test_cuda_formal_integral.py:    # The function calculate_spectrum sets this property, but in order to test the CUDA.
tardis/spectrum/tests/test_cuda_formal_integral.py:    formal_integrator_cuda.interpolate_shells = max(
tardis/spectrum/tests/test_cuda_formal_integral.py:        2 * formal_integrator_cuda.simulation_state.no_of_shells, 80
tardis/spectrum/tests/test_cuda_formal_integral.py:    res_cuda = formal_integrator_cuda.make_source_function()
tardis/spectrum/tests/test_cuda_formal_integral.py:    att_S_ul_cuda = res_cuda[0].flatten(order="F")
tardis/spectrum/tests/test_cuda_formal_integral.py:    Jred_lu_cuda = res_cuda[1].values.flatten(order="F")
tardis/spectrum/tests/test_cuda_formal_integral.py:    Jblue_lu_cuda = res_cuda[2].flatten(order="F")
tardis/spectrum/tests/test_cuda_formal_integral.py:    # as it is automatically set to the CUDA version when there is a GPU available
tardis/spectrum/tests/test_cuda_formal_integral.py:    formal_integrator_cuda.generate_numba_objects()
tardis/spectrum/tests/test_cuda_formal_integral.py:    L_cuda = formal_integrator_cuda.integrator.formal_integral(
tardis/spectrum/tests/test_cuda_formal_integral.py:        formal_integrator_cuda.simulation_state.t_inner,
tardis/spectrum/tests/test_cuda_formal_integral.py:        att_S_ul_cuda,
tardis/spectrum/tests/test_cuda_formal_integral.py:        Jred_lu_cuda,
tardis/spectrum/tests/test_cuda_formal_integral.py:        Jblue_lu_cuda,
tardis/spectrum/tests/test_cuda_formal_integral.py:        formal_integrator_cuda.transport.tau_sobolevs_integ,
tardis/spectrum/tests/test_cuda_formal_integral.py:        formal_integrator_cuda.transport.electron_densities_integ,
tardis/spectrum/tests/test_cuda_formal_integral.py:        formal_integrator_cuda.points,
tardis/spectrum/tests/test_cuda_formal_integral.py:    ntest.assert_allclose(L_cuda, L_numba, rtol=1e-14)
tardis/spectrum/formal_integral_cuda.py:from numba import cuda
tardis/spectrum/formal_integral_cuda.py:@cuda.jit
tardis/spectrum/formal_integral_cuda.py:def cuda_vector_integrator(L, I_nu, N, R_max):
tardis/spectrum/formal_integral_cuda.py:    The CUDA Vectorized integrator over second axis
tardis/spectrum/formal_integral_cuda.py:    nu_idx = cuda.grid(1)
tardis/spectrum/formal_integral_cuda.py:        8 * M_PI * M_PI * trapezoid_integration_cuda(I_nu[nu_idx], R_max / N)
tardis/spectrum/formal_integral_cuda.py:@cuda.jit
tardis/spectrum/formal_integral_cuda.py:def cuda_formal_integral(
tardis/spectrum/formal_integral_cuda.py:    The CUDA version of numba_formal_integral that can run
tardis/spectrum/formal_integral_cuda.py:    on a NVIDIA GPU.
tardis/spectrum/formal_integral_cuda.py:    nu_idx, p_idx = cuda.grid(2)  # 2D Cuda Grid, nu x p
tardis/spectrum/formal_integral_cuda.py:    # Check to see if CUDA is out of bounds
tardis/spectrum/formal_integral_cuda.py:    size_z = populate_z_cuda(
tardis/spectrum/formal_integral_cuda.py:        I_nu_thread[p_idx] = intensity_black_body_cuda(nu * z_thread[0], iT)
tardis/spectrum/formal_integral_cuda.py:    idx_nu_start = line_search_cuda(line_list_nu, nu_start, size_line)
tardis/spectrum/formal_integral_cuda.py:        nu_end_idx = line_search_cuda(line_list_nu, nu_end, len(line_list_nu))
tardis/spectrum/formal_integral_cuda.py:class CudaFormalIntegrator:
tardis/spectrum/formal_integral_cuda.py:    with CUDA.
tardis/spectrum/formal_integral_cuda.py:        Simple wrapper for the CUDA implementation of the formal integral
tardis/spectrum/formal_integral_cuda.py:        # These are device objects stored on the GPU
tardis/spectrum/formal_integral_cuda.py:        d_L = cuda.device_array((inu_size,), dtype=np.float64)
tardis/spectrum/formal_integral_cuda.py:        d_I_nu = cuda.device_array((inu_size, N), dtype=np.float64)
tardis/spectrum/formal_integral_cuda.py:        z = cuda.to_device(z)
tardis/spectrum/formal_integral_cuda.py:        shell_id = cuda.to_device(shell_id)
tardis/spectrum/formal_integral_cuda.py:        pp = cuda.to_device(pp)
tardis/spectrum/formal_integral_cuda.py:        exp_tau = cuda.to_device(exp_tau)
tardis/spectrum/formal_integral_cuda.py:        r_inner = cuda.to_device(self.geometry.r_inner)
tardis/spectrum/formal_integral_cuda.py:        r_outer = cuda.to_device(self.geometry.r_outer)
tardis/spectrum/formal_integral_cuda.py:        line_list_nu = cuda.to_device(self.plasma.line_list_nu)
tardis/spectrum/formal_integral_cuda.py:        inu = cuda.to_device(inu.value)
tardis/spectrum/formal_integral_cuda.py:        att_S_ul = cuda.to_device(att_S_ul)
tardis/spectrum/formal_integral_cuda.py:        Jred_lu = cuda.to_device(Jred_lu)
tardis/spectrum/formal_integral_cuda.py:        Jblue_lu = cuda.to_device(Jblue_lu)
tardis/spectrum/formal_integral_cuda.py:        tau_sobolev = cuda.to_device(tau_sobolev)
tardis/spectrum/formal_integral_cuda.py:        electron_density = cuda.to_device(electron_density)
tardis/spectrum/formal_integral_cuda.py:        cuda_formal_integral[
tardis/spectrum/formal_integral_cuda.py:        cuda_vector_integrator[blocks_per_grid_nu, THREADS_PER_BLOCK_NU](
tardis/spectrum/formal_integral_cuda.py:@cuda.jit(device=True)
tardis/spectrum/formal_integral_cuda.py:def populate_z_cuda(r_inner, r_outer, time_explosion, p, oz, oshell_id):
tardis/spectrum/formal_integral_cuda.py:            oz[i] = 1 - calculate_z_cuda(r_outer[i], p, inv_t)
tardis/spectrum/formal_integral_cuda.py:            z = calculate_z_cuda(r_outer[i], p, inv_t)
tardis/spectrum/formal_integral_cuda.py:@cuda.jit(device=True)
tardis/spectrum/formal_integral_cuda.py:def calculate_z_cuda(r, p, inv_t):
tardis/spectrum/formal_integral_cuda.py:@cuda.jit(device=True)
tardis/spectrum/formal_integral_cuda.py:def line_search_cuda(nu, nu_insert, number_of_lines):
tardis/spectrum/formal_integral_cuda.py:        result = reverse_binary_search_cuda(nu, nu_insert, imin, imax)
tardis/spectrum/formal_integral_cuda.py:@cuda.jit(device=True)
tardis/spectrum/formal_integral_cuda.py:def reverse_binary_search_cuda(x, x_insert, imin, imax):
tardis/spectrum/formal_integral_cuda.py:@cuda.jit(device=True)
tardis/spectrum/formal_integral_cuda.py:def trapezoid_integration_cuda(arr, dx):
tardis/spectrum/formal_integral_cuda.py:@cuda.jit(device=True)
tardis/spectrum/formal_integral_cuda.py:def intensity_black_body_cuda(nu, temperature):
tardis/spectrum/formal_integral.py:from tardis.spectrum.formal_integral_cuda import (
tardis/spectrum/formal_integral.py:    CudaFormalIntegrator,
tardis/spectrum/formal_integral.py:    If there is a NVIDIA CUDA GPU available,
tardis/spectrum/formal_integral.py:    on it. If multiple GPUs are available, it will
tardis/spectrum/formal_integral.py:    read more about selecting different GPUs on
tardis/spectrum/formal_integral.py:    Numba's CUDA documentation.
tardis/spectrum/formal_integral.py:        if self.transport.use_gpu:
tardis/spectrum/formal_integral.py:            self.integrator = CudaFormalIntegrator(
tardis/io/configuration/schemas/spectrum.yml:            It defaults to the Numba version, but it can be run on NVIDIA Cuda
tardis/io/configuration/schemas/spectrum.yml:            GPUs. GPU will make it only run on a NVIDIA Cuda GPU, so if one is 
tardis/io/configuration/schemas/spectrum.yml:            an acceptable GPU, and if none is found it will run on the CPU.
tardis/io/configuration/schemas/spectrum.yml:                - "GPU"
tardis/transport/montecarlo/base.py:from numba import cuda, set_num_threads
tardis/transport/montecarlo/base.py:        use_gpu=False,
tardis/transport/montecarlo/base.py:        self.use_gpu = use_gpu
tardis/transport/montecarlo/base.py:        if running_mode == "GPU":
tardis/transport/montecarlo/base.py:            if cuda.is_available():
tardis/transport/montecarlo/base.py:                use_gpu = True
tardis/transport/montecarlo/base.py:                    """The GPU option was selected for the formal_integral,
tardis/transport/montecarlo/base.py:                    but no CUDA GPU is available."""
tardis/transport/montecarlo/base.py:            use_gpu = bool(cuda.is_available())
tardis/transport/montecarlo/base.py:            use_gpu = False
tardis/transport/montecarlo/base.py:                valid values are 'GPU', 'CPU', and 'Automatic'."""
tardis/transport/montecarlo/base.py:            use_gpu=use_gpu,
CHANGELOG.md:- [1919](https://github.com/tardis-sn/tardis/pull/1919) GPU Options (1919) (@KevinCawley)
CHANGELOG.md:- [1907](https://github.com/tardis-sn/tardis/pull/1907) Add note on GPU selection to formal integral (1907) (@KevinCawley)
CHANGELOG.md:- [1837](https://github.com/tardis-sn/tardis/pull/1837) CUDA Version of the Formal Integral (1837) (@KevinCawley)

```
