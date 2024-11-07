# https://github.com/kerkelae/disimpy

```console
README.rst:walk simulations that run massively parallel on Nvidia CUDA-capable GPUs. If
development_environment.yml:  - cudatoolkit=11.8.0=h6a678d5_0
docs/source/installation.rst:You need an Nvidia CUDA-capable GPU with compute capability 2.0 or above with
docs/source/installation.rst:the appropriate Nvidia driver. You can check if your GPU is supported on
docs/source/installation.rst:`Nvidia's website <https://developer.nvidia.com/cuda-gpus>`_.
docs/source/installation.rst:You need the CUDA Toolkit (version 8.0 or above), which can be installed `using
docs/source/installation.rst:Conda <https://numba.pydata.org/numba-doc/dev/cuda/overview.html#software>`_
docs/source/installation.rst:(recommended) or from `Nvidia <https://developer.nvidia.com/cuda-toolkit>`_.
docs/source/installation.rst:Make sure that the version you install supports your Nvidia driver or upgrade
docs/source/installation.rst:the driver. The driver requirements of each CUDA Toolkit version can be found in
docs/source/installation.rst:the `release notes <https://developer.nvidia.com/cuda-toolkit-archive>`_. If you
docs/source/installation.rst:use the CUDA Toolkit not installed by Conda and encounter issues, check that the
docs/source/installation.rst:<https://numba.pydata.org/numba-doc/dev/cuda/overview.html#setting-cuda-installation-path>`_.
disimpy/tests/test_simulations.py:from numba import cuda
disimpy/tests/test_simulations.py:from numba.cuda.random import (
disimpy/tests/test_simulations.py:def test__cuda_dot_product():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        dp[thread_id] = simulations._cuda_dot_product(a[thread_id, :], b[thread_id, :])
disimpy/tests/test_simulations.py:        stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_cross_product():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        simulations._cuda_cross_product(
disimpy/tests/test_simulations.py:        stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_normalize_vector():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        simulations._cuda_normalize_vector(a[thread_id, :])
disimpy/tests/test_simulations.py:        stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_triangle_normal():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        simulations._cuda_triangle_normal(triangle[thread_id, :], normal[thread_id, :])
disimpy/tests/test_simulations.py:        stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_random_step():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        simulations._cuda_random_step(steps[thread_id, :], rng_states, thread_id)
disimpy/tests/test_simulations.py:        stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_mat_mul():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        simulations._cuda_mat_mul(R, a[thread_id, :])
disimpy/tests/test_simulations.py:        stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_line_circle_intersection():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        d[thread_id] = simulations._cuda_line_circle_intersection(
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_line_sphere_intersection():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        d[thread_id] = simulations._cuda_line_sphere_intersection(
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_line_ellipsoid_intersection():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        d[thread_id] = simulations._cuda_line_ellipsoid_intersection(
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_ray_triangle_intersection_check():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        ds[thread_id, :] = simulations._cuda_ray_triangle_intersection_check(
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_reflection():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        simulations._cuda_reflection(
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        d = simulations._cuda_ray_triangle_intersection_check(triangle, r0, step)
disimpy/tests/test_simulations.py:            normal = cuda.local.array(3, numba.float64)
disimpy/tests/test_simulations.py:            simulations._cuda_triangle_normal(triangle, normal)
disimpy/tests/test_simulations.py:            simulations._cuda_reflection(r0, step, d, normal, epsilon)
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/tests/test_simulations.py:def test__cuda_crossing():
disimpy/tests/test_simulations.py:    @cuda.jit()
disimpy/tests/test_simulations.py:        thread_id = cuda.grid(1)
disimpy/tests/test_simulations.py:        d = simulations._cuda_ray_triangle_intersection_check(triangle, r0, step)
disimpy/tests/test_simulations.py:            normal = cuda.local.array(3, numba.float64)
disimpy/tests/test_simulations.py:            simulations._cuda_triangle_normal(triangle, normal)
disimpy/tests/test_simulations.py:            simulations._cuda_crossing(r0, step, d, normal, epsilon)
disimpy/tests/test_simulations.py:    stream = cuda.stream()
disimpy/simulations.py:from numba import cuda
disimpy/simulations.py:from numba.cuda.random import (
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_dot_product(a, b):
disimpy/simulations.py:    a : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    b : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_cross_product(a, b, c):
disimpy/simulations.py:    a : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    b : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    c : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_normalize_vector(v):
disimpy/simulations.py:    v : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    length = math.sqrt(_cuda_dot_product(v, v))
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_triangle_normal(triangle, normal):
disimpy/simulations.py:    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    v = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    k = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    _cuda_cross_product(v, k, normal)
disimpy/simulations.py:    _cuda_normalize_vector(normal)
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_get_triangle(i, vertices, faces, triangle):
disimpy/simulations.py:    vertices : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    faces : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_random_step(step, rng_states, thread_id):
disimpy/simulations.py:    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    rng_states : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    _cuda_normalize_vector(step)
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_mat_mul(R, v):
disimpy/simulations.py:    R : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    v : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    rotated_v = cuda.local.array(3, numba.float64)
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_line_circle_intersection(r0, step, radius):
disimpy/simulations.py:    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_line_sphere_intersection(r0, step, radius):
disimpy/simulations.py:    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    dp = _cuda_dot_product(step, r0)
disimpy/simulations.py:    d = -dp + math.sqrt(dp**2 - (_cuda_dot_product(r0, r0) - radius**2))
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_line_ellipsoid_intersection(r0, step, semiaxes):
disimpy/simulations.py:    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    semiaxes : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:def _cuda_ray_triangle_intersection_check(triangle, r0, step):
disimpy/simulations.py:    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    T = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    E_1 = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    E_2 = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    P = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    Q = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    _cuda_cross_product(step, E_2, P)
disimpy/simulations.py:    _cuda_cross_product(T, E_1, Q)
disimpy/simulations.py:    det = _cuda_dot_product(P, E_1)
disimpy/simulations.py:        t = 1 / det * _cuda_dot_product(Q, E_2)
disimpy/simulations.py:        u = 1 / det * _cuda_dot_product(P, T)
disimpy/simulations.py:        v = 1 / det * _cuda_dot_product(Q, step)
disimpy/simulations.py:def _cuda_reflection(r0, step, d, normal, epsilon):
disimpy/simulations.py:    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    semiaxes : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    intersection = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    v = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    dp = _cuda_dot_product(v, normal)
disimpy/simulations.py:        dp = _cuda_dot_product(v, normal)
disimpy/simulations.py:    _cuda_normalize_vector(step)
disimpy/simulations.py:def _cuda_crossing(r0, step, d, normal, epsilon):
disimpy/simulations.py:    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    semiaxes : numba.cuda.cudadrv.devicearray.DeviceNDArray
disimpy/simulations.py:    intersection = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    v = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    dp = _cuda_dot_product(v, normal)
disimpy/simulations.py:@cuda.jit()
disimpy/simulations.py:def _cuda_fill_mesh(
disimpy/simulations.py:    thread_id = cuda.grid(1)
disimpy/simulations.py:    point = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    ray = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    lls = cuda.local.array(3, numba.int64)
disimpy/simulations.py:    uls = cuda.local.array(3, numba.int64)
disimpy/simulations.py:    triangle = cuda.local.array((3, 3), numba.float64)
disimpy/simulations.py:    triangles = cuda.local.array(1000, numba.int64)
disimpy/simulations.py:                    _cuda_get_triangle(triangle_indices[i], vertices, faces, triangle)
disimpy/simulations.py:                    d = _cuda_ray_triangle_intersection_check(triangle, point, ray)
disimpy/simulations.py:def _fill_mesh(n_points, substrate, intra, seed, cuda_bs=128):
disimpy/simulations.py:    cuda_bs : int, optional
disimpy/simulations.py:    bs = cuda_bs
disimpy/simulations.py:    stream = cuda.stream()
disimpy/simulations.py:        d_vertices = cuda.to_device(substrate.vertices, stream=stream)
disimpy/simulations.py:        d_faces = cuda.to_device(substrate.faces, stream=stream)
disimpy/simulations.py:        d_subvoxel_indices = cuda.to_device(substrate.subvoxel_indices, stream=stream)
disimpy/simulations.py:        d_triangle_indices = cuda.to_device(substrate.triangle_indices, stream=stream)
disimpy/simulations.py:        d_vertices = cuda.to_device(vertices, stream=stream)
disimpy/simulations.py:        d_faces = cuda.to_device(faces, stream=stream)
disimpy/simulations.py:        d_subvoxel_indices = cuda.to_device(subvoxel_indices, stream=stream)
disimpy/simulations.py:        d_triangle_indices = cuda.to_device(triangle_indices, stream=stream)
disimpy/simulations.py:    d_voxel_size = cuda.to_device(substrate.voxel_size, stream=stream)
disimpy/simulations.py:    d_xs = cuda.to_device(substrate.xs, stream=stream)
disimpy/simulations.py:    d_ys = cuda.to_device(substrate.ys, stream=stream)
disimpy/simulations.py:    d_zs = cuda.to_device(substrate.zs, stream=stream)
disimpy/simulations.py:    d_n_sv = cuda.to_device(substrate.n_sv, stream=stream)
disimpy/simulations.py:        d_points = cuda.to_device(new_points, stream=stream)
disimpy/simulations.py:        _cuda_fill_mesh[gs, bs, stream](
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:@cuda.jit(device=True)
disimpy/simulations.py:@cuda.jit()
disimpy/simulations.py:def _cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, t, step_l, dt):
disimpy/simulations.py:    thread_id = cuda.grid(1)
disimpy/simulations.py:    step = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    _cuda_random_step(step, rng_states, thread_id)
disimpy/simulations.py:@cuda.jit()
disimpy/simulations.py:def _cuda_step_sphere(
disimpy/simulations.py:    thread_id = cuda.grid(1)
disimpy/simulations.py:    step = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    _cuda_random_step(step, rng_states, thread_id)
disimpy/simulations.py:        d = _cuda_line_sphere_intersection(r0, step, radius)
disimpy/simulations.py:            normal = cuda.local.array(3, numba.float64)
disimpy/simulations.py:            _cuda_normalize_vector(normal)
disimpy/simulations.py:            _cuda_reflection(r0, step, d, normal, epsilon)
disimpy/simulations.py:@cuda.jit()
disimpy/simulations.py:def _cuda_step_cylinder(
disimpy/simulations.py:    thread_id = cuda.grid(1)
disimpy/simulations.py:    step = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    _cuda_random_step(step, rng_states, thread_id)
disimpy/simulations.py:    _cuda_mat_mul(R, r0)  # Move to cylinder frame
disimpy/simulations.py:        d = _cuda_line_circle_intersection(r0[1:3], step[1:3], radius)
disimpy/simulations.py:            normal = cuda.local.array(3, numba.float64)
disimpy/simulations.py:            _cuda_normalize_vector(normal)
disimpy/simulations.py:            _cuda_reflection(r0, step, d, normal, epsilon)
disimpy/simulations.py:    _cuda_mat_mul(R_inv, step)  # Move back to lab frame
disimpy/simulations.py:    _cuda_mat_mul(R_inv, r0)  # Move back to lab frame
disimpy/simulations.py:@cuda.jit()
disimpy/simulations.py:def _cuda_step_ellipsoid(
disimpy/simulations.py:    thread_id = cuda.grid(1)
disimpy/simulations.py:    step = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    _cuda_random_step(step, rng_states, thread_id)
disimpy/simulations.py:    _cuda_mat_mul(R, r0)  # Move to ellipsoid frame
disimpy/simulations.py:        d = _cuda_line_ellipsoid_intersection(r0, step, semiaxes)
disimpy/simulations.py:            normal = cuda.local.array(3, numba.float64)
disimpy/simulations.py:            _cuda_normalize_vector(normal)
disimpy/simulations.py:            _cuda_reflection(r0, step, d, normal, epsilon)
disimpy/simulations.py:    _cuda_mat_mul(R_inv, step)  # Move back to lab frame
disimpy/simulations.py:    _cuda_mat_mul(R_inv, r0)  # Move back to lab frame
disimpy/simulations.py:@cuda.jit()
disimpy/simulations.py:def _cuda_step_mesh(
disimpy/simulations.py:    thread_id = cuda.grid(1)
disimpy/simulations.py:    step = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    lls = cuda.local.array(3, numba.int64)
disimpy/simulations.py:    uls = cuda.local.array(3, numba.int64)
disimpy/simulations.py:    triangle = cuda.local.array((3, 3), numba.float64)
disimpy/simulations.py:    normal = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    shifts = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    temp_r0 = cuda.local.array(3, numba.float64)
disimpy/simulations.py:    _cuda_random_step(step, rng_states, thread_id)
disimpy/simulations.py:                        _cuda_get_triangle(
disimpy/simulations.py:                        d = _cuda_ray_triangle_intersection_check(
disimpy/simulations.py:            _cuda_get_triangle(closest_triangle_index, vertices, faces, triangle)
disimpy/simulations.py:            _cuda_triangle_normal(triangle, normal)
disimpy/simulations.py:            _cuda_reflection(r0, step, min_d, normal, epsilon)
disimpy/simulations.py:            _cuda_get_triangle(closest_triangle_index, vertices, faces, triangle)
disimpy/simulations.py:            _cuda_triangle_normal(triangle, normal)
disimpy/simulations.py:            _cuda_crossing(r0, step, min_d, normal, epsilon)
disimpy/simulations.py:    cuda_bs=128,
disimpy/simulations.py:    cuda_bs : int, optional
disimpy/simulations.py:        The size of the one-dimensional CUDA thread block.
disimpy/simulations.py:    # Confirm that Numba detects the GPU wihtout printing it
disimpy/simulations.py:            cuda.detect()
disimpy/simulations.py:                "Numba was unable to detect a CUDA GPU. To run the simulation,"
disimpy/simulations.py:                + " check that the requirements are met and CUDA installation"
disimpy/simulations.py:                + "https://numba.pydata.org/numba-doc/dev/cuda/overview.html"
disimpy/simulations.py:    if not isinstance(cuda_bs, int) or cuda_bs <= 0:
disimpy/simulations.py:        raise ValueError("Incorrect value (%s) for cuda_bs" % cuda_bs)
disimpy/simulations.py:    # Set up CUDA stream
disimpy/simulations.py:    bs = cuda_bs  # Threads per block
disimpy/simulations.py:    stream = cuda.stream()
disimpy/simulations.py:    # Move arrays to the GPU
disimpy/simulations.py:    d_g_x = cuda.to_device(np.ascontiguousarray(gradient[:, :, 0]), stream=stream)
disimpy/simulations.py:    d_g_y = cuda.to_device(np.ascontiguousarray(gradient[:, :, 1]), stream=stream)
disimpy/simulations.py:    d_g_z = cuda.to_device(np.ascontiguousarray(gradient[:, :, 2]), stream=stream)
disimpy/simulations.py:    d_phases = cuda.to_device(np.zeros((gradient.shape[0], n_walkers)), stream=stream)
disimpy/simulations.py:    d_iter_exc = cuda.to_device(np.zeros(n_walkers).astype(bool))
disimpy/simulations.py:        d_positions = cuda.to_device(positions, stream=stream)
disimpy/simulations.py:            _cuda_step_free[gs, bs, stream](
disimpy/simulations.py:        d_R = cuda.to_device(R)
disimpy/simulations.py:        d_R_inv = cuda.to_device(R_inv)
disimpy/simulations.py:        d_positions = cuda.to_device(positions, stream=stream)
disimpy/simulations.py:            _cuda_step_cylinder[gs, bs, stream](
disimpy/simulations.py:        d_positions = cuda.to_device(positions, stream=stream)
disimpy/simulations.py:            _cuda_step_sphere[gs, bs, stream](
disimpy/simulations.py:        d_semiaxes = cuda.to_device(substrate.semiaxes)
disimpy/simulations.py:        d_R_inv = cuda.to_device(R_inv)
disimpy/simulations.py:        d_R = cuda.to_device(np.linalg.inv(R_inv))
disimpy/simulations.py:        d_positions = cuda.to_device(positions, stream=stream)
disimpy/simulations.py:            _cuda_step_ellipsoid[gs, bs, stream](
disimpy/simulations.py:        # Move arrays to the GPU
disimpy/simulations.py:        d_vertices = cuda.to_device(substrate.vertices, stream=stream)
disimpy/simulations.py:        d_faces = cuda.to_device(substrate.faces, stream=stream)
disimpy/simulations.py:        d_xs = cuda.to_device(substrate.xs, stream=stream)
disimpy/simulations.py:        d_ys = cuda.to_device(substrate.ys, stream=stream)
disimpy/simulations.py:        d_zs = cuda.to_device(substrate.zs, stream=stream)
disimpy/simulations.py:        d_triangle_indices = cuda.to_device(substrate.triangle_indices, stream=stream)
disimpy/simulations.py:        d_subvoxel_indices = cuda.to_device(substrate.subvoxel_indices, stream=stream)
disimpy/simulations.py:        d_n_sv = cuda.to_device(substrate.n_sv, stream=stream)
disimpy/simulations.py:        d_positions = cuda.to_device(positions, stream=stream)
disimpy/simulations.py:            _cuda_step_mesh[gs, bs, stream](
disimpy/__init__.py:# Numba warns about GPU under-utilization and, at least in version 5.7.0, importing
disimpy/__init__.py:# numba.cuda.random leads to many deprecation warnings. Let's ignore these:
.gitignore:# Cuda

```
