# https://github.com/sblunt/orbitize

```console
docs/index.rst:- GPU Kepler solver (@devincody)
orbitize/kernels/newton.cu:__global__ void newton_gpu(const double *manom, 
orbitize/kernels/mikkola.cu:__global__ void mikkola_gpu(const double *manom, const double *ecc, double *eanom){
orbitize/kepler.py:from orbitize import cuda_ext, cext
orbitize/kepler.py:if cuda_ext:
orbitize/kepler.py:    # Configure GPU context for CUDA accelerated compute
orbitize/kepler.py:    from orbitize import gpu_context
orbitize/kepler.py:    kep_gpu_ctx = gpu_context.gpu_context()
orbitize/kepler.py:  max_iter=100, use_c=True, use_gpu=False
orbitize/kepler.py:        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False
orbitize/kepler.py:    eanom = _calc_ecc_anom(manom, ecc_arr, tolerance=tolerance, max_iter=max_iter, use_c=use_c, use_gpu=use_gpu)
orbitize/kepler.py:def _calc_ecc_anom(manom, ecc, tolerance=1e-9, max_iter=100, use_c=False, use_gpu=False):
orbitize/kepler.py:        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False
orbitize/kepler.py:        eanom[ind_low] = _newton_solver_wrapper(manom[ind_low], ecc[ind_low], tolerance, max_iter, use_c, use_gpu)
orbitize/kepler.py:    ind_high = np.where(~ecc_zero & ~ecc_low | (eanom == -1)) # The C and CUDA solvers return the unphysical value -1 if they fail to converge
orbitize/kepler.py:        eanom[ind_high] = _mikkola_solver_wrapper(manom[ind_high], ecc[ind_high], use_c, use_gpu)
orbitize/kepler.py:def _newton_solver_wrapper(manom, ecc, tolerance, max_iter, use_c=False, use_gpu=False):
orbitize/kepler.py:    Wrapper for the various (Python, C, CUDA) implementations of the Newton-Raphson solver 
orbitize/kepler.py:        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False
orbitize/kepler.py:    if cuda_ext and use_gpu:
orbitize/kepler.py:        # the CUDA solver returns eanom = -1 if it doesnt converge after max_iter iterations
orbitize/kepler.py:        eanom = _CUDA_newton_solver(manom, ecc, tolerance=tolerance, max_iter=max_iter)
orbitize/kepler.py:def _CUDA_newton_solver(manom, ecc, tolerance=1e-9, max_iter=100, eanom0=None):
orbitize/kepler.py:    Helper function for calling the CUDA implementation of the Newton-Raphson solver for eccentric anomaly.
orbitize/kepler.py:    global kep_gpu_ctx
orbitize/kepler.py:    kep_gpu_ctx.newton(manom, ecc, eanom, eanom0, tolerance, max_iter)
orbitize/kepler.py:def _mikkola_solver_wrapper(manom, ecc, use_c=False, use_gpu=False):
orbitize/kepler.py:    Wrapper for the various (Python, C, CUDA) implementations of Analtyical Mikkola solver 
orbitize/kepler.py:        use_gpu (bool, optional): Use the GPU solver if configured. Defaults to False
orbitize/kepler.py:    if cuda_ext and use_gpu:
orbitize/kepler.py:        eanom = _CUDA_mikkola_solver(manom, ecc)
orbitize/kepler.py:def _CUDA_mikkola_solver(manom, ecc):
orbitize/kepler.py:    Helper function for calling the CUDA implementation of the Analtyical Mikkola solver for the eccentric anomaly.
orbitize/kepler.py:    global kep_gpu_ctx
orbitize/kepler.py:    kep_gpu_ctx.mikkola(manom, ecc, eanom)
orbitize/__init__.py:# Detect a valid CUDA environment
orbitize/__init__.py:    import pycuda.driver as cuda
orbitize/__init__.py:    import pycuda.autoinit
orbitize/__init__.py:    from pycuda.compiler import SourceModule
orbitize/__init__.py:    cuda_ext = True
orbitize/__init__.py:    cuda_ext = False
orbitize/sampler.py:from orbitize import cuda_ext
orbitize/gpu_context.py:import pycuda.driver as cuda
orbitize/gpu_context.py:import pycuda.autoinit
orbitize/gpu_context.py:from pycuda.compiler import SourceModule
orbitize/gpu_context.py:class gpu_context:
orbitize/gpu_context.py:    GPU context which manages the allocation of memory, the movement of memory between python and the GPU, 
orbitize/gpu_context.py:    and the calling of GPU funcitons
orbitize/gpu_context.py:    def __init__(self, len_gpu_arrays = 10000000):
orbitize/gpu_context.py:        self.gpu_initalized = False
orbitize/gpu_context.py:        self.len_gpu_arrays = len_gpu_arrays
orbitize/gpu_context.py:            self.newton_gpu = mod_newton.get_function("newton_gpu")
orbitize/gpu_context.py:            self.mikkola_gpu = mod_mikkola.get_function("mikkola_gpu")
orbitize/gpu_context.py:            print("Allocating with {} bytes".format(self.len_gpu_arrays))
orbitize/gpu_context.py:            self.d_manom = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:            self.d_ecc = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:            self.d_eanom = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:            self.d_tol = cuda.mem_alloc(self.tolerance.nbytes)
orbitize/gpu_context.py:            self.d_max_iter = cuda.mem_alloc(self.max_iter.nbytes)
orbitize/gpu_context.py:            print("Copying parameters to GPU")
orbitize/gpu_context.py:            cuda.memcpy_htod(self.d_tol, self.tolerance)
orbitize/gpu_context.py:            cuda.memcpy_htod(self.d_max_iter, self.max_iter)
orbitize/gpu_context.py:            gpu_initalized = True
orbitize/gpu_context.py:            print("Error: KEPLER: Unable to initialize Kepler GPU solver context")
orbitize/gpu_context.py:        Moves numpy arrays onto the GPU memory, calls the Newton-Raphson solver for eccentric anomaly
orbitize/gpu_context.py:        if (self.len_gpu_arrays < manom.nbytes):
orbitize/gpu_context.py:            self.len_gpu_arrays = manom.nbytes
orbitize/gpu_context.py:            self.d_manom = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:            self.d_ecc = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:            self.d_eanom = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:        cuda.memcpy_htod(self.d_manom, manom)
orbitize/gpu_context.py:        cuda.memcpy_htod(self.d_ecc, ecc)
orbitize/gpu_context.py:        cuda.memcpy_htod(self.d_tol, tolerance)
orbitize/gpu_context.py:        cuda.memcpy_htod(self.d_max_iter, max_iter)
orbitize/gpu_context.py:            cuda.memcpy_dtod(self.d_eanom, self.d_manom, self.len_gpu_arrays)
orbitize/gpu_context.py:            cuda.memcpy_htod(self.d_eanom, eanom0)
orbitize/gpu_context.py:        self.newton_gpu(self.d_manom, self.d_ecc, self.d_eanom, self.d_max_iter, self.d_tol, grid = (len(manom)//64+1,1,1), block = (64,1,1))
orbitize/gpu_context.py:        cuda.memcpy_dtoh(eanom, self.d_eanom)
orbitize/gpu_context.py:        Moves numpy arrays onto the GPU memory, calls the analtyical Mikkola solver for eccentric anomaly
orbitize/gpu_context.py:        if (self.len_gpu_arrays < manom.nbytes):
orbitize/gpu_context.py:            self.len_gpu_arrays = manom.nbytes
orbitize/gpu_context.py:            self.d_manom = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:            self.d_ecc = cuda.mem_alloc(self.len_gpu_arrays)
orbitize/gpu_context.py:        cuda.memcpy_htod(self.d_manom, manom)
orbitize/gpu_context.py:        cuda.memcpy_htod(self.d_ecc, ecc)
orbitize/gpu_context.py:        self.mikkola_gpu(self.d_manom, self.d_ecc, self.d_eanom, grid = (len(manom)//64+1,1,1), block = (64,1,1))
orbitize/gpu_context.py:        cuda.memcpy_dtoh(eanom, self.d_eanom)
tests/test_kepler_solver.py:from orbitize import cuda_ext
tests/test_kepler_solver.py:def test_analytical_ecc_anom_solver(use_c = False, use_gpu = False):
tests/test_kepler_solver.py:        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, tolerance=1e-9, use_c=use_c, use_gpu = use_gpu)
tests/test_kepler_solver.py:def test_iterative_ecc_anom_solver(use_c = False, use_gpu = False):
tests/test_kepler_solver.py:        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, tolerance=1e-9, use_c=use_c, use_gpu = use_gpu)
tests/test_kepler_solver.py:def test_pycuda_ecc_anom_solver():
tests/test_kepler_solver.py:    if cuda_ext:
tests/test_kepler_solver.py:        test_iterative_ecc_anom_solver(use_gpu = True)
tests/test_kepler_solver.py:        test_analytical_ecc_anom_solver(use_gpu = True)
tests/test_kepler_solver.py:def profile_iterative_ecc_anom_solver(n_orbits = 1000, use_c = True, use_gpu = False):
tests/test_kepler_solver.py:        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, tolerance=1e-9, use_c = use_c, use_gpu = use_gpu)
tests/test_kepler_solver.py:def profile_mikkola_ecc_anom_solver(n_orbits = 1000, use_c = True, use_gpu = False):
tests/test_kepler_solver.py:        ecc_anoms = kepler._calc_ecc_anom(mean_anoms, ee, use_c = use_c, use_gpu = use_gpu)
tests/test_kepler_solver.py:        if cuda_ext:
tests/test_kepler_solver.py:            cProfile.runctx("profile_iterative_ecc_anom_solver(n_orbits = n_orbits, use_c = False, use_gpu = True)", globals(), locals(), profile_name)
tests/test_kepler_solver.py:                print("Profiling Newton: CUDA with {} orbits".format(n_orbits**2))
tests/test_kepler_solver.py:            d["Newton GPU Solver"] = s.__dict__["total_tt"]
tests/test_kepler_solver.py:            print("System not configured for CUDA")
tests/test_kepler_solver.py:        if cuda_ext:
tests/test_kepler_solver.py:            cProfile.runctx("profile_mikkola_ecc_anom_solver(n_orbits = n_orbits, use_c = False, use_gpu = True)", globals(), locals(), profile_name)
tests/test_kepler_solver.py:                print("Profiling Mikkola: CUDA with {} orbits".format(n_orbits**2))
tests/test_kepler_solver.py:            d["Mikkola GPU Solver"] = s.__dict__["total_tt"]
tests/test_kepler_solver.py:            cProfile.runctx("profile_mikkola_ecc_anom_solver(n_orbits = n_orbits, use_c = True, use_gpu = False)", globals(), locals(), profile_name)
tests/test_kepler_solver.py:        cProfile.runctx("profile_mikkola_ecc_anom_solver(n_orbits = n_orbits, use_c = False, use_gpu = False)", globals(), locals(), profile_name)
tests/test_kepler_solver.py:        test_pycuda_ecc_anom_solver()
tests/test_OFTI.py:    import pycuda.driver
tests/test_OFTI.py:    # pycuda.driver.initialize_profiler()
tests/test_OFTI.py:    pycuda.driver.start_profiler()
tests/test_OFTI.py:    orbitize.cuda_ext = True
tests/test_OFTI.py:    print("CUDA Runtime: " + str(end - start) + " s")
tests/test_OFTI.py:    orbitize.cuda_ext = False
tests/test_OFTI.py:    pycuda.driver.stop_profiler()
tests/test_OFTI.py:    orbitize.cuda_ext = False
tests/test_OFTI.py:    pycuda.driver.stop_profiler()
tests/test_OFTI.py:    pycuda.autoinit.context.detach()

```
