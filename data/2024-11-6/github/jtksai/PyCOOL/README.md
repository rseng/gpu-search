# https://github.com/jtksai/PyCOOL

```console
cuda_templates/rho_pres_new.cu:Part of this code adapted from CUDAEASY
cuda_templates/rho_pres_new.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/rho_pres_new.cu:    // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:        // In a multi-gpu implementation these values could be loaded fjennifer lopez heightrom a different device
cuda_templates/rho_pres_new.cu:    // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/rho_pres_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:Part of this code adapted from CUDAEASY
cuda_templates/kernel_gws_new.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/kernel_gws_new.cu:Nvidia SDK FDTD3dGPU kernel
cuda_templates/kernel_gws_new.cu:(See http://developer.nvidia.com/gpu-computing-sdk .) 
cuda_templates/kernel_gws_new.cu:    // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:        // In a multi-gpu implementation these values could be loaded fjennifer lopez heightrom a different device
cuda_templates/kernel_gws_new.cu:    // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_gws_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H2.cu:Part of this code adapted from CUDAEASY
cuda_templates/kernel_H2.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/kernel_linear_evo.cu:Part of this code adapted from CUDAEASY
cuda_templates/kernel_linear_evo.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/spatial_corr.cu:Part of this code adapted from CUDAEASY
cuda_templates/spatial_corr.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/pd_kernel.cu:Part of this code adapted from CUDAEASY
cuda_templates/pd_kernel.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/rho_pres.cu:Part of this code adapted from CUDAEASY
cuda_templates/rho_pres.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/kernel_k.cu:__global__ void gpu_k_vec({{ real_name_c  }} *k_x, {{ real_name_c  }} *k_y, {{ real_name_c  }} *k_z, {{ real_name_c  }} *k2_abs)
cuda_templates/kernel_gws.cu:Part of this code adapted from CUDAEASY
cuda_templates/kernel_gws.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/kernel_k2.cu:__global__ void gpu_k2({{ real_name_c  }} *g_output)
cuda_templates/kernel_k2.cu:__global__ void gpu_k_vec({{ real_name_c  }} *k_x, {{ real_name_c  }} *k_y, {{ real_name_c  }} *k_z, {{ real_name_c  }} *k2_abs)
cuda_templates/kernel_k2.cu:__global__ void gpu_k2_to_bin({{ real_name_c  }} *k2_array, {{ real_name_c  }} *k2_bins, int *k2_bin_id, int bins)
cuda_templates/kernel_k2.cu:__global__ void gpu_evolve_lin_fields({{ complex_name_c }} *Fk_1_m{% for i in range(2,fields_c+1) %}, {{ complex_name_c }} *Fk_{{i}}_m{% endfor %}, {{ complex_name_c }} *Pik_1_m{% for i in range(2,fields_c+1) %}, {{ complex_name_c }} *Pik_{{i}}_m{% endfor %}, {{ real_name_c }} *f_lin_01_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_01_{{i}}_m{% endfor %}, {{ real_name_c }} *pi_lin_01_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_01_{{i}}_m{% endfor %}, {{ real_name_c }} *f_lin_10_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *f_lin_10_{{i}}_m{% endfor %}, {{ real_name_c }} *pi_lin_10_1_m{% for i in range(2,fields_c+1) %}, {{ real_name_c }} *pi_lin_10_{{i}}_m{% endfor %}, int *k2_bin_id)
cuda_templates/kernel_H3.cu:Part of this code adapted from CUDAEASY
cuda_templates/kernel_H3.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/kernel_H3_new.cu:Part of this code adapted from CUDAEASY
cuda_templates/kernel_H3_new.cu:http://www.physics.utu.fi/tiedostot/theory/particlecosmology/cudaeasy/
cuda_templates/kernel_H3_new.cu:Nvidia SDK FDTD3dGPU kernel
cuda_templates/kernel_H3_new.cu:(See http://developer.nvidia.com/gpu-computing-sdk .) 
cuda_templates/kernel_H3_new.cu:    // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:        // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:    // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
cuda_templates/kernel_H3_new.cu:            // In a multi-gpu implementation these values could be loaded from a different device
init/gpu_3dconv.cu:__global__ void gpu_3dconv(double *g_output,
init/field_init.py:from pycuda.tools import make_default_context
init/field_init.py:import pycuda.gpuarray as gpuarray
init/field_init.py:import pycuda.driver as cuda
init/field_init.py:import pycuda.autoinit
init/field_init.py:from pycuda.compiler import SourceModule
init/field_init.py:    func = name of Cuda kernel
init/field_init.py:    ker_gpu = gpuarray.to_gpu(ker)
init/field_init.py:    tmp_gpu = gpuarray.to_gpu(tmp)
init/field_init.py:    func(tmp_gpu, ker_gpu, np.uint32(nn), np.float64(os),
init/field_init.py:         block = lat.cuda_block_1, grid = lat.cuda_grid)
init/field_init.py:    tmp += tmp_gpu.get()
init/field_init.py:    ker_gpu.gpudata.free()
init/field_init.py:    tmp_gpu.gpudata.free()
init/field_init.py:    func = name of Cuda kernel
init/field_init.py:    ker_gpu = gpuarray.to_gpu(ker)
init/field_init.py:    tmp_gpu = gpuarray.to_gpu(tmp)
init/field_init.py:    func(tmp_gpu, ker_gpu, np.uint32(nn), np.float64(os),
init/field_init.py:         block = lat.cuda_block_1, grid = lat.cuda_grid)
init/field_init.py:    tmp += tmp_gpu.get()
init/field_init.py:def sample_defrost_gpu(lat, func, gamma, m2_eff):
init/field_init.py:    func = name of Cuda kernel
init/field_init.py:    import scikits.cuda.fft as fft
init/field_init.py:    Fk_gpu = gpuarray.zeros((n/2+1,n,n), dtype = lat.prec_complex)
init/field_init.py:    ker_gpu = gpuarray.to_gpu(ker)
init/field_init.py:    tmp_gpu = gpuarray.zeros((n,n,n),dtype = lat.prec_real)
init/field_init.py:    plan = fft.Plan(tmp_gpu.shape, lat.prec_real, lat.prec_complex)
init/field_init.py:    plan2 = fft.Plan(tmp_gpu.shape, lat.prec_complex, lat.prec_real)
init/field_init.py:    func(tmp_gpu, ker_gpu, np.uint32(nn), np.float64(os),
init/field_init.py:         block = lat.cuda_block_1, grid = lat.cuda_grid)
init/field_init.py:    fft.fft(tmp_gpu, Fk_gpu, plan)
init/field_init.py:    rr1 = (np.random.normal(size=Fk_gpu.shape)+
init/field_init.py:           np.random.normal(size=Fk_gpu.shape)*1j)
init/field_init.py:    Fk = Fk_gpu.get()
init/field_init.py:    Fk_gpu = gpuarray.to_gpu(Fk)
init/field_init.py:    fft.ifft(Fk_gpu, tmp_gpu, plan2)
init/field_init.py:    res = (tmp_gpu.get()).astype(lat.prec_real)
init/field_init.py:    falg_gpu = 'cpu' or 'gpu'. If 'gpu' then Fast Fourier Transforms calculated
init/field_init.py:               on the gpu.
init/field_init.py:    "Open and compile Cuda kernels"
init/field_init.py:    f = codecs.open('init/gpu_3dconv.cu','r',encoding='utf-8')
init/field_init.py:    gpu_3dconv = f.read()
init/field_init.py:    mod = SourceModule(gpu_3dconv)
init/field_init.py:    gpu_conv = mod.get_function("gpu_3dconv")
init/field_init.py:    if flag_method=='defrost_gpu':
init/field_init.py:        f = sample_defrost_gpu(lat, gpu_conv,-0.25, m2_eff) + c*field0
init/field_init.py:        print "\nField " + repr(field_i)+ " init on gpu done"
init/field_init.py:        f = sample_defrost_cpu(lat, gpu_conv,-0.25, m2_eff) + c*field0
init/field_init.py:        f = sample_defrost_cpu2(lat, gpu_conv,-0.25, m2_eff) + c*field0
init/field_init.py:    falg_gpu = 'defrost_cpu' or 'defrost_gpu'. If 'gpu' then Fast Fourier Transforms calculated
init/field_init.py:               on the gpu.
init/field_init.py:    "Open and compile Cuda kernels"
init/field_init.py:    f = codecs.open('init/gpu_3dconv.cu','r',encoding='utf-8')
init/field_init.py:    gpu_3dconv = f.read()
init/field_init.py:    mod = SourceModule(gpu_3dconv)
init/field_init.py:    gpu_conv = mod.get_function("gpu_3dconv")
init/field_init.py:    if flag_method=='defrost_gpu':
init/field_init.py:        fp = sample_defrost_gpu(lat, gpu_conv, 0.25, m2_eff) + c*pi0
init/field_init.py:        print "Field " + repr(field_i)+ " time derivative init on gpu done"
init/field_init.py:        fp = sample_defrost_cpu(lat, gpu_conv, 0.25, m2_eff) + c*pi0
init/field_init.py:        fp = sample_defrost_cpu2(lat, gpu_conv, 0.25, m2_eff) + c*pi0
integrator/symp_integrator.py:import pycuda.driver as cuda
integrator/symp_integrator.py:import pycuda.autoinit
integrator/symp_integrator.py:import pycuda.gpuarray as gpuarray
integrator/symp_integrator.py:from pycuda.compiler import SourceModule
integrator/symp_integrator.py:def kernel_H2_gpu_code(lat, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/kernel_H2.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_H3_gpu_code(lat, field_i, V, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/kernel_H3.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_H3_new_gpu_code(lat, field_i, V, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/kernel_H3_new.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_lin_evo_gpu_code(lat, V, sim, write_code=True):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/kernel_linear_evo.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_k2_gpu_code(lat, V, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/'+kernel_name+'.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_rho_pres_gpu_code(lat, field_i, V, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/rho_pres.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_rho_pres_new_gpu_code(lat, field_i, V, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/rho_pres_new.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_gws_gpu_code(lat, tensor_ij, V, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/kernel_gws.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_gws_new_gpu_code(lat, tensor_ij, V, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/kernel_gws_new.cu','r',encoding='utf-8')
integrator/symp_integrator.py:def kernel_spat_corr_gpu_code(lat, write_code=False):
integrator/symp_integrator.py:    f = codecs.open('cuda_templates/spatial_corr.cu','r',encoding='utf-8')
integrator/symp_integrator.py:        self.mod = kernel_H2_gpu_code(lat, write_code)
integrator/symp_integrator.py:        cuda.memcpy_htod_async(self.hc_add[0],x, stream=None)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.hc_add[0])
integrator/symp_integrator.py:            self.mod = kernel_H3_gpu_code(lat, field_i, V, write_code)
integrator/symp_integrator.py:            self.mod = kernel_H3_new_gpu_code(lat, field_i, V, write_code)
integrator/symp_integrator.py:            self.mod = kernel_H3_new_gpu_code(lat, field_i, V, write_code)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.cc_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.cc_add[0])
integrator/symp_integrator.py:        cuda.memcpy_htod(self.cd_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.cd_add[0])
integrator/symp_integrator.py:        cuda.memcpy_htod(self.dc_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.dc_add[0])
integrator/symp_integrator.py:        cuda.memcpy_htod_async(self.fc_add[0], x, stream=None)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.fc_add[0])
integrator/symp_integrator.py:            self.mod = kernel_gws_gpu_code(lat, tensor_ij, V, write_code)
integrator/symp_integrator.py:            self.mod = kernel_gws_new_gpu_code(lat, tensor_ij, V, write_code)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.cc_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.cc_add[0])
integrator/symp_integrator.py:        cuda.memcpy_htod_async(self.gwc_add[0], x, stream=None)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.gwc_add[0])
integrator/symp_integrator.py:            self.mod = kernel_lin_evo_gpu_code(lat, V, sim, write_code)
integrator/symp_integrator.py:        self.mod2 = kernel_k2_gpu_code(lat, V, write_code)
integrator/symp_integrator.py:        self.k2_calc = self.mod2.get_function('gpu_k2')
integrator/symp_integrator.py:        self.k_vec_calc = self.mod2.get_function('gpu_k_vec')
integrator/symp_integrator.py:        self.k2_bins_calc = self.mod2.get_function('gpu_k2_to_bin')
integrator/symp_integrator.py:        self.lin_field_evo = self.mod2.get_function('gpu_evolve_lin_fields')
integrator/symp_integrator.py:            self.mod = kernel_rho_pres_gpu_code(lat, field_i, V, write_code)
integrator/symp_integrator.py:            self.mod = kernel_rho_pres_new_gpu_code(lat, field_i, V, write_code)
integrator/symp_integrator.py:            self.mod = kernel_rho_pres_new_gpu_code(lat, field_i, V, write_code)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.cc_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.cc_add[0])
integrator/symp_integrator.py:        cuda.memcpy_htod(self.dc_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.dc_add[0])
integrator/symp_integrator.py:        cuda.memcpy_htod_async(self.gc_add[0], x, stream)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.gc_add[0])
integrator/symp_integrator.py:        self.mod = kernel_spat_corr_gpu_code(lat, write_code)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.cc_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.cc_add[0])
integrator/symp_integrator.py:        cuda.memcpy_htod(self.cor_c_add[0],x)
integrator/symp_integrator.py:        cuda.memcpy_dtoh(x,self.cor_c_add[0])
integrator/symp_integrator.py:        - total energy density field rho_gpu
integrator/symp_integrator.py:        - total pressure density field pres_gpu
integrator/symp_integrator.py:        - sum over z-direction of energy density rhosum_gpu
integrator/symp_integrator.py:        - sum over z-direction of pressure density pressum_gpu
integrator/symp_integrator.py:        - sum_gpu that is used when updating canonical momentum p
integrator/symp_integrator.py:        - pa_gpu used in calc_wk_conf function. It equals p'/a.
integrator/symp_integrator.py:        self.t_gpu = gpuarray.to_gpu(np.array(self.t, dtype = lat.prec_real))
integrator/symp_integrator.py:        self.a_gpu = gpuarray.to_gpu(np.array(self.a, dtype = lat.prec_real))
integrator/symp_integrator.py:        self.p_gpu = gpuarray.to_gpu(np.array(self.p, dtype = lat.prec_real))
integrator/symp_integrator.py:            self.k2_field_gpu = gpuarray.to_gpu(np.zeros(lat.dims_k,
integrator/symp_integrator.py:            self.k2_bins_gpu = gpuarray.to_gpu(self.k2_bins)
integrator/symp_integrator.py:            self.k2_bin_id = gpuarray.to_gpu(self.zeros_i)
integrator/symp_integrator.py:        """Number of steps in linearized CUDA kernel.
integrator/symp_integrator.py:           If only one CUDA device in the system more than 1e6 steps
integrator/symp_integrator.py:        if cuda.Device.count() == 1 and steps > 1000000:
integrator/symp_integrator.py:            print 'Only one CUDA device detected. Set steps to 1000000.\n'
integrator/symp_integrator.py:        elif cuda.Device.count() == 1 and steps <= 1000000:
integrator/symp_integrator.py:        elif cuda.Device.count() > 1:
integrator/symp_integrator.py:        "Various GPU-memory arrays:"
integrator/symp_integrator.py:        self.rho_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.rho_host = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.pres_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.rhosum_gpu = gpuarray.zeros(lat.dims_xy, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.rhosum_host = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.pressum_gpu = gpuarray.zeros(lat.dims_xy, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.pressum_host = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.sum_gpu = gpuarray.zeros(lat.dims_xy, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.sum_host = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.inter_sum_gpu = gpuarray.zeros(lat.dims_xy, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.inter_sum_host = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.sum_nabla_rho_gpu = gpuarray.zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.sum_nabla_rho_h = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.sum_rho_squ_gpu = gpuarray.zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.sum_rho_squ_host = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.pd_gpu = gpuarray.zeros(shape = lat.dims_xy,
integrator/symp_integrator.py:            self.u11_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u12_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u22_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u13_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u23_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u33_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu11_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu12_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu22_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu13_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu23_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu33_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:        self.d_array = cuda.pagelocked_empty(V.d_coeff_l,
integrator/symp_integrator.py:        self.f_array = cuda.pagelocked_empty(V.f_coeff_l,
integrator/symp_integrator.py:        self.g_array = cuda.pagelocked_empty(V.g_coeff_l,
integrator/symp_integrator.py:        self.h_array = cuda.pagelocked_empty(V.h_coeff_l,
integrator/symp_integrator.py:        self.lin_array = cuda.pagelocked_empty(V.lin_coeff_l,
integrator/symp_integrator.py:        self.gw_array = cuda.pagelocked_empty(1,dtype = lat.prec_real)
integrator/symp_integrator.py:        "Create a Cuda Stream for the simulation:"
integrator/symp_integrator.py:        self.stream = cuda.Stream()
integrator/symp_integrator.py:            f = field.f_gpu.get()
integrator/symp_integrator.py:            pi = field.pi_gpu.get()
integrator/symp_integrator.py:            cuda.memcpy_htod(field.f_gpu.gpudata, field.f)
integrator/symp_integrator.py:            cuda.memcpy_htod(field.pi_gpu.gpudata, field.pi)
integrator/symp_integrator.py:        k2n = lat.dx**2.*(self.k2_field_gpu.get())
integrator/symp_integrator.py:        "Used CUDA block size:"
integrator/symp_integrator.py:        block = lat.cuda_lin_block[0]
integrator/symp_integrator.py:        """Make k2_bin a multiple of CUDA block size. What this means is
integrator/symp_integrator.py:        self.k2_bins_gpu = gpuarray.to_gpu(self.k2_bins)
integrator/symp_integrator.py:                dset = subgroup.create_dataset('f', data=field.f_gpu.get())
integrator/symp_integrator.py:                dset2 = subgroup.create_dataset('pi', data=field.pi_gpu.get())
integrator/symp_integrator.py:            dset = subgroup.create_dataset('rho', data=self.rho_gpu.get())
integrator/symp_integrator.py:            dset2 = subgroup.create_dataset('pres', data=self.pres_gpu.get())
integrator/symp_integrator.py:                                   np.asarray(field.f_gpu.get(), order="F"),
integrator/symp_integrator.py:                                   field.f_gpu.shape,
integrator/symp_integrator.py:                                   np.asarray(field.pi_gpu.get(), order="F"),
integrator/symp_integrator.py:                                   field.pi_gpu.shape,
integrator/symp_integrator.py:                                       np.asarray(field.rho_gpu.get(),
integrator/symp_integrator.py:                                       field.rho_gpu.shape,
integrator/symp_integrator.py:                               np.asarray(c1*self.rho_gpu.get(), order="F"),
integrator/symp_integrator.py:                               self.rho_gpu.shape,
integrator/symp_integrator.py:                               np.asarray(c1*self.pres_gpu.get(), order="F"),
integrator/symp_integrator.py:                               self.pres_gpu.shape,
integrator/symp_integrator.py:                cuda.memcpy_htod(field.f_gpu.gpudata, field.f)
integrator/symp_integrator.py:                cuda.memcpy_htod(field.pi_gpu.gpudata, field.pi)
integrator/symp_integrator.py:            cuda.memcpy_htod(self.rho_gpu.gpudata, f['rp']['rho'].value)
integrator/symp_integrator.py:            cuda.memcpy_htod(self.pres_gpu.gpudata, f['rp']['pres'].value)
integrator/symp_integrator.py:                cuda.memcpy_htod(field.f_gpu.gpudata, field.f)
integrator/symp_integrator.py:                cuda.memcpy_htod(field.pi_gpu.gpudata, field.pi)
integrator/symp_integrator.py:            cuda.memcpy_htod(self.rho_gpu.gpudata,
integrator/symp_integrator.py:            cuda.memcpy_htod(self.pres_gpu.gpudata,
integrator/symp_integrator.py:        #self.t_gpu = gpuarray.to_gpu(np.array(self.t, dtype = lat.prec_real))
integrator/symp_integrator.py:        #self.a_gpu = gpuarray.to_gpu(np.array(self.a, dtype = lat.prec_real))
integrator/symp_integrator.py:        #self.p_gpu = gpuarray.to_gpu(np.array(self.p, dtype = lat.prec_real))
integrator/symp_integrator.py:            cuda.memcpy_htod(self.a_gpu.gpudata, np.array(a,lat.prec_real))
integrator/symp_integrator.py:            cuda.memcpy_htod(self.p_gpu.gpudata, np.array(self.p,lat.prec_real))
integrator/symp_integrator.py:            cuda.memcpy_htod(self.t_gpu.gpudata,
integrator/symp_integrator.py:            self.u11_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u12_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u22_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u13_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u23_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.u33_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu11_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu12_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu22_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu13_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu23_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:            self.piu33_gpu = gpuarray.zeros(lat.dims_xyz, dtype = lat.prec_real)
integrator/symp_integrator.py:                cuda.memcpy_htod(field.f0_gpu.gpudata, field.f0_field)
integrator/symp_integrator.py:                cuda.memcpy_htod(field.pi0_gpu.gpudata, field.pi0_field)
integrator/symp_integrator.py:        - self.f_gpu = field variable in device memory.
integrator/symp_integrator.py:        - self.pi_gpu = canonical momentum in device memory.
integrator/symp_integrator.py:        - self.f_perturb0_gpu = gpu_array of zeros used in the linearized
integrator/symp_integrator.py:        - self.pi_perturb0_gpu = gpu_array of zeros used in the linearized
integrator/symp_integrator.py:        - self.f_perturb1_gpu = gpu_array of ones used in the linearized
integrator/symp_integrator.py:        - self.pi_perturb1_gpu = gpu_array of ones used in the linearized
integrator/symp_integrator.py:        - self.d2V_sum_gpu = sum over z-direction of effective mass."""
integrator/symp_integrator.py:        self.f_gpu = gpuarray.to_gpu(self.f)
integrator/symp_integrator.py:        self.pi_gpu = gpuarray.to_gpu(self.pi)
integrator/symp_integrator.py:        self.f0_gpu = gpuarray.to_gpu(self.f0_field)
integrator/symp_integrator.py:        self.pi0_gpu = gpuarray.to_gpu(self.pi0_field)
integrator/symp_integrator.py:           automatically in the CUDA kernel before
integrator/symp_integrator.py:        self.f_lin_01_gpu = gpuarray.to_gpu(self.zeros)
integrator/symp_integrator.py:        self.pi_lin_01_gpu = gpuarray.to_gpu(self.zeros)
integrator/symp_integrator.py:        self.f_lin_10_gpu = gpuarray.to_gpu(self.ones)
integrator/symp_integrator.py:        self.pi_lin_10_gpu = gpuarray.to_gpu(self.ones)
integrator/symp_integrator.py:        self.rho_sum_gpu = gpuarray.zeros(shape = lat.dims_xy,
integrator/symp_integrator.py:        self.pres_sum_gpu = gpuarray.zeros(shape = lat.dims_xy,
integrator/symp_integrator.py:            self.rho_gpu = gpuarray.zeros(lat.dims_xyz,
integrator/symp_integrator.py:            #self.pres_gpu = gpuarray.zeros(lat.dims_xyz,
integrator/symp_integrator.py:            self.sum_nabla_rho_gpu = gpuarray.zeros(lat.dims_xy,
integrator/symp_integrator.py:            self.sum_nabla_rho_h = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:            self.sum_rho_squ_gpu = gpuarray.zeros(lat.dims_xy,
integrator/symp_integrator.py:            self.sum_rho_squ_h = cuda.pagelocked_zeros(lat.dims_xy,
integrator/symp_integrator.py:        self.d2V_sum_gpu = gpuarray.zeros(shape = lat.dims_xy,
integrator/symp_integrator.py:        "Subtract the homogeneous value from the f_gpu and pi_gpu arrays"
integrator/symp_integrator.py:        tmp = self.f_gpu.get()
integrator/symp_integrator.py:        tmp2 = self.pi_gpu.get()
integrator/symp_integrator.py:        cuda.memcpy_htod(self.f_gpu.gpudata, tmp)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.pi_gpu.gpudata, tmp2)
integrator/symp_integrator.py:        "Subtract the homogeneous value from the f_gpu and pi_gpu arrays"
integrator/symp_integrator.py:        tmp = self.f_gpu.get() - self.f0
integrator/symp_integrator.py:        tmp2 = self.pi_gpu.get() - self.pi0
integrator/symp_integrator.py:        cuda.memcpy_htod(self.f_gpu.gpudata, tmp)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.pi_gpu.gpudata, tmp2)
integrator/symp_integrator.py:        "Add the homogeneous value to the f_gpu and pi_gpu arrays"
integrator/symp_integrator.py:        tmp = self.f_gpu.get() + self.f0
integrator/symp_integrator.py:        tmp2 = self.pi_gpu.get() + self.pi0
integrator/symp_integrator.py:        cuda.memcpy_htod(self.f_gpu.gpudata, tmp)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.pi_gpu.gpudata, tmp2)
integrator/symp_integrator.py:        """Fourier transform of f_gpu and pi_gpu. Note that this will change
integrator/symp_integrator.py:           the id of f_gpu and pi_gpu. Update function has to be called after
integrator/symp_integrator.py:        tmp = fft.rfftn(self.f_gpu.get()).transpose()
integrator/symp_integrator.py:        tmp2 = fft.rfftn(self.pi_gpu.get()).transpose()
integrator/symp_integrator.py:        self.f_gpu = gpuarray.to_gpu(np.array(tmp))
integrator/symp_integrator.py:        self.pi_gpu = gpuarray.to_gpu(np.array(tmp2))
integrator/symp_integrator.py:        """Inverse Fourier transform of f_gpu and pi_gpu.
integrator/symp_integrator.py:           Note that this will change the id of f_gpu and pi_gpu.
integrator/symp_integrator.py:        tmp = fft.irfftn(self.f_gpu.get().transpose())
integrator/symp_integrator.py:        tmp2 = fft.irfftn(self.pi_gpu.get().transpose())
integrator/symp_integrator.py:        self.f_gpu = gpuarray.to_gpu(np.array(tmp))
integrator/symp_integrator.py:        self.pi_gpu = gpuarray.to_gpu(np.array(tmp2))
integrator/symp_integrator.py:        cuda.memcpy_htod(self.f_gpu.gpudata, self.f)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.pi_gpu.gpudata, self.pi)
integrator/symp_integrator.py:        """Set f_lin_0_gpu and pi_lin_0_gpu arrays to zero and
integrator/symp_integrator.py:           Set f_lin_1_gpu and pi_lin_1_gpu arrays to one."""
integrator/symp_integrator.py:        cuda.memcpy_htod(self.f_lin_0_gpu.gpudata, self.zeros)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.pi_lin_0_gpu.gpudata, self.zeros)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.f_lin_1_gpu.gpudata, self.ones)
integrator/symp_integrator.py:        cuda.memcpy_htod(self.pi_lin_1_gpu.gpudata, self.ones)
integrator/symp_integrator.py:        self.H2_kernel.evo1.set_cache_config(cuda.func_cache.PREFER_L1)
integrator/symp_integrator.py:        self.H2_kernel.evo2.set_cache_config(cuda.func_cache.PREFER_L1)
integrator/symp_integrator.py:            kernel.evo.set_cache_config(cuda.func_cache.PREFER_L1)
integrator/symp_integrator.py:                kernel.evo.set_cache_config(cuda.func_cache.PREFER_L1)
integrator/symp_integrator.py:        "Cuda function arguments used in H2 and H3:"
integrator/symp_integrator.py:        self.cuda_H2_arg = [[sim.sum_gpu]]
integrator/symp_integrator.py:            self.cuda_H2_arg[0].append(f.f_gpu)
integrator/symp_integrator.py:            self.cuda_H2_arg[0].append(f.pi_gpu)
integrator/symp_integrator.py:        "Cuda function argument used in tensor perturbation kernel:"
integrator/symp_integrator.py:            self.cuda_H2_arg.append([sim.u11_gpu,
integrator/symp_integrator.py:                                     sim.u12_gpu,
integrator/symp_integrator.py:                                     sim.u22_gpu,
integrator/symp_integrator.py:                                     sim.u13_gpu,
integrator/symp_integrator.py:                                     sim.u23_gpu,
integrator/symp_integrator.py:                                     sim.u33_gpu,
integrator/symp_integrator.py:                                     sim.piu11_gpu,
integrator/symp_integrator.py:                                     sim.piu12_gpu,
integrator/symp_integrator.py:                                     sim.piu22_gpu,
integrator/symp_integrator.py:                                     sim.piu13_gpu,
integrator/symp_integrator.py:                                     sim.piu23_gpu,
integrator/symp_integrator.py:                                     sim.piu33_gpu])
integrator/symp_integrator.py:        self.cuda_H3_arg = [[sim.sum_gpu]]
integrator/symp_integrator.py:            self.cuda_H3_arg[0].append(f.f_gpu)
integrator/symp_integrator.py:            self.cuda_H3_arg[0].append(f.pi_gpu)
integrator/symp_integrator.py:            self.cuda_H3_arg[0].extend([sim.piu11_gpu,sim.piu12_gpu,
integrator/symp_integrator.py:                                        sim.piu22_gpu,sim.piu13_gpu,
integrator/symp_integrator.py:                                        sim.piu23_gpu,sim.piu33_gpu])
integrator/symp_integrator.py:        "Cuda function arguments used in tensor perturbation Laplacian kernel:"
integrator/symp_integrator.py:            self.cuda_H3_arg.append([sim.u11_gpu,
integrator/symp_integrator.py:                                     sim.u12_gpu,
integrator/symp_integrator.py:                                     sim.u22_gpu,
integrator/symp_integrator.py:                                     sim.u13_gpu,
integrator/symp_integrator.py:                                     sim.u23_gpu,
integrator/symp_integrator.py:                                     sim.u33_gpu,
integrator/symp_integrator.py:                                     sim.piu11_gpu,
integrator/symp_integrator.py:                                     sim.piu12_gpu,
integrator/symp_integrator.py:                                     sim.piu22_gpu,
integrator/symp_integrator.py:                                     sim.piu13_gpu,
integrator/symp_integrator.py:                                     sim.piu23_gpu,
integrator/symp_integrator.py:                                     sim.piu33_gpu])
integrator/symp_integrator.py:        self.cuda_param_H2 = dict(block=lat.cuda_block_1,
integrator/symp_integrator.py:                                  grid=lat.cuda_grid,
integrator/symp_integrator.py:            self.cuda_param_H3 = dict(block=lat.cuda_block_2,
integrator/symp_integrator.py:                                      grid=lat.cuda_grid,
integrator/symp_integrator.py:            self.cuda_param_H3 = dict(block=lat.cuda_block_1,
integrator/symp_integrator.py:                                      grid=lat.cuda_grid,
integrator/symp_integrator.py:            self.lin_evo_kernel.k2_calc(sim.k2_field_gpu, **self.cuda_param_H2)
integrator/symp_integrator.py:            sim.k2_field = (sim.k2_field_gpu.get()).astype(np.float64)
integrator/symp_integrator.py:            "Calculate into which bin an element of sim.k2_field_gpu belongs:"
integrator/symp_integrator.py:            self.k2_arg.append(sim.k2_field_gpu)
integrator/symp_integrator.py:            self.k2_arg.append(sim.k2_bins_gpu)
integrator/symp_integrator.py:            self.lin_evo_kernel.k2_bins_calc(*self.k2_arg, **self.cuda_param_H2)
integrator/symp_integrator.py:                field.f_lin_01_gpu = gpuarray.zeros_like(sim.k2_bins_gpu)
integrator/symp_integrator.py:                field.pi_lin_01_gpu = gpuarray.zeros_like(sim.k2_bins_gpu)
integrator/symp_integrator.py:                field.f_lin_10_gpu = gpuarray.zeros_like(sim.k2_bins_gpu)
integrator/symp_integrator.py:                field.pi_lin_10_gpu = gpuarray.zeros_like(sim.k2_bins_gpu)
integrator/symp_integrator.py:            "Cuda function arguments used in lin_evo:"
integrator/symp_integrator.py:                self.lin_e_arg.append(f.f0_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.pi0_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.f_lin_01_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.pi_lin_01_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.f_lin_10_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.pi_lin_10_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.a_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.p_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.t_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.k2_bins_gpu)
integrator/symp_integrator.py:            grid_lin = len(sim.k2_bins_gpu)/lat.cuda_lin_block[0]
integrator/symp_integrator.py:            self.cuda_param_lin_e = dict(block=lat.cuda_lin_block,
integrator/symp_integrator.py:            self.lin_evo_kernel.k2_calc(sim.k2_field_gpu, **self.cuda_param_H2)
integrator/symp_integrator.py:            sim.k2_field = (sim.k2_field_gpu.get()).astype(np.float64)
integrator/symp_integrator.py:            sim.k2_field_gpu.gpudata.free()
integrator/symp_integrator.py:        "Cuda function arguments used in rho and pressure kernels:"
integrator/symp_integrator.py:        self.rp_arg = [sim.rho_gpu, sim.pres_gpu, sim.rhosum_gpu,
integrator/symp_integrator.py:                       sim.pressum_gpu, sim.inter_sum_gpu]
integrator/symp_integrator.py:            self.rp_arg.append(f.f_gpu)
integrator/symp_integrator.py:            self.rp_arg.append(f.pi_gpu)
integrator/symp_integrator.py:            self.rp_arg.append(f.rho_sum_gpu)
integrator/symp_integrator.py:            self.rp_arg.append(f.pres_sum_gpu)
integrator/symp_integrator.py:                self.rp_arg.append(f.rho_gpu)
integrator/symp_integrator.py:            self.cuda_param_rp = dict(block=lat.cuda_block_2,
integrator/symp_integrator.py:                                      grid=lat.cuda_grid)
integrator/symp_integrator.py:            self.cuda_param_rp = dict(block=lat.cuda_block_1,
integrator/symp_integrator.py:                                      grid=lat.cuda_grid)
integrator/symp_integrator.py:        "Cuda function arguments used in spatial correlation kernel:"
integrator/symp_integrator.py:        self.sc_arg = [sim.rho_gpu, sim.sum_nabla_rho_gpu,
integrator/symp_integrator.py:                       sim.sum_rho_squ_gpu]
integrator/symp_integrator.py:        self.cuda_param_sc = dict(block=lat.cuda_block_2, grid=lat.cuda_grid)
integrator/symp_integrator.py:                      self.rp_kernels, self.cuda_param_rp, self.rp_arg,
integrator/symp_integrator.py:                      self.sc_kernel, self.cuda_param_sc, self.sc_arg,
integrator/symp_integrator.py:                   self.cuda_param_H2, self.cuda_param_H3,
integrator/symp_integrator.py:                   self.cuda_H2_arg, self.cuda_H3_arg, dt)
integrator/symp_integrator.py:                   self.cuda_param_H2, self.cuda_param_H3,
integrator/symp_integrator.py:                   self.cuda_H2_arg, self.cuda_H3_arg, dt)
integrator/symp_integrator.py:                   self.cuda_param_H2, self.cuda_param_H3,
integrator/symp_integrator.py:                   self.cuda_H2_arg, self.cuda_H3_arg, dt)
integrator/symp_integrator.py:                   self.cuda_param_H2, self.cuda_param_H3,
integrator/symp_integrator.py:                   self.cuda_H2_arg, self.cuda_H3_arg, dt)
integrator/symp_integrator.py:                 self.cuda_param_lin_e)
integrator/symp_integrator.py:        sim.a = sim.a_gpu.get().item()
integrator/symp_integrator.py:        sim.p = sim.p_gpu.get().item()
integrator/symp_integrator.py:        sim.t = sim.t_gpu.get().item()
integrator/symp_integrator.py:            field.f0 = field.f0_gpu.get().item()
integrator/symp_integrator.py:            field.pi0 = field.pi0_gpu.get().item()
integrator/symp_integrator.py:        """Print ids of the arrays in different cuda_args. This can be used to
integrator/symp_integrator.py:           verify that the Cuda functions are pointing to correct arrays."""
integrator/symp_integrator.py:            res = [id(x) for x in self.cuda_H2_arg[0]]
integrator/symp_integrator.py:            res = [id(x) for x in self.cuda_H3_arg[0]]
integrator/symp_integrator.py:            args.append(field.f_gpu)
integrator/symp_integrator.py:            args.append(field.pi_gpu)
integrator/symp_integrator.py:            args.append(field.f_lin_01_gpu)
integrator/symp_integrator.py:            args.append(field.pi_lin_01_gpu)
integrator/symp_integrator.py:            args.append(field.f_lin_10_gpu)
integrator/symp_integrator.py:            args.append(field.pi_lin_10_gpu)
integrator/symp_integrator.py:        params = dict(block=lat.cuda_block_1, grid=lat.cuda_grid,
integrator/symp_integrator.py:        "Cuda function arguments used in H2 and H3:"
integrator/symp_integrator.py:        self.cuda_H2_arg = [[sim.sum_gpu]]
integrator/symp_integrator.py:            self.cuda_H2_arg[0].append(f.f_gpu)
integrator/symp_integrator.py:            self.cuda_H2_arg[0].append(f.pi_gpu)
integrator/symp_integrator.py:        "Cuda function argument used in tensor perturbation kernel:"
integrator/symp_integrator.py:            self.cuda_H2_arg.append([sim.u11_gpu,
integrator/symp_integrator.py:                                     sim.u12_gpu,
integrator/symp_integrator.py:                                     sim.u22_gpu,
integrator/symp_integrator.py:                                     sim.u13_gpu,
integrator/symp_integrator.py:                                     sim.u23_gpu,
integrator/symp_integrator.py:                                     sim.u33_gpu,
integrator/symp_integrator.py:                                     sim.piu11_gpu,
integrator/symp_integrator.py:                                     sim.piu12_gpu,
integrator/symp_integrator.py:                                     sim.piu22_gpu,
integrator/symp_integrator.py:                                     sim.piu13_gpu,
integrator/symp_integrator.py:                                     sim.piu23_gpu,
integrator/symp_integrator.py:                                     sim.piu33_gpu])
integrator/symp_integrator.py:        self.cuda_H3_arg = [[sim.sum_gpu]]
integrator/symp_integrator.py:            self.cuda_H3_arg[0].append(f.f_gpu)
integrator/symp_integrator.py:            self.cuda_H3_arg[0].append(f.pi_gpu)
integrator/symp_integrator.py:            self.cuda_H3_arg[0].extend([sim.piu11_gpu,sim.piu12_gpu,
integrator/symp_integrator.py:                                        sim.piu22_gpu,sim.piu13_gpu,
integrator/symp_integrator.py:                                        sim.piu23_gpu,sim.piu33_gpu])
integrator/symp_integrator.py:        "Cuda function arguments used in tensor perturbation Laplacian kernel:"
integrator/symp_integrator.py:            self.cuda_H3_arg.append([sim.u11_gpu,
integrator/symp_integrator.py:                                     sim.u12_gpu,
integrator/symp_integrator.py:                                     sim.u22_gpu,
integrator/symp_integrator.py:                                     sim.u13_gpu,
integrator/symp_integrator.py:                                     sim.u23_gpu,
integrator/symp_integrator.py:                                     sim.u33_gpu,
integrator/symp_integrator.py:                                     sim.piu11_gpu,
integrator/symp_integrator.py:                                     sim.piu12_gpu,
integrator/symp_integrator.py:                                     sim.piu22_gpu,
integrator/symp_integrator.py:                                     sim.piu13_gpu,
integrator/symp_integrator.py:                                     sim.piu23_gpu,
integrator/symp_integrator.py:                                     sim.piu33_gpu])
integrator/symp_integrator.py:        self.cuda_param_H2 = dict(block=lat.cuda_block_1, grid=lat.cuda_grid,
integrator/symp_integrator.py:            self.cuda_param_H3 = dict(block=lat.cuda_block_2,
integrator/symp_integrator.py:                                      grid=lat.cuda_grid,
integrator/symp_integrator.py:            self.cuda_param_H3 = dict(block=lat.cuda_block_1,
integrator/symp_integrator.py:                                      grid=lat.cuda_grid,
integrator/symp_integrator.py:            "Cuda function arguments used in lin_evo:"
integrator/symp_integrator.py:                self.lin_e_arg.append(f.f0_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.pi0_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.f_lin_01_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.pi_lin_01_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.f_lin_10_gpu)
integrator/symp_integrator.py:                self.lin_e_arg.append(f.pi_lin_10_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.a_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.p_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.t_gpu)
integrator/symp_integrator.py:            self.lin_e_arg.append(sim.k2_bins_gpu)
integrator/symp_integrator.py:            grid_lin = len(sim.k2_bins_gpu)/lat.cuda_lin_block[0]
integrator/symp_integrator.py:            self.cuda_param_lin_e = dict(block=lat.cuda_lin_block,
integrator/symp_integrator.py:        "Cuda function arguments used in rho and pressure kernels:"
integrator/symp_integrator.py:        self.rp_arg = [sim.rho_gpu, sim.pres_gpu, sim.rhosum_gpu,
integrator/symp_integrator.py:                       sim.pressum_gpu, sim.inter_sum_gpu]
integrator/symp_integrator.py:            self.rp_arg.append(f.f_gpu)
integrator/symp_integrator.py:            self.rp_arg.append(f.pi_gpu)
integrator/symp_integrator.py:            self.rp_arg.append(f.rho_sum_gpu)
integrator/symp_integrator.py:            self.rp_arg.append(f.pres_sum_gpu)
integrator/symp_integrator.py:                self.rp_arg.append(f.rho_gpu)
integrator/symp_integrator.py:        self.cuda_param_rp = dict(block=lat.cuda_block_2, grid=lat.cuda_grid)
integrator/symp_integrator.py:def calc_rho_pres(lat, V, sim, rp_list, cuda_param_rp, cuda_args,
integrator/symp_integrator.py:                  corr_kernel, cuda_param_sc, cuda_sc_args,
integrator/symp_integrator.py:        kernel.calc(*cuda_args, **cuda_param_rp)
integrator/symp_integrator.py:    cuda.memcpy_dtoh_async(sim.rhosum_host, sim.rhosum_gpu.gpudata, sim.stream)
integrator/symp_integrator.py:    cuda.memcpy_dtoh_async(sim.pressum_host, sim.pressum_gpu.gpudata,
integrator/symp_integrator.py:    cuda.memcpy_dtoh_async(sim.inter_sum_host,
integrator/symp_integrator.py:                           sim.inter_sum_gpu.gpudata, sim.stream)
integrator/symp_integrator.py:    field_rho_avgs = [sum(sum(x.rho_sum_gpu.get()))/VL for x in sim.fields]
integrator/symp_integrator.py:    w = [sum(sum(x.pres_sum_gpu.get()))/sum(sum(x.rho_sum_gpu.get()))
integrator/symp_integrator.py:    corr_kernel.calc(*cuda_sc_args, **cuda_param_sc)
integrator/symp_integrator.py:    cuda.memcpy_dtoh_async(sim.sum_nabla_rho_h,
integrator/symp_integrator.py:                           sim.sum_nabla_rho_gpu.gpudata,
integrator/symp_integrator.py:    cuda.memcpy_dtoh_async(sim.sum_rho_squ_host,
integrator/symp_integrator.py:                           sim.sum_rho_squ_gpu.gpudata,
integrator/symp_integrator.py:            corr_kernel.calc(field.rho_gpu,
integrator/symp_integrator.py:                             field.sum_nabla_rho_gpu,
integrator/symp_integrator.py:                             field.sum_rho_squ_gpu,
integrator/symp_integrator.py:                             **cuda_param_sc)
integrator/symp_integrator.py:            cuda.memcpy_dtoh_async(field.sum_nabla_rho_h,
integrator/symp_integrator.py:                           field.sum_nabla_rho_gpu.gpudata,
integrator/symp_integrator.py:            cuda.memcpy_dtoh_async(field.sum_rho_squ_h,
integrator/symp_integrator.py:                           field.sum_rho_squ_gpu.gpudata,
integrator/symp_integrator.py:def H2_step1(lat, sim, H2_kernel, cuda_args, cuda_param_H2, dt):
integrator/symp_integrator.py:    H2_kernel.evo1(*cuda_args[0], **cuda_param_H2)
integrator/symp_integrator.py:        H2_kernel.gw_evo(*cuda_args[1], **cuda_param_H2)
integrator/symp_integrator.py:def H2_step2(lat, sim, H2_kernel, cuda_args, cuda_param_H2, dt):
integrator/symp_integrator.py:    H2_kernel.evo2(*cuda_args[0], **cuda_param_H2)
integrator/symp_integrator.py:    cuda.memcpy_dtoh_async(sim.sum_host, sim.sum_gpu.gpudata, sim.stream)
integrator/symp_integrator.py:    #sim.p += sum(sum(sim.sum_gpu.get())) - lat.VL*sim.rho_m0*(2*dt)
integrator/symp_integrator.py:        H2_kernel.gw_evo(*cuda_args[1], **cuda_param_H2)
integrator/symp_integrator.py:def H3_step(lat, V, sim, H3_list, cuda_args, cuda_param_H3, dt):
integrator/symp_integrator.py:        kernel.evo(*cuda_args[0], **cuda_param_H3)
integrator/symp_integrator.py:            kernel.evo(*cuda_args[1], **cuda_param_H3)
integrator/symp_integrator.py:def evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, dt):
integrator/symp_integrator.py:    H2_step1(lat, sim, H2_kernel, cuda_H2_arg, cuda_param_H2, dt/2)
integrator/symp_integrator.py:    H3_step(lat, V, sim, H3_list, cuda_H3_arg, cuda_param_H3, dt)
integrator/symp_integrator.py:    H2_step2(lat, sim, H2_kernel, cuda_H2_arg, cuda_param_H2, dt/2)
integrator/symp_integrator.py:def evo_step_4(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, dt):
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c1*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c0*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c1*dt)
integrator/symp_integrator.py:def evo_step_6_slow(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, dt):
integrator/symp_integrator.py:    evo_step_4(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c1*dt)
integrator/symp_integrator.py:    evo_step_4(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c0*dt)
integrator/symp_integrator.py:    evo_step_4(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c1*dt)
integrator/symp_integrator.py:def evo_step_6(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, dt):
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w1*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w2*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w3*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w4*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w3*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w2*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w1*dt)
integrator/symp_integrator.py:def evo_step_8_slow(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, dt):
integrator/symp_integrator.py:    evo_step_6(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c1*dt)
integrator/symp_integrator.py:    evo_step_6(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c0*dt)
integrator/symp_integrator.py:    evo_step_6(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, c1*dt)
integrator/symp_integrator.py:def evo_step_8(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, dt):
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w1*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w2*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w3*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w4*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w5*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w6*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w7*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w8*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w7*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w6*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w5*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w4*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w3*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w2*dt)
integrator/symp_integrator.py:    evo_step_2(lat, V, sim, H2_kernel, H3_list, cuda_param_H2, cuda_param_H3,
integrator/symp_integrator.py:               cuda_H2_arg, cuda_H3_arg, w1*dt)
integrator/symp_integrator.py:def lin_step(lat, V, sim, lin_evo_kernel, lin_args, cuda_param_lin_evo):
integrator/symp_integrator.py:    lin_evo_kernel.evo(*lin_args, **cuda_param_lin_evo)
main_program.py:import pycuda.driver as cuda
main_program.py:import pycuda.gpuarray as gpuarray
main_program.py:"GPU memory status:"
main_program.py:show_GPU_mem()
main_program.py:start = cuda.Event()
main_program.py:end = cuda.Event()
misc_functions.py:function given in string format to a form suitable to CUDA (i.e. C).
misc_functions.py:    i.e. write the multiplication open for CUDA."""
misc_functions.py:def format_to_cuda(V,var_list,C_list,D_list,n):
misc_functions.py:    """Format a polynomial function V into a suitable form for CUDA.
misc_functions.py:       will not function in CUDA code:"""
misc_functions.py:    tmp = (format_to_cuda(str(new_terms),power_list,C_list,D_list,n)
misc_functions.py:    r_cuda_V = {}
misc_functions.py:        r_cuda_V.update({'C'+str(i+1):const_name+'['+str(i0+i)+']'})
misc_functions.py:        r_cuda_V.update({'D'+str(i+1):const2_name+'['+str(i1+i)+']'})
misc_functions.py:    r_cuda_dV = {}
misc_functions.py:        r_cuda_dV.update({'C'+str(i+1):const_name+'['+str(i0+i)+']'})
misc_functions.py:        r_cuda_dV.update({'D'+str(i+1):const2_name+'['+str(i1+i)+']'})
misc_functions.py:    r_cuda_d2V = {}
misc_functions.py:        r_cuda_d2V.update({'C'+str(i+1):const_name+'['+str(i0+i)+']'})
misc_functions.py:        r_cuda_d2V.update({'D'+str(i+1):const2_name+'['+str(i1+i)+']'})
misc_functions.py:        tmp2 = format_to_cuda(tmp, power_list, C_list, D_list, n)
misc_functions.py:        res = replace_all(tmp2, r_cuda_V)
misc_functions.py:        tmp2 = format_to_cuda(tmp, power_list, C_list, D_list, n)
misc_functions.py:        res = replace_all(tmp2, r_cuda_dV)
misc_functions.py:        tmp2 = format_to_cuda(tmp, power_list, C_list, D_list, n)
misc_functions.py:        res = replace_all(tmp2, r_cuda_d2V)
misc_functions.py:        tmp2 = format_to_cuda(str(tmp), power_list, C_list, D_list, n)
misc_functions.py:        tmp2 = format_to_cuda(str(tmp), power_list, C_list, D_list, n)
misc_functions.py:        tmp2 = format_to_cuda(str(tmp), power_list, C_list, D_list, n)
misc_functions.py:# Misc CUDA functions
misc_functions.py:def show_GPU_mem():
misc_functions.py:    import pycuda.driver as cuda
misc_functions.py:    mem_free = float(cuda.mem_get_info()[0])
misc_functions.py:    mem_free_per = mem_free/float(cuda.mem_get_info()[1])
misc_functions.py:    mem_used = float(cuda.mem_get_info()[1] - cuda.mem_get_info()[0])
misc_functions.py:    mem_used_per = mem_used/float(cuda.mem_get_info()[1])
misc_functions.py:    print '\nGPU memory available {0} Mbytes, {1} % of total \n'.format(
misc_functions.py:    print 'GPU memory used {0} Mbytes, {1} % of total \n'.format(
solvers.py:import pycuda.driver as cuda
solvers.py:       start, end = Cuda timing functions
lattice.py:    Defines lattice dimensions and CUDA grid dimensions.
lattice.py:    init_m = Determine if the initialization fft:s are calculated on gpu
lattice.py:             'defrost_gpu'. Default value = 'defrost_cpu'.
lattice.py:    max_reg = Maximum number of registers per CUDA thread. For complicated
lattice.py:    Note that the calculations in the H3_kernel (with cuda_block_2) are done
lattice.py:        #Different CUDA grid and block choises:
lattice.py:        self.cuda_grid = (self.grid_x, self.grid_y)
lattice.py:        self.cuda_block_1 = (self.block_x, self.block_y, self.block_z)
lattice.py:        self.cuda_block_2 = (self.block_x2, self.block_y2, self.block_z2)
lattice.py:        self.cuda_lin_block = (self.block_x,1,1)
lattice.py:        self.cuda_g_lin_H = (self.gridx_lin_H, self.gridy_lin_H)
lattice.py:        self.cuda_b_lin_H = (self.block_lin_Hx, self.block_lin_Hy,
lattice.py:        "Replacement list used in format_to_cuda function:"
lattice.py:           forms needed in the different CUDA kernels:"""
lattice.py:        """Temporary variables in CUDA form used in the H3 kernel:"""
lattice.py:        "Potential function V_{i} of field i in CUDA form used in H3 part:"
lattice.py:        """Interaction term V_{int} of the fields in CUDA form used in
lattice.py:            "Use zero to avoid CUDA error messages:"
lattice.py:        "Derivative dV/df_i for all field variables f_i in CUDA form:"
lattice.py:        """Potential function V_{i} of field i in CUDA form used in rho and
lattice.py:        """Interaction term V_{int} of the fields in CUDA form used in
lattice.py:            "Use zero to avoid CUDA error messages:"
lattice.py:        "Derivative d2V/df_i^2 for all field variables f_i in CUDA form:"
lattice.py:        self.d2V_Cuda = [V_calc(self.V , n, self.f_list, i+1,
lattice.py:            "Use zero to avoid CUDA error messages:"
lattice.py:        "Derivative d2V/df_i^2 for all field variables f_i in CUDA form:"
postprocess/procedures.py:import pycuda.driver as cuda
postprocess/procedures.py:import pycuda.autoinit
postprocess/procedures.py:import pycuda.gpuarray as gpuarray
postprocess/procedures.py:from pycuda.compiler import SourceModule
postprocess/procedures.py:def kernel_pd_gpu_code(lat, V, field_i, write_code=False):
postprocess/procedures.py:    f = codecs.open('cuda_templates/pd_kernel.cu','r',encoding='utf-8')
postprocess/procedures.py:def kernel_k_gpu_code(lat, V, write_code=False):
postprocess/procedures.py:    f = codecs.open('cuda_templates/'+kernel_name+'.cu','r',encoding='utf-8')
postprocess/procedures.py:        self.mod = kernel_pd_gpu_code(lat, V, field_i, write_code)
postprocess/procedures.py:        cuda.memcpy_htod(self.cc_add[0],x)
postprocess/procedures.py:        cuda.memcpy_dtoh(x,self.cc_add[0])
postprocess/procedures.py:        cuda.memcpy_htod(self.dc_add[0],x)
postprocess/procedures.py:        cuda.memcpy_dtoh(x,self.dc_add[0])
postprocess/procedures.py:        cuda.memcpy_htod(self.pc_add[0],x)
postprocess/procedures.py:        cuda.memcpy_dtoh(x,self.pc_add[0])
postprocess/procedures.py:        self.mod = kernel_k_gpu_code(lat, V, write_code)
postprocess/procedures.py:        self.k_vec_calc = self.mod.get_function('gpu_k_vec')
postprocess/procedures.py:            pd_args.append(field.f_gpu)
postprocess/procedures.py:            pd_args.append(field.pi_gpu)
postprocess/procedures.py:        pd_args.append(sim.pd_gpu)
postprocess/procedures.py:            pd_args.append(field.d2V_sum_gpu)
postprocess/procedures.py:        self.pd_params = dict(block=lat.cuda_block_2,
postprocess/procedures.py:                              grid=lat.cuda_grid,
postprocess/procedures.py:            sum1 = sum(sum(field.d2V_sum_gpu.get()))/lat.VL
postprocess/procedures.py:            sum2 = sum(sum(sim.pd_gpu.get()))/(6.*lat.VL)
postprocess/procedures.py:            f = field.f_gpu.get()
postprocess/procedures.py:            pi_f = field.pi_gpu.get()
postprocess/procedures.py:        data = sim.rho_gpu.get().flatten()/(3.*sim.H**2.)
postprocess/procedures.py:                data = field.rho_gpu.get().flatten()/(3.*sim.H**2.)
postprocess/procedures.py:            data = field.f_gpu.get().flatten()
postprocess/procedures.py:            u11 = sim.u11_gpu.get()
postprocess/procedures.py:            u12 = sim.u12_gpu.get()
postprocess/procedures.py:            u13 = sim.u13_gpu.get()
postprocess/procedures.py:            u22 = sim.u22_gpu.get()
postprocess/procedures.py:            u23 = sim.u23_gpu.get()
postprocess/procedures.py:            u33 = sim.u33_gpu.get()
postprocess/procedures.py:        piu11 = sim.piu11_gpu.get()
postprocess/procedures.py:        piu12 = sim.piu12_gpu.get()
postprocess/procedures.py:        piu13 = sim.piu13_gpu.get()
postprocess/procedures.py:        piu22 = sim.piu22_gpu.get()
postprocess/procedures.py:        piu23 = sim.piu23_gpu.get()
postprocess/procedures.py:        piu33 = sim.piu33_gpu.get()
README:functions in the Python and CUDA codes in order to make the program
README:PyCOOL is a Python + CUDA program that solves the evolution of interacting
README:scalar fields in an expanding universe. PyCOOL uses modern GPUs
README:See http://arxiv.org/abs/0911.5692 for more information on the GPU
README:  - CUDA and template files:
README:    - gpu_3dconv.cu
README:PyCOOL naturally needs CUDA drivers installed into your machine
README:and preferably a fast NVIDIA GPU (GTX 470 and Tesla C2050 cards tested)
README:PyCUDA http://mathema.tician.de/software/pycuda which needs to be installed.
README:The easiest way to install PyCUDA and Pyvisfile is to use
README:git clone http://git.tiker.net/trees/pycuda.git
README:in the downloaded package folders. Further PyCUDA install instructions
README:are available in http://wiki.tiker.net/PyCuda/Installation .
README:PyCOOL uses then textual templating to write the necessary CUDA files. This is
README:  - PyCUDA (and CUDA)
README:potential functions. It might however fail to write CUDA compatible code in
README:uses format_to_cuda function (found in misc_functions.py) to write these terms
README:functions into suitable CUDA form,
README:The last two of these use an identical CUDA implementation and are also more
README:suitable for a multi-GPU implementation.
README:- Multi-GPU support should be included. This might however take some
README:- OpenCL support would allow to run the simulations also on AMD Radeon cards.
README:CUDADRV_LIBNAME = ['cuda']
README:CUDADRV_LIB_DIR = ['/usr/lib']
README:CUDA_ENABLE_GL = False
README:CUDA_ROOT = '/usr/local/cuda'
README:CUDA_TRACE = False

```
