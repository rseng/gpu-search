# https://github.com/FabioRigamonti/BANG

```console
tests/utils_numba_1D_32bit.py:from numba import cuda,float64,float32,int16
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import acosf as acos
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import atan2f as atan2
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_cosf as cos
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_sinf as sin
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_expf as exp
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_logf as log
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_log10f as log10
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import sqrtf as sqrt
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_exp10f
tests/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fadd_rn,fast_fdividef,fsub_rn,hypotf,fmul_rn,fmaf_rn,frcp_rn,fast_powf
tests/utils_numba_1D_32bit.py:It contains all the relevant function of the code written for GPU usage.
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,\
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
tests/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32,int16)',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],int16[::1])',
tests/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
tests/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit.py:            cuda.atomic.add(likelihood,0,tmp)
tests/network.py:    device = "cuda" if torch.cuda.is_available() else "cpu"
tests/configuration.py:    def __init__(self,config_path,gpu_device=0):
tests/configuration.py:                    gpu_device: int16
tests/configuration.py:                        integer referring to the gpu machine. Useful only
tests/configuration.py:                        in case of multi-gpu usage
tests/configuration.py:                parameter_estimation(gpu_device=0):
tests/configuration.py:                              gpu_device,
tests/configuration.py:    def parameter_estimation(self,gpu_device=0):
tests/configuration.py:                gpu_device: int16
tests/configuration.py:                    ID of the GPU. If multi-gpu is not needed, set it to zero.
tests/configuration.py:                        gpu_device,
tests/utils_python_GH.py:    CPU/GPU FUNCTION
tests/utils_python_GH.py:    CPU/GPU FUNCTION
tests/utils_python_GH.py:    CPU/GPU FUNCTION
tests/utils_python_GH.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tests/config.yaml:#          - "GPU"
tests/config.yaml:    name: 'GPU'
tests/model_creation.py:#THE STRUCTURE CAN be changed, all model on gpu can be easily put togheter.
tests/model_creation.py:from numba import cuda
tests/model_creation.py:        - GPU with fastmath:
tests/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
tests/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
tests/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE
tests/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE
tests/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
tests/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE
tests/model_creation.py:        - GPU without fastmath:
tests/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
tests/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
tests/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE 
tests/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE 
tests/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
tests/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE 
tests/model_creation.py:        - GPU + Super Resolution:
tests/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_superresolution_gpu")         UNSTABLE with new convolution
tests/model_creation.py:class model_gpu_B_BH():
tests/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
tests/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
tests/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
tests/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
tests/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
tests/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
tests/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
tests/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
tests/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
tests/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
tests/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
tests/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
tests/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
tests/model_creation.py:class model_gpu_BD_BH():
tests/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
tests/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
tests/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
tests/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
tests/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
tests/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
tests/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
tests/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
tests/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
tests/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
tests/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
tests/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
tests/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
tests/model_creation.py:class model_gpu_BD():
tests/model_creation.py:                 gpu_device,
tests/model_creation.py:        A class for representing a "Bulge + Disc + Halo" model on GPU
tests/model_creation.py:                    gpu_device : int
tests/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
tests/model_creation.py:                        Compute # blocks, # theads for the Gpu
tests/model_creation.py:                        Load all data on GPU
tests/model_creation.py:                        Allocate memory for GPU operations
tests/model_creation.py:        self.gpu_device = gpu_device
tests/model_creation.py:        Compute # blocks, # theads for the Gpu
tests/model_creation.py:         Load all data on GPU
tests/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
tests/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
tests/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
tests/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        Allocate memory for GPU operations
tests/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
tests/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
tests/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
tests/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
tests/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
tests/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
tests/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
tests/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
tests/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
tests/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
tests/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
tests/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
tests/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
tests/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
tests/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
tests/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
tests/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
tests/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
tests/model_creation.py:class model_gpu_DD():
tests/model_creation.py:                 gpu_device,
tests/model_creation.py:        A class for representing a "Disc + Disc + Halo" model on GPU
tests/model_creation.py:                    gpu_device : int
tests/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
tests/model_creation.py:                        Compute # blocks, # theads for the Gpu
tests/model_creation.py:                        Load all data on GPU
tests/model_creation.py:                        Allocate memory for GPU operations
tests/model_creation.py:        self.gpu_device = gpu_device
tests/model_creation.py:        Compute # blocks, # theads for the Gpu
tests/model_creation.py:         Load all data on GPU
tests/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
tests/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
tests/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
tests/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        Allocate memory for GPU operations
tests/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
tests/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
tests/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
tests/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
tests/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
tests/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
tests/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
tests/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
tests/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
tests/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
tests/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
tests/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
tests/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
tests/model_creation.py:class model_gpu_BH():
tests/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
tests/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
tests/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
tests/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
tests/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
tests/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
tests/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
tests/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
tests/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
tests/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
tests/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
tests/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
tests/model_creation.py:class model_gpu():
tests/model_creation.py:                 gpu_device,
tests/model_creation.py:        A class for representing a "Bulge + Disc + Disc + Halo" model on GPU
tests/model_creation.py:                    gpu_device : int
tests/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
tests/model_creation.py:                        Compute # blocks, # theads for the Gpu
tests/model_creation.py:                        Load all data on GPU
tests/model_creation.py:                        Allocate memory for GPU operations
tests/model_creation.py:        self.gpu_device = gpu_device
tests/model_creation.py:        Compute # blocks, # theads for the Gpu
tests/model_creation.py:         Load all data on GPU
tests/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
tests/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
tests/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
tests/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
tests/model_creation.py:        self.vD1_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.vD2_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        Allocate memory for GPU operations
tests/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
tests/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
tests/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
tests/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
tests/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
tests/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
tests/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
tests/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
tests/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
tests/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
tests/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
tests/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
tests/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
tests/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
tests/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
tests/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
tests/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
tests/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
tests/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:            cuda.select_device(self.gpu_device)
tests/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
tests/model_creation.py:#class model_superresolution_gpu():
tests/model_creation.py:#        This is not full on gpu but it combines this... probably the fastest combination is 
tests/model_creation.py:#        lr img on GPU
tests/model_creation.py:#        self.device = torch.device('cuda')
tests/model_creation.py:#        self.x_device = cuda.to_device(np.float32(self.x_g))
tests/model_creation.py:#        self.y_device = cuda.to_device(np.float32(self.y_g))
tests/model_creation.py:#        self.r_proj_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.r_true_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.phi_device           = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.all_rhoB_device      = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.all_sigma2_device    = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.all_rhoD1_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.all_rhoD2_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.all_v_device         = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.all_v2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.Xs_device            = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.v_exp1_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.v_exp2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.v_bulge_device       = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.v_halo_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
tests/model_creation.py:#        self.LM_avg_device            = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
tests/model_creation.py:#        self.rho_avg_device           = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
tests/model_creation.py:#        self.v_avg_device             = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
tests/model_creation.py:#        self.sigma_avg_device         = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
tests/model_creation.py:#        self.tot = cuda.device_array((4,1,self.new_N,self.new_J),dtype=np.float32)
tests/model_creation.py:#        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
tests/model_creation.py:#        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
tests/model_creation.py:#        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
tests/model_creation.py:#        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
tests/model_creation.py:#        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
tests/model_creation.py:#        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
tests/model_creation.py:#        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
tests/model_creation.py:#        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
tests/model_creation.py:#        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
tests/model_creation.py:#        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
tests/model_creation.py:class model(model_gpu,model_cpu,model_gpu_BH,model_cpu_BH,
tests/model_creation.py:            model_gpu_BD,model_cpu_BD,model_gpu_BD_BH,model_cpu_BD_BH,model_gpu_DD,
tests/model_creation.py:            model_gpu_B_BH,model_cpu_B_BH,model_cpu_DD):#,
tests/model_creation.py:            #model_superresolution_cpu,model_superresolution_gpu):
tests/model_creation.py:                 gpu_device,
tests/model_creation.py:                        Type of device (gpu/cpu) 
tests/model_creation.py:                    gpu_device : int
tests/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
tests/model_creation.py:        if self.device == 'GPU':
tests/model_creation.py:                model_gpu_BH.__init__(self,
tests/model_creation.py:                self.__likelihood = model_gpu_BH.likelihood
tests/model_creation.py:                self.__model      = model_gpu_BH.model 
tests/model_creation.py:                model_gpu.__init__(self,
tests/model_creation.py:                                   gpu_device,
tests/model_creation.py:                self.__likelihood = model_gpu.likelihood
tests/model_creation.py:                self.__model      = model_gpu.model  
tests/model_creation.py:                model_gpu_BD_BH.__init__(self,
tests/model_creation.py:                self.__likelihood = model_gpu_BD_BH.likelihood
tests/model_creation.py:                self.__model      = model_gpu_BD_BH.model                 
tests/model_creation.py:                model_gpu_BD.__init__(self,
tests/model_creation.py:                                   gpu_device,
tests/model_creation.py:                self.__likelihood = model_gpu_BD.likelihood
tests/model_creation.py:                self.__model      = model_gpu_BD.model 
tests/model_creation.py:                model_gpu_DD.__init__(self,
tests/model_creation.py:                                   gpu_device,
tests/model_creation.py:                self.__likelihood = model_gpu_DD.likelihood
tests/model_creation.py:                self.__model      = model_gpu_DD.model   
tests/model_creation.py:                model_gpu_B_BH.__init__(self,
tests/model_creation.py:                self.__likelihood = model_gpu_B_BH.likelihood
tests/model_creation.py:                self.__model      = model_gpu_B_BH.model                 
tests/model_creation.py:        elif self.device == 'super resolution GPU':
tests/model_creation.py:            #    model_superresolution_gpu.__init__(self,
tests/model_creation.py:            #                       'GPU',
tests/model_creation.py:            #    self.__likelihood = model_superresolution_gpu.likelihood
tests/model_creation.py:            #    self.__model      = model_superresolution_gpu.model 
tests/model_creation.py:    gpu_device = 0
tests/model_creation.py:    my_model_GPU = model('GPU',
tests/model_creation.py:                    gpu_device,
tests/model_creation.py:    rho_GPU,rho_B_GPU,rho_D1_GPU,rho_D2_GPU,\
tests/model_creation.py:    v_GPU,v_B_GPU,v_D1_GPU,v_D2_GPU,\
tests/model_creation.py:    sigma_GPU,sigma_B_GPU,sigma_D1_GPU,sigma_D2_GPU,\
tests/model_creation.py:    LM_GPU,LM_B_GPU,LM_D1_GPU,LM_D2_GPU = my_model_GPU.model(all_par) 
tests/model_creation.py:    #lk = my_model_GPU.likelihood(all_par)   
tests/model_creation.py:    fig,ax,pmesh,cbar = data_obj.brightness_map(np.log10(rho_GPU))
tests/model_creation.py:    fig.savefig('test_model/B_gpu.png')
tests/model_creation.py:    fig,ax,pmesh,cbar = data_obj.velocity_map(v_GPU)
tests/model_creation.py:    fig.savefig('test_model/v_gpu.png')
tests/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_GPU)
tests/model_creation.py:    fig.savefig('test_model/sigma_gpu.png')
tests/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_B_GPU)
tests/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D1_GPU)
tests/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D2_GPU)
tests/model_creation.py:    fig,ax,pmesh,cbar = data_obj.ML_map(1.0/LM_GPU)
tests/model_creation.py:    fig.savefig('test_model/ML_gpu.png')
tests/model_creation.py:                    gpu_device,
tests/example_script.py:- run a script similar to the one below and wait for the result. On a standard gpu you parameter
tests/utils_numba_1D_32bit_no_proxy.py:from numba import cuda,float64,float32,int16
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32,float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1])',
tests/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
tests/utils_numba_1D_32bit_no_proxy.py:        cuda.atomic.add(likelihood,0,tmp)
tests/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
tests/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
tests/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
tests/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
tests/utils_python.py:    CPU/GPU FUNCTION
tests/utils_python.py:    CPU/GPU FUNCTION
tests/utils_python.py:    CPU/GPU FUNCTION
tests/utils_python.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tests/parameter_search.py:from numba import cuda
tests/parameter_search.py:                gpu_device,
tests/parameter_search.py:                        Type of device (gpu/cpu) 
tests/parameter_search.py:                    gpu_device : int
tests/parameter_search.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
tests/parameter_search.py:                                         gpu_device,
tests/parameter_search.py:                              gpu_device,
tests/parameter_search.py:    gpu_device = 0
tests/parameter_search.py:                    gpu_device,
README.md:BANG is a GPU/CPU-python code for modelling both the photometry and kinematics of galaxies.
README.md:We strongly suggest to run BANG on GPU. CPU parameter estimation can take
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba import cuda,float64,float32,int16
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import acosf as acos
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import atan2f as atan2
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_cosf as cos
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_sinf as sin
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_expf as exp
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_logf as log
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_log10f as log10
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import sqrtf as sqrt
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_exp10f
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fadd_rn,fast_fdividef,fsub_rn,hypotf,fmul_rn,fmaf_rn,frcp_rn,fast_powf
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:It contains all the relevant function of the code written for GPU usage.
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32,int16)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],int16[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit.py:            cuda.atomic.add(likelihood,0,tmp)
build/lib.linux-x86_64-cpython-38/src/BANG/network.py:    device = "cuda" if torch.cuda.is_available() else "cpu"
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:    def __init__(self,config_path,gpu_device=0):
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                    gpu_device: int16
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                        integer referring to the gpu machine. Useful only
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                        in case of multi-gpu usage
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                parameter_estimation(gpu_device=0):
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                              gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:    def parameter_estimation(self,gpu_device=0):
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                gpu_device: int16
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                    ID of the GPU. If multi-gpu is not needed, set it to zero.
build/lib.linux-x86_64-cpython-38/src/BANG/configuration.py:                        gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python_GH.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#THE STRUCTURE CAN be changed, all model on gpu can be easily put togheter.
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:from numba import cuda
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        - GPU with fastmath:
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        - GPU without fastmath:
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        - GPU + Super Resolution:
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_superresolution_gpu")         UNSTABLE with new convolution
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:class model_gpu_B_BH():
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:class model_gpu_BD_BH():
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:class model_gpu_BD():
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        A class for representing a "Bulge + Disc + Halo" model on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Load all data on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Allocate memory for GPU operations
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.gpu_device = gpu_device
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:         Load all data on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        Allocate memory for GPU operations
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:class model_gpu_DD():
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        A class for representing a "Disc + Disc + Halo" model on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Load all data on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Allocate memory for GPU operations
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.gpu_device = gpu_device
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:         Load all data on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        Allocate memory for GPU operations
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:class model_gpu_BH():
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:class model_gpu():
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        A class for representing a "Bulge + Disc + Disc + Halo" model on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Load all data on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Allocate memory for GPU operations
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.gpu_device = gpu_device
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:         Load all data on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.vD1_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.vD2_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        Allocate memory for GPU operations
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#class model_superresolution_gpu():
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        This is not full on gpu but it combines this... probably the fastest combination is 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        lr img on GPU
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.device = torch.device('cuda')
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.r_proj_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.r_true_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.phi_device           = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.all_rhoB_device      = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.all_sigma2_device    = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.all_rhoD1_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.all_rhoD2_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.all_v_device         = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.all_v2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.Xs_device            = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.v_exp1_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.v_exp2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.v_bulge_device       = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.v_halo_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.LM_avg_device            = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.rho_avg_device           = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.v_avg_device             = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.sigma_avg_device         = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.tot = cuda.device_array((4,1,self.new_N,self.new_J),dtype=np.float32)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:#        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:class model(model_gpu,model_cpu,model_gpu_BH,model_cpu_BH,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            model_gpu_BD,model_cpu_BD,model_gpu_BD_BH,model_cpu_BD_BH,model_gpu_DD,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            model_gpu_B_BH,model_cpu_B_BH,model_cpu_DD):#,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            #model_superresolution_cpu,model_superresolution_gpu):
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Type of device (gpu/cpu) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        if self.device == 'GPU':
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                model_gpu_BH.__init__(self,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__likelihood = model_gpu_BH.likelihood
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__model      = model_gpu_BH.model 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                model_gpu.__init__(self,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                                   gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__likelihood = model_gpu.likelihood
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__model      = model_gpu.model  
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                model_gpu_BD_BH.__init__(self,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__likelihood = model_gpu_BD_BH.likelihood
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__model      = model_gpu_BD_BH.model                 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                model_gpu_BD.__init__(self,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                                   gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__likelihood = model_gpu_BD.likelihood
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__model      = model_gpu_BD.model 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                model_gpu_DD.__init__(self,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                                   gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__likelihood = model_gpu_DD.likelihood
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__model      = model_gpu_DD.model   
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                model_gpu_B_BH.__init__(self,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__likelihood = model_gpu_B_BH.likelihood
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                self.__model      = model_gpu_B_BH.model                 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:        elif self.device == 'super resolution GPU':
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            #    model_superresolution_gpu.__init__(self,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            #                       'GPU',
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            #    self.__likelihood = model_superresolution_gpu.likelihood
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:            #    self.__model      = model_superresolution_gpu.model 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    gpu_device = 0
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    my_model_GPU = model('GPU',
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                    gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    rho_GPU,rho_B_GPU,rho_D1_GPU,rho_D2_GPU,\
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    v_GPU,v_B_GPU,v_D1_GPU,v_D2_GPU,\
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    sigma_GPU,sigma_B_GPU,sigma_D1_GPU,sigma_D2_GPU,\
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    LM_GPU,LM_B_GPU,LM_D1_GPU,LM_D2_GPU = my_model_GPU.model(all_par) 
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    #lk = my_model_GPU.likelihood(all_par)   
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.brightness_map(np.log10(rho_GPU))
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig.savefig('test_model/B_gpu.png')
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.velocity_map(v_GPU)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig.savefig('test_model/v_gpu.png')
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_GPU)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig.savefig('test_model/sigma_gpu.png')
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_B_GPU)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D1_GPU)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D2_GPU)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.ML_map(1.0/LM_GPU)
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:    fig.savefig('test_model/ML_gpu.png')
build/lib.linux-x86_64-cpython-38/src/BANG/model_creation.py:                    gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/example_script.py:- run a script similar to the one below and wait for the result. On a standard gpu you parameter
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:from numba import cuda,float64,float32,int16
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32,float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1])',
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:        cuda.atomic.add(likelihood,0,tmp)
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-cpython-38/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-cpython-38/src/BANG/utils_python.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:from numba import cuda
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:                gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:                        Type of device (gpu/cpu) 
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:                    gpu_device : int
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:                                         gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:                              gpu_device,
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:    gpu_device = 0
build/lib.linux-x86_64-cpython-38/src/BANG/parameter_search.py:                    gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba import cuda,float64,float32,int16
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import acosf as acos
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import atan2f as atan2
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_cosf as cos
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_sinf as sin
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_expf as exp
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_logf as log
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_log10f as log10
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import sqrtf as sqrt
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_exp10f
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fadd_rn,fast_fdividef,fsub_rn,hypotf,fmul_rn,fmaf_rn,frcp_rn,fast_powf
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:It contains all the relevant function of the code written for GPU usage.
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32,int16)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],int16[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit.py:            cuda.atomic.add(likelihood,0,tmp)
build/lib.linux-x86_64-3.8/src/BANG/network.py:    device = "cuda" if torch.cuda.is_available() else "cpu"
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:    def __init__(self,config_path,gpu_device=0):
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                    gpu_device: int16
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                        integer referring to the gpu machine. Useful only
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                        in case of multi-gpu usage
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                parameter_estimation(gpu_device=0):
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                              gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:    def parameter_estimation(self,gpu_device=0):
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                gpu_device: int16
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                    ID of the GPU. If multi-gpu is not needed, set it to zero.
build/lib.linux-x86_64-3.8/src/BANG/configuration.py:                        gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-3.8/src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-3.8/src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-3.8/src/BANG/utils_python_GH.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#THE STRUCTURE CAN be changed, all model on gpu can be easily put togheter.
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:from numba import cuda
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        - GPU with fastmath:
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        - GPU without fastmath:
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        - GPU + Super Resolution:
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_superresolution_gpu")         UNSTABLE with new convolution
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:class model_gpu_B_BH():
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:class model_gpu_BD_BH():
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:class model_gpu_BD():
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        A class for representing a "Bulge + Disc + Halo" model on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Load all data on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Allocate memory for GPU operations
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.gpu_device = gpu_device
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:         Load all data on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        Allocate memory for GPU operations
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:class model_gpu_DD():
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        A class for representing a "Disc + Disc + Halo" model on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Load all data on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Allocate memory for GPU operations
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.gpu_device = gpu_device
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:         Load all data on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        Allocate memory for GPU operations
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:class model_gpu_BH():
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:class model_gpu():
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        A class for representing a "Bulge + Disc + Disc + Halo" model on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Load all data on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Allocate memory for GPU operations
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.gpu_device = gpu_device
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:         Load all data on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.vD1_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.vD2_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        Allocate memory for GPU operations
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#class model_superresolution_gpu():
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        This is not full on gpu but it combines this... probably the fastest combination is 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        lr img on GPU
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.device = torch.device('cuda')
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.x_device = cuda.to_device(np.float32(self.x_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.y_device = cuda.to_device(np.float32(self.y_g))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.r_proj_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.r_true_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.phi_device           = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.all_rhoB_device      = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.all_sigma2_device    = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.all_rhoD1_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.all_rhoD2_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.all_v_device         = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.all_v2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.Xs_device            = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.v_exp1_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.v_exp2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.v_bulge_device       = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.v_halo_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.LM_avg_device            = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.rho_avg_device           = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.v_avg_device             = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.sigma_avg_device         = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.tot = cuda.device_array((4,1,self.new_N,self.new_J),dtype=np.float32)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:#        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:class model(model_gpu,model_cpu,model_gpu_BH,model_cpu_BH,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            model_gpu_BD,model_cpu_BD,model_gpu_BD_BH,model_cpu_BD_BH,model_gpu_DD,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            model_gpu_B_BH,model_cpu_B_BH,model_cpu_DD):#,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            #model_superresolution_cpu,model_superresolution_gpu):
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                 gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Type of device (gpu/cpu) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                    gpu_device : int
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        if self.device == 'GPU':
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                model_gpu_BH.__init__(self,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__likelihood = model_gpu_BH.likelihood
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__model      = model_gpu_BH.model 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                model_gpu.__init__(self,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                                   gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__likelihood = model_gpu.likelihood
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__model      = model_gpu.model  
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                model_gpu_BD_BH.__init__(self,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__likelihood = model_gpu_BD_BH.likelihood
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__model      = model_gpu_BD_BH.model                 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                model_gpu_BD.__init__(self,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                                   gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__likelihood = model_gpu_BD.likelihood
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__model      = model_gpu_BD.model 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                model_gpu_DD.__init__(self,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                                   gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__likelihood = model_gpu_DD.likelihood
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__model      = model_gpu_DD.model   
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                model_gpu_B_BH.__init__(self,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__likelihood = model_gpu_B_BH.likelihood
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                self.__model      = model_gpu_B_BH.model                 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:        elif self.device == 'super resolution GPU':
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            #    model_superresolution_gpu.__init__(self,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            #                       'GPU',
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            #    self.__likelihood = model_superresolution_gpu.likelihood
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:            #    self.__model      = model_superresolution_gpu.model 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    gpu_device = 0
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    my_model_GPU = model('GPU',
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                    gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    rho_GPU,rho_B_GPU,rho_D1_GPU,rho_D2_GPU,\
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    v_GPU,v_B_GPU,v_D1_GPU,v_D2_GPU,\
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    sigma_GPU,sigma_B_GPU,sigma_D1_GPU,sigma_D2_GPU,\
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    LM_GPU,LM_B_GPU,LM_D1_GPU,LM_D2_GPU = my_model_GPU.model(all_par) 
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    #lk = my_model_GPU.likelihood(all_par)   
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.brightness_map(np.log10(rho_GPU))
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig.savefig('test_model/B_gpu.png')
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.velocity_map(v_GPU)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig.savefig('test_model/v_gpu.png')
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_GPU)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig.savefig('test_model/sigma_gpu.png')
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_B_GPU)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D1_GPU)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D2_GPU)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.ML_map(1.0/LM_GPU)
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:    fig.savefig('test_model/ML_gpu.png')
build/lib.linux-x86_64-3.8/src/BANG/model_creation.py:                    gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/example_script.py:- run a script similar to the one below and wait for the result. On a standard gpu you parameter
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:from numba import cuda,float64,float32,int16
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32,float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1])',
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:        cuda.atomic.add(likelihood,0,tmp)
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-3.8/src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
build/lib.linux-x86_64-3.8/src/BANG/utils_python.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-3.8/src/BANG/utils_python.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-3.8/src/BANG/utils_python.py:    CPU/GPU FUNCTION
build/lib.linux-x86_64-3.8/src/BANG/utils_python.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:from numba import cuda
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:                gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:                        Type of device (gpu/cpu) 
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:                    gpu_device : int
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:                                         gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:                              gpu_device,
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:    gpu_device = 0
build/lib.linux-x86_64-3.8/src/BANG/parameter_search.py:                    gpu_device,
src/BANGal.egg-info/PKG-INFO:BANG is a GPU/CPU-python code for modelling both the photometry and kinematics of galaxies.
src/BANGal.egg-info/PKG-INFO:We strongly suggest to run BANG on GPU. CPU parameter estimation can take
src/BANG/utils_numba_1D_32bit.py:from numba import cuda,float64,float32,int16
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import acosf as acos
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import atan2f as atan2
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_cosf as cos
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_sinf as sin
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_expf as exp
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_logf as log
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_log10f as log10
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import sqrtf as sqrt
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fast_exp10f
src/BANG/utils_numba_1D_32bit.py:from numba.cuda.libdevice import fadd_rn,fast_fdividef,fsub_rn,hypotf,fmul_rn,fmaf_rn,frcp_rn,fast_powf
src/BANG/utils_numba_1D_32bit.py:It contains all the relevant function of the code written for GPU usage.
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,\
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32)',
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32)',
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32,int16)',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1],float32,int16)',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],int16[::1])',
src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],\
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:#@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
src/BANG/utils_numba_1D_32bit.py:#    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit.py:            cuda.atomic.add(likelihood,0,tmp)
src/BANG/network.py:    device = "cuda" if torch.cuda.is_available() else "cpu"
src/BANG/configuration.py:    def __init__(self,config_path,gpu_device=0):
src/BANG/configuration.py:                    gpu_device: int16
src/BANG/configuration.py:                        integer referring to the gpu machine. Useful only
src/BANG/configuration.py:                        in case of multi-gpu usage
src/BANG/configuration.py:                parameter_estimation(gpu_device=0):
src/BANG/configuration.py:                              gpu_device,
src/BANG/configuration.py:    def parameter_estimation(self,gpu_device=0):
src/BANG/configuration.py:                gpu_device: int16
src/BANG/configuration.py:                    ID of the GPU. If multi-gpu is not needed, set it to zero.
src/BANG/configuration.py:                        gpu_device,
src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
src/BANG/utils_python_GH.py:    CPU/GPU FUNCTION
src/BANG/utils_python_GH.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src/BANG/config.yaml:#          - "GPU"
src/BANG/config.yaml:    name: 'GPU'
src/BANG/model_creation.py:#THE STRUCTURE CAN be changed, all model on gpu can be easily put togheter.
src/BANG/model_creation.py:from numba import cuda
src/BANG/model_creation.py:        - GPU with fastmath:
src/BANG/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
src/BANG/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
src/BANG/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE
src/BANG/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE
src/BANG/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE
src/BANG/model_creation.py:        - GPU without fastmath:
src/BANG/model_creation.py:                - Bulge+Black Hole+Halo ("model_gpu_B_BH")                   UNSTABLE with new convolution
src/BANG/model_creation.py:                - Bulge+Disc+Black Hole+Halo ("model_gpu_BD_BH")             UNSTABLE with new convolution
src/BANG/model_creation.py:                - Bulge+Disc+Halo ("model_gpu_BD")                           STABLE 
src/BANG/model_creation.py:                - Disc+Disc+Halo ("model_gpu_DD")                            STABLE 
src/BANG/model_creation.py:                - Bulge+Disc+Disc+Black Hole+Halo ("model_gpu_BH")           UNSTABLE with new convolution
src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_gpu")                         STABLE 
src/BANG/model_creation.py:        - GPU + Super Resolution:
src/BANG/model_creation.py:                - Bulge+Disc+Disc+Halo ("model_superresolution_gpu")         UNSTABLE with new convolution
src/BANG/model_creation.py:class model_gpu_B_BH():
src/BANG/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
src/BANG/model_creation.py:class model_gpu_BD_BH():
src/BANG/model_creation.py:    # PROBABLY IT CAN BE UNIFIED WITH model_gpu
src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
src/BANG/model_creation.py:class model_gpu_BD():
src/BANG/model_creation.py:                 gpu_device,
src/BANG/model_creation.py:        A class for representing a "Bulge + Disc + Halo" model on GPU
src/BANG/model_creation.py:                    gpu_device : int
src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
src/BANG/model_creation.py:                        Load all data on GPU
src/BANG/model_creation.py:                        Allocate memory for GPU operations
src/BANG/model_creation.py:        self.gpu_device = gpu_device
src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
src/BANG/model_creation.py:         Load all data on GPU
src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        Allocate memory for GPU operations
src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
src/BANG/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
src/BANG/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
src/BANG/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
src/BANG/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
src/BANG/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
src/BANG/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
src/BANG/model_creation.py:class model_gpu_DD():
src/BANG/model_creation.py:                 gpu_device,
src/BANG/model_creation.py:        A class for representing a "Disc + Disc + Halo" model on GPU
src/BANG/model_creation.py:                    gpu_device : int
src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
src/BANG/model_creation.py:                        Load all data on GPU
src/BANG/model_creation.py:                        Allocate memory for GPU operations
src/BANG/model_creation.py:        self.gpu_device = gpu_device
src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
src/BANG/model_creation.py:         Load all data on GPU
src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        Allocate memory for GPU operations
src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
src/BANG/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
src/BANG/model_creation.py:class model_gpu_BH():
src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.v_BH_device          = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X1_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X2_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.X3_dot_device        = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Y_1_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.Y_2_device           = cuda.device_array((self.N*self.J*self.K),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
src/BANG/model_creation.py:        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
src/BANG/model_creation.py:class model_gpu():
src/BANG/model_creation.py:                 gpu_device,
src/BANG/model_creation.py:        A class for representing a "Bulge + Disc + Disc + Halo" model on GPU
src/BANG/model_creation.py:                    gpu_device : int
src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
src/BANG/model_creation.py:                        Compute # blocks, # theads for the Gpu
src/BANG/model_creation.py:                        Load all data on GPU
src/BANG/model_creation.py:                        Allocate memory for GPU operations
src/BANG/model_creation.py:        self.gpu_device = gpu_device
src/BANG/model_creation.py:        Compute # blocks, # theads for the Gpu
src/BANG/model_creation.py:         Load all data on GPU
src/BANG/model_creation.py:        self.x_device = cuda.to_device(np.float32(self.x_g))
src/BANG/model_creation.py:        self.y_device = cuda.to_device(np.float32(self.y_g))
src/BANG/model_creation.py:        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
src/BANG/model_creation.py:        self.r_proj_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.r_true_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.phi_device           = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoB_device      = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_sigma2_device    = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD1_device     = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_rhoD2_device     = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v_device         = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.all_v2_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.Xs_device            = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp1_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_exp2_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_bulge_device       = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.v_halo_device        = cuda.device_array((self.size),dtype=np.float32)
src/BANG/model_creation.py:        self.vD1_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.vD2_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        Allocate memory for GPU operations
src/BANG/model_creation.py:        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
src/BANG/model_creation.py:        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
src/BANG/model_creation.py:        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
src/BANG/model_creation.py:        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
src/BANG/model_creation.py:        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
src/BANG/model_creation.py:        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
src/BANG/model_creation.py:        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
src/BANG/model_creation.py:        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
src/BANG/model_creation.py:        self.kinpsf_device              = cuda.to_device(np.float32(self.kin_psf.ravel()))
src/BANG/model_creation.py:        self.phopsf_device              = cuda.to_device(np.float32(self.pho_psf.ravel()))
src/BANG/model_creation.py:        self.indexing_device            = cuda.to_device(np.int16(self.indexing.ravel()))
src/BANG/model_creation.py:        self.goodpix_device             = cuda.to_device(np.float32(self.goodpix))
src/BANG/model_creation.py:        self.LM_avg_B_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_D1_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.LM_avg_D2_device            = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_B_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_D1_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.rho_avg_D2_device           = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_B_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_D1_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.v_avg_D2_device             = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_B_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_D1_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.sigma_avg_D2_device         = cuda.device_array((self.N*self.J),dtype=np.float32)
src/BANG/model_creation.py:        self.s_grid_device        = cuda.to_device(np.float32(s_grid))
src/BANG/model_creation.py:        self.gamma_grid_device    = cuda.to_device(np.float32(gamma_grid))
src/BANG/model_creation.py:        self.rho_grid_device      = cuda.to_device(np.float32(rho_grid))
src/BANG/model_creation.py:        self.sigma_grid_device    = cuda.to_device(np.float32(sigma_grid))
src/BANG/model_creation.py:        self.central_rho_device   = cuda.to_device(np.float32(central_rho))
src/BANG/model_creation.py:        self.central_sigma_device = cuda.to_device(np.float32(central_sigma))
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:            cuda.select_device(self.gpu_device)
src/BANG/model_creation.py:        model_goodpix = cuda.to_device(np.float32(np.ones_like(self.ibrightness_data)))
src/BANG/model_creation.py:#class model_superresolution_gpu():
src/BANG/model_creation.py:#        This is not full on gpu but it combines this... probably the fastest combination is 
src/BANG/model_creation.py:#        lr img on GPU
src/BANG/model_creation.py:#        self.device = torch.device('cuda')
src/BANG/model_creation.py:#        self.x_device = cuda.to_device(np.float32(self.x_g))
src/BANG/model_creation.py:#        self.y_device = cuda.to_device(np.float32(self.y_g))
src/BANG/model_creation.py:#        self.r_proj_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.r_true_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.phi_device           = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.all_rhoB_device      = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.all_sigma2_device    = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.all_rhoD1_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.all_rhoD2_device     = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.all_v_device         = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.all_v2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.Xs_device            = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.v_exp1_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.v_exp2_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.v_bulge_device       = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.v_halo_device        = cuda.device_array((self.new_N*self.new_J*self.K),dtype=np.float32)
src/BANG/model_creation.py:#        self.LM_avg_device            = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
src/BANG/model_creation.py:#        self.rho_avg_device           = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
src/BANG/model_creation.py:#        self.v_avg_device             = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
src/BANG/model_creation.py:#        self.sigma_avg_device         = cuda.device_array((self.new_N*self.new_J),dtype=np.float32)
src/BANG/model_creation.py:#        self.tot = cuda.device_array((4,1,self.new_N,self.new_J),dtype=np.float32)
src/BANG/model_creation.py:#        self.lk_device            = cuda.to_device(np.float32(np.zeros(1)))
src/BANG/model_creation.py:#        self.psf_device                 = cuda.to_device(np.float32(self.psf.ravel()))
src/BANG/model_creation.py:#        self.ibrightness_data_device    = cuda.to_device(np.float32(self.ibrightness_data)) 
src/BANG/model_creation.py:#        self.v_data_device              = cuda.to_device(np.float32(self.v_data)) 
src/BANG/model_creation.py:#        self.sigma_data_device          = cuda.to_device(np.float32(self.sigma_data)) 
src/BANG/model_creation.py:#        self.ML_data_device             = cuda.to_device(np.float32(self.ML_data)) 
src/BANG/model_creation.py:#        self.ibrightness_error_device   = cuda.to_device(np.float32(self.ibrightness_error)) 
src/BANG/model_creation.py:#        self.v_error_device             = cuda.to_device(np.float32(self.v_error)) 
src/BANG/model_creation.py:#        self.sigma_error_device         = cuda.to_device(np.float32(self.sigma_error))
src/BANG/model_creation.py:#        self.ML_error_device            = cuda.to_device(np.float32(self.ML_error))
src/BANG/model_creation.py:class model(model_gpu,model_cpu,model_gpu_BH,model_cpu_BH,
src/BANG/model_creation.py:            model_gpu_BD,model_cpu_BD,model_gpu_BD_BH,model_cpu_BD_BH,model_gpu_DD,
src/BANG/model_creation.py:            model_gpu_B_BH,model_cpu_B_BH,model_cpu_DD):#,
src/BANG/model_creation.py:            #model_superresolution_cpu,model_superresolution_gpu):
src/BANG/model_creation.py:                 gpu_device,
src/BANG/model_creation.py:                        Type of device (gpu/cpu) 
src/BANG/model_creation.py:                    gpu_device : int
src/BANG/model_creation.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
src/BANG/model_creation.py:        if self.device == 'GPU':
src/BANG/model_creation.py:                model_gpu_BH.__init__(self,
src/BANG/model_creation.py:                self.__likelihood = model_gpu_BH.likelihood
src/BANG/model_creation.py:                self.__model      = model_gpu_BH.model 
src/BANG/model_creation.py:                model_gpu.__init__(self,
src/BANG/model_creation.py:                                   gpu_device,
src/BANG/model_creation.py:                self.__likelihood = model_gpu.likelihood
src/BANG/model_creation.py:                self.__model      = model_gpu.model  
src/BANG/model_creation.py:                model_gpu_BD_BH.__init__(self,
src/BANG/model_creation.py:                self.__likelihood = model_gpu_BD_BH.likelihood
src/BANG/model_creation.py:                self.__model      = model_gpu_BD_BH.model                 
src/BANG/model_creation.py:                model_gpu_BD.__init__(self,
src/BANG/model_creation.py:                                   gpu_device,
src/BANG/model_creation.py:                self.__likelihood = model_gpu_BD.likelihood
src/BANG/model_creation.py:                self.__model      = model_gpu_BD.model 
src/BANG/model_creation.py:                model_gpu_DD.__init__(self,
src/BANG/model_creation.py:                                   gpu_device,
src/BANG/model_creation.py:                self.__likelihood = model_gpu_DD.likelihood
src/BANG/model_creation.py:                self.__model      = model_gpu_DD.model   
src/BANG/model_creation.py:                model_gpu_B_BH.__init__(self,
src/BANG/model_creation.py:                self.__likelihood = model_gpu_B_BH.likelihood
src/BANG/model_creation.py:                self.__model      = model_gpu_B_BH.model                 
src/BANG/model_creation.py:        elif self.device == 'super resolution GPU':
src/BANG/model_creation.py:            #    model_superresolution_gpu.__init__(self,
src/BANG/model_creation.py:            #                       'GPU',
src/BANG/model_creation.py:            #    self.__likelihood = model_superresolution_gpu.likelihood
src/BANG/model_creation.py:            #    self.__model      = model_superresolution_gpu.model 
src/BANG/model_creation.py:    gpu_device = 0
src/BANG/model_creation.py:    my_model_GPU = model('GPU',
src/BANG/model_creation.py:                    gpu_device,
src/BANG/model_creation.py:    rho_GPU,rho_B_GPU,rho_D1_GPU,rho_D2_GPU,\
src/BANG/model_creation.py:    v_GPU,v_B_GPU,v_D1_GPU,v_D2_GPU,\
src/BANG/model_creation.py:    sigma_GPU,sigma_B_GPU,sigma_D1_GPU,sigma_D2_GPU,\
src/BANG/model_creation.py:    LM_GPU,LM_B_GPU,LM_D1_GPU,LM_D2_GPU = my_model_GPU.model(all_par) 
src/BANG/model_creation.py:    #lk = my_model_GPU.likelihood(all_par)   
src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.brightness_map(np.log10(rho_GPU))
src/BANG/model_creation.py:    fig.savefig('test_model/B_gpu.png')
src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.velocity_map(v_GPU)
src/BANG/model_creation.py:    fig.savefig('test_model/v_gpu.png')
src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_GPU)
src/BANG/model_creation.py:    fig.savefig('test_model/sigma_gpu.png')
src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_B_GPU)
src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D1_GPU)
src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.dispersion_map(sigma_D2_GPU)
src/BANG/model_creation.py:    fig,ax,pmesh,cbar = data_obj.ML_map(1.0/LM_GPU)
src/BANG/model_creation.py:    fig.savefig('test_model/ML_gpu.png')
src/BANG/model_creation.py:                    gpu_device,
src/BANG/example_script.py:- run a script similar to the one below and wait for the result. On a standard gpu you parameter
src/BANG/utils_numba_1D_32bit_no_proxy.py:from numba import cuda,float64,float32,int16
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32)',
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32)',
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('float32(float32,float32,float32,float32,float32,float32)',
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32,float32[::1],float32[::1],float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32,float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32,float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32,float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32,float32[::1],float32[::1],float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32,float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:@cuda.jit('void(float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32[::1],float32,float32[::1])',
src/BANG/utils_numba_1D_32bit_no_proxy.py:    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
src/BANG/utils_numba_1D_32bit_no_proxy.py:        cuda.atomic.add(likelihood,0,tmp)
src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
src/BANG/utils_numba_1D_32bit_no_proxy.py:        Arrays allocated on device(GPU):  
src/BANG/utils_python.py:    CPU/GPU FUNCTION
src/BANG/utils_python.py:    CPU/GPU FUNCTION
src/BANG/utils_python.py:    CPU/GPU FUNCTION
src/BANG/utils_python.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src/BANG/parameter_search.py:from numba import cuda
src/BANG/parameter_search.py:                gpu_device,
src/BANG/parameter_search.py:                        Type of device (gpu/cpu) 
src/BANG/parameter_search.py:                    gpu_device : int
src/BANG/parameter_search.py:                        Index of gpu device, needed in case of multi gpu (pass 0 otherwise)  
src/BANG/parameter_search.py:                                         gpu_device,
src/BANG/parameter_search.py:                              gpu_device,
src/BANG/parameter_search.py:    gpu_device = 0
src/BANG/parameter_search.py:                    gpu_device,
BANG_Fabio_Rigamonti.egg-info/PKG-INFO:BANG is a GPU/CPU-python code for modelling both the photometry and kinematics of galaxies.
BANG_Fabio_Rigamonti.egg-info/PKG-INFO:We strongly suggest to run BANG on GPU. CPU parameter estimation can take

```
