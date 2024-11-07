# https://github.com/exoclime/THOR

```console
mjolnir/hamarr.py:    import pycuda.driver as cuda
mjolnir/hamarr.py:    import pycuda.autoinit
mjolnir/hamarr.py:    from pycuda.compiler import SourceModule
mjolnir/hamarr.py:    import pycuda.gpuarray as gpuarray
mjolnir/hamarr.py:    has_pycuda = True
mjolnir/hamarr.py:    has_pycuda = False
mjolnir/hamarr.py:if has_pycuda:
mjolnir/hamarr.py:    # cuda functions and sizes
mjolnir/hamarr.py:    gridGPU = (np.int(np.floor(num_ll / 256)) + 1, 1, 1)
mjolnir/hamarr.py:    blockGPU = (256, 1, 1)
mjolnir/hamarr.py:    find_nearest(cuda.Out(near), cuda.In(v_ll.ravel()), cuda.In(v_ico.ravel()),
mjolnir/hamarr.py:                 block=blockGPU, grid=gridGPU)
mjolnir/hamarr.py:    calc_weights(cuda.Out(weight3.ravel()), cuda.InOut(near3.ravel()),
mjolnir/hamarr.py:                 cuda.In(v_ll.ravel()), cuda.In(v_ico.ravel()),
mjolnir/hamarr.py:                 cuda.In(grid.pntloc.ravel()), np.int32(num_ll),
mjolnir/hamarr.py:                 np.int32(grid.point_num), block=blockGPU, grid=gridGPU)
mjolnir/hamarr.py:    # handles set up and running of vertical interpolation in pycuda
mjolnir/hamarr.py:    gridgpu = (np.int(np.floor(len(x_non_mono_check.ravel()) / 256)) + 1, 1, 1)
mjolnir/hamarr.py:    blockgpu = (256, 1, 1)
mjolnir/hamarr.py:    vert_lin_interp(cuda.In(x), cuda.In(y), cuda.In(xnew),
mjolnir/hamarr.py:                    cuda.Out(ynew.ravel()), np.int32(len(xnew)),
mjolnir/hamarr.py:                    cuda.InOut(x_non_mono_check.ravel()),
mjolnir/hamarr.py:                    cuda.InOut(xnew_non_mono_check), grid=gridgpu, block=blockgpu)
mjolnir/hamarr.py:    # points to the destination point. PyCuda is used to accelerate the process.
mjolnir/hamarr.py:    # Remapping from height to pressure is done on all columns via pycuda
Makefile:obj_cuda   := esp.o grid.o esp_initial.o simulation_setup.o thor_driver.o profx_driver.o esp_output.o debug_helpers.o profx_globdiag.o reduction_add.o phy_modules_device.o ultrahot_thermo.o profx_sponge.o cuda_device_memory.o insolation.o chemistry.o phy_modules.o radiative_transfer.o boundary_layer.o diagnostics.o
Makefile:obj := $(obj_cpp) $(obj_cuda)
Makefile:CUDA_PATH := /usr/lib/cuda/
Makefile:CUDA_LIBS := /usr/lib/x86-64-linux-gnu/
Makefile:	cuda_dependencies_flags = --generate-dependencies
Makefile:	cuda_flags := $(ccbin) --compiler-options  -Wall -std=c++14 -DDEVICE_SM=$(SM)
Makefile:	cuda_dep_flags := $(ccbin) -std=c++14
Makefile:	arch := --cuda-gpu-arch=sm_$(SM)
Makefile:	cuda_dependencies_flags := -MP -MMD -MF $(OBJDIR)/$(OUTPUTDIR)/$.d
Makefile:	cuda_flags := -Wall -std=c++14 -DDEVICE_SM=$(SM) --cuda-path=$(CUDA_PATH)
Makefile:	cuda_dep_flags := -std=c++14 --cuda-path=$(CUDA_PATH)
Makefile:	link_flags = --cuda-path=$(CUDA_PATH) -L$(CUDA_LIBS) -lcudart_static -ldl -lrt -pthread
Makefile:	cuda_flags += $(profiling_flags) -DBUILD_LEVEL="\"profiling\""
Makefile:	cuda_flags += $(debug_flags) -DBUILD_LEVEL="\"debug\""
Makefile:	cuda_flags += $(release_flags) -DBUILD_LEVEL="\"release\""
Makefile:	cuda_flags += $(debug_flags) -DBUILD_LEVEL="\"test\""
Makefile:	cuda_flags += $(release_flags) -DBUILD_LEVEL="\"release\""
Makefile:$(info cuda_flags: $(cuda_flags) )
Makefile:# CUDA files
Makefile:	$(CC) $(cuda_dependencies_flags) $(arch)  $(cuda_flags) $(h5include) -I$(includedir)  -I$(OBJDIR) $(CDB) $(ALFRODULL_FLAGS) $< > $(OBJDIR)/${OUTPUTDIR}/${DEPDIR}/esp.d.$$$$; \
Makefile:	$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) -I$(includedir) -I$(OBJDIR) $(CDB) $(ALFRODULL_FLAGS) -o $@ $<
Makefile:	$(CC) $(cuda_dependencies_flags) $(arch)  $(cuda_flags) $(h5include) $(includeflags) $(CDB) $(ALFRODULL_FLAGS) $< > $(OBJDIR)/${OUTPUTDIR}/${DEPDIR}/$*.d.$$$$; \
Makefile:	$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) $(includeflags) $(CDB) $(ALFRODULL_FLAGS) -o $@ $<
Makefile:		sed -i.bak s/-xcuda/-xc++/g compile_commands.json; \
README.md:*THOR* is a GCM that solves the three-dimensional non-hydrostatic Euler equations on an icosahedral grid. *THOR* was designed to run on Graphics Processing Units (GPUs).
README.md:   $ sudo apt-get install git make gcc g++ cmake nvidia-cuda-toolkit nvidia-utils-390 libhdf5-dev libhdf5-100  libhdf5-serial-dev libhdf5-cpp-100
README.md:Find the `SM` value of your Nvidia GPU. Decide if you want to run without any physics module `empty` physics module, or the one with radiative transfer, the `multi` module. Then open `Makefile.conf` in a text editor and edit like so:
tools/slurm_batch_run.py:# gpu_key = gpu:gtx1080ti:1
tools/slurm_batch_run.py:# partition = gpu
tools/slurm_batch_run.py:               'gpu_key': 'gpu:1',                   # slurm argument for gpu selection  ( --gres )
tools/slurm_batch_run.py:parser.add_argument("-par", "--partition", nargs=1, default=['gpu-invest'],help='specify partition')
tools/slurm_batch_run.py:parser.add_argument("-qos","--qos_preempt",action="store_true",default=False, help='add qos=job_gpu_preempt argument')
tools/slurm_batch_run.py:parser.add_argument("-g","--gpu_type",nargs=1,default=['gtx1080ti'],type=str,help="type of GPU:'gtx1080ti','rtx2080ti','rtx3090','teslaP100'")
tools/slurm_batch_run.py:#    qos = 'job_gpu_preempt'
tools/slurm_batch_run.py:#    qos = 'job_gpu'
tools/slurm_batch_run.py:gres_argument = 'gpu:'+args.gpu_type[0]+':1'
tools/slurm_batch_run.py:    sbatch_args.append('--qos=job_gpu_preempt')
tools/slurm_batch_run.py:                   '--gres', config_data['gpu_key'],
tools/slurm_batch_run.py:                   '--gres', config_data['gpu_key'],
tools/check_cuda.cu:    cudaDeviceProp dP;
tools/check_cuda.cu:    int rc = cudaGetDeviceProperties(&dP, 0);
tools/check_cuda.cu:    if (rc != cudaSuccess) {
tools/check_cuda.cu:        cudaError_t error = cudaGetLastError();
tools/check_cuda.cu:        printf("CUDA error: %s", cudaGetErrorString(error));
tools/check_cuda.cu:        printf("Min Compute Capability of %2.1f required:  %d.%d found\n Not Building CUDA Code",
CMakeLists.txt:# the script should autodetect the CUDA architecture, when run alone
CMakeLists.txt:project (THOR CUDA CXX)
CMakeLists.txt:  find_package(CUDA REQUIRED)
CMakeLists.txt:  set(USING_CUDA_LANG_SUPPORT False)
CMakeLists.txt:set(SM "0" CACHE STRING "GPU SM value")
CMakeLists.txt:# if CUDA wants gcc/g++ 5
CMakeLists.txt:project (THOR CUDA CXX)
CMakeLists.txt:  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS_DEBUG})
CMakeLists.txt:string (APPEND CMAKE_CUDA_FLAGS " --compiler-options -Wall ")
CMakeLists.txt:check_language(CUDA)
CMakeLists.txt:  # Find CUDA
CMakeLists.txt:  #  find_package(CUDA REQUIRED)
CMakeLists.txt:  #if (CUDA_FOUND AND SM MATCHES "0")
CMakeLists.txt:  #  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
CMakeLists.txt:  #  message(STATUS "CUDA Architecture manually set to: -arch=sm_${SM}")
CMakeLists.txt:#LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
CMakeLists.txt:#LIST(APPEND CUDA_NVCC_FLAGS ${COMPILE_FLAGS})
CMakeLists.txt:#LIST(APPEND CUDA_NVCC_FLAGS "-std c++14")
CMakeLists.txt:#LIST(APPEND CUDA_NVCC_FLAGS "-ccbin gcc-5")
CMakeLists.txt:string (APPEND CMAKE_CUDA_FLAGS " -cudart shared" )
CMakeLists.txt:set(CMAKE_CUDA_STANDARD 14)
CMakeLists.txt:set(CMAKE_CUDA_STANDARD_REQUIRED ON)
CMakeLists.txt:  src/utils/cuda_device_memory.cu
ifile/deephj.thr:# GPU ID number
ifile/deephj.thr:GPU_ID_N = 0
ifile/earth_acoustic_test.thr:# GPU ID number
ifile/earth_acoustic_test.thr:GPU_ID_N = 0
ifile/earth_hstest.thr:# GPU ID number
ifile/earth_hstest.thr:GPU_ID_N = 0
ifile/repo_benchmarks/earth_hstest.thr:# GPU ID number
ifile/repo_benchmarks/earth_hstest.thr:GPU_ID_N = 0
ifile/repo_benchmarks/hd189b_constg.thr:# GPU ID number
ifile/repo_benchmarks/hd189b_constg.thr:GPU_ID_N = 0
ifile/repo_benchmarks/sync_rot_pbl_test.thr:# GPU ID number
ifile/repo_benchmarks/sync_rot_pbl_test.thr:GPU_ID_N = 0
ifile/repo_benchmarks/earth_rt_dc_g5.thr:# GPU ID number
ifile/repo_benchmarks/earth_rt_dc_g5.thr:GPU_ID_N = 0
ifile/repo_benchmarks/wasp43_ni_i2s.thr:# GPU ID number
ifile/repo_benchmarks/wasp43_ni_i2s.thr:GPU_ID_N = 0
ifile/repo_benchmarks/wasp43b_ex.thr:# GPU ID number
ifile/repo_benchmarks/wasp43b_ex.thr:GPU_ID_N = 0
ifile/repo_benchmarks/hd189b_fullg.thr:# GPU ID number
ifile/repo_benchmarks/hd189b_fullg.thr:GPU_ID_N = 0
ifile/earth_gwave_test.thr:# GPU ID number
ifile/earth_gwave_test.thr:GPU_ID_N = 0
ifile/earth_rt_dc_g5.thr:# GPU ID number
ifile/earth_rt_dc_g5.thr:GPU_ID_N = 0
ifile/shallowhj.thr:# GPU ID number
ifile/shallowhj.thr:GPU_ID_N = 0
ifile/picket_fence_wasp43b_hydro.thr:# GPU ID number
ifile/picket_fence_wasp43b_hydro.thr:GPU_ID_N = 0
ifile/earth_sync.thr:# GPU ID number
ifile/earth_sync.thr:GPU_ID_N = 0
ifile/wasp43b_ex.thr:# GPU ID number
ifile/wasp43b_ex.thr:GPU_ID_N = 0
ifile/input_template.thr:# Use the highest number your GPU will allow!
ifile/input_template.thr:# GPU ID number
ifile/input_template.thr:GPU_ID_N = 0
ifile/earth_localRi_g5_dc.thr:# GPU ID number
ifile/earth_localRi_g5_dc.thr:GPU_ID_N = 0
ifile/wasp43b_tsrt.thr:# GPU ID number
ifile/wasp43b_tsrt.thr:GPU_ID_N = 0
ifile/alf_wasp_alf.thr:# GPU ID number
ifile/alf_wasp_alf.thr:GPU_ID_N = 0
src/test/reduction_add_test.cu:template<int BLOCK_SIZE> bool cpu_gpu_test(double *s, long size) {
src/test/reduction_add_test.cu:        double output_val = gpu_sum_from_host<BLOCK_SIZE>(s, compute_size);
src/test/reduction_add_test.cu:        auto duration_gpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
src/test/reduction_add_test.cu:        printf("[%ld] [%s] Computed in: CPU: %ld us, GPU: %ld us, CPU/GPU ratio: %f\n",
src/test/reduction_add_test.cu:               duration_gpu.count(),
src/test/reduction_add_test.cu:               double(duration_cpu.count()) / double(duration_gpu.count()));
src/test/reduction_add_test.cu:            printf("GPU reduction sum: %32.15f\n", output_val);
src/test/reduction_add_test.cu:    overall_result &= cpu_gpu_test<512>(s, size);
src/test/reduction_add_test.cu:    overall_result &= cpu_gpu_test<1024>(s, size);
src/test/reduction_add_test.cu:    double *vbar_d, *vbar_h_cpu, *vbar_h_gpu, *vbar_h_gpu2, *utmp, *vtmp, *wtmp, *utmp_h, *vtmp_h,
src/test/reduction_add_test.cu:    vbar_h_gpu  = (double *)malloc(3 * nv * nlat_bins * sizeof(double));
src/test/reduction_add_test.cu:    vbar_h_gpu2 = (double *)malloc(3 * nv * nlat_bins * sizeof(double));
src/test/reduction_add_test.cu:    cudaMalloc((void **)&utmp, nv * nlat_bins * max_count * sizeof(double));
src/test/reduction_add_test.cu:    cudaMalloc((void **)&vtmp, nv * nlat_bins * max_count * sizeof(double));
src/test/reduction_add_test.cu:    cudaMalloc((void **)&wtmp, nv * nlat_bins * max_count * sizeof(double));
src/test/reduction_add_test.cu:    cudaMalloc((void **)&vbar_d, nv * nlat_bins * 3 * sizeof(double));
src/test/reduction_add_test.cu:            vbar_h_gpu[ilat * nv * 3 + lev * 3 + 0] = 0.0;
src/test/reduction_add_test.cu:            vbar_h_gpu[ilat * nv * 3 + lev * 3 + 1] = 0.0;
src/test/reduction_add_test.cu:            vbar_h_gpu[ilat * nv * 3 + lev * 3 + 2] = 0.0;
src/test/reduction_add_test.cu:            vbar_h_gpu2[ilat * nv * 3 + lev * 3 + 0] = 0.0;
src/test/reduction_add_test.cu:            vbar_h_gpu2[ilat * nv * 3 + lev * 3 + 1] = 0.0;
src/test/reduction_add_test.cu:            vbar_h_gpu2[ilat * nv * 3 + lev * 3 + 2] = 0.0;
src/test/reduction_add_test.cu:    cudaMemcpy(utmp, utmp_h, nv * nlat_bins * max_count * sizeof(double), cudaMemcpyHostToDevice);
src/test/reduction_add_test.cu:    cudaMemcpy(vtmp, vtmp_h, nv * nlat_bins * max_count * sizeof(double), cudaMemcpyHostToDevice);
src/test/reduction_add_test.cu:    cudaMemcpy(wtmp, wtmp_h, nv * nlat_bins * max_count * sizeof(double), cudaMemcpyHostToDevice);
src/test/reduction_add_test.cu:            vbar_h_gpu[ilat * nv * 3 + lev * 3 + 0] = gpu_sum_on_device<1024>(
src/test/reduction_add_test.cu:            vbar_h_gpu[ilat * nv * 3 + lev * 3 + 1] = gpu_sum_on_device<1024>(
src/test/reduction_add_test.cu:            vbar_h_gpu[ilat * nv * 3 + lev * 3 + 2] = gpu_sum_on_device<1024>(
src/test/reduction_add_test.cu:    cudaMemcpy(vbar_d, vbar_h_gpu, 3 * nlat_bins * nv * sizeof(double), cudaMemcpyHostToDevice);
src/test/reduction_add_test.cu:                         - vbar_h_gpu[ilat * nv * 3 + lev * 3 + xyz])
src/test/reduction_add_test.cu:                                - vbar_h_gpu[ilat * nv * 3 + lev * 3 + xyz])
src/test/reduction_add_test.cu:            //             - vbar_h_gpu[ilat * nv * 3 + lev * 3 + 2])
src/test/reduction_add_test.cu:    gpu_sum_on_device_sponge<1024>(utmp, max_count, vbar_h_gpu2, nv, nlat_bins, 0);
src/test/reduction_add_test.cu:    gpu_sum_on_device_sponge<1024>(vtmp, max_count, vbar_h_gpu2, nv, nlat_bins, 1);
src/test/reduction_add_test.cu:    gpu_sum_on_device_sponge<1024>(wtmp, max_count, vbar_h_gpu2, nv, nlat_bins, 2);
src/test/reduction_add_test.cu:    cudaMemcpy(vbar_d, vbar_h_gpu2, 3 * nlat_bins * nv * sizeof(double), cudaMemcpyHostToDevice);
src/test/reduction_add_test.cu:                         - vbar_h_gpu2[ilat * nv * 3 + lev * 3 + xyz])
src/test/reduction_add_test.cu:                                - vbar_h_gpu2[ilat * nv * 3 + lev * 3 + xyz])
src/test/reduction_add_test.cu:            //             - vbar_h_gpu[ilat * nv * 3 + lev * 3 + 2])
src/ESP/esp_initial.cu:    cudaMalloc((void **)&boundary_flux_d, 6 * point_num * nv * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&point_local_d, 6 * point_num * sizeof(int));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&maps_d, (nl_region + 2) * (nl_region + 2) * nr * sizeof(int));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&nvecoa_d, 6 * 3 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&nvecti_d, 6 * 3 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&nvecte_d, 6 * 3 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&areasT_d, point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&areasTr_d, 6 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&areas_d, 3 * 6 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&func_r_d, 3 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&div_d, 7 * 3 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&grad_d, 7 * 3 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Altitude_d, nv * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Altitudeh_d, nvi * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&lonlat_d, 2 * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Mh_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&W_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Wh_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Rho_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&pressure_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&pressureh_d, (nv + 1) * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&dT_conv_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Mh_mean_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Wh_mean_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Rho_mean_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&pressure_mean_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Mh_start_dt_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Mh_profx_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Rho_start_dt_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Rho_profx_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Rd_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Cp_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&temperature_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&pt_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&pth_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&epotential_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&epotentialh_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&ekinetic_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&ekinetich_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Etotal_tau_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&h_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&hh_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Adv_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&v_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&gtil_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&gtilh_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&SlowMh_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&SlowWh_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&SlowRho_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Slowpressure_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&pressures_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Rhos_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Mhs_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Ws_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Whs_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&pressurek_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Rhok_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Mhk_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Wk_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Whk_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Sp_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Sd_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Kdhz_d, nv * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Kdh4_d, nv * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Kdvz_d, nv * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Kdv6_d, nv * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&DivM_d, nv * point_num * 3 * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffpr_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffmh_d, 3 * nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffw_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffrh_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diff_d, 6 * nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&diff_sponge_d, 6 * nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&divg_Mh_d, 3 * nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Kdh2_d, nv * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffprv_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffmv_d, 3 * nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffwv_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diffrv_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&diff2_d, 6 * (nv + 2) * point_num * sizeof(double));
src/ESP/esp_initial.cu:    // cudaMalloc((void **)&diffv_d2, 6 * nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&profx_Qheat_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&profx_dMh_d, 3 * nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&profx_dWh_d, nvi * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&profx_dW_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&check_d, sizeof(bool));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&vbar_d, 3 * nv * nlat_bins * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&zonal_mean_tab_d, 3 * point_num * sizeof(int));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&utmp, nv * nlat_bins * max_count * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&vtmp, nv * nlat_bins * max_count * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&wtmp, nv * nlat_bins * max_count * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Tbar_d, nv * nlat_bins * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Ttmp, nv * nlat_bins * max_count * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Etotal_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Entropy_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&Mass_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&AngMomx_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&AngMomy_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&AngMomz_d, nv * point_num * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&GlobalE_d, 1 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&GlobalEnt_d, 1 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&GlobalMass_d, 1 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&GlobalAMx_d, 1 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&GlobalAMy_d, 1 * sizeof(double));
src/ESP/esp_initial.cu:        cudaMalloc((void **)&GlobalAMz_d, 1 * sizeof(double));
src/ESP/esp_initial.cu:            cudaMalloc((void **)&Esurf_d, point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&Tsurface_d, point_num * sizeof(double));
src/ESP/esp_initial.cu:    cudaMalloc((void **)&dTsurf_dt_d, point_num * sizeof(double));
src/ESP/esp_initial.cu:            cudaMemcpy(Altitude_d, Altitude_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:            cudaMemcpy(
src/ESP/esp_initial.cu:                pressure_d, pressure_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:            cudaMemcpy(Mh_d, Mh_h, 3 * point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:            cudaMemcpy(Rho_d, Rho_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:            cudaMemcpy(temperature_d,
src/ESP/esp_initial.cu:                       cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:            cudaMemcpy(lonlat_d, lonlat_h, 2 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:            cudaMemcpy(Mh_h, Mh_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_initial.cu:            cudaMemcpy(temperature_h,
src/ESP/esp_initial.cu:                       cudaMemcpyDeviceToHost);
src/ESP/esp_initial.cu:            cudaMemcpy(
src/ESP/esp_initial.cu:                pressure_h, pressure_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_initial.cu:            cudaMemcpy(Rho_h, Rho_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_initial.cu:    cudaMemcpy(point_local_d, point_local_h, 6 * point_num * sizeof(int), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(maps_d,
src/ESP/esp_initial.cu:               cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Altitude_d, Altitude_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Altitudeh_d, Altitudeh_h, nvi * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(nvecoa_d, nvecoa_h, 6 * 3 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(nvecti_d, nvecti_h, 6 * 3 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(nvecte_d, nvecte_h, 6 * 3 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(areasTr_d, areasTr_h, 6 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(areasT_d, areasT_h, point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(areas_d, areas_h, 3 * 6 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(lonlat_d, lonlat_h, 2 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(func_r_d, func_r_h, 3 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(
src/ESP/esp_initial.cu:        temperature_d, temperature_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Mh_d, Mh_h, point_num * nv * 3 * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    //cudaMemcpy(diffmh_d, diffmh_h, point_num * nv * 3 * sizeof(double), cudaMemcpyHostToDevice);// i think this is not needed -RD
src/ESP/esp_initial.cu:    cudaMemcpy(W_d, W_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Wh_d, Wh_h, point_num * nvi * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Rho_d, Rho_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(pressure_d, pressure_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(div_d, div_h, 7 * 3 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(grad_d, grad_h, 7 * 3 * point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Kdhz_d, Kdhz_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Kdh4_d, Kdh4_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Kdvz_d, Kdvz_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Kdv6_d, Kdv6_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Kdh2_d, Kdh2_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:        cudaMemcpy(Mh_mean_d, Mh_h, point_num * nv * 3 * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:        cudaMemcpy(
src/ESP/esp_initial.cu:            pressure_mean_d, pressure_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:        cudaMemcpy(Wh_mean_d, Wh_h, point_num * nvi * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:        cudaMemcpy(Rho_mean_d, Rho_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:        cudaMemcpy(zonal_mean_tab_d,
src/ESP/esp_initial.cu:                   cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Rd_d, Rd_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(Cp_d, Cp_h, point_num * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(GibbsT_d, GibbsT, GibbsN * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemcpy(GibbsdG_d, GibbsdG, GibbsN * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:        cudaMemcpy(Tsurface_d, Tsurface_h, point_num * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/esp_initial.cu:    cudaMemset(Adv_d, 0, sizeof(double) * 3 * point_num * nv);
src/ESP/esp_initial.cu:    cudaMemset(v_d, 0, sizeof(double) * nv * point_num * 3);
src/ESP/esp_initial.cu:    cudaMemset(pt_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(pth_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    // cudaMemset(pt_tau_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(epotential_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(epotentialh_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    cudaMemset(ekinetic_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(ekinetich_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Etotal_tau_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(SlowMh_d, 0, sizeof(double) * nv * point_num * 3);
src/ESP/esp_initial.cu:    cudaMemset(SlowWh_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    cudaMemset(SlowRho_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Slowpressure_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(h_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(hh_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Rhos_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(pressures_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Mhs_d, 0, sizeof(double) * nv * point_num * 3);
src/ESP/esp_initial.cu:    cudaMemset(Ws_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Whs_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    cudaMemset(gtil_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(gtilh_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Rhok_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(pressurek_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Mhk_d, 0, sizeof(double) * nv * point_num * 3);
src/ESP/esp_initial.cu:    cudaMemset(Wk_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Whk_d, 0, sizeof(double) * nvi * point_num);
src/ESP/esp_initial.cu:    cudaMemset(Sp_d, 0, sizeof(double) * point_num * nv);
src/ESP/esp_initial.cu:    cudaMemset(Sd_d, 0, sizeof(double) * point_num * nv);
src/ESP/esp_initial.cu:    cudaMemset(DivM_d, 0, sizeof(double) * point_num * 3 * nv);
src/ESP/esp_initial.cu:    cudaMemset(diffpr_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diffmh_d, 0, sizeof(double) * 3 * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diffw_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diffrh_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diff_d, 0, sizeof(double) * 6 * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(divg_Mh_d, 0, sizeof(double) * 3 * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diffprv_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diffmv_d, 0, sizeof(double) * 3 * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diffwv_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diffrv_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(diff2_d, 0, sizeof(double) * 6 * (nv + 2) * point_num);
src/ESP/esp_initial.cu:    // cudaMemset(diffv_d2, 0, sizeof(double) * 6 * nv * point_num);
src/ESP/esp_initial.cu:        cudaMemset(Mh_start_dt_d, 0, sizeof(double) * nv * point_num * 3);
src/ESP/esp_initial.cu:        cudaMemset(Mh_profx_d, 0, sizeof(double) * nv * point_num * 3);
src/ESP/esp_initial.cu:        cudaMemset(Rho_start_dt_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:        cudaMemset(Rho_profx_d, 0, sizeof(double) * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(boundary_flux_d, 0, sizeof(double) * 6 * nv * point_num);
src/ESP/esp_initial.cu:    cudaMemset(profx_dMh_d, 0, sizeof(double) * 3 * point_num * nv);
src/ESP/esp_initial.cu:    cudaMemset(profx_dWh_d, 0, sizeof(double) * point_num * nvi);
src/ESP/esp_initial.cu:    cudaMemset(profx_dW_d, 0, sizeof(double) * point_num * nv);
src/ESP/esp_initial.cu:    cudaFree(point_local_d);
src/ESP/esp_initial.cu:    cudaFree(maps_d);
src/ESP/esp_initial.cu:    cudaFree(Altitude_d);
src/ESP/esp_initial.cu:    cudaFree(Altitudeh_d);
src/ESP/esp_initial.cu:    cudaFree(nvecoa_d);
src/ESP/esp_initial.cu:    cudaFree(nvecti_d);
src/ESP/esp_initial.cu:    cudaFree(nvecte_d);
src/ESP/esp_initial.cu:    cudaFree(areasT_d);
src/ESP/esp_initial.cu:    cudaFree(areasTr_d);
src/ESP/esp_initial.cu:    cudaFree(areas_d);
src/ESP/esp_initial.cu:    cudaFree(lonlat_d);
src/ESP/esp_initial.cu:    cudaFree(div_d);
src/ESP/esp_initial.cu:    cudaFree(grad_d);
src/ESP/esp_initial.cu:    cudaFree(func_r_d);
src/ESP/esp_initial.cu:    cudaFree(temperature_d);
src/ESP/esp_initial.cu:    cudaFree(Mh_d);
src/ESP/esp_initial.cu:    cudaFree(W_d);
src/ESP/esp_initial.cu:    cudaFree(Wh_d);
src/ESP/esp_initial.cu:    cudaFree(Rho_d);
src/ESP/esp_initial.cu:    cudaFree(pressure_d);
src/ESP/esp_initial.cu:    cudaFree(pressureh_d);
src/ESP/esp_initial.cu:    cudaFree(h_d);
src/ESP/esp_initial.cu:    cudaFree(hh_d);
src/ESP/esp_initial.cu:    cudaFree(Adv_d);
src/ESP/esp_initial.cu:    cudaFree(gtil_d);
src/ESP/esp_initial.cu:    cudaFree(gtilh_d);
src/ESP/esp_initial.cu:    cudaFree(v_d);
src/ESP/esp_initial.cu:    cudaFree(pt_d);
src/ESP/esp_initial.cu:    cudaFree(pth_d);
src/ESP/esp_initial.cu:    cudaFree(SlowMh_d);
src/ESP/esp_initial.cu:    cudaFree(SlowWh_d);
src/ESP/esp_initial.cu:    cudaFree(SlowRho_d);
src/ESP/esp_initial.cu:    cudaFree(Slowpressure_d);
src/ESP/esp_initial.cu:    cudaFree(Rhok_d);
src/ESP/esp_initial.cu:    cudaFree(pressurek_d);
src/ESP/esp_initial.cu:    cudaFree(Mhk_d);
src/ESP/esp_initial.cu:    cudaFree(Whk_d);
src/ESP/esp_initial.cu:    cudaFree(Wk_d);
src/ESP/esp_initial.cu:    cudaFree(Rhos_d);
src/ESP/esp_initial.cu:    cudaFree(pressures_d);
src/ESP/esp_initial.cu:    cudaFree(Mhs_d);
src/ESP/esp_initial.cu:    cudaFree(Whs_d);
src/ESP/esp_initial.cu:    cudaFree(Ws_d);
src/ESP/esp_initial.cu:    cudaFree(Sd_d);
src/ESP/esp_initial.cu:    cudaFree(Sp_d);
src/ESP/esp_initial.cu:    cudaFree(Kdhz_d);
src/ESP/esp_initial.cu:    cudaFree(Kdh4_d);
src/ESP/esp_initial.cu:    cudaFree(Kdvz_d);
src/ESP/esp_initial.cu:    cudaFree(Kdv6_d);
src/ESP/esp_initial.cu:    cudaFree(DivM_d);
src/ESP/esp_initial.cu:    cudaFree(diffpr_d);
src/ESP/esp_initial.cu:    cudaFree(diffmh_d);
src/ESP/esp_initial.cu:    cudaFree(diffw_d);
src/ESP/esp_initial.cu:    cudaFree(diffrh_d);
src/ESP/esp_initial.cu:    cudaFree(diff_d);
src/ESP/esp_initial.cu:    cudaFree(divg_Mh_d);
src/ESP/esp_initial.cu:    cudaFree(Kdh2_d);
src/ESP/esp_initial.cu:    cudaFree(diffprv_d);
src/ESP/esp_initial.cu:    cudaFree(diffmv_d);
src/ESP/esp_initial.cu:    cudaFree(diffwv_d);
src/ESP/esp_initial.cu:    cudaFree(diffrv_d);
src/ESP/esp_initial.cu:    cudaFree(diff2_d);
src/ESP/esp_initial.cu:    // cudaFree(diffv_d2);
src/ESP/esp_initial.cu:    cudaFree(Etotal_d);
src/ESP/esp_initial.cu:    cudaFree(Entropy_d);
src/ESP/esp_initial.cu:    cudaFree(Mass_d);
src/ESP/esp_initial.cu:    cudaFree(AngMomx_d);
src/ESP/esp_initial.cu:    cudaFree(AngMomy_d);
src/ESP/esp_initial.cu:    cudaFree(AngMomz_d);
src/ESP/esp_initial.cu:    cudaFree(GlobalE_d);
src/ESP/esp_initial.cu:    cudaFree(GlobalEnt_d);
src/ESP/esp_initial.cu:    cudaFree(GlobalMass_d);
src/ESP/esp_initial.cu:    cudaFree(GlobalAMx_d);
src/ESP/esp_initial.cu:    cudaFree(GlobalAMy_d);
src/ESP/esp_initial.cu:    cudaFree(GlobalAMz_d);
src/ESP/esp_initial.cu:    cudaFree(check_d);
src/ESP/esp_initial.cu:    cudaFree(vbar_d);
src/ESP/esp_initial.cu:    cudaFree(zonal_mean_tab_d);
src/ESP/esp_initial.cu:    cudaFree(Tbar_d);
src/ESP/esp_initial.cu:    cudaFree(utmp);
src/ESP/esp_initial.cu:    cudaFree(vtmp);
src/ESP/esp_initial.cu:    cudaFree(wtmp);
src/ESP/esp_initial.cu:    cudaFree(Ttmp);
src/ESP/esp_initial.cu:    cudaFree(profx_Qheat_d);
src/ESP/esp_initial.cu:    cudaFree(profx_dMh_d);
src/ESP/esp_initial.cu:    cudaFree(profx_dWh_d);
src/ESP/esp_initial.cu:    cudaFree(profx_dW_d);
src/ESP/esp_initial.cu:    cudaFree(epotential_d);
src/ESP/esp_initial.cu:    cudaFree(epotentialh_d);
src/ESP/esp_initial.cu:    cudaFree(ekinetic_d);
src/ESP/esp_initial.cu:    cudaFree(ekinetich_d);
src/ESP/esp_initial.cu:    cudaFree(Etotal_tau_d);
src/ESP/esp_initial.cu:    cudaFree(Rd_d);
src/ESP/esp_initial.cu:    cudaFree(Cp_d);
src/ESP/esp_initial.cu:    cudaFree(GibbsT_d);
src/ESP/esp_initial.cu:    cudaFree(GibbsdG_d);
src/ESP/esp_initial.cu:    cudaFree(boundary_flux_d);
src/ESP/esp_initial.cu:    cudaFree(Tsurface_d);
src/ESP/esp_initial.cu:    cudaFree(dTsurf_dt_d);
src/ESP/esp_initial.cu:        cudaFree(dT_conv_d);
src/ESP/profx_driver.cu:    cudaMemset(profx_Qheat_d, 0, sizeof(double) * point_num * nv);
src/ESP/profx_driver.cu:    cudaMemset(dTsurf_dt_d, 0, sizeof(double) * point_num);
src/ESP/profx_driver.cu:        cudaMemcpy(
src/ESP/profx_driver.cu:            Mh_start_dt_d, Mh_d, point_num * nv * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/profx_driver.cu:        cudaMemcpy(
src/ESP/profx_driver.cu:            Rho_start_dt_d, Rho_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/profx_driver.cu:    cudaDeviceSynchronize();
src/ESP/profx_driver.cu:            cudaMemset(utmp, 0, sizeof(double) * nlat_bins * nv * max_count);
src/ESP/profx_driver.cu:            cudaMemset(vtmp, 0, sizeof(double) * nlat_bins * nv * max_count);
src/ESP/profx_driver.cu:            cudaDeviceSynchronize();
src/ESP/profx_driver.cu:            cudaMemset(wtmp, 0, sizeof(double) * nlat_bins * nv * max_count);
src/ESP/profx_driver.cu:            cudaDeviceSynchronize();
src/ESP/profx_driver.cu:            gpu_sum_on_device_sponge<1024>(utmp, max_count, vbar_h, nv, nlat_bins, 0);
src/ESP/profx_driver.cu:            gpu_sum_on_device_sponge<1024>(vtmp, max_count, vbar_h, nv, nlat_bins, 1);
src/ESP/profx_driver.cu:            gpu_sum_on_device_sponge<1024>(wtmp, max_count, vbar_h, nv, nlat_bins, 2);
src/ESP/profx_driver.cu:            gpu_sum_on_device_sponge<1024>(utmp, max_count, vbar_h, nv, nlat_bins, 0);
src/ESP/profx_driver.cu:            gpu_sum_on_device_sponge<1024>(vtmp, max_count, vbar_h, nv, nlat_bins, 1);
src/ESP/profx_driver.cu:            gpu_sum_on_device_sponge<1024>(wtmp, max_count, vbar_h, nv, nlat_bins, 2);
src/ESP/profx_driver.cu:            cudaMemset(vbar_d, 0, sizeof(double) * 3 * nlat_bins * nv);
src/ESP/profx_driver.cu:            cudaMemcpy(vbar_d, vbar_h, 3 * nlat_bins * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/profx_driver.cu:            cudaDeviceSynchronize();
src/ESP/profx_driver.cu:                    Tbar_h[ilat * nv + lev] = gpu_sum_on_device<1024>(
src/ESP/profx_driver.cu:            cudaMemcpy(Tbar_d, Tbar_h, nlat_bins * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/profx_driver.cu:        cudaMemcpy(check_d, &check_h, sizeof(bool), cudaMemcpyHostToDevice);
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:        cudaMemcpy(&check_h, check_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/ESP/profx_driver.cu:    // cudaMemcpy(W_h, W_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:    cudaMemcpy(check_d, &check_h, sizeof(bool), cudaMemcpyHostToDevice);
src/ESP/profx_driver.cu:    cudaMemcpy(&check_h, check_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:    cudaDeviceSynchronize();
src/ESP/profx_driver.cu:        cudaDeviceSynchronize();
src/ESP/profx_driver.cu:    cudaDeviceSynchronize();
src/ESP/profx_driver.cu:    cudaMemcpy(check_d, &check_h, sizeof(bool), cudaMemcpyHostToDevice);
src/ESP/profx_driver.cu:    cudaMemcpy(&check_h, check_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/ESP/profx_driver.cu:        cudaMemcpy(Mh_profx_d, Mh_d, point_num * nv * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/profx_driver.cu:        cudaMemcpy(Rho_profx_d, Rho_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/profx_driver.cu:    cudaMemset(GlobalE_d, 0, sizeof(double));
src/ESP/profx_driver.cu:    cudaMemset(GlobalMass_d, 0, sizeof(double));
src/ESP/profx_driver.cu:    cudaMemset(GlobalAMx_d, 0, sizeof(double));
src/ESP/profx_driver.cu:    cudaMemset(GlobalAMy_d, 0, sizeof(double));
src/ESP/profx_driver.cu:    cudaMemset(GlobalAMz_d, 0, sizeof(double));
src/ESP/profx_driver.cu:    cudaMemset(GlobalEnt_d, 0, sizeof(double));
src/ESP/profx_driver.cu:    GlobalE_h    = gpu_sum_on_device<1024>(Etotal_d, point_num * nv);
src/ESP/profx_driver.cu:    GlobalMass_h = gpu_sum_on_device<1024>(Mass_d, point_num * nv);
src/ESP/profx_driver.cu:    GlobalAMx_h  = gpu_sum_on_device<1024>(AngMomx_d, point_num * nv);
src/ESP/profx_driver.cu:    GlobalAMy_h  = gpu_sum_on_device<1024>(AngMomy_d, point_num * nv);
src/ESP/profx_driver.cu:    GlobalAMz_h  = gpu_sum_on_device<1024>(AngMomz_d, point_num * nv);
src/ESP/profx_driver.cu:    GlobalEnt_h  = gpu_sum_on_device<1024>(Entropy_d, point_num * nv);
src/ESP/profx_driver.cu:        GlobalEsurf        = gpu_sum_on_device<1024>(Esurf_d, point_num);
src/ESP/grid.cu:    int divide_face; // Used to split memory on the GPU
src/ESP/ultrahot_thermo.cu:// Known limitations: - Runs in a single GPU.
src/ESP/ultrahot_thermo.cu:    cudaMalloc((void **)&GibbsT_d, nlines * sizeof(double));
src/ESP/ultrahot_thermo.cu:    cudaMalloc((void **)&GibbsdG_d, nlines * sizeof(double));
src/ESP/phy_modules_device.cu:// Known limitations: - Runs in a single GPU.
src/ESP/phy_modules_device.cu:    cudaMemcpyToSymbol(
src/ESP/phy_modules_device.cu:        cudaError_t err = cudaGetLastError();
src/ESP/phy_modules_device.cu:        if (err != cudaSuccess) {
src/ESP/phy_modules_device.cu:            log::printf("phy: array cuda error: %s\n", cudaGetErrorString(err));
src/ESP/phy_modules_device.cu:    cudaMemcpyToSymbol(num_dynamical_arrays, &datasize, sizeof(int));
src/ESP/phy_modules_device.cu:        cudaError_t err = cudaGetLastError();
src/ESP/phy_modules_device.cu:        if (err != cudaSuccess) {
src/ESP/phy_modules_device.cu:            log::printf("'phy: num' cuda error: %s\n", cudaGetErrorString(err));
src/ESP/phy_modules_device.cu:    cudaDeviceSynchronize();
src/ESP/thor_driver.cu://   - Operational in just one GPU.
src/ESP/thor_driver.cu:    cudaDeviceSynchronize();
src/ESP/thor_driver.cu:    cudaMemcpy(Mhk_d, Mh_d, point_num * nv * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(Whk_d, Wh_d, point_num * nvi * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(Wk_d, W_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(Rhok_d, Rho_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(pressurek_d, pressure_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemset(Mhs_d, 0, sizeof(double) * 3 * point_num * nv);
src/ESP/thor_driver.cu:    cudaMemset(Rhos_d, 0, sizeof(double) * point_num * nv);
src/ESP/thor_driver.cu:    cudaMemset(Whs_d, 0, sizeof(double) * point_num * nvi);
src/ESP/thor_driver.cu:    cudaMemset(Ws_d, 0, sizeof(double) * point_num * nv);
src/ESP/thor_driver.cu:    cudaMemset(pressures_d, 0, sizeof(double) * point_num * nv);
src/ESP/thor_driver.cu:        cudaMemset(Adv_d, 0, sizeof(double) * 3 * point_num * nv); // Sets every value of Adv_d to
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaMemcpy(check_d, &check_h, sizeof(bool), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaMemcpy(&check_h, check_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaMemset(SlowMh_d, 0, sizeof(double) * 3 * point_num * nv);
src/ESP/thor_driver.cu:        cudaMemset(SlowWh_d, 0, sizeof(double) * point_num * nvi);
src/ESP/thor_driver.cu:        cudaMemset(SlowRho_d, 0, sizeof(double) * point_num * nv);
src/ESP/thor_driver.cu:        cudaMemset(Slowpressure_d, 0, sizeof(double) * point_num * nv);
src/ESP/thor_driver.cu:                cudaMemcpy(Kdhz_d, Kdhz_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:                cudaMemcpy(Kdh4_d, Kdh4_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:                cudaMemcpy(Kdv6_d, Kdv6_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:                cudaMemcpy(Kdhz_d, Kdhz_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:                cudaMemcpy(Kdh4_d, Kdh4_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:                cudaMemcpy(Kdv6_d, Kdv6_h, nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:            cudaMemset(diff_d, 0, sizeof(double) * 6 * point_num * nv);
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:                cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaMemset(diffrh_d, 0, sizeof(double) * point_num * nv);
src/ESP/thor_driver.cu:            cudaMemset(boundary_flux_d, 0, sizeof(double) * 6 * point_num * nv);
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:                cudaMemset(diff_d, 0, sizeof(double) * 6 * point_num * nv);
src/ESP/thor_driver.cu:                cudaMemset(diff2_d, 0, sizeof(double) * 6 * point_num * nv);
src/ESP/thor_driver.cu:                cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaMemset(DivM_d, 0, sizeof(double) * point_num * 3 * nv);
src/ESP/thor_driver.cu:        cudaMemset(divg_Mh_d, 0, sizeof(double) * point_num * 3 * nv);
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaMemset(profx_dMh_d, 0, sizeof(double) * 3 * point_num * nv);
src/ESP/thor_driver.cu:            cudaMemset(profx_dWh_d, 0, sizeof(double) * point_num * nvi);
src/ESP/thor_driver.cu:            cudaMemset(profx_dW_d, 0, sizeof(double) * point_num * nv);
src/ESP/thor_driver.cu:                    cudaMemset(utmp, 0, sizeof(double) * nlat_bins * nv * max_count);
src/ESP/thor_driver.cu:                    cudaMemset(vtmp, 0, sizeof(double) * nlat_bins * nv * max_count);
src/ESP/thor_driver.cu:                    cudaDeviceSynchronize();
src/ESP/thor_driver.cu:                    cudaMemset(wtmp, 0, sizeof(double) * nlat_bins * nv * max_count);
src/ESP/thor_driver.cu:                    cudaDeviceSynchronize();
src/ESP/thor_driver.cu:                    gpu_sum_on_device_sponge<1024>(utmp, max_count, vbar_h, nv, nlat_bins, 0);
src/ESP/thor_driver.cu:                    gpu_sum_on_device_sponge<1024>(vtmp, max_count, vbar_h, nv, nlat_bins, 1);
src/ESP/thor_driver.cu:                    gpu_sum_on_device_sponge<1024>(wtmp, max_count, vbar_h, nv, nlat_bins, 2);
src/ESP/thor_driver.cu:                    gpu_sum_on_device_sponge<1024>(utmp, max_count, vbar_h, nv, nlat_bins, 0);
src/ESP/thor_driver.cu:                    gpu_sum_on_device_sponge<1024>(vtmp, max_count, vbar_h, nv, nlat_bins, 1);
src/ESP/thor_driver.cu:                    gpu_sum_on_device_sponge<1024>(wtmp, max_count, vbar_h, nv, nlat_bins, 2);
src/ESP/thor_driver.cu:                    cudaMemset(vbar_d, 0, sizeof(double) * 3 * nlat_bins * nv);
src/ESP/thor_driver.cu:                    cudaMemcpy(vbar_d,
src/ESP/thor_driver.cu:                               cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:                    cudaDeviceSynchronize();
src/ESP/thor_driver.cu:                            Tbar_h[ilat * nv + lev] = gpu_sum_on_device<1024>(
src/ESP/thor_driver.cu:                    cudaMemcpy(
src/ESP/thor_driver.cu:                        Tbar_d, Tbar_h, nlat_bins * nv * sizeof(double), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaMemset(DivM_d, 0, sizeof(double) * point_num * 3 * nv);
src/ESP/thor_driver.cu:            cudaMemset(divg_Mh_d, 0, sizeof(double) * point_num * 3 * nv);
src/ESP/thor_driver.cu:                cudaDeviceSynchronize();
src/ESP/thor_driver.cu:                cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaError_t err = cudaGetLastError();
src/ESP/thor_driver.cu:            if (err != cudaSuccess) {
src/ESP/thor_driver.cu:                log::printf("[%s:%d] CUDA error check reports error: %s\n",
src/ESP/thor_driver.cu:                            cudaGetErrorString(err));
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:        cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaMemcpy(check_d, &check_h, sizeof(bool), cudaMemcpyHostToDevice);
src/ESP/thor_driver.cu:            cudaDeviceSynchronize();
src/ESP/thor_driver.cu:            cudaMemcpy(&check_h, check_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/ESP/thor_driver.cu:    cudaDeviceSynchronize();
src/ESP/thor_driver.cu:    cudaMemcpy(Mh_d, Mhk_d, point_num * nv * 3 * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(Wh_d, Whk_d, point_num * nvi * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(W_d, Wk_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(Rho_d, Rhok_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/thor_driver.cu:    cudaMemcpy(pressure_d, pressurek_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToDevice);
src/ESP/profx_globdiag.cu:// Known limitations: - Runs in a single GPU.
src/ESP/insolation.cu:                        cudaDeviceSynchronize();
src/ESP/insolation.cu:                        cuda_check_status_or_exit(__FILE__, __LINE__);
src/ESP/insolation.cu:                        cudaDeviceSynchronize();
src/ESP/insolation.cu:                        cuda_check_status_or_exit(__FILE__, __LINE__);
src/ESP/insolation.cu:        cudaDeviceSynchronize();
src/ESP/insolation.cu:        cuda_check_status_or_exit(__FILE__, __LINE__);
src/ESP/esp_output.cu:    cudaMemcpy(Etotal_h, Etotal_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Entropy_h, Entropy_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Mass_h, Mass_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(AngMomx_h, AngMomx_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(AngMomy_h, AngMomy_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(AngMomz_h, AngMomz_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:        cudaMemcpy(Esurf_h, Esurf_d, point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(&GlobalE_h, GlobalE_d, sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(&GlobalEnt_h, GlobalEnt_d, sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(&GlobalMass_h, GlobalMass_d, sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(&GlobalAMx_h, GlobalAMx_d, sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(&GlobalAMy_h, GlobalAMy_d, sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(&GlobalAMz_h, GlobalAMz_d, sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Rho_h, Rho_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Wh_h, Wh_d, point_num * nvi * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(pressure_h, pressure_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Mh_h, Mh_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Rd_h, Rd_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Cp_h, Cp_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(
src/ESP/esp_output.cu:        profx_Qheat_h, profx_Qheat_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Tsurface_h, Tsurface_d, point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Rho_mean_h, Rho_mean_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Wh_mean_h, Wh_mean_d, point_num * nvi * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(
src/ESP/esp_output.cu:        pressure_mean_h, pressure_mean_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Mh_mean_h, Mh_mean_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(
src/ESP/esp_output.cu:        Mh_start_dt_h, Mh_start_dt_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Mh_profx_h, Mh_profx_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(
src/ESP/esp_output.cu:        Rho_start_dt_h, Rho_start_dt_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(Rho_profx_h, Rho_profx_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffmh_h, diffmh_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffw_h, diffw_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffrh_h, diffrh_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffpr_h, diffpr_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(DivM_h, DivM_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffmv_h, diffmv_d, 3 * point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffwv_h, diffwv_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffrv_h, diffrv_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/ESP/esp_output.cu:    cudaMemcpy(diffprv_h, diffprv_d, point_num * nv * sizeof(double), cudaMemcpyDeviceToHost);
src/esp.cu:// Known limitations: - Runs in a single GPU.
src/esp.cu:#include "cuda_device_memory.h"
src/esp.cu:    cuda_device_memory_manager::get_instance().deallocate();
src/esp.cu:void get_cuda_mem_usage(size_t& total_bytes, size_t& free_bytes) {
src/esp.cu:    // show memory usage of GPU
src/esp.cu:    cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
src/esp.cu:    if (cudaSuccess != cuda_status) {
src/esp.cu:        log::printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
src/esp.cu:    argparser.add_arg("g", "gpu_id", 0, "GPU_ID to run on");
src/esp.cu:    int GPU_ID_N = 0;
src/esp.cu:    config_reader.append_config_var("GPU_ID_N", GPU_ID_N, GPU_ID_N_default);
src/esp.cu:    int GPU_ID_N_arg;
src/esp.cu:    if (argparser.get_arg("gpu_id", GPU_ID_N_arg))
src/esp.cu:        GPU_ID_N = GPU_ID_N_arg;
src/esp.cu:    config_OK &= check_greater("GPU_ID_N", GPU_ID_N, -1);
src/esp.cu:                    cuda_device_memory_manager::get_instance().deallocate();
src/esp.cu:                cuda_device_memory_manager::get_instance().deallocate();
src/esp.cu:            cuda_device_memory_manager::get_instance().deallocate();
src/esp.cu:    //  Set the GPU device.
src/esp.cu:    cudaError_t error;
src/esp.cu:    log::printf(" Using GPU #%d\n", GPU_ID_N);
src/esp.cu:    cudaSetDevice(GPU_ID_N);
src/esp.cu:    cudaError_t err = cudaGetDeviceCount(&ndevices);
src/esp.cu:    if (err != cudaSuccess) {
src/esp.cu:        log::printf("%s\n", cudaGetErrorString(err));
src/esp.cu:        log::printf("\n CUDA Device #%d\n", i);
src/esp.cu:        cudaDeviceProp devPp;
src/esp.cu:        cudaGetDeviceProperties(&devPp, i);
src/esp.cu:        if (i == GPU_ID_N)
src/esp.cu:    if (ndevices < 1 || err != cudaSuccess) {
src/esp.cu:    if (GPU_ID_N >= ndevices) {
src/esp.cu:        log::printf("Asked for device #%d but only found %d devices.\n", GPU_ID_N, ndevices);
src/esp.cu:                    GPU_ID_N);
src/esp.cu:    // takes care of freeing cuda memory in case we catch an error and bail out with exit(EXIT_FAILURE)
src/esp.cu:            get_cuda_mem_usage(total_bytes, free_bytes);
src/esp.cu:            double pressure_min = gpu_min_on_device<1024>(X.pressure_d, X.point_num * X.nv);
src/esp.cu:            double pressure_min = gpu_min_on_device<1024>(X.pressure_d, X.point_num * X.nv);
src/esp.cu:    cudaDeviceSynchronize();
src/esp.cu:    error = cudaGetLastError();
src/esp.cu:    log::printf("CudaMalloc error = %d = %s\n\n", error, cudaGetErrorString(error));
src/esp.cu:    cuda_device_memory_manager::get_instance().deallocate();
src/physics/managers/empty/Makefile:cuda_flags ?= unset
src/physics/managers/empty/Makefile:$(info cuda_flags: $(cuda_flags))
src/physics/managers/empty/Makefile:	$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) $(INCLUDE_DIRS) $(CDB) -o $@ $<; \
src/physics/managers/multi/Makefile:cuda_flags ?= unset
src/physics/managers/multi/Makefile:$(info cuda_flags: $(cuda_flags))
src/physics/managers/multi/Makefile:	$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) -I$(includedir) $(INCLUDE_DIRS) $(CDB)  -o $@ $<
src/physics/managers/multi/Makefile:	$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) -I$(includedir)  $(INCLUDE_DIRS) $(CDB)  -o $@ $<
src/physics/managers/multi/Makefile:	$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) -I$(includedir) $(INCLUDE_DIRS) $(CDB)  -o $@ $<
src/physics/managers/multi/Makefile:	$(CC) $(CC_comp_flag) $(arch)  $(cuda_flags) $(h5include) -I$(includedir) $(INCLUDE_DIRS) $(CDB) $(ALFRODULL_FLAGS) -o $@ $<
src/physics/modules/inc/radiative_transfer.h:// Known limitations: - Runs in a single GPU.
src/physics/modules/inc/chemistry.h:// Known limitations: - Runs in a single GPU.
src/physics/modules/inc/boundary_layer.h:// Known limitations: - Runs in a single GPU.
src/physics/modules/src/chemistry.cu:// Known limitations: - Runs in a single GPU.
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&tracer_d, esp.nv * esp.point_num * ntr * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&tracers_d, esp.nv * esp.point_num * ntr * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&tracerk_d, esp.nv * esp.point_num * ntr * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&coeq_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&co2eq_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&ch4eq_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&h2oeq_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&nh3eq_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&tauco_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&tauco2_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&tauch4_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&tauh2o_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&taunh3_d, num_trace_pts * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&P_che_d, num_a * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&T_che_d, num_b * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaMalloc((void **)&difftr_d, esp.nv * esp.point_num * ntr * sizeof(double));
src/physics/modules/src/chemistry.cu:    cudaFree(ch4eq_d);
src/physics/modules/src/chemistry.cu:    cudaFree(coeq_d);
src/physics/modules/src/chemistry.cu:    cudaFree(h2oeq_d);
src/physics/modules/src/chemistry.cu:    cudaFree(co2eq_d);
src/physics/modules/src/chemistry.cu:    cudaFree(nh3eq_d);
src/physics/modules/src/chemistry.cu:    cudaFree(tauch4_d);
src/physics/modules/src/chemistry.cu:    cudaFree(tauco_d);
src/physics/modules/src/chemistry.cu:    cudaFree(tauh2o_d);
src/physics/modules/src/chemistry.cu:    cudaFree(tauco2_d);
src/physics/modules/src/chemistry.cu:    cudaFree(taunh3_d);
src/physics/modules/src/chemistry.cu:    cudaFree(tracer_d);
src/physics/modules/src/chemistry.cu:    cudaFree(tracers_d);
src/physics/modules/src/chemistry.cu:    cudaFree(tracerk_d);
src/physics/modules/src/chemistry.cu:    cudaFree(P_che_d);
src/physics/modules/src/chemistry.cu:    cudaFree(T_che_d);
src/physics/modules/src/chemistry.cu:    cudaFree(difftr_d);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(coeq_d, coeq_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(ch4eq_d, ch4eq_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(h2oeq_d, h2oeq_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(co2eq_d, co2eq_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(nh3eq_d, nh3eq_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(tauco_d, tauco_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(tauch4_d, tauch4_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(tauh2o_d, tauh2o_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(tauco2_d, tauco2_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(taunh3_d, taunh3_h, 7425 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(P_che_d, P_che_h, 135 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(T_che_d, T_che_h, 55 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(
src/physics/modules/src/chemistry.cu:        tracer_d, tracer_h, esp.point_num * esp.nv * ntr * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemset(tracers_d, 0, sizeof(double) * esp.nv * esp.point_num * ntr);
src/physics/modules/src/chemistry.cu:    cudaMemset(tracerk_d, 0, sizeof(double) * esp.nv * esp.point_num * ntr);
src/physics/modules/src/chemistry.cu:    cudaMemcpy(tracerk_d,
src/physics/modules/src/chemistry.cu:               cudaMemcpyDeviceToDevice);
src/physics/modules/src/chemistry.cu:    cudaMemset(tracers_d, 0, sizeof(double) * esp.point_num * esp.nv * ntr);
src/physics/modules/src/chemistry.cu:        cudaMemset(esp.diff_d, 0, sizeof(double) * 6 * esp.point_num * esp.nv);
src/physics/modules/src/chemistry.cu:        cudaDeviceSynchronize();
src/physics/modules/src/chemistry.cu:        cudaDeviceSynchronize();
src/physics/modules/src/chemistry.cu:    cudaDeviceSynchronize();
src/physics/modules/src/chemistry.cu:    cudaMemcpy(tracer_d,
src/physics/modules/src/chemistry.cu:               cudaMemcpyDeviceToDevice);
src/physics/modules/src/chemistry.cu:    cudaDeviceSynchronize();
src/physics/modules/src/chemistry.cu:    cudaDeviceSynchronize();
src/physics/modules/src/chemistry.cu:    cudaMemcpy(
src/physics/modules/src/chemistry.cu:        tracer_h, tracer_d, esp.point_num * esp.nv * ntr * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:// Known limitations: - Runs in a single GPU.
src/physics/modules/src/radiative_transfer.cu:    cudaMalloc((void **)&ASR_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:    cudaMalloc((void **)&OLR_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&phtemp, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&dtemp, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&qheat_d, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&insol_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&surf_flux_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&k_IR_2_nv_d, 2 * esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&k_V_3_nv_d, 3 * esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&gam_V_3_d, 3 * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&gam_1_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&gam_2_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Beta_V_3_d, 3 * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Beta_2_d, 2 * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&net_F_nvi_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&AB_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&OpaTableTemperature_d, 1060 * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&OpaTablePressure_d, 1060 * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&OpaTableKappa_d, 1060 * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&tau_Ve__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&tau_IRe__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Te__df_e,
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&be__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&sw_down__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&sw_down_b__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&sw_up__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&lw_down__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&lw_down_b__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&lw_up__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&lw_up_b__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&lw_net__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&sw_net__df_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&dtau__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&del__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&edel__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&e0i__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&e1i__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Bm__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Am__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&lw_up_g__dff_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&lw_down_g__dff_e, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Gp__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Bp__dff_l, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&Te__df_e,
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&flw_up_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&flw_dn_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&fsw_up_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&fsw_dn_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&tau_d, esp.nv * esp.point_num * 2 * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&phtemp, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&thtemp, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&ttemp, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&dtemp, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&qheat_d, esp.nv * esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&insol_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&insol_ann_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        cudaMalloc((void **)&surf_flux_d, esp.point_num * sizeof(double));
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(phtemp);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(dtemp);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(qheat_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(insol_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(surf_flux_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(ASR_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(OLR_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(k_IR_2_nv_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(k_V_3_nv_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(gam_V_3_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(gam_1_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(gam_2_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Beta_V_3_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Beta_2_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(net_F_nvi_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(AB_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(OpaTableTemperature_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(OpaTablePressure_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(OpaTableKappa_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(tau_Ve__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(tau_IRe__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Te__df_e); // as well used for dry convective adjustment
src/physics/modules/src/radiative_transfer.cu:        cudaFree(be__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(sw_down__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(sw_down_b__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(sw_up__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(lw_down__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(lw_down_b__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(lw_up__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(lw_up_b__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(lw_net__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(sw_net__df_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(dtau__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(del__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(edel__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(e0i__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(e1i__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Bm__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Am__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(lw_up_g__dff_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(lw_down_g__dff_e);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Gp__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Bp__dff_l);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(Te__df_e); // as well used for dry convective adjustment
src/physics/modules/src/radiative_transfer.cu:        cudaFree(flw_up_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(flw_dn_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(fsw_up_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(fsw_dn_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(tau_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(phtemp);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(thtemp);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(ttemp);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(dtemp);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(qheat_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(insol_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(insol_ann_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(surf_flux_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(ASR_d);
src/physics/modules/src/radiative_transfer.cu:        cudaFree(OLR_d);
src/physics/modules/src/radiative_transfer.cu:    //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMemset(surf_flux_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/radiative_transfer.cu:    //         cudaMemset(surf_flux_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/radiative_transfer.cu:    //     cudaMemcpy(Tsurface_d, Tsurface_h, esp.point_num * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            bool cudaStatus;
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:                gam_V_3_d, gam_V__h, 3 * esp.point_num * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "gam_V_3_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:                Beta_V_3_d, Beta_V__h, 3 * esp.point_num * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "Beta_V_3_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:                Beta_2_d, Beta__h, 2 * esp.point_num * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "Beta_2_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:                gam_1_d, gam_1__h, esp.point_num * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "gam_1_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:                gam_2_d, gam_2__h, esp.point_num * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "gam_2_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus =
src/physics/modules/src/radiative_transfer.cu:                cudaMemcpy(AB_d, AB__h, esp.point_num * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "AB_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(OpaTableTemperature_d,
src/physics/modules/src/radiative_transfer.cu:                                    cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "OpaTableTemperature_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(OpaTablePressure_d,
src/physics/modules/src/radiative_transfer.cu:                                    cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "OpaTablePressure_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaStatus = cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:                OpaTableKappa_d, OpaTableKappa__h, 1060 * sizeof(double), cudaMemcpyHostToDevice);
src/physics/modules/src/radiative_transfer.cu:            if (cudaStatus != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                fprintf(stderr, "OpaTableKappa_d cudaMemcpyHostToDevice failed!");
src/physics/modules/src/radiative_transfer.cu:            cudaError_t error = cudaGetLastError();
src/physics/modules/src/radiative_transfer.cu:            if (error != cudaSuccess) {
src/physics/modules/src/radiative_transfer.cu:                // print the CUDA error message and exit
src/physics/modules/src/radiative_transfer.cu:                printf("CUDA error: %s\n", cudaGetErrorString(error));
src/physics/modules/src/radiative_transfer.cu:        ASR_tot = gpu_sum_on_device<1024>(ASR_d, esp.point_num);
src/physics/modules/src/radiative_transfer.cu:        OLR_tot = gpu_sum_on_device<1024>(OLR_d, esp.point_num);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:    cudaDeviceSynchronize();
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(insol_h, insol_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(lw_net__h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(sw_net__h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(flw_up_h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(fsw_up_h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(flw_dn_h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(fsw_dn_h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        //cudaMemcpy(tau_h, tau_Ve__df_e, esp.nvi * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            qheat_h, qheat_d, esp.nv * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        // cudaMemcpy(Tsurface_h, Tsurface_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(k_IR_2__h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(k_V_3__h,
src/physics/modules/src/radiative_transfer.cu:                   cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(insol_h, insol_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            insol_ann_h, insol_ann_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            flw_up_h, flw_up_d, esp.nvi * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            fsw_up_h, fsw_up_d, esp.nvi * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            flw_dn_h, flw_dn_d, esp.nvi * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            fsw_dn_h, fsw_dn_d, esp.nvi * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            tau_h, tau_d, esp.nv * esp.point_num * 2 * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        cudaMemcpy(
src/physics/modules/src/radiative_transfer.cu:            qheat_h, qheat_d, esp.nv * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        // cudaMemcpy(Tsurface_h, Tsurface_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:        //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/radiative_transfer.cu:    //cuda_check_status_or_exit(__FILE__, __LINE__);
src/physics/modules/src/boundary_layer.cu:// Known limitations: - Runs in a single GPU.
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&cpr_tmp, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&dpr_tmp, 3 * esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&KM_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&KH_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&CM_d, esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&CH_d, esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&vh_lowest_d, esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&pt_surf_d, esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&F_sens_d, esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&RiGrad_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&bl_top_lev_d, esp.point_num * sizeof(int));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&bl_top_height_d, esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&Rho_int_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMalloc((void **)&p_int_d, esp.nvi * esp.point_num * sizeof(double));
src/physics/modules/src/boundary_layer.cu:    cudaMemset(cpr_tmp, 0, sizeof(double) * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(dpr_tmp, 0, sizeof(double) * 3 * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(F_sens_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(KM_d, 0, sizeof(double) * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(KH_d, 0, sizeof(double) * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(bl_top_lev_d, 0, sizeof(int) * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(bl_top_height_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(CM_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(CH_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(vh_lowest_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(pt_surf_d, 0, sizeof(double) * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(Rho_int_d, 0, sizeof(double) * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(p_int_d, 0, sizeof(double) * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:    cudaMemset(RiGrad_d, 0, sizeof(double) * esp.nvi * esp.point_num);
src/physics/modules/src/boundary_layer.cu:    cudaFree(cpr_tmp);
src/physics/modules/src/boundary_layer.cu:    cudaFree(dpr_tmp);
src/physics/modules/src/boundary_layer.cu:    cudaFree(KM_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(KH_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(CM_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(CH_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(vh_lowest_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(pt_surf_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(bl_top_lev_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(F_sens_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(bl_top_height_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(Rho_int_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(p_int_d);
src/physics/modules/src/boundary_layer.cu:    cudaFree(RiGrad_d);
src/physics/modules/src/boundary_layer.cu:        cudaDeviceSynchronize();
src/physics/modules/src/boundary_layer.cu:        cudaDeviceSynchronize();
src/physics/modules/src/boundary_layer.cu:        cudaMemset(cpr_tmp, 0, sizeof(double) * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:        cudaMemset(dpr_tmp, 0, sizeof(double) * 3 * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:        cudaDeviceSynchronize();
src/physics/modules/src/boundary_layer.cu:        cudaMemset(cpr_tmp, 0, sizeof(double) * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:        cudaMemset(dpr_tmp, 0, sizeof(double) * 3 * esp.point_num * esp.nvi);
src/physics/modules/src/boundary_layer.cu:        cudaDeviceSynchronize();
src/physics/modules/src/boundary_layer.cu:        cudaMemcpy(
src/physics/modules/src/boundary_layer.cu:            RiGrad_h, RiGrad_d, esp.nvi * esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/boundary_layer.cu:        cudaMemcpy(CM_h, CM_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/boundary_layer.cu:        cudaMemcpy(CH_h, CH_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/boundary_layer.cu:        cudaMemcpy(KM_h, KM_d, esp.point_num * esp.nvi * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/boundary_layer.cu:        cudaMemcpy(KH_h, KH_d, esp.point_num * esp.nvi * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/boundary_layer.cu:        cudaMemcpy(F_sens_h, F_sens_d, esp.point_num * sizeof(double), cudaMemcpyDeviceToHost);
src/physics/modules/src/boundary_layer.cu:        cudaMemcpy(bl_top_lev_d, bl_top_lev_h, esp.point_num * sizeof(int), cudaMemcpyHostToDevice);
src/utils/directories.cpp:// Known limitations: - Runs in a single GPU.
src/utils/directories.cpp:// #include "cuda_device_memory.h"
src/utils/directories.cpp:            // cuda_device_memory_manager::get_instance().deallocate();
src/utils/cuda_device_memory.cu:#include "cuda_device_memory.h"
src/utils/cuda_device_memory.cu:cuda_device_memory_manager& cuda_device_memory_manager::get_instance() {
src/utils/cuda_device_memory.cu:  static cuda_device_memory_manager cdmm;
src/utils/cuda_device_memory.cu:cuda_device_memory_manager::cuda_device_memory_manager() {
src/utils/cuda_device_memory.cu:cuda_device_memory_manager::~cuda_device_memory_manager() {
src/utils/cuda_device_memory.cu:void cuda_device_memory_manager::register_mem(cuda_device_memory_interface * cdm)
src/utils/cuda_device_memory.cu:void cuda_device_memory_manager::unregister_mem(cuda_device_memory_interface * cdm)
src/utils/cuda_device_memory.cu:  vector<cuda_device_memory_interface*>::iterator position = std::find(device_memory.begin(), device_memory.end(), cdm);
src/utils/cuda_device_memory.cu:void cuda_device_memory_manager::deallocate()
src/utils/binary_test.cpp:#    ifdef BENCH_CHECK_LAST_CUDA_ERROR
src/utils/binary_test.cpp:    check_last_cuda_error(ref_name);
src/utils/binary_test.cpp:#    endif // BENCH_CHECK_LAST_CUDA_ERROR
src/utils/binary_test.cpp:            check_last_cuda_error(string("output ") + ref_name + string(" ") + def.name);
src/utils/debug_helpers.cu:    cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
src/utils/debug_helpers.cu:    cudaMalloc((void **)&ptr, sizeof(bool));
src/utils/debug_helpers.cu:    cudaFree(ptr);
src/utils/debug_helpers.cu:        cudaMemcpy(check_d, &check_h, sizeof(bool), cudaMemcpyHostToDevice);
src/utils/debug_helpers.cu:        cudaMemcpy(&check_h, check_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/utils/debug_helpers.cu:        cudaMemcpy(ptr_h, def.data, def.size * sizeof(double), cudaMemcpyDeviceToHost);
src/utils/debug_helpers.cu:void check_last_cuda_error(std::string ref_name) {
src/utils/debug_helpers.cu:    cudaError_t err = cudaGetLastError();
src/utils/debug_helpers.cu:    if (err != cudaSuccess) {
src/utils/debug_helpers.cu:        log::printf("'%s' cuda error: %s\n", ref_name.c_str(), cudaGetErrorString(err));
src/utils/debug_helpers.cu:void cuda_check_status_or_exit() {
src/utils/debug_helpers.cu:    cudaError_t err = cudaGetLastError();
src/utils/debug_helpers.cu:    if (err != cudaSuccess) {
src/utils/debug_helpers.cu:        log::printf("[%s:%d] CUDA error check reports error: %s\n",
src/utils/debug_helpers.cu:                    cudaGetErrorString(err));
src/utils/debug_helpers.cu:void cuda_check_status_or_exit(const char *filename, const int &line) {
src/utils/debug_helpers.cu:    cudaError_t err = cudaGetLastError();
src/utils/debug_helpers.cu:    if (err != cudaSuccess) {
src/utils/debug_helpers.cu:        log::printf("[%s:%d] CUDA error check reports error: %s\n",
src/utils/debug_helpers.cu:                    cudaGetErrorString(err));
src/headers/debug.h:// #define BENCH_CHECK_LAST_CUDA_ERROR
src/headers/reduction_add.h:// GPU reduction sum kernel
src/headers/reduction_add.h:template<int BLOCK_SIZE> __global__ void gpu_reduction_sum(double *d, double *o, long length) {
src/headers/reduction_add.h:template<int BLOCK_SIZE> __host__ double gpu_sum_on_device(double *in_d, long length) {
src/headers/reduction_add.h:    cudaMalloc((void **)&out_d, num_blocks * sizeof(double));
src/headers/reduction_add.h:    cudaError_t err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("Malloc: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    gpu_reduction_sum<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(in_d, out_d, length);
src/headers/reduction_add.h:    err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("krnl: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    cudaMemcpy(out_h, out_d, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
src/headers/reduction_add.h:    err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("cpyD2H: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    cudaFree(out_d);
src/headers/reduction_add.h:template<int BLOCK_SIZE> __host__ double gpu_sum_from_host(double *d, long length) {
src/headers/reduction_add.h:    cudaMalloc((void **)&in_d, length * sizeof(double));
src/headers/reduction_add.h:    cudaError_t err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("Malloc: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    cudaMemcpy(in_d, d, length * sizeof(double), cudaMemcpyHostToDevice);
src/headers/reduction_add.h:    err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("cpyH2D: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    double out = gpu_sum_on_device<BLOCK_SIZE>(in_d, length);
src/headers/reduction_add.h:    cudaFree(in_d);
src/headers/reduction_add.h:__global__ void gpu_reduction_sum_sponge(double *d, double *o, long length) {
src/headers/reduction_add.h:__host__ void gpu_sum_on_device_sponge(double *in_d,
src/headers/reduction_add.h:    cudaMalloc((void **)&out_d, num_blocks * nv * nlat_bins * sizeof(double));
src/headers/reduction_add.h:    cudaError_t err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("Malloc: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    gpu_reduction_sum_sponge<BLOCK_SIZE><<<NB, BLOCK_SIZE>>>(in_d, out_d, length);
src/headers/reduction_add.h:    err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("krnl: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    cudaMemcpy(out_h, out_d, num_blocks * nv * nlat_bins * sizeof(double), cudaMemcpyDeviceToHost);
src/headers/reduction_add.h:    err = cudaGetLastError();
src/headers/reduction_add.h:    if (err != cudaSuccess)
src/headers/reduction_add.h:        log::printf("cpyD2H: %s\n", cudaGetErrorString(err));
src/headers/reduction_add.h:    cudaDeviceSynchronize();
src/headers/reduction_add.h:    cudaFree(out_d);
src/headers/define.h:// GPU ID
src/headers/define.h:#define GPU_ID_N_default 0 // Set GPU ID number
src/headers/debug_helpers.h:// helper to copy data from device to host (from any place, without cuda dependencies)
src/headers/debug_helpers.h:void check_last_cuda_error(string ref_name);
src/headers/debug_helpers.h:void cuda_check_status_or_exit();
src/headers/debug_helpers.h:void cuda_check_status_or_exit(const char *filename, const int &line);
src/headers/cmdargs.h:// int: e.g. num. iteration, GPU_ID
src/headers/insolation.h:#include "cuda_device_memory.h"
src/headers/insolation.h:    cuda_device_memory<double> cos_zenith_daily; //daily averaged zenith angles
src/headers/insolation.h:    cuda_device_memory<double> day_start_time; //start time of each day (relative to start of orbit)
src/headers/insolation.h:    cuda_device_memory<double> cos_zenith_angles;
src/headers/dyn/phy_modules_device.h:// Known limitations: - Runs in a single GPU.
src/headers/directories.h:// Known limitations: - Runs in a single GPU.
src/headers/reduction_min.h:// GPU reduction sum kernel
src/headers/reduction_min.h:template<int BLOCK_SIZE> __global__ void gpu_reduction_min(double *d, double *o, long length) {
src/headers/reduction_min.h:template<int BLOCK_SIZE> __host__ double gpu_min_on_device(double *in_d, long length) {
src/headers/reduction_min.h:    cudaMalloc((void **)&out_d, num_blocks * sizeof(double));
src/headers/reduction_min.h:    cudaError_t err = cudaGetLastError();
src/headers/reduction_min.h:    if (err != cudaSuccess)
src/headers/reduction_min.h:        log::printf("Malloc: %s\n", cudaGetErrorString(err));
src/headers/reduction_min.h:    gpu_reduction_min<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(in_d, out_d, length);
src/headers/reduction_min.h:    err = cudaGetLastError();
src/headers/reduction_min.h:    if (err != cudaSuccess)
src/headers/reduction_min.h:        log::printf("krnl: %s\n", cudaGetErrorString(err));
src/headers/reduction_min.h:    cudaMemcpy(out_h, out_d, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
src/headers/reduction_min.h:    err = cudaGetLastError();
src/headers/reduction_min.h:    if (err != cudaSuccess)
src/headers/reduction_min.h:        log::printf("cpyD2H: %s\n", cudaGetErrorString(err));
src/headers/reduction_min.h:    cudaFree(out_d);
src/headers/diagnostics.h:#include "cuda_device_memory.h"
src/headers/diagnostics.h:    cuda_device_memory<diag_data> diagnostics;
src/headers/diagnostics.h:    cuda_device_memory<unsigned int> diagnostics_global_flag;
src/headers/cuda_device_memory.h:template<class T> std::shared_ptr<T[]> get_cuda_data(T* device_ptr, size_t size) {
src/headers/cuda_device_memory.h:    cudaError_t ret =
src/headers/cuda_device_memory.h:        cudaMemcpy(host_mem.get(), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
src/headers/cuda_device_memory.h:class cuda_device_memory_interface
src/headers/cuda_device_memory.h:// necessary to free device memory before the whole cuda environment is tear down.
src/headers/cuda_device_memory.h:class cuda_device_memory_manager
src/headers/cuda_device_memory.h:    static cuda_device_memory_manager& get_instance();
src/headers/cuda_device_memory.h:    ~cuda_device_memory_manager();
src/headers/cuda_device_memory.h:    cuda_device_memory_manager(cuda_device_memory_manager const&) = delete;
src/headers/cuda_device_memory.h:    void operator=(cuda_device_memory_manager const&) = delete;
src/headers/cuda_device_memory.h:    void register_mem(cuda_device_memory_interface* cdm);
src/headers/cuda_device_memory.h:    void unregister_mem(cuda_device_memory_interface* cdm);
src/headers/cuda_device_memory.h:    cuda_device_memory_manager();
src/headers/cuda_device_memory.h:    vector<cuda_device_memory_interface*> device_memory;
src/headers/cuda_device_memory.h:template<typename T> class cuda_device_memory : cuda_device_memory_interface
src/headers/cuda_device_memory.h:    cuda_device_memory() {
src/headers/cuda_device_memory.h:    cuda_device_memory(size_t size) {
src/headers/cuda_device_memory.h:    cuda_device_memory(size_t size, bool host_mem) {
src/headers/cuda_device_memory.h:    ~cuda_device_memory() {
src/headers/cuda_device_memory.h:            cudaError_t ret = cudaFree(device_ptr);
src/headers/cuda_device_memory.h:            if (ret != cudaSuccess)
src/headers/cuda_device_memory.h:                printf("CudaDeviceMemory: device free error\n");
src/headers/cuda_device_memory.h:        cudaError_t ret = cudaMalloc((void**)&device_ptr, size_in * sizeof(T));
src/headers/cuda_device_memory.h:        if (ret == cudaSuccess) {
src/headers/cuda_device_memory.h:        return ret == cudaSuccess;
src/headers/cuda_device_memory.h:            cudaError_t ret =
src/headers/cuda_device_memory.h:                cudaMemcpy(device_ptr, &(host_ptr[0]), size * sizeof(T), cudaMemcpyHostToDevice);
src/headers/cuda_device_memory.h:            return ret == cudaSuccess;
src/headers/cuda_device_memory.h:            cudaError_t ret = cudaMemset(device_ptr, 0, sizeof(T) * size);
src/headers/cuda_device_memory.h:            return ret == cudaSuccess;
src/headers/cuda_device_memory.h:            cudaError_t ret =
src/headers/cuda_device_memory.h:                cudaMemcpy(host_ptr.get(), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
src/headers/cuda_device_memory.h:            return ret == cudaSuccess;
src/headers/cuda_device_memory.h:        cudaError_t ret =
src/headers/cuda_device_memory.h:            cudaMemcpy(&(data_ptr[0]), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
src/headers/cuda_device_memory.h:        return ret == cudaSuccess;
src/headers/cuda_device_memory.h:        cudaError_t ret =
src/headers/cuda_device_memory.h:            cudaMemcpy(device_ptr, &(data_ptr[0]), size * sizeof(T), cudaMemcpyHostToDevice);
src/headers/cuda_device_memory.h:        return ret == cudaSuccess;
src/headers/cuda_device_memory.h:        cudaError_t ret =
src/headers/cuda_device_memory.h:            cudaMemcpy(device_ptr, data_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
src/headers/cuda_device_memory.h:        return ret == cudaSuccess;
src/headers/cuda_device_memory.h:        cuda_device_memory_manager& cdmm = cuda_device_memory_manager::get_instance();
src/headers/cuda_device_memory.h:        cuda_device_memory_manager& cdmm = cuda_device_memory_manager::get_instance();
src/headers/phy_modules.h:// Known limitations: - Runs in a single GPU.

```
