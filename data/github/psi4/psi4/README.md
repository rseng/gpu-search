# https://github.com/psi4/psi4

```console
devtools/scripts/ci_print_failing.py:                'dkh', 'gcp', 'gdma', 'simint', 'snsmp2', 'v2rdm_casscf', 'gpu_dfcc'
cdash/CDashTSan.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_TSAN=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
cdash/CDashUBSan.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_UBSAN=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
cdash/CDashValgrind.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_MEMCHECK=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
cdash/CDashMSan.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_MSAN=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
cdash/CDashTSanGCC.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_TSAN=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
cdash/CDashUBSanGCC.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_UBSAN=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
cdash/CDashASanGCC.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_ASAN=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
cdash/CDashASan.cmake:    set(CTEST_CONFIGURE_COMMAND "cmake  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DENABLE_MPI=OFF -DENABLE_SGI_MPT=OFF -DENABLE_OMP=ON -DENABLE_VECTORIZATION=OFF -DENABLE_CSR=OFF -DENABLE_SCALAPACK=OFF -DENABLE_SCALASCA=OFF -DENABLE_UNIT_TESTS=OFF -DENABLE_STATIC_LINKING=OFF -DENABLE_PLUGINS=ON -DENABLE_LIBERD=OFF -DENABLE_GPU_DFCC=OFF -DENABLE_DUMMY_PLUGIN=OFF -DENABLE_CXX11_SUPPORT=ON -DLIBINT_OPT_AM=5 -DENABLE_ASAN=ON -DCMAKE_INSTALL_PREFIX=/usr/local/psi4 -DCMAKE_BUILD_TYPE=debug -DPYTHON_EXECUTABLE=/global/apps/python/2.7.3/bin/python ${CTEST_SOURCE_DIRECTORY}")
external/upstream/gauxc/CMakeLists.txt:        #== check if GPU-enabled GauXC is found, if GPU is specified ==#
external/upstream/gauxc/CMakeLists.txt:        get_property(_gpu TARGET gauxc::gauxc PROPERTY GAUXC_HAS_DEVICE)
external/upstream/gauxc/CMakeLists.txt:        if(${gauxc_ENABLE_GPU} AND (NOT ${_gpu}))
external/upstream/gauxc/CMakeLists.txt:           message(FATAL_ERROR "gauxc_ENABLE_GPU turned on, but selected GauXC install is not built for GPU support!")
external/upstream/gauxc/CMakeLists.txt:        elseif((NOT ${gauxc_ENABLE_GPU}) AND ${_gpu})
external/upstream/gauxc/CMakeLists.txt:           message(WARNING "gauxc_ENABLE_GPU turned off, but selected GauXC install is built for GPU support! If you want GPU support for the Psi4/GauXC interface, make sure to explicitly turn on gauxc_ENABLE_GPU!")
external/upstream/gauxc/CMakeLists.txt:        if(${gauxc_ENABLE_MAGMA} AND (NOT ${GAUXC_ENABLE_GPU}))
external/upstream/gauxc/CMakeLists.txt:           message(FATAL_ERROR "gauxc_ENABLE_MAGMA turned on, but gauxc_ENABLE_GPU is turned off! Magma only works with device-enabled GauXC instances.")
external/upstream/gauxc/CMakeLists.txt:        if(${gauxc_ENABLE_GPU} AND (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES}))
external/upstream/gauxc/CMakeLists.txt:           message(FATAL_ERROR "Internal GPU-enabled GauXC construction specified, but CMAKE_CUDA_ARCHITECTURES is not defined!")
external/upstream/gauxc/CMakeLists.txt:                   -DGAUXC_ENABLE_CUDA=${gauxc_ENABLE_GPU}
external/upstream/gauxc/CMakeLists.txt:                   -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
external/downstream/gpu_dfcc/CMakeLists.txt:if(${ENABLE_gpu_dfcc})
external/downstream/gpu_dfcc/CMakeLists.txt:    find_package(gpu_dfcc 0.3 CONFIG QUIET)
external/downstream/gpu_dfcc/CMakeLists.txt:    if(${gpu_dfcc_FOUND})
external/downstream/gpu_dfcc/CMakeLists.txt:        get_property(_loc TARGET gpu_dfcc::gpu_dfcc PROPERTY LOCATION)
external/downstream/gpu_dfcc/CMakeLists.txt:        message(STATUS "${Cyan}Found gpu_dfcc${ColourReset}: ${_loc} (found version ${gpu_dfcc_VERSION})")
external/downstream/gpu_dfcc/CMakeLists.txt:        add_library(gpu_dfcc_external INTERFACE)  # dummy
external/downstream/gpu_dfcc/CMakeLists.txt:        if(${CMAKE_INSIST_FIND_PACKAGE_gpu_dfcc})
external/downstream/gpu_dfcc/CMakeLists.txt:            message(FATAL_ERROR "Suitable gpu_dfcc could not be externally located as user insists")
external/downstream/gpu_dfcc/CMakeLists.txt:        message(STATUS "Suitable gpu_dfcc could not be located, ${Magenta}Building gpu_dfcc${ColourReset} instead.")
external/downstream/gpu_dfcc/CMakeLists.txt:        ExternalProject_Add(gpu_dfcc_external
external/downstream/gpu_dfcc/CMakeLists.txt:            URL https://github.com/edeprince3/gpu_dfcc/archive/b98c6f1.tar.gz  # v0.3 + 1 (install fix)
external/downstream/gpu_dfcc/CMakeLists.txt:                       -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
external/downstream/gpu_dfcc/CMakeLists.txt:                       -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
external/downstream/gpu_dfcc/CMakeLists.txt:                             -DCMAKE_CUDA_FLAGS:STRING=${CMAKE_CUDA_FLAGS}
external/downstream/gpu_dfcc/CMakeLists.txt:        set(gpu_dfcc_DIR ${STAGED_INSTALL_PREFIX}/share/cmake/gpu_dfcc CACHE PATH "path to internally built gpu_dfccConfig.cmake" FORCE)
external/downstream/gpu_dfcc/CMakeLists.txt:    add_library(gpu_dfcc_external INTERFACE)  # dummy
external/downstream/CMakeLists.txt:            gpu_dfcc
psi4/extras.py:    "gpu_dfcc": which_import("gpu_dfcc", return_bool=True),
psi4/driver/endorsed_plugins.py:    import gpu_dfcc
psi4/src/psi4/libfock/SplitJK.h:    // are we running snLinK on GPUs?
psi4/src/psi4/libfock/SplitJK.h:    bool use_gpu_; 
psi4/src/psi4/libfock/snLinK.cc:    use_gpu_ = options_.get_bool("SNLINK_USE_GPU"); 
psi4/src/psi4/libfock/snLinK.cc:    auto ex = use_gpu_ ? GauXC::ExecutionSpace::Device : GauXC::ExecutionSpace::Host;  
psi4/src/psi4/libfock/snLinK.cc:    if (use_gpu_) {
psi4/src/psi4/libfock/snLinK.cc:        rt = std::make_unique<GauXC::DeviceRuntimeEnvironment>( GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.01*options_.get_int("SNLINK_GPU_MEM"));
psi4/src/psi4/libfock/snLinK.cc:    // this is required for GPU execution when using spherical harmonic basis sets
psi4/src/psi4/libfock/snLinK.cc:    if (use_gpu_ && !force_cartesian && primary_->has_puream()) {
psi4/src/psi4/libfock/snLinK.cc:        outfile->Printf("    INFO: GPU snLinK must be executed with SNLINK_FORCE_CARTESIAN=true when using spherical harmonic basis sets!\n");  
psi4/src/psi4/libfock/snLinK.cc:        outfile->Printf("    K Execution Space: %s\n", (use_gpu_) ? "Device (GPU)" : "Host (CPU)");
psi4/src/psi4/fnocc/fnocc.cc:#ifdef GPUCC
psi4/src/psi4/fnocc/fnocc.cc:        auto ccsd = std::make_shared<GPUDFCoupledCluster>(wfn, options);
psi4/src/psi4/fnocc/ccsd.cc:        // TODO: this was a problem with cuda 3.2 vs 4.0
psi4/src/read_options.cc:    options.add_int("NUM_GPUS", 1);
psi4/src/read_options.cc:    /*- Whether to enable using the BrianQC GPU module -*/
psi4/src/read_options.cc:        /*- Use GPU for GauXC? -*/
psi4/src/read_options.cc:        options.add_bool("SNLINK_USE_GPU", false);
psi4/src/read_options.cc:        /*- Proportion (in %) of available GPU memory to allocate to snLinK. !expert-*/
psi4/src/read_options.cc:        options.add_bool("SNLINK_GPU_MEM", 90);
psi4/src/read_options.cc:        GauXC also has NCCL, but it is incompatible with Psi4 due to requiring MPI. -*/
psi4/src/read_options.cc:        due to compile-time issues and requiring very modern CUDA CCs (>=80) -*/
codedeps.yaml:      required_note: "Allows using the BrianQC GPU module."
codedeps.yaml:    # ENABLE_BrianQC "Enables the BrianQC GPU module (requires CUDA; requires separate installation and licensing of the BrianQC module)"
codedeps.yaml:    #  gauxc_ENABLE_GPU 
tests/dfremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/freq-isotope1/forcibly-corrected-molpro.out: Using default tuning parameters: mindgm=1; mindgv=20; mindgc=4; mindgr=1; noblas=0; mincuda=1000; minvec=7
tests/freq-isotope1/molpro.out: Using default tuning parameters: mindgm=1; mindgv=20; mindgc=4; mindgr=1; noblas=0; mincuda=1000; minvec=7
tests/dfccsd-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/dfccsdt2/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/dforemp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/dfremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/pytests/test_addons.py:from addons import hardware_nvidia_gpu, uusing
tests/pytests/test_addons.py:@hardware_nvidia_gpu
tests/pytests/test_addons.py:@uusing("gpu_dfcc")
tests/pytests/test_addons.py:def test_gpu_dfcc():
tests/pytests/test_addons.py:    """gpu_dfcc/tests/gpu_dfcc1"""
tests/pytests/test_addons.py:    #! cc-pvdz (H2O)2 Test DF-CCSD vs GPU-DF-CCSD
tests/pytests/test_addons.py:    import gpu_dfcc
tests/pytests/test_addons.py:      'num_gpus': 1,
tests/pytests/test_addons.py:    en_gpu_dfcc = psi4.energy('gpu-df-ccsd', molecule=H20)
tests/pytests/test_addons.py:    assert psi4.compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD total energy")
tests/pytests/test_addons_qcschema.py:from addons import hardware_nvidia_gpu, uusing, using
tests/pytests/test_addons_qcschema.py:#@hardware_nvidia_gpu
tests/pytests/test_addons_qcschema.py:#@uusing("gpu_dfcc")
tests/pytests/test_addons_qcschema.py:#def test_gpu_dfcc():
tests/pytests/test_addons_qcschema.py:#    """gpu_dfcc/tests/gpu_dfcc1"""
tests/pytests/test_addons_qcschema.py:#    #! cc-pvdz (H2O)2 Test DF-CCSD vs GPU-DF-CCSD
tests/pytests/test_addons_qcschema.py:#    import gpu_dfcc
tests/pytests/test_addons_qcschema.py:#      'num_gpus': 1,
tests/pytests/test_addons_qcschema.py:#    en_gpu_dfcc = psi4.energy('gpu-df-ccsd', molecule=H20)
tests/pytests/test_addons_qcschema.py:#    assert psi4.compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD total energy")
tests/pytests/addons.py:    "hardware_nvidia_gpu",
tests/pytests/addons.py:def is_nvidia_gpu_present():
tests/pytests/addons.py:        import GPUtil
tests/pytests/addons.py:            import gpu_dfcc
tests/pytests/addons.py:            return gpu_dfcc.cudaGetDeviceCount() > 0
tests/pytests/addons.py:            ngpu = len(GPUtil.getGPUs())
tests/pytests/addons.py:            # no `nvidia-smi`
tests/pytests/addons.py:            return ngpu > 0
tests/pytests/addons.py:    "gpu_dfcc": which_import("gpu_dfcc", return_bool=True),
tests/pytests/addons.py:hardware_nvidia_gpu = pytest.mark.skipif(
tests/pytests/addons.py:    True,  #is_nvidia_gpu_present() is False,
tests/pytests/addons.py:    reason='Psi4 not detecting Nvidia GPU via `nvidia-smi`. Install one')
tests/dfccsd-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/dfccsd-t-grad2/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/remp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/gpu_dfcc/gpudfcc1/input.dat:#! aug-cc-pvdz H2O Test DF-CCSD vs GPU-DF-CCSD
tests/gpu_dfcc/gpudfcc1/input.dat:import gpu_dfcc
tests/gpu_dfcc/gpudfcc1/input.dat:en_gpu_dfcc = energy('gpu-df-ccsd')
tests/gpu_dfcc/gpudfcc1/input.dat:compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD total energy") # TEST
tests/gpu_dfcc/gpudfcc1/CMakeLists.txt:add_regression_test(gpu_dfcc-gpudfcc1 "psi;gpu_dfcc;addon")
tests/gpu_dfcc/gpudfcc1/test_input.py:@uusing("gpu_dfcc")
tests/gpu_dfcc/gpudfcc1/test_input.py:def test_gpu_dfcc_gpudfcc1():
tests/gpu_dfcc/gpudfcc1/output.ref:                         Git: Rev {enable_gpu_dfcc} b3dff61 dirty
tests/gpu_dfcc/gpudfcc1/output.ref:    PSIDATADIR: /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4
tests/gpu_dfcc/gpudfcc1/output.ref:#! aug-cc-pvdz H2O Test DF-CCSD vs GPU-DF-CCSD
tests/gpu_dfcc/gpudfcc1/output.ref:import gpu_dfcc
tests/gpu_dfcc/gpudfcc1/output.ref:en_gpu_dfcc = energy('gpu-df-ccsd')
tests/gpu_dfcc/gpudfcc1/output.ref:compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD total energy") # TEST
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   250 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    36 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   204 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    30 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    Method 'GPU-DF-CCSD' requires SCF_TYPE = DISK_DF, setting.
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   250 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    36 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 1   entry O          line   204 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:    atoms 2-3 entry H          line    30 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc1/output.ref:Reading options from the GPU_DFCC block
tests/gpu_dfcc/gpudfcc1/output.ref:Calling plugin gpu_dfcc.so.
tests/gpu_dfcc/gpudfcc1/output.ref:  CUDA device properties:
tests/gpu_dfcc/gpudfcc1/output.ref:  allocating gpu memory...  Total memory requirements:            4.13 mb
tests/gpu_dfcc/gpudfcc2/input.dat:#! aug-cc-pvdz H2O Test DF-CCSD(T) vs GPU-DF-CCSD(T)
tests/gpu_dfcc/gpudfcc2/input.dat:import gpu_dfcc
tests/gpu_dfcc/gpudfcc2/input.dat:en_gpu_dfcc = energy('gpu-df-ccsd(t)')
tests/gpu_dfcc/gpudfcc2/input.dat:compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD(T) total energy") # TEST
tests/gpu_dfcc/gpudfcc2/CMakeLists.txt:add_regression_test(gpu_dfcc-gpudfcc2 "psi;gpu_dfcc;addon")
tests/gpu_dfcc/gpudfcc2/test_input.py:@uusing("gpu_dfcc")
tests/gpu_dfcc/gpudfcc2/test_input.py:def test_gpu_dfcc_gpudfcc2():
tests/gpu_dfcc/gpudfcc2/output.ref:                         Git: Rev {enable_gpu_dfcc} b3dff61 dirty
tests/gpu_dfcc/gpudfcc2/output.ref:    PSIDATADIR: /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4
tests/gpu_dfcc/gpudfcc2/output.ref:#! aug-cc-pvdz H2O Test DF-CCSD(T) vs GPU-DF-CCSD(T)
tests/gpu_dfcc/gpudfcc2/output.ref:import gpu_dfcc
tests/gpu_dfcc/gpudfcc2/output.ref:en_gpu_dfcc = energy('gpu-df-ccsd(t)')
tests/gpu_dfcc/gpudfcc2/output.ref:compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD(T) total energy") # TEST
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   250 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    36 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   204 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    30 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    Method 'GPU-DF-CCSD(T)' requires SCF_TYPE = DISK_DF, setting.
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   250 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    36 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   270 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    70 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-jkfit.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 1   entry O          line   204 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:    atoms 2-3 entry H          line    30 file /edfs/users/deprince/edeprince3/psi4/install/gnu-gpu-dfcc/share/psi4/basis/aug-cc-pvdz-ri.gbs 
tests/gpu_dfcc/gpudfcc2/output.ref:Reading options from the GPU_DFCC block
tests/gpu_dfcc/gpudfcc2/output.ref:Calling plugin gpu_dfcc.so.
tests/gpu_dfcc/gpudfcc2/output.ref:  CUDA device properties:
tests/gpu_dfcc/gpudfcc2/output.ref:  allocating gpu memory...  Total memory requirements:            4.13 mb
tests/gpu_dfcc/CMakeLists.txt:add_subdirectory(gpudfcc1)
tests/gpu_dfcc/CMakeLists.txt:add_subdirectory(gpudfcc2)
tests/cdoremp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy2/output.ref:  NUM_GPUS                      => (empty)          
tests/CMakeLists.txt:# <<<  GPU_DFCC  >>>
tests/CMakeLists.txt:if(ENABLE_gpu_dfcc)
tests/CMakeLists.txt:    add_subdirectory(gpu_dfcc)
tests/CMakeLists.txt:    message(STATUS "Adding test cases: Psi4 + gpu_dfcc")
tests/cbs-xtpl-energy-conv/output.ref:  NUM_GPUS                      => (empty)          
tests/cbs-xtpl-energy-conv/output.ref:  NUM_GPUS                      => (empty)          
tests/cbs-xtpl-energy-conv/output.ref:  NUM_GPUS                      => (empty)          
tests/cbs-xtpl-energy-conv/output.ref:  NUM_GPUS                      => (empty)          
tests/cbs-xtpl-energy-conv/output.ref:  NUM_GPUS                      => (empty)          
tests/cbs-xtpl-energy-conv/output.ref:  NUM_GPUS                      => (empty)          
tests/mp2-h/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/oremp-grad1/output.ref:  NUM_GPUS                      => (empty)          
tests/dft-custom-dhdf/ref_outputs/orca_b2gpplyp.ref:/usr/qc/openmpi2//lib64:/usr/qc/gcc63/lib64:/usr/qc/gcc63/lib:/usr/local/cuda80/lib64/:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/tbb/lib/intel64_lin/gcc4.7:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin:/usr/qc/openmpi.1.6.5/lib64:/usr/qc/openmpi.1.6.5/include/:/lib/:/include://usr/qc/libxc-2.1.2/libxc/lib64:/lib
tests/dft-custom-dhdf/ref_outputs/orca_b2plyp.ref:/usr/qc/openmpi2//lib64:/usr/qc/gcc63/lib64:/usr/qc/gcc63/lib:/usr/local/cuda80/lib64/:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/tbb/lib/intel64_lin/gcc4.7:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin:/usr/qc/openmpi.1.6.5/lib64:/usr/qc/openmpi.1.6.5/include/:/lib/:/include://usr/qc/libxc-2.1.2/libxc/lib64:/lib
tests/dft-custom-dhdf/ref_outputs/orca_dsdblyp.ref:/usr/qc/openmpi2//lib64:/usr/qc/gcc63/lib64:/usr/qc/gcc63/lib:/usr/local/cuda80/lib64/:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/tbb/lib/intel64_lin/gcc4.7:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin:/usr/qc/openmpi.1.6.5/lib64:/usr/qc/openmpi.1.6.5/include/:/lib/:/include://usr/qc/libxc-2.1.2/libxc/lib64:/lib
tests/dft-custom-dhdf/ref_outputs/orca_pwpb95.ref:/usr/qc/openmpi2//lib64:/usr/local/cuda80/lib64/:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/tbb/lib/intel64_lin/gcc4.7:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/usr/qc/mkl2017/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin:/usr/qc/openmpi.1.6.5/lib64:/usr/qc/openmpi.1.6.5/include/:/lib/:/include://usr/qc/libxc-2.1.2/libxc/lib64:/lib
tests/cdremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-2/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdoremp-energy1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/cdremp-1/output.ref:  NUM_GPUS                      => (empty)          
tests/dfccsdat2/output.ref:  NUM_GPUS                      => (empty)          
samples/gpu_dfcc/gpudfcc1/test.in:#! aug-cc-pvdz H2O Test DF-CCSD vs GPU-DF-CCSD
samples/gpu_dfcc/gpudfcc1/test.in:import gpu_dfcc
samples/gpu_dfcc/gpudfcc1/test.in:en_gpu_dfcc = energy('gpu-df-ccsd')
samples/gpu_dfcc/gpudfcc1/test.in:compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD total energy") # TEST
samples/gpu_dfcc/gpudfcc1/input.dat:#! aug-cc-pvdz H2O Test DF-CCSD vs GPU-DF-CCSD
samples/gpu_dfcc/gpudfcc1/input.dat:import gpu_dfcc
samples/gpu_dfcc/gpudfcc1/input.dat:en_gpu_dfcc = energy('gpu-df-ccsd')
samples/gpu_dfcc/gpudfcc2/test.in:#! aug-cc-pvdz H2O Test DF-CCSD(T) vs GPU-DF-CCSD(T)
samples/gpu_dfcc/gpudfcc2/test.in:import gpu_dfcc
samples/gpu_dfcc/gpudfcc2/test.in:en_gpu_dfcc = energy('gpu-df-ccsd(t)')
samples/gpu_dfcc/gpudfcc2/test.in:compare_values(en_gpu_dfcc, en_dfcc, 8, "CCSD(T) total energy") # TEST
samples/gpu_dfcc/gpudfcc2/input.dat:#! aug-cc-pvdz H2O Test DF-CCSD(T) vs GPU-DF-CCSD(T)
samples/gpu_dfcc/gpudfcc2/input.dat:import gpu_dfcc
samples/gpu_dfcc/gpudfcc2/input.dat:en_gpu_dfcc = energy('gpu-df-ccsd(t)')
doc/sphinxman/source/brianqc.rst:Interface to the BrianQC GPU module by the BrianQC team
doc/sphinxman/source/brianqc.rst:|PSIfour| contains code to interface to the BrianQC GPU module developed
doc/sphinxman/source/brianqc.rst:to download the BrianQC GPU module and obtain a license.
doc/sphinxman/source/brianqc.rst:having a supported GPU available in the computing node and having the
doc/sphinxman/source/brianqc.rst:proper GPU drivers installed. Please refer to the `BrianQC manual <https://www.brianqc.com/download/>`_
doc/sphinxman/source/build_planning.rst:* gpu_dfcc |w---w| https://github.com/edeprince3/gpu_dfcc/commits/master
doc/sphinxman/source/testsuite.rst:   autodoc_testsuite_gpu_dfcc
doc/sphinxman/source/bibliography.rst:   "Distributed memory, GPU accelerated Fock construction for hybrid, Gaussian basis density functional theory"
doc/sphinxman/source/scf.rst:    on Graphics Processing Units (GPUs). See :ref:`sec:scfsnlink` for more information.
doc/sphinxman/source/scf.rst:general contraction kernel. Additionally, GauXC's sn-LinK code supports execution on GPUs, allowing for
doc/sphinxman/source/scf.rst:GPU-enabled construction of the Exchange matrix, while |PSIfour|'s COSX does not. In general,
doc/sphinxman/source/scf.rst:* :makevar:`gauxc_ENABLE_GPU`: Enable GPU support for the Psi4-GauXC interface class. When building GauXC internally within Psi4, this keyword controls whether to enable GPU support on the internally-built GauXC instance. When using an external GauXC build, this keyword must align with the GPU capabilities of the external GauXC install.  
doc/sphinxman/source/scf.rst:  |scf__snlinK_use_gpu|: Select whether to execute the sn-LinK algorithm on GPU or not. Setting this option to ``true`` will fail unless the Psi4-GauXC interface is compiled with GPU support.
doc/sphinxman/document_tests.pl:   "gpu_dfcc/"  => "gpu_dfcc",
doc/sphinxman/document_tests.pl:    next if $File =~ /^gpu_dfcc$/;
pytest.ini:    gpu_dfcc
CMakeLists.txt:option(ENABLE_gpu_dfcc "Enables GPU_DFCC plugin for gpu-accelerated df-cc (requires CUDA; can also be added at runtime)" OFF)
CMakeLists.txt:option(ENABLE_BrianQC "Enables the BrianQC GPU module (requires CUDA; requires separate installation and licensing of the BrianQC module)" OFF)
CMakeLists.txt:option_with_default(gauxc_ENABLE_GPU "Enable GPU support for GauXC" OFF)
CMakeLists.txt:option_with_default(gauxc_ENABLE_MAGMA "Enable MAGMA support for GauXC. Required GPU-enabled GauXC" OFF)

```
