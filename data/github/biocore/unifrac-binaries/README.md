# https://github.com/biocore/unifrac-binaries

```console
README.md:    For GPU-accelerated UniFrac, please see:
README.md:An example of installing UniFrac, and using it with CPUs as well as GPUs, can be be found on [Google Colabs](https://colab.research.google.com/drive/1yL0MdF1zNAkPg1_yESI1iABUH4ZHNGwj?usp=sharing).
README.md:For GPU-enabled code, you will need the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) compiler.
README.md:## GPU support
README.md:On Linux platforms, Unifrac will run on a GPU, if one is found. 
README.md:To disable GPU offload, and thus force CPU-only execution, one can set:
README.md:    export UNIFRAC_USE_GPU=N
README.md:    export UNIFRAC_GPU_INFO=Y
README.md:Finally, Unifrac will only use one GPU at a time. 
README.md:If more than one GPU is present, one can select the one to use by setting:
README.md:    export ACC_DEVICE_NUM=gpunum
README.md:Note that there is no GPU support for MacOS.
README.md:        GPU offload can be disabled with UNIFRAC_USE_GPU=N. By default, if a NVIDIA GPU is detected, it will be used.
README.md:        A specific GPU can be selected with ACC_DEVICE_NUM. If not defined, the first GPU will be used.
scripts/install_hpc_sdk.sh:# This is a helper script for installing the NVIDIA HPC SDK 
scripts/install_hpc_sdk.sh:# needed to compile a GPU-enabled version of unifrac.
scripts/install_hpc_sdk.sh:# since NVIDIA HPC SDK does not use the env variables
scripts/install_hpc_sdk.sh:# Install the NVIDIA HPC SDK
scripts/install_hpc_sdk.sh:  NV_URL=https://developer.download.nvidia.com/hpc-sdk/23.5/nvhpc_2023_235_Linux_x86_64_cuda_multi.tar.gz
scripts/install_hpc_sdk.sh:echo "Downloading the NVIDIA HPC SDK"
scripts/install_hpc_sdk.sh:echo "Installing NVIDIA HPC SDK"
scripts/install_hpc_sdk.sh:for f in nvhpc_*/install_components/install nvhpc_*/install_components/*/*/compilers/bin/makelocalrc nvhpc_*/install_components/install_cuda; do
combined/ssu:# Need at least AVX to support GPUs
combined/ssu:   if [ "${UNIFRAC_GPU_INFO}" == "Y" ]; then
combined/ssu:     echo "INFO (unifrac): CPU too old, disabling GPU"
combined/libssu.c:      const char* env_gpu_info = getenv("UNIFRAC_GPU_INFO");
combined/libssu.c:      if ((env_gpu_info!=NULL) && (env_gpu_info[0]=='Y')) {
combined/libssu.c:         printf("INFO (unifrac): CPU too old, disabling GPU\n");
combined/faithpd:# Need at least AVX to support GPUs
src/unifrac_cmp.hpp:  // Returns True iff a GPU can be used
src/unifrac_cmp.hpp:  bool found_gpu();
src/test_su.cpp:    std::vector<uint32_t> exp_openclose;
src/test_su.cpp:    exp_openclose.push_back(7);
src/test_su.cpp:    exp_openclose.push_back(6);
src/test_su.cpp:    exp_openclose.push_back(3);
src/test_su.cpp:    exp_openclose.push_back(2);
src/test_su.cpp:    exp_openclose.push_back(5);
src/test_su.cpp:    exp_openclose.push_back(4);
src/test_su.cpp:    exp_openclose.push_back(1);
src/test_su.cpp:    exp_openclose.push_back(0);
src/test_su.cpp:    ASSERT(tree.get_openclose() == exp_openclose);
src/test_su.cpp:    std::vector<uint32_t> exp_openclose;
src/test_su.cpp:    exp_openclose.push_back(7);
src/test_su.cpp:    exp_openclose.push_back(6);
src/test_su.cpp:    exp_openclose.push_back(3);
src/test_su.cpp:    exp_openclose.push_back(2);
src/test_su.cpp:    exp_openclose.push_back(5);
src/test_su.cpp:    exp_openclose.push_back(4);
src/test_su.cpp:    exp_openclose.push_back(1);
src/test_su.cpp:    exp_openclose.push_back(0);
src/test_su.cpp:    ASSERT(tree.get_openclose() == exp_openclose);
src/test_su.cpp:    std::vector<uint32_t> exp_openclose;
src/test_su.cpp:    exp_openclose.push_back(5);
src/test_su.cpp:    exp_openclose.push_back(4);
src/test_su.cpp:    exp_openclose.push_back(3);
src/test_su.cpp:    exp_openclose.push_back(2);
src/test_su.cpp:    exp_openclose.push_back(1);
src/test_su.cpp:    exp_openclose.push_back(0);
src/test_su.cpp:    ASSERT(tree.get_openclose() == exp_openclose);
src/test_su.cpp:    std::vector<uint32_t> exp_openclose;
src/test_su.cpp:    exp_openclose.push_back(7);
src/test_su.cpp:    exp_openclose.push_back(6);
src/test_su.cpp:    exp_openclose.push_back(3);
src/test_su.cpp:    exp_openclose.push_back(2);
src/test_su.cpp:    exp_openclose.push_back(5);
src/test_su.cpp:    exp_openclose.push_back(4);
src/test_su.cpp:    exp_openclose.push_back(1);
src/test_su.cpp:    exp_openclose.push_back(0);
src/test_su.cpp:    ASSERT(tree.get_openclose() == exp_openclose);
src/tree.hpp:            std::vector<uint32_t> get_openclose();
src/tree.hpp:            std::vector<uint32_t> openclose;      // cache'd mapping between parentheses
src/tree.hpp:            void structure_to_openclose();  // set the cache mapping between parentheses pairs
src/su.cpp:    std::cout << "    GPU offload can be disabled with UNIFRAC_USE_GPU=N. By default, if a NVIDIA GPU is detected, it will be used." << std::endl;
src/su.cpp:    std::cout << "    A specific GPU can be selected with ACC_DEVICE_NUM. If not defined, the first GPU will be used." << std::endl;
src/Makefile:ifndef NOGPU
src/Makefile:   ifeq ($(BUILD_OFFLOAD),ompgpu)
src/Makefile:		CPPFLAGS += -DUNIFRAC_ENABLE_OMPGPU=1
src/Makefile:                UNIFRAC_FILES += unifrac_cmp_ompgpu.o
src/Makefile:		OMPGPUCPPFLAGS += -mp=gpu -DOMPGPU=1 -noacc
src/Makefile:	            OMPGPUCPPFLAGS += -gpu=ccall
src/Makefile:	            OMPGPUCPPFLAGS += -gpu=ccnative
src/Makefile:		    OMPGPUCPPFLAGS += -Minfo=accel
src/Makefile:	        LDDFLAGS += -shlib -mp=gpu -Bstatic_pgi
src/Makefile:	        EXEFLAGS += -mp=gpu -Bstatic_pgi
src/Makefile:	            ACCCPPFLAGS += -gpu=ccall
src/Makefile:	            ACCCPPFLAGS += -gpu=ccnative
src/Makefile:unifrac_cmp_ompgpu.o: unifrac_cmp.cpp unifrac_cmp.hpp unifrac_internal.hpp unifrac.hpp unifrac_task.cpp unifrac_task.hpp biom_interface.hpp tree.hpp
src/Makefile:	$(CXX) $(CPPFLAGS) $(OMPGPUCPPFLAGS) -c $< -o $@
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#define SUCMP_NM su_ompgpu
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#ifndef SMALLGPU
src/unifrac_task.hpp:  // defaultt on larger alignment, which improves performance on GPUs like V100
src/unifrac_task.hpp:  // smaller GPUs prefer smaller allignment 
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:        // alternate buffer only needed in async environments, like openacc
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:        // The parallel nature of GPUs needs a largish step
src/unifrac_task.hpp:  #ifndef SMALLGPU
src/unifrac_task.hpp:        // default to larger step, which makes a big difference for bigger GPUs like V100
src/unifrac_task.hpp:        // smaller GPUs prefer a slightly smaller step
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/unifrac_task.hpp:#elif defined(_OPENACC)
src/unifrac_task.hpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.hpp:#if defined(OMPGPU)
src/tree.cpp:    openclose = std::vector<uint32_t>();
src/tree.cpp:    openclose.resize(nparens);
src/tree.cpp:    structure_to_openclose();
src/tree.cpp:    openclose = std::vector<uint32_t>();
src/tree.cpp:    openclose.resize(nparens);
src/tree.cpp:    structure_to_openclose();
src/tree.cpp:    openclose = std::vector<uint32_t>();
src/tree.cpp:    openclose.resize(nparens);
src/tree.cpp:    structure_to_openclose();
src/tree.cpp:    return structure[i] ? i : openclose[i];
src/tree.cpp:    return structure[i] ? openclose[i] : i;
src/tree.cpp:void BPTree::structure_to_openclose() {
src/tree.cpp:            openclose[i] = open_idx;
src/tree.cpp:            openclose[open_idx] = i;
src/tree.cpp:std::vector<uint32_t> BPTree::get_openclose() {
src/tree.cpp:    return openclose;
src/unifrac_cmp.cpp:#if defined(OMPGPU)
src/unifrac_cmp.cpp:bool SUCMP_NM::found_gpu() {
src/unifrac_cmp.cpp:#elif defined(_OPENACC)
src/unifrac_cmp.cpp:#include <openacc.h>
src/unifrac_cmp.cpp:bool SUCMP_NM::found_gpu() {
src/unifrac_cmp.cpp:bool SUCMP_NM::found_gpu() {
src/unifrac_cmp.cpp:    // no processor affinity whenusing openacc or openmp
src/unifrac_cmp.cpp:#if defined(OMPGPU)
src/unifrac_cmp.cpp:          // TODO: Change if we ever implement async in OMPGPU
src/unifrac_cmp.cpp:#elif defined(_OPENACC)
src/unifrac_cmp.cpp:#if defined(OMPGPU)
src/unifrac_cmp.cpp:    // TODO: Change if we ever implement async in OMPGPU
src/unifrac_cmp.cpp:#elif defined(_OPENACC)
src/unifrac_cmp.cpp:    // no processor affinity whenusing openacc or openmp
src/unifrac_cmp.cpp:#if defined(OMPGPU)
src/unifrac_cmp.cpp:#elif defined(_OPENACC)
src/unifrac_cmp.cpp:#if defined(OMPGPU)
src/unifrac_cmp.cpp:          // TODO: Change if we ever implement async in OMPGPU
src/unifrac_cmp.cpp:#elif defined(_OPENACC)
src/unifrac_cmp.cpp:#if defined(OMPGPU)
src/unifrac_cmp.cpp:    // TODO: Change if we ever implement async in OMPGPU
src/unifrac_cmp.cpp:#elif defined(_OPENACC)
src/unifrac_cmp.cpp:#if defined(OMPGPU)
src/unifrac_cmp.cpp:#elif defined(_OPENACC)
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#elif defined(_OPENACC)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac_task.cpp:#if !(defined(_OPENACC) || defined(OMPGPU))
src/unifrac_task.cpp:            // keep reads in the same place to maximize GPU warp performance
src/unifrac_task.cpp:          // keep all writes in a single place, to maximize GPU warp performance
src/unifrac_task.cpp:#if !(defined(_OPENACC) || defined(OMPGPU))
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:      // SIMT-based GPU work great one at a time (HW will deal with parallelism)
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac_task.cpp:          // keep all writes in a single place, to maximize GPU warp performance
src/unifrac_task.cpp:#if !(defined(_OPENACC) || defined(OMPGPU))
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:      // SIMT-based GPU work great one at a time (HW will deal with parallelism)
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac_task.cpp:#if !(defined(_OPENACC) || defined(OMPGPU))
src/unifrac_task.cpp:                 // GPU/SIMT faster if we just go ahead will full at all times
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#elif defined(_OPENACC)
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#elif defined(_OPENACC)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#elif defined(_OPENACC)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#elif defined(_OPENACC)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#elif defined(_OPENACC)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:    // openacc only works well with local variables
src/unifrac_task.cpp:#if defined(_OPENACC) || defined(OMPGPU)
src/unifrac_task.cpp:#if defined(OMPGPU)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac_task.cpp:#if !defined(OMPGPU) && defined(_OPENACC)
src/unifrac.cpp:#ifdef UNIFRAC_ENABLE_OMPGPU
src/unifrac.cpp:#define OMPGPU
src/unifrac.cpp:#define SUCMP_NM  su_ompgpu
src/unifrac.cpp:#undef OMPGPU
src/unifrac.cpp:#ifdef UNIFRAC_ENABLE_OMPGPU
src/unifrac.cpp:static int proc_use_ompgpu = -1;
src/unifrac.cpp:inline bool use_ompgpu() {
src/unifrac.cpp: if (proc_use_ompgpu!=-1) return (proc_use_ompgpu!=0);
src/unifrac.cpp: if (const char* env_p = std::getenv("UNIFRAC_GPU_INFO")) {
src/unifrac.cpp: if (!su_ompgpu::found_gpu()) {
src/unifrac.cpp:   if (print_info) printf("INFO (unifrac): GPU not found, using CPU\n");
src/unifrac.cpp:   proc_use_ompgpu=0;
src/unifrac.cpp: if (const char* env_p = std::getenv("UNIFRAC_USE_GPU")) {
src/unifrac.cpp:     if (print_info) printf("INFO (unifrac): Use of GPU explicitly disabled, using CPU\n");
src/unifrac.cpp:     proc_use_ompgpu=0;
src/unifrac.cpp: if (print_info) printf("INFO (unifrac): Using GPU\n");
src/unifrac.cpp: proc_use_ompgpu=1;
src/unifrac.cpp: int has_nvidia_gpu_rc = access("/proc/driver/nvidia/gpus", F_OK);
src/unifrac.cpp: if (const char* env_p = std::getenv("UNIFRAC_GPU_INFO")) {
src/unifrac.cpp: if (has_nvidia_gpu_rc == 0) {
src/unifrac.cpp:    if (!su_acc::found_gpu()) {
src/unifrac.cpp:       has_nvidia_gpu_rc  = 1;
src/unifrac.cpp:       if (print_info) printf("INFO (unifrac): NVIDIA GPU listed but OpenACC cannot use it.\n");
src/unifrac.cpp: if (has_nvidia_gpu_rc != 0) {
src/unifrac.cpp:   if (print_info) printf("INFO (unifrac): GPU not found, using CPU\n");
src/unifrac.cpp: if (const char* env_p = std::getenv("UNIFRAC_USE_GPU")) {
src/unifrac.cpp:     if (print_info) printf("INFO (unifrac): Use of GPU explicitly disabled, using CPU\n");
src/unifrac.cpp: if (print_info) printf("INFO (unifrac): Using GPU\n");
src/unifrac.cpp:#if defined(UNIFRAC_ENABLE_OMPGPU)
src/unifrac.cpp:  if (use_ompgpu()) {
src/unifrac.cpp:    su_ompgpu::unifrac(table, tree, unifrac_method, dm_stripes, dm_stripes_total, task_p);
src/unifrac.cpp:  // TODO: Should we support both OMPGPU and ACC at the same time?
src/unifrac.cpp:#if defined(UNIFRAC_ENABLE_OMPGPU)
src/unifrac.cpp:  if (use_ompgpu()) {
src/unifrac.cpp:   su_ompgpu::unifrac_vaw(table, tree, unifrac_method, dm_stripes, dm_stripes_total, task_p);
src/unifrac.cpp:  // TODO: Should we support both OMPGPU and ACC at the same time?
src/unifrac.cpp:    // cannot use threading with openacc or openmp

```
