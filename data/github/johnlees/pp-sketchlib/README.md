# https://github.com/bacpop/pp-sketchlib

```console
vendor/highfive/include/highfive/bits/H5Node_traits_misc.hpp:#include <H5Gpublic.h>
test/clean_test.py:    "tmp.gpu.query.dists.txt",
test/clean_test.py:    "tmp.gpu.self.dists.txt"
test/gpu_test.py:"""Tests for pp-sketchlib GPU"""
test/gpu_test.py:def compare_dists_files(cpu_file, gpu_file):
test/gpu_test.py:    with open(cpu_file, 'r') as cpu_dists, open(gpu_file, 'r') as gpu_dists:
test/gpu_test.py:        gpu_header = gpu_dists.readline()
test/gpu_test.py:        if (cpu_header != gpu_header):
test/gpu_test.py:        for cpu_line, gpu_line in zip(cpu_dists, gpu_dists):
test/gpu_test.py:            g_rname, g_qname, g_core_dist, g_acc_dist = gpu_line.rstrip().split("\t")
test/gpu_test.py:sys.stderr.write("Testing self distances on GPU\n")
test/gpu_test.py:subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass --read-k --print --use-gpu", shell=True, check=True)
test/gpu_test.py:subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass --read-k --print --use-gpu > tmp.gpu.self.dists.txt", shell=True, check=True)
test/gpu_test.py:fail_self = compare_dists_files("tmp.cpu.self.dists.txt", "tmp.gpu.self.dists.txt")
test/gpu_test.py:sys.stderr.write("Testing query distances on GPU\n")
test/gpu_test.py:subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass_plus1 --read-k --print --use-gpu", shell=True, check=True)
test/gpu_test.py:subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass_plus1 --read-k --print --use-gpu > tmp.gpu.query.dists.txt", shell=True, check=True)
test/gpu_test.py:fail_query = compare_dists_files("tmp.cpu.query.dists.txt", "tmp.gpu.query.dists.txt")
environment.yml:  - cudatoolkit==11.2 # This is pinned due to version install on azure, see azure-pipelines.yml
pp_sketch/__main__.py:  sketchlib sketch <files>... -o <output> [-k <kseq>|--kmer <k>] [-s <size>] [--single-strand] [--codon-phased] [--min-count <count>] [--exact-counter] [--cpus <cpus>] [--gpu <gpu>]
pp_sketch/__main__.py:  sketchlib sketch -l <file-list> -o <output> [-k <kseq>|--kmer <k>] [-s <size>] [--single-strand] [--codon-phased] [--min-count <count>] [--exact-counter] [--cpus <cpus>] [--gpu <gpu>]
pp_sketch/__main__.py:  sketchlib query dist <db1> [<db2>] [-o <output>] [--adj-random] [--subset <file>] [--cpus <cpus>] [--gpu <gpu>]
pp_sketch/__main__.py:  sketchlib query sparse <db1> (--kNN <k>|--threshold <max>) [-o <output>] [--accessory] [--adj-random] [--subset <file>] [--cpus <cpus>] [--gpu <gpu>]
pp_sketch/__main__.py:  --gpu <gpu>    Use GPU with specified device ID [default: -1].
pp_sketch/__main__.py:    arguments['--gpu'] = int(arguments['--gpu'])
pp_sketch/__main__.py:    if int(arguments['--gpu']) >= 0:
pp_sketch/__main__.py:        arguments['--use-gpu'] = True
pp_sketch/__main__.py:        arguments['--use-gpu'] = False
pp_sketch/__main__.py:                                       use_gpu=args['--use-gpu'],
pp_sketch/__main__.py:                                       device_id=args['--gpu'])
pp_sketch/__main__.py:                                              use_gpu=args['--use-gpu'],
pp_sketch/__main__.py:                                              device_id=args['--gpu'])
pp_sketch/__main__.py:                              use_gpu=args['--use-gpu'],
pp_sketch/__main__.py:                              device_id=args['--gpu'])
pp_sketch/__main__.py:                                                 use_gpu=args['--use-gpu'],
pp_sketch/__main__.py:                                                 device_id=args['--gpu'])
README.md:If you wish to compile the GPU code you will also need the CUDA toolkit
README.md:For calculating sketches of read datasets, or large numbers of distances, and you have a CUDA compatible GPU,
README.md:you can calculate distances on your graphics device even more quickly. Add the `--gpu` option with the desired
README.md:sketchlib sketch -l rfiles.txt -o listeria --cpus 4 --gpu 0
README.md:sketchlib query dist listeria --gpu 0
README.md:Both CPU parallelism and the GPU will be used, so be sure to add
README.md:both `--cpus` and `--gpu` for maximum speed. This is particularly efficient
README.md:on a CPU node, or 2 minutes on a GPU. Assigning new queries is twice as fast.
README.md:|             | CPU & GPU       | 49 genomes per minute           |
README.md:|             | GPU             | 6000k distances per second      |
README.md:GPU tested using an NVIDIA RTX 2080 Ti GPU (4352 CUDA cores @ 1.35GHz).
README.md:jaccard, cpus, use_gpu, deviceid)
README.md:- GPU sketching is only supported for reads. If a mix of reads and assemblies,
README.md:- GPU sketching filters out any read containing an N, which may give slightly
README.md:- GPU sketching with variable read lengths is unsupported. Illumina data only for now!
README.md:- GPU distances use lower precision than the CPU code, so slightly different results
README.md:- `all` (default): builds test executables `sketch_test`, `matrix_test`, `read_test` and `gpu_dist_test`
README.md:- `GPU=1` also build CUDA code (assumes `/usr/local/cuda-11.1/` and SM v8.6)
README.md:The repository key for the ubuntu CUDA install is periodically updated, which may cause build failures. See https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/ and update in `azure-pipelines.yml`.
azure-pipelines.yml:# This just checks the package can be installed using CUDA, no testing
azure-pipelines.yml:    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
azure-pipelines.yml:    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
azure-pipelines.yml:    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
azure-pipelines.yml:    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
azure-pipelines.yml:    sudo apt-get -y install cuda=11.2.2-1
azure-pipelines.yml:    export CUDA_HOME=/usr/local/cuda-11.2
azure-pipelines.yml:    export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
azure-pipelines.yml:    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
azure-pipelines.yml:    export CUDA_HOME=/usr/local/cuda-11.2
azure-pipelines.yml:    export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
azure-pipelines.yml:    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
CMakeLists.txt:  cmake_policy(SET CMP0104 OLD) # Can't get CUDA_ARCHITECTURES to work with NEW
CMakeLists.txt:execute_process(COMMAND cat ${VERSION_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/sketch.cu
CMakeLists.txt:# Check for CUDA and compiles GPU library
CMakeLists.txt:check_language(CUDA)
CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 11.0)
CMakeLists.txt:        message(STATUS "CUDA >11.0 found, compiling both GPU and CPU code")
CMakeLists.txt:        # --cudart static: static linking of the CUDA libraries
CMakeLists.txt:        set(CUDA_OPTS "-Xcompiler -fPIC --relocatable-device-code=true --expt-relaxed-constexpr")
CMakeLists.txt:            string(APPEND CUDA_OPTS " -dlto -arch=sm_86")
CMakeLists.txt:            string(APPEND CUDA_OPTS " -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")
CMakeLists.txt:            string(APPEND CUDA_OPTS " -G")
CMakeLists.txt:        set(CMAKE_CUDA_FLAGS "${CUDA_OPTS}")
CMakeLists.txt:        add_compile_definitions(GPU_AVAILABLE)
CMakeLists.txt:        add_library("${TARGET_NAME}_CUDA" OBJECT src/gpu/dist.cu
CMakeLists.txt:                                                src/gpu/sketch.cu
CMakeLists.txt:                                                src/gpu/device_memory.cu
CMakeLists.txt:                                                src/gpu/gpu_countmin.cu
CMakeLists.txt:                                                src/gpu/device_reads.cu)
CMakeLists.txt:        target_include_directories("${TARGET_NAME}_CUDA" PRIVATE "${EIGEN3_INCLUDE_DIR}" "${pybind11_INCLUDE_DIRS}")
CMakeLists.txt:        set_property(TARGET "${TARGET_NAME}_CUDA"
CMakeLists.txt:                    CUDA_SEPARABLE_COMPILATION ON
CMakeLists.txt:                    CUDA_RESOLVE_DEVICE_SYMBOLS ON   # try and ensure device link with nvcc
CMakeLists.txt:                    CUDA_VISIBILITY_PRESET "hidden"
CMakeLists.txt:                CUDA_RUNTIME_LIBRARY Static)
CMakeLists.txt:              #CUDA_ARCHITECTURES OFF) # set off as done explicitly above (due to dlto complexities)
CMakeLists.txt:        # CPU code/gcc compiled code needed by cuda lib
CMakeLists.txt:        target_sources("${TARGET_NAME}" PRIVATE src/gpu/gpu_api.cpp)
CMakeLists.txt:        message(STATUS "CUDA >=11.0 required, compiling CPU code only")
CMakeLists.txt:    message(STATUS "CUDA not found, compiling CPU code only")
CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
CMakeLists.txt:    target_link_libraries("${TARGET_NAME}" PRIVATE "${TARGET_NAME}_CUDA")
CMakeLists.txt:    set_property(TARGET "${TARGET_NAME}" PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMakeLists.txt:    #set_property(TARGET "${TARGET_NAME}" PROPERTY CUDA_ARCHITECTURES OFF)
src/dist/matrix_idx.hpp:#ifndef __CUDA_ARCH__
src/sketchlib_bindings.cpp:                       const size_t num_threads = 1, const bool use_gpu = false,
src/sketchlib_bindings.cpp:#ifdef GPU_AVAILABLE
src/sketchlib_bindings.cpp:  if (use_gpu) {
src/sketchlib_bindings.cpp:          "Codon phased seeds not yet implemented for GPU sketching");
src/sketchlib_bindings.cpp:    ref_sketches = create_sketches_cuda(db_name, sample_names, file_names,
src/sketchlib_bindings.cpp:                          const bool use_gpu = false, const int device_id = 0) {
src/sketchlib_bindings.cpp:  if (use_gpu && (jaccard || kmer_lengths.size() < 2)) {
src/sketchlib_bindings.cpp:        "Extracting Jaccard distances not supported on GPU");
src/sketchlib_bindings.cpp:#ifdef GPU_AVAILABLE
src/sketchlib_bindings.cpp:  if (use_gpu) {
src/sketchlib_bindings.cpp:    dists = query_db_cuda(ref_sketches, query_sketches, kmer_lengths, random,
src/sketchlib_bindings.cpp:                const bool use_gpu = false, const int device_id = 0) {
src/sketchlib_bindings.cpp:  if (use_gpu && (jaccard || kmer_lengths.size() < 2 || dist_col > 1)) {
src/sketchlib_bindings.cpp:        "Extracting Jaccard distances not supported on GPU");
src/sketchlib_bindings.cpp:                                      false, num_threads, use_gpu, device_id);
src/sketchlib_bindings.cpp:#ifdef GPU_AVAILABLE
src/sketchlib_bindings.cpp:    if (use_gpu) {
src/sketchlib_bindings.cpp:          query_db_sparse_cuda(ref_sketches, kmer_lengths, random, kNN,
src/sketchlib_bindings.cpp:        py::arg("use_gpu") = false, py::arg("device_id") = 0);
src/sketchlib_bindings.cpp:        py::arg("use_gpu") = false, py::arg("device_id") = 0);
src/sketchlib_bindings.cpp:        py::arg("use_gpu") = false, py::arg("device_id") = 0);
src/Makefile:  CUDAFLAGS = -g -G
src/Makefile:  CUDAFLAGS = -O2 -pg -lineinfo
src/Makefile:  CUDAFLAGS = -O3
src/Makefile:CUDA_LDLIBS=-lcudadevrt -lcudart_static $(LDLIBS)
src/Makefile:CUDA_LDFLAGS =-L$(LIBLOC)/lib -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib
src/Makefile:CUDAFLAGS +=-std=c++17 -Xcompiler -fPIC --cudart static --relocatable-device-code=true --expt-relaxed-constexpr -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
src/Makefile:ifdef GPU
src/Makefile:	CXXFLAGS += -DGPU_AVAILABLE
src/Makefile:	CUDAFLAGS += -gencode arch=compute_86,code=sm_86
src/Makefile:	CUDA_LDFLAGS += -L/usr/local/cuda-11.2/lib64
src/Makefile:python: CPPFLAGS += -DGPU_AVAILABLE -DPYTHON_EXT -DNDEBUG -Dpp_sketchlib_EXPORTS $(shell python3 -m pybind11 --includes)
src/Makefile:PROGRAMS=sketch_test matrix_test read_test gpu_dist_test
src/Makefile:GPU_SKETCH_OBJS=gpu/gpu_api.o
src/Makefile:CUDA_OBJS=gpu/dist.cu.o gpu/sketch.cu.o gpu/device_reads.cu.o gpu/gpu_countmin.cu.o gpu/device_memory.cu.o
src/Makefile:	$(RM) $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) $(WEB_OBJS) *.o *.so version.h ~* $(PROGRAMS)
src/Makefile:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
src/Makefile:read_test: $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) test/read_test.o
src/Makefile:	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
src/Makefile:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ device_link.o -o $@ $(CUDA_LDLIBS)
src/Makefile:gpu_dist_test: $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) test/gpu_dist_test.o
src/Makefile:	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
src/Makefile:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ device_link.o -o $@ $(CUDA_LDLIBS)
src/Makefile:	cat sketch/*.cpp sketch/*.hpp gpu/sketch.cu | openssl sha1 | awk '{print "#define SKETCH_VERSION \"" $$2 "\""}' > version.h
src/Makefile:$(PYTHON_LIB): $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) sketchlib_bindings.o
src/Makefile:	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
src/Makefile:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) -shared $^ device_link.o -o $(PYTHON_LIB) $(CUDA_LDLIBS)
src/Makefile:gpu/dist.cu.o:
src/Makefile:	echo ${CUDAFLAGS}
src/Makefile:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/dist.cu -o $@
src/Makefile:gpu/sketch.cu.o:
src/Makefile:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/sketch.cu -o $@
src/Makefile:gpu/device_memory.cu.o:
src/Makefile:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/device_memory.cu -o $@
src/Makefile:gpu/device_reads.cu.o:
src/Makefile:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/device_reads.cu -o $@
src/Makefile:gpu/gpu_countmin.cu.o:
src/Makefile:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/gpu_countmin.cu -o $@
src/Makefile_fedora38:  CUDAFLAGS = -g -G
src/Makefile_fedora38:  CUDAFLAGS = -O2 -pg -lineinfo
src/Makefile_fedora38:  CUDAFLAGS+= -O3
src/Makefile_fedora38:LDFLAGS+= -L$(LIBLOC)/lib -L"/home/linuxbrew/.linuxbrew/lib" -L/usr/local/cuda-12.3/lib64
src/Makefile_fedora38:CUDA_LDLIBS=-lcudadevrt -lcudart_static $(LDLIBS)
src/Makefile_fedora38:CUDA_LDFLAGS =-L$(LIBLOC)/lib -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib
src/Makefile_fedora38:CUDAFLAGS +=-ccbin /home/linuxbrew/.linuxbrew/bin/g++-11 -std=c++17 -Xcompiler -fPIC --cudart static --relocatable-device-code=true --expt-relaxed-constexpr -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
src/Makefile_fedora38:ifdef GPU
src/Makefile_fedora38:	CXXFLAGS += -DGPU_AVAILABLE
src/Makefile_fedora38:	CUDAFLAGS += -gencode arch=compute_86,code=sm_86
src/Makefile_fedora38:	CUDA_LDFLAGS += -L/usr/local/cuda-12.3/lib64
src/Makefile_fedora38:python: CPPFLAGS += -DGPU_AVAILABLE -DPYTHON_EXT -DNDEBUG -Dpp_sketchlib_EXPORTS $(shell python3 -m pybind11 --includes)
src/Makefile_fedora38:PROGRAMS=sketch_test matrix_test read_test gpu_dist_test
src/Makefile_fedora38:GPU_SKETCH_OBJS=gpu/gpu_api.o
src/Makefile_fedora38:CUDA_OBJS=gpu/dist.cu.o gpu/sketch.cu.o gpu/device_reads.cu.o gpu/gpu_countmin.cu.o gpu/device_memory.cu.o
src/Makefile_fedora38:	$(RM) $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) $(WEB_OBJS) *.o *.so version.h ~* $(PROGRAMS)
src/Makefile_fedora38:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
src/Makefile_fedora38:read_test: $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) test/read_test.o
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
src/Makefile_fedora38:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ device_link.o -o $@ $(CUDA_LDLIBS)
src/Makefile_fedora38:gpu_dist_test: $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) test/gpu_dist_test.o
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
src/Makefile_fedora38:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) $^ device_link.o -o $@ $(CUDA_LDLIBS)
src/Makefile_fedora38:	cat sketch/*.cpp sketch/*.hpp gpu/sketch.cu | openssl sha1 | awk '{print "#define SKETCH_VERSION \"" $$2 "\""}' > version.h
src/Makefile_fedora38:$(PYTHON_LIB): $(SKETCH_OBJS) $(GPU_SKETCH_OBJS) $(CUDA_OBJS) sketchlib_bindings.o
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
src/Makefile_fedora38:	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) -shared $^ device_link.o -o $(PYTHON_LIB) $(CUDA_LDLIBS)
src/Makefile_fedora38:gpu/dist.cu.o:
src/Makefile_fedora38:	echo ${CUDAFLAGS}
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/dist.cu -o $@
src/Makefile_fedora38:gpu/sketch.cu.o:
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/sketch.cu -o $@
src/Makefile_fedora38:gpu/device_memory.cu.o:
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/device_memory.cu -o $@
src/Makefile_fedora38:gpu/device_reads.cu.o:
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/device_reads.cu -o $@
src/Makefile_fedora38:gpu/gpu_countmin.cu.o:
src/Makefile_fedora38:	nvcc $(CUDAFLAGS) $(CPPFLAGS) -DGPU_AVAILABLE -x cu -c gpu/gpu_countmin.cu -o $@
src/api.cpp:#include "gpu/gpu.hpp"
src/reference.hpp:  // Initialise from GPU sketch
src/random/random_match.cpp:// Helper functions for loading onto GPU
src/random/random_match.hpp:#include "gpu/gpu.hpp"
src/random/random_match.hpp:  // GPU helper functions to flatten
src/gpu/gpu_api.cpp: * gpu_api.cpp
src/gpu/gpu_api.cpp: * PopPUNK dists using CUDA
src/gpu/gpu_api.cpp:#include "gpu.hpp"
src/gpu/gpu_api.cpp:std::vector<Reference> create_sketches_cuda(
src/gpu/gpu_api.cpp:    std::cerr << "Sketching " << files.size() << " read sets on GPU device "
src/gpu/gpu_api.cpp:    GPUCountMin countmin_filter;
src/gpu/gpu_api.cpp:      // Run the sketch on the GPU (serially over the batch)
src/gpu/gpu_api.cpp:          std::tie(usigs, seq_length, densified) = sketch_gpu(
src/gpu/gpu_api.cpp:dist_params cuda_dists_init(const std::vector<Reference> &ref_sketches,
src/gpu/gpu_api.cpp:  std::cerr << "Calculating distances on GPU device " << device_id << std::endl;
src/gpu/gpu_api.cpp:NumpyMatrix query_db_cuda(std::vector<Reference> &ref_sketches,
src/gpu/gpu_api.cpp:    cuda_dists_init(ref_sketches, query_sketches, kmer_lengths, device_id);
src/gpu/gpu_api.cpp:sparse_coo query_db_sparse_cuda(std::vector<Reference> &ref_sketches,
src/gpu/gpu_api.cpp:    cuda_dists_init(ref_sketches, ref_sketches, kmer_lengths, device_id);
src/gpu/gpu_countmin.cu:#include "gpu_countmin.cuh"
src/gpu/gpu_countmin.cu:#include "cuda.cuh"
src/gpu/gpu_countmin.cu:GPUCountMin::GPUCountMin()
src/gpu/gpu_countmin.cu:  : _table_width_bits(cuda_table_width_bits),
src/gpu/gpu_countmin.cu:    _table_width(cuda_table_width), _hash_per_hash(cuda_hash_per_hash),
src/gpu/gpu_countmin.cu:    _table_rows(cuda_table_rows), _table_cells(cuda_table_cells) {
src/gpu/gpu_countmin.cu:  CUDA_CALL(cudaMalloc((void **)&_d_countmin_table,
src/gpu/gpu_countmin.cu:GPUCountMin::~GPUCountMin() { CUDA_CALL(cudaFree(_d_countmin_table)); }
src/gpu/gpu_countmin.cu:void GPUCountMin::reset() {
src/gpu/gpu_countmin.cu:  CUDA_CALL(
src/gpu/gpu_countmin.cu:    cudaMemset(_d_countmin_table, 0, _table_cells * sizeof(unsigned int)));
src/gpu/gpu_countmin.cu:// Loop variables are global constants defined in gpu.hpp
src/gpu/gpu_countmin.cu:  for (int hash_nr = 0; hash_nr < cuda_table_rows; hash_nr += cuda_hash_per_hash) {
src/gpu/gpu_countmin.cu:    for (uint i = 0; i < cuda_hash_per_hash; i++) {
src/gpu/gpu_countmin.cu:      uint32_t hash_val_masked = current_hash & cuda_table_width;
src/gpu/gpu_countmin.cu:          atomicInc(d_countmin_table + (hash_nr + i) * cuda_table_width +
src/gpu/gpu_countmin.cu:      current_hash = current_hash >> cuda_table_width_bits;
src/gpu/gpu.hpp: * gpu.hpp
src/gpu/gpu.hpp: * functions using CUDA
src/gpu/gpu.hpp:#include "gpu_countmin.cuh"
src/gpu/gpu.hpp:#ifdef GPU_AVAILABLE
src/gpu/gpu.hpp:// Small struct used in cuda_dists_init
src/gpu/gpu.hpp:std::vector<uint64_t> get_signs(DeviceReads &reads, GPUCountMin &countmin,
src/gpu/containers.cuh:#include "cuda.cuh"
src/gpu/containers.cuh:    CUDA_CALL(cudaMalloc((void **)&data_, size_ * sizeof(T)));
src/gpu/containers.cuh:    CUDA_CALL(cudaMemset(data_, 0, size_ * sizeof(T)));
src/gpu/containers.cuh:    CUDA_CALL(cudaMalloc((void **)&data_, size_ * sizeof(T)));
src/gpu/containers.cuh:    CUDA_CALL(
src/gpu/containers.cuh:        cudaMemcpy(data_, data.data(), size_ * sizeof(T), cudaMemcpyDefault));
src/gpu/containers.cuh:    CUDA_CALL(cudaMalloc((void **)&data_, size_ * sizeof(T)));
src/gpu/containers.cuh:    CUDA_CALL(
src/gpu/containers.cuh:        cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDefault));
src/gpu/containers.cuh:      CUDA_CALL(cudaFree(data_));
src/gpu/containers.cuh:      CUDA_CALL(cudaMalloc((void **)&data_, size_ * sizeof(T)));
src/gpu/containers.cuh:      CUDA_CALL(
src/gpu/containers.cuh:          cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDefault));
src/gpu/containers.cuh:      CUDA_CALL(cudaFree(data_));
src/gpu/containers.cuh:  ~device_array() { CUDA_CALL_NOTHROW(cudaFree(data_)); }
src/gpu/containers.cuh:    CUDA_CALL(cudaMemcpy(dst.data(), data_, dst.size() * sizeof(T),
src/gpu/containers.cuh:                         cudaMemcpyDefault));
src/gpu/containers.cuh:    CUDA_CALL(
src/gpu/containers.cuh:        cudaMemcpy(data_, src.data(), size_ * sizeof(T), cudaMemcpyDefault));
src/gpu/containers.cuh:    CUDA_CALL(cudaMemcpy(data_, src, size_ * sizeof(T), cudaMemcpyDefault));
src/gpu/containers.cuh:  void get_array_async(T *value, const size_t size, cudaStream_t stream) const {
src/gpu/containers.cuh:    CUDA_CALL(cudaMemcpyAsync((void *)value, data_, sizeof(T) * size,
src/gpu/containers.cuh:                              cudaMemcpyDefault, stream));
src/gpu/containers.cuh:  void set_array_async(const T *value, const size_t size, cudaStream_t stream) {
src/gpu/containers.cuh:    CUDA_CALL(cudaMemcpyAsync(data_, (void *)value, sizeof(T) * size,
src/gpu/containers.cuh:                              cudaMemcpyDefault, stream));
src/gpu/containers.cuh:      CUDA_CALL(cudaMalloc((void **)&data_, size_));
src/gpu/containers.cuh:  ~device_array() { CUDA_CALL_NOTHROW(cudaFree(data_)); }
src/gpu/containers.cuh:      CUDA_CALL(cudaFree(data_));
src/gpu/containers.cuh:        CUDA_CALL(cudaMalloc((void **)&data_, size_));
src/gpu/containers.cuh:class cuda_stream {
src/gpu/containers.cuh:  cuda_stream() { CUDA_CALL(cudaStreamCreate(&stream_)); }
src/gpu/containers.cuh:  ~cuda_stream() {
src/gpu/containers.cuh:      CUDA_CALL_NOTHROW(cudaStreamDestroy(stream_));
src/gpu/containers.cuh:  cudaStream_t stream() { return stream_; }
src/gpu/containers.cuh:  void sync() { CUDA_CALL(cudaStreamSynchronize(stream_)); }
src/gpu/containers.cuh:  cuda_stream(const cuda_stream &) = delete;
src/gpu/containers.cuh:  cuda_stream(cuda_stream &&) = delete;
src/gpu/containers.cuh:  cudaStream_t stream_;
src/gpu/align.hpp:#if defined(__CUDACC__) // NVCC
src/gpu/sketch.cu: * CUDA version of bindash sketch method
src/gpu/sketch.cu:#include "cuda.cuh"
src/gpu/sketch.cu:#include "gpu.hpp"
src/gpu/sketch.cu:  CUDA_CALL(
src/gpu/sketch.cu:      cudaMemcpyToSymbolAsync(d_seedTab, seedTab, 256 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_A33r, A33r, 33 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_A31l, A31l, 31 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_C33r, C33r, 33 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_C31l, C31l, 31 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_G33r, G33r, 33 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_G31l, G31l, 31 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_T33r, T33r, 33 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_T31l, T31l, 31 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_N33r, N33r, 33 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_N31l, N31l, 31 * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaDeviceSynchronize());
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&A33r_ptr, d_A33r));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&A31l_ptr, d_A31l));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&C33r_ptr, d_C33r));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&C31l_ptr, d_C31l));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&G33r_ptr, d_G33r));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&G31l_ptr, d_G31l));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&T33r_ptr, d_T33r));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&T31l_ptr, d_T31l));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&N33r_ptr, d_N33r));
src/gpu/sketch.cu:  CUDA_CALL(cudaGetSymbolAddress((void **)&N31l_ptr, d_N31l));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab31l, d_addr_msTab31l,
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpyToSymbolAsync(d_msTab33r, d_addr_msTab33r,
src/gpu/sketch.cu:  CUDA_CALL(cudaDeviceSynchronize());
src/gpu/sketch.cu:std::vector<uint64_t> get_signs(DeviceReads &reads, GPUCountMin &countmin,
src/gpu/sketch.cu:  CUDA_CALL(cudaMalloc((void **)&d_signs, nbins * sizeof(uint64_t)));
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpy(d_signs, signs.data(), nbins * sizeof(uint64_t),
src/gpu/sketch.cu:                       cudaMemcpyDefault));
src/gpu/sketch.cu:    CUDA_CALL(cudaDeviceSynchronize()); // Make sure copy is finished
src/gpu/sketch.cu:    if (cudaGetLastError() != cudaSuccess) {
src/gpu/sketch.cu:  CUDA_CALL(cudaMemcpy(signs.data(), d_signs, nbins * sizeof(uint64_t),
src/gpu/sketch.cu:                       cudaMemcpyDefault));
src/gpu/sketch.cu:  CUDA_CALL(cudaFree(d_signs));
src/gpu/cuda.cuh:static_assert(__CUDACC_VER_MAJOR__ >= 11, "CUDA >=11.0 required");
src/gpu/cuda.cuh:#include <cuda.h>
src/gpu/cuda.cuh:#include <cuda_runtime.h>
src/gpu/cuda.cuh:static void HandleCUDAError(const char *file, int line,
src/gpu/cuda.cuh:                            cudaError_t status = cudaGetLastError()) {
src/gpu/cuda.cuh:  cudaDeviceSynchronize();
src/gpu/cuda.cuh:  if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess) {
src/gpu/cuda.cuh:    if (status == cudaErrorUnknown) {
src/gpu/cuda.cuh:      printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
src/gpu/cuda.cuh:    printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line,
src/gpu/cuda.cuh:           cudaGetErrorString(status));
src/gpu/cuda.cuh:    throw std::runtime_error("CUDA error");
src/gpu/cuda.cuh:#define CUDA_CALL(err) (HandleCUDAError(__FILE__, __LINE__, err))
src/gpu/cuda.cuh:#define CUDA_CALL_NOTHROW( err ) (err)
src/gpu/cuda.cuh:    CUDA_CALL(cudaMallocManaged((void**)&(managed_ptrs.blocks_complete), sizeof(int)));
src/gpu/cuda.cuh:    CUDA_CALL(cudaMallocManaged((void**)&(managed_ptrs.kill_kernel), sizeof(bool)));
src/gpu/cuda.cuh:    CUDA_CALL(cudaFree((void *)managed_ptrs.blocks_complete));
src/gpu/cuda.cuh:    CUDA_CALL(cudaFree(managed_ptrs.kill_kernel));
src/gpu/device_reads.cu:#include "cuda.cuh"
src/gpu/device_reads.cu:  CUDA_CALL(cudaMemGetInfo(&mem_free, &mem_total));
src/gpu/device_reads.cu:  CUDA_CALL_NOTHROW(cudaHostRegister(host_buffer.data(),
src/gpu/device_reads.cu:                                     cudaHostRegisterDefault));
src/gpu/device_reads.cu:  CUDA_CALL(
src/gpu/device_reads.cu:      cudaMalloc((void **)&d_reads, buffer_size * read_length * sizeof(char)));
src/gpu/device_reads.cu:  CUDA_CALL_NOTHROW(cudaHostUnregister(host_buffer.data()));
src/gpu/device_reads.cu:  CUDA_CALL_NOTHROW(cudaFree(d_reads));
src/gpu/device_reads.cu:      CUDA_CALL(cudaMemcpyAsync(d_reads, host_buffer.data(),
src/gpu/device_reads.cu:                                cudaMemcpyDefault));
src/gpu/device_memory.cu:#include "cuda.cuh"
src/gpu/device_memory.cu:#include "gpu.hpp"
src/gpu/device_memory.cu:  CUDA_CALL(
src/gpu/device_memory.cu:      cudaMalloc((void **)&d_ref_sketches, flat_ref.size() * sizeof(uint64_t)));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMemcpy(d_ref_sketches, flat_ref.data(),
src/gpu/device_memory.cu:                       flat_ref.size() * sizeof(uint64_t), cudaMemcpyDefault));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMalloc((void **)&d_random_table,
src/gpu/device_memory.cu:  CUDA_CALL(cudaMemcpy(d_random_table, std::get<1>(flat_random).data(),
src/gpu/device_memory.cu:                       cudaMemcpyDefault));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMalloc((void **)&d_ref_random,
src/gpu/device_memory.cu:  CUDA_CALL(
src/gpu/device_memory.cu:      cudaMemcpy(d_ref_random, ref_random_idx.data() + sample_slice.ref_offset,
src/gpu/device_memory.cu:                 sample_slice.ref_size * sizeof(uint16_t), cudaMemcpyDefault));
src/gpu/device_memory.cu:    CUDA_CALL(cudaMalloc((void **)&d_query_sketches,
src/gpu/device_memory.cu:    CUDA_CALL(cudaMemcpy(d_query_sketches, flat_query.data(),
src/gpu/device_memory.cu:                         cudaMemcpyDefault));
src/gpu/device_memory.cu:    CUDA_CALL(cudaMalloc((void **)&d_query_random,
src/gpu/device_memory.cu:    CUDA_CALL(cudaMemcpy(
src/gpu/device_memory.cu:        sample_slice.query_size * sizeof(uint16_t), cudaMemcpyDefault));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMalloc((void **)&d_kmers, kmer_ints.size() * sizeof(int)));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMemcpy(d_kmers, kmer_ints.data(),
src/gpu/device_memory.cu:                       kmer_ints.size() * sizeof(int), cudaMemcpyDefault));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMalloc((void **)&d_dist_mat, _n_dists * sizeof(float)));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMemset(d_dist_mat, 0, _n_dists * sizeof(float)));
src/gpu/device_memory.cu:  CUDA_CALL(cudaFree(d_ref_sketches));
src/gpu/device_memory.cu:  CUDA_CALL(cudaFree(d_query_sketches));
src/gpu/device_memory.cu:  CUDA_CALL(cudaFree(d_random_table));
src/gpu/device_memory.cu:  CUDA_CALL(cudaFree(d_ref_random));
src/gpu/device_memory.cu:  CUDA_CALL(cudaFree(d_query_random));
src/gpu/device_memory.cu:  CUDA_CALL(cudaFree(d_kmers));
src/gpu/device_memory.cu:  CUDA_CALL(cudaFree(d_dist_mat));
src/gpu/device_memory.cu:  CUDA_CALL(cudaMemcpy(dists.data(), d_dist_mat, _n_dists * sizeof(float),
src/gpu/device_memory.cu:                       cudaMemcpyDefault));
src/gpu/gpu_countmin.cuh:const unsigned int cuda_table_width_bits =
src/gpu/gpu_countmin.cuh:constexpr uint64_t cuda_table_width{0x3FFFFFFF}; // 30 lowest bits ON
src/gpu/gpu_countmin.cuh:const int cuda_hash_per_hash =
src/gpu/gpu_countmin.cuh:const int cuda_table_rows =
src/gpu/gpu_countmin.cuh:constexpr uint64_t cuda_table_cells = cuda_table_rows * cuda_table_width;
src/gpu/gpu_countmin.cuh:class GPUCountMin {
src/gpu/gpu_countmin.cuh:  GPUCountMin();
src/gpu/gpu_countmin.cuh:  ~GPUCountMin();
src/gpu/gpu_countmin.cuh:  GPUCountMin(const GPUCountMin &) = delete;
src/gpu/gpu_countmin.cuh:  GPUCountMin(GPUCountMin &&) = delete;
src/gpu/device_memory.cuh:#ifdef GPU_AVAILABLE
src/gpu/dist.cu: * PopPUNK dists using CUDA
src/gpu/dist.cu:#include <cuda/barrier>
src/gpu/dist.cu:#include "cuda.cuh"
src/gpu/dist.cu:#include "gpu.hpp"
src/gpu/dist.cu:// CUDA version of bindash dist function (see dist.cpp)
src/gpu/dist.cu:    samebits += __popcll(bits); // CUDA 64-bit popcnt
src/gpu/dist.cu:    // CUDA fast-math intrinsics on floats, which give comparable accuracy
src/gpu/dist.cu:    // https://stackoverflow.com/questions/15468059/copy-to-the-shared-memory-in-cuda)
src/gpu/dist.cu:    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
src/gpu/dist.cu:          cuda::memcpy_async(query_shared + lidx,
src/gpu/dist.cu:        // Would normally break here, but gives no advantage on a GPU as causes
src/gpu/dist.cu:// Get the blockSize and blockCount for CUDA call
src/gpu/dist.cu:        fprintf(stderr, "%cProgress (GPU): %.1lf%%", 13, kern_progress * 100);
src/gpu/dist.cu:  CUDA_CALL(cudaSetDevice(device_id));
src/gpu/dist.cu:  CUDA_CALL(cudaMemGetInfo(&mem_free, &mem_total));
src/gpu/dist.cu:  CUDA_CALL(cudaDeviceGetAttribute(
src/gpu/dist.cu:      &shared_size, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
src/gpu/dist.cu:  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
src/gpu/dist.cu:  CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
src/gpu/dist.cu:  CUDA_CALL(cudaGetLastError());
src/gpu/dist.cu:  fprintf(stderr, "%cProgress (GPU): 100.0%%\n", 13);
src/gpu/dist.cu:  CUDA_CALL(cudaDeviceSynchronize());
src/gpu/dist.cu:// NB cuda graph not needed as API calls faster than ops here
src/gpu/dist.cu:  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
src/gpu/dist.cu:  CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
src/gpu/dist.cu:    CUDA_CALL(cudaHostRegister((void *)ref_sketches[chunk_idx].data(),
src/gpu/dist.cu:                               cudaHostRegisterReadOnly));
src/gpu/dist.cu:  // Four CUDA streams used in loops
src/gpu/dist.cu:  cuda_stream dist_stream, idx_stream, mem_stream, sort_stream;
src/gpu/dist.cu:  fprintf(stderr, "%cProgress (GPU): %.1lf%%", 13, 0.0f);
src/gpu/dist.cu:      fprintf(stderr, "%cProgress (GPU): %.1lf%%", 13, 100 * blocks_done / total_blocks);
src/gpu/dist.cu:  fprintf(stderr, "%cProgress (GPU): 100.0%%\n", 13);
src/gpu/dist.cu:  CUDA_CALL(cudaDeviceSynchronize());
src/gpu/dist.cu:    CUDA_CALL(cudaHostUnregister((void *)ref_sketches[chunk_idx].data()));
src/reference.cpp:// Initialise from GPU sketch
src/api.hpp:#ifdef GPU_AVAILABLE
src/api.hpp:// defined in gpu_api.cpp
src/api.hpp:std::vector<Reference> create_sketches_cuda(const std::string &db_name,
src/api.hpp:NumpyMatrix query_db_cuda(std::vector<Reference> &ref_sketches,
src/api.hpp:sparse_coo query_db_sparse_cuda(std::vector<Reference> &ref_sketches,
src/sketch/seqio.hpp:  // Aligns memory to warp size when using on GPU
src/sketch/sketch.hpp:#ifdef GPU_AVAILABLE
src/sketch/sketch.hpp:class GPUCountMin;
src/sketch/sketch.hpp:sketch_gpu(const std::shared_ptr<SeqBuf> &seq, GPUCountMin &countmin,
src/sketch/sketch.cpp:#include "gpu/gpu.hpp"
src/sketch/sketch.cpp:#ifdef GPU_AVAILABLE
src/sketch/sketch.cpp:sketch_gpu(const std::shared_ptr<SeqBuf> &seq, GPUCountMin &countmin,

```
