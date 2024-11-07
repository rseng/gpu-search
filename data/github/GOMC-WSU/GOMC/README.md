# https://github.com/GOMC-WSU/GOMC

```console
CMake/GOMCCUDASetup.cmake:# Find CUDA is enabled, set it up
CMake/GOMCCUDASetup.cmake:	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G --keep")
CMake/GOMCCUDASetup.cmake:set(GEN_COMP_flag "-DGOMC_CUDA -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT ")
CMake/GOMCCUDASetup.cmake:	message("-- Enabling profiling with NVTX for GPU")
CMake/GOMCCUDASetup.cmake:include_directories(src/GPU)
CMake/GOMCCUDASetup.cmake:set(GPU_NPT_flags "-DENSEMBLE=4 ${GEN_COMP_flag}")
CMake/GOMCCUDASetup.cmake:set(GPU_NPT_name "GOMC_GPU_NPT")
CMake/GOMCCUDASetup.cmake:set(GPU_GC_flags "-DENSEMBLE=3 ${GEN_COMP_flag}")
CMake/GOMCCUDASetup.cmake:set(GPU_GC_name "GOMC_GPU_GCMC")
CMake/GOMCCUDASetup.cmake:set(GPU_GE_flags "-DENSEMBLE=2 ${GEN_COMP_flag}")
CMake/GOMCCUDASetup.cmake:set(GPU_GE_name "GOMC_GPU_GEMC")
CMake/GOMCCUDASetup.cmake:set(GPU_NVT_flags "-DENSEMBLE=1 ${GEN_COMP_flag}")
CMake/GOMCCUDASetup.cmake:set(GPU_NVT_name "GOMC_GPU_NVT")
CMake/GOMCCUDASetup.cmake:set(CMAKE_CUDA_STANDARD 14)
CMake/GOMCCUDASetup.cmake:set(CMAKE_CUDA_STANDARD_REQUIRED true)
CMake/GOMCCUDASetup.cmake:set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CCBIN} -Wno-deprecated-gpu-targets" )
CMake/GOMCCUDASetup.cmake:include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
CMake/GOMCCUDASetup.cmake:if(ENSEMBLE_GPU_NVT)
CMake/GOMCCUDASetup.cmake:    add_executable(GPU_NVT ${cudaSources} ${cudaHeaders}
CMake/GOMCCUDASetup.cmake:    set_target_properties(GPU_NVT PROPERTIES
CMake/GOMCCUDASetup.cmake:        CUDA_SEPARABLE_COMPILATION ON
CMake/GOMCCUDASetup.cmake:        OUTPUT_NAME ${GPU_NVT_name}
CMake/GOMCCUDASetup.cmake:        CUDA_ARCHITECTURES "35;60;70;80"
CMake/GOMCCUDASetup.cmake:        COMPILE_FLAGS "${GPU_NVT_flags}")
CMake/GOMCCUDASetup.cmake:		message("-- Debug build type detected, GPU_NVT setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
CMake/GOMCCUDASetup.cmake:    	set_property(TARGET GPU_NVT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMake/GOMCCUDASetup.cmake:        target_link_libraries(GPU_NVT ws2_32)
CMake/GOMCCUDASetup.cmake:	    target_link_libraries(GPU_NVT ${MPI_LIBRARIES})
CMake/GOMCCUDASetup.cmake:if(ENSEMBLE_GPU_GEMC)
CMake/GOMCCUDASetup.cmake:    add_executable(GPU_GEMC ${cudaSources} ${cudaHeaders} ${sources}
CMake/GOMCCUDASetup.cmake:    set_target_properties(GPU_GEMC PROPERTIES
CMake/GOMCCUDASetup.cmake:        CUDA_SEPARABLE_COMPILATION ON
CMake/GOMCCUDASetup.cmake:        OUTPUT_NAME ${GPU_GE_name}
CMake/GOMCCUDASetup.cmake:        CUDA_ARCHITECTURES "35;60;70;80"
CMake/GOMCCUDASetup.cmake:        COMPILE_FLAGS "${GPU_GE_flags}")
CMake/GOMCCUDASetup.cmake:		message("-- Debug build type detected, GPU_GEMC setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
CMake/GOMCCUDASetup.cmake:    	set_property(TARGET GPU_GEMC PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMake/GOMCCUDASetup.cmake:        target_link_libraries(GPU_GEMC ws2_32)
CMake/GOMCCUDASetup.cmake:	    target_link_libraries(GPU_GEMC ${MPI_LIBRARIES})
CMake/GOMCCUDASetup.cmake:if(ENSEMBLE_GPU_GCMC)
CMake/GOMCCUDASetup.cmake:    add_executable(GPU_GCMC ${cudaSources} ${cudaHeaders} ${sources}
CMake/GOMCCUDASetup.cmake:    set_target_properties(GPU_GCMC PROPERTIES
CMake/GOMCCUDASetup.cmake:        CUDA_SEPARABLE_COMPILATION ON
CMake/GOMCCUDASetup.cmake:        OUTPUT_NAME ${GPU_GC_name}
CMake/GOMCCUDASetup.cmake:        CUDA_ARCHITECTURES "35;60;70;80"
CMake/GOMCCUDASetup.cmake:        COMPILE_FLAGS "${GPU_GC_flags}")
CMake/GOMCCUDASetup.cmake:		message("-- Debug build type detected, GPU_GCMC setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
CMake/GOMCCUDASetup.cmake:    	set_property(TARGET GPU_GCMC PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMake/GOMCCUDASetup.cmake:        target_link_libraries(GPU_GCMC ws2_32)
CMake/GOMCCUDASetup.cmake:	    target_link_libraries(GPU_GCMC ${MPI_LIBRARIES})
CMake/GOMCCUDASetup.cmake:if(ENSEMBLE_GPU_NPT)
CMake/GOMCCUDASetup.cmake:    add_executable(GPU_NPT ${cudaSources} ${cudaHeaders} ${sources}
CMake/GOMCCUDASetup.cmake:    set_target_properties(GPU_NPT PROPERTIES
CMake/GOMCCUDASetup.cmake:        CUDA_SEPARABLE_COMPILATION ON
CMake/GOMCCUDASetup.cmake:        OUTPUT_NAME ${GPU_NPT_name}
CMake/GOMCCUDASetup.cmake:        CUDA_ARCHITECTURES "35;60;70;80"
CMake/GOMCCUDASetup.cmake:        COMPILE_FLAGS "${GPU_NPT_flags}")
CMake/GOMCCUDASetup.cmake:		message("-- Debug build type detected, GPU_NPT setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
CMake/GOMCCUDASetup.cmake:    	set_property(TARGET GPU_NPT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CMake/GOMCCUDASetup.cmake:        target_link_libraries(GPU_NPT ws2_32)
CMake/GOMCCUDASetup.cmake:	    target_link_libraries(GPU_NPT ${MPI_LIBRARIES})
CMake/FileLists.cmake:set(cudaHeaders
CMake/FileLists.cmake:    src/GPU/ConstantDefinitionsCUDAKernel.cuh
CMake/FileLists.cmake:    src/GPU/CalculateMinImageCUDAKernel.cuh
CMake/FileLists.cmake:    src/GPU/CalculateEnergyCUDAKernel.cuh
CMake/FileLists.cmake:    src/GPU/CalculateForceCUDAKernel.cuh
CMake/FileLists.cmake:    src/GPU/CalculateEwaldCUDAKernel.cuh
CMake/FileLists.cmake:    src/GPU/CUDAMemoryManager.cuh
CMake/FileLists.cmake:    src/GPU/TransformParticlesCUDAKernel.cuh
CMake/FileLists.cmake:    src/GPU/VariablesCUDA.cuh)
CMake/FileLists.cmake:set(cudaSources
CMake/FileLists.cmake:    src/GPU/CalculateEnergyCUDAKernel.cu
CMake/FileLists.cmake:    src/GPU/CalculateForceCUDAKernel.cu
CMake/FileLists.cmake:    src/GPU/CalculateEwaldCUDAKernel.cu
CMake/FileLists.cmake:    src/GPU/ConstantDefinitionsCUDAKernel.cu
CMake/FileLists.cmake:    src/GPU/CUDAMemoryManager.cu
CMake/FileLists.cmake:    src/GPU/TransformParticlesCUDAKernel.cu)
CMake/FileLists.cmake:source_group("CUDA Header Files" FILES ${cudaHeaders})
CMake/FileLists.cmake:source_group("CUDA Source Files" FILES ${cudaSources})
test/Run_Examples.py:GPU_binaries_dict = {}
test/Run_Examples.py:binaries_gpu_new = sorted(glob.glob('GOMC_GPU_*'), key=os.path.getmtime)
test/Run_Examples.py:for binary in binaries_gpu_new:
test/Run_Examples.py:binaries_gpu_ref = sorted(glob.glob('GOMC_GPU_*'), key=os.path.getmtime)
test/Run_Examples.py:for binary in binaries_gpu_ref:
test/Run_Examples.py:            cpuOrGpu = ""
test/Run_Examples.py:                cpuOrGpu = "CPU_"
test/Run_Examples.py:                cpuOrGpu = "CPU_"
test/Run_Examples.py:            elif "new_gpu" in path:
test/Run_Examples.py:                cpuOrGpu = "GPU_"
test/Run_Examples.py:            elif "ref_gpu" in path:
test/Run_Examples.py:                cpuOrGpu = "GPU_"
test/Run_Examples.py:            if cpuOrGpu+"NVT"+newOrRef in binaries_dict and 'NVT' in path and 'NVT_GEMC' not in path:
test/Run_Examples.py:                command = binaries_dict[cpuOrGpu+"NVT"+newOrRef],os.path.abspath(root),cpuOrGpu+"NVT"+newOrRef,cpuOrGpu,newOrRef,"NVT_"+os.path.basename(root)
test/Run_Examples.py:            elif cpuOrGpu+"NPT"+newOrRef in binaries_dict and 'NPT' in path and 'NPT_GEMC' not in path:
test/Run_Examples.py:                command = binaries_dict[cpuOrGpu+"NPT"+newOrRef],os.path.abspath(root),cpuOrGpu+"NPT"+newOrRef,cpuOrGpu,newOrRef,"NPT_"+os.path.basename(root)
test/Run_Examples.py:            elif cpuOrGpu+"GCMC"+newOrRef in binaries_dict and 'GCMC' in path:
test/Run_Examples.py:                command = binaries_dict[cpuOrGpu+"GCMC"+newOrRef],os.path.abspath(root),cpuOrGpu+"GCMC"+newOrRef,cpuOrGpu,newOrRef,"GCMC_"+os.path.basename(root)
test/Run_Examples.py:            elif cpuOrGpu+"GEMC"+newOrRef in binaries_dict and 'NVT_GEMC' in path:
test/Run_Examples.py:                command = binaries_dict[cpuOrGpu+"GEMC"+newOrRef],os.path.abspath(root),cpuOrGpu+"NVT_GEMC"+newOrRef,cpuOrGpu,newOrRef,"NVT_GEMC_"+os.path.basename(root)
test/Run_Examples.py:            elif cpuOrGpu+"GEMC"+newOrRef in binaries_dict and 'NPT_GEMC' in path:
test/Run_Examples.py:                command = binaries_dict[cpuOrGpu+"GEMC"+newOrRef],os.path.abspath(root),cpuOrGpu+"NPT_GEMC"+newOrRef,cpuOrGpu,newOrRef,"NPT_GEMC_"+os.path.basename(root)
test/Run_Examples.py:colNames = ['PathToBinary', 'PathToExample', 'Binary', 'CPU_or_GPU','New_or_Ref','Example']
test/Run_Examples.py:CPU_v_GPU_global = True
test/Run_Examples.py:CPU_v_GPU_exists_global = False
test/Run_Examples.py:    CPU_v_GPU = True
test/Run_Examples.py:    CPU_v_GPU_exists = False
test/Run_Examples.py:            if ((row['CPU_or_GPU_x'] != row['CPU_or_GPU_y']) and (row['New_or_Ref_x'] == row['New_or_Ref_y'])):
test/Run_Examples.py:                CPU_v_GPU_exists = True
test/Run_Examples.py:                CPU_v_GPU_exists_global = True
test/Run_Examples.py:                CPU_v_GPU = CPU_v_GPU and result
test/Run_Examples.py:                CPU_v_GPU_global = CPU_v_GPU_global and result
test/Run_Examples.py:            elif ((row['CPU_or_GPU_x'] == row['CPU_or_GPU_y']) and (row['New_or_Ref_x'] != row['New_or_Ref_y'])):
test/Run_Examples.py:            elif ((row['CPU_or_GPU_x'] != row['CPU_or_GPU_y']) and (row['New_or_Ref_x'] != row['New_or_Ref_y'])):
test/Run_Examples.py:    if(CPU_v_GPU_exists):
test/Run_Examples.py:        if(CPU_v_GPU):
test/Run_Examples.py:            Log_Template_file.write("---------{}\n".format("CPU_v_GPU: "+ OKGREEN + "PASS" + ENDC))
test/Run_Examples.py:            Log_Template_file.write("---------{}\n".format("CPU_v_GPU: "+ FAIL + "FAIL" + ENDC))
test/Run_Examples.py:            Log_Template_file.write("---------{}\n".format("CPU v GPU X New v Ref: "+ OKGREEN + "PASS" + ENDC))
test/Run_Examples.py:            Log_Template_file.write("---------{}\n".format("CPU v GPU X New v Ref: "+ FAIL + "FAIL" + ENDC))
test/Run_Examples.py:if(CPU_v_GPU_exists_global):
test/Run_Examples.py:    if(CPU_v_GPU_global):
test/Run_Examples.py:        Log_Template_file.write(str("CPU_v_GPU Global: "+ OKGREEN + "PASS" + ENDC))
test/Run_Examples.py:        Log_Template_file.write(str("CPU_v_GPU Global: "+ FAIL + "FAIL" + ENDC))
test/Run_Examples.py:        Log_Template_file.write(str("CPU vs GPU X New vs Ref Global: "+ OKGREEN + "PASS" + ENDC))
test/Run_Examples.py:        Log_Template_file.write(str("CPU vs GPU X New vs Ref Global: "+ FAIL + "FAIL" + ENDC))
test/WriteLog.py:GPU_binaries_dict = {}
test/WriteLog.py:binaries_gpu_new = sorted(glob.glob('GOMC_GPU_*'), key=os.path.getmtime)
test/WriteLog.py:for binary in binaries_gpu_new:
test/WriteLog.py:binaries_gpu_ref = sorted(glob.glob('GOMC_GPU_*'), key=os.path.getmtime)
test/WriteLog.py:for binary in binaries_gpu_ref:
test/WriteLog.py:            cpuOrGpu = ""
test/WriteLog.py:                cpuOrGpu = "CPU_"
test/WriteLog.py:                cpuOrGpu = "CPU_"
test/WriteLog.py:            elif "new_gpu" in path:
test/WriteLog.py:                cpuOrGpu = "GPU_"
test/WriteLog.py:            elif "ref_gpu" in path:
test/WriteLog.py:                cpuOrGpu = "GPU_"
test/WriteLog.py:            if cpuOrGpu+"NVT"+newOrRef in binaries_dict and 'NVT' in path and 'NVT_GEMC' not in path:
test/WriteLog.py:                command = binaries_dict[cpuOrGpu+"NVT"+newOrRef],os.path.abspath(root),cpuOrGpu+"NVT"+newOrRef,cpuOrGpu,newOrRef,"NVT_"+os.path.basename(root)
test/WriteLog.py:            elif cpuOrGpu+"NPT"+newOrRef in binaries_dict and 'NPT' in path and 'NPT_GEMC' not in path:
test/WriteLog.py:                command = binaries_dict[cpuOrGpu+"NPT"+newOrRef],os.path.abspath(root),cpuOrGpu+"NPT"+newOrRef,cpuOrGpu,newOrRef,"NPT_"+os.path.basename(root)
test/WriteLog.py:            elif cpuOrGpu+"GCMC"+newOrRef in binaries_dict and 'GCMC' in path:
test/WriteLog.py:                command = binaries_dict[cpuOrGpu+"GCMC"+newOrRef],os.path.abspath(root),cpuOrGpu+"GCMC"+newOrRef,cpuOrGpu,newOrRef,"GCMC_"+os.path.basename(root)
test/WriteLog.py:            elif cpuOrGpu+"GEMC"+newOrRef in binaries_dict and 'NVT_GEMC' in path:
test/WriteLog.py:                command = binaries_dict[cpuOrGpu+"GEMC"+newOrRef],os.path.abspath(root),cpuOrGpu+"NVT_GEMC"+newOrRef,cpuOrGpu,newOrRef,"NVT_GEMC_"+os.path.basename(root)
test/WriteLog.py:            elif cpuOrGpu+"GEMC"+newOrRef in binaries_dict and 'NPT_GEMC' in path:
test/WriteLog.py:                command = binaries_dict[cpuOrGpu+"GEMC"+newOrRef],os.path.abspath(root),cpuOrGpu+"NPT_GEMC"+newOrRef,cpuOrGpu,newOrRef,"NPT_GEMC_"+os.path.basename(root)
test/WriteLog.py:colNames = ['PathToBinary', 'PathToExample', 'Binary', 'CPU_or_GPU','New_or_Ref','Example']
test/WriteLog.py:CPU_v_GPU_global = True
test/WriteLog.py:CPU_v_GPU_exists_global = False
test/WriteLog.py:    CPU_v_GPU = True
test/WriteLog.py:    CPU_v_GPU_exists = False
test/WriteLog.py:            if ((row['CPU_or_GPU_x'] != row['CPU_or_GPU_y']) and (row['New_or_Ref_x'] == row['New_or_Ref_y'])):
test/WriteLog.py:                CPU_v_GPU_exists = True
test/WriteLog.py:                CPU_v_GPU_exists_global = True
test/WriteLog.py:                CPU_v_GPU = CPU_v_GPU and result
test/WriteLog.py:                CPU_v_GPU_global = CPU_v_GPU_global and result
test/WriteLog.py:            elif ((row['CPU_or_GPU_x'] == row['CPU_or_GPU_y']) and (row['New_or_Ref_x'] != row['New_or_Ref_y'])):
test/WriteLog.py:            elif ((row['CPU_or_GPU_x'] != row['CPU_or_GPU_y']) and (row['New_or_Ref_x'] != row['New_or_Ref_y'])):
test/WriteLog.py:    if(CPU_v_GPU_exists):
test/WriteLog.py:        if(CPU_v_GPU):
test/WriteLog.py:            Log_Template_file.write("---------{}\n".format("CPU_v_GPU: "+ OKGREEN + "PASS" + ENDC))
test/WriteLog.py:            Log_Template_file.write("---------{}\n".format("CPU_v_GPU: "+ FAIL + "FAIL" + ENDC))
test/WriteLog.py:            Log_Template_file.write("---------{}\n".format("CPU v GPU X New v Ref: "+ OKGREEN + "PASS" + ENDC))
test/WriteLog.py:            Log_Template_file.write("---------{}\n".format("CPU v GPU X New v Ref: "+ FAIL + "FAIL" + ENDC))
test/WriteLog.py:if(CPU_v_GPU_exists_global):
test/WriteLog.py:    if(CPU_v_GPU_global):
test/WriteLog.py:        Log_Template_file.write(str("CPU_v_GPU Global: "+ OKGREEN + "PASS" + ENDC))
test/WriteLog.py:        Log_Template_file.write(str("CPU_v_GPU Global: "+ FAIL + "FAIL" + ENDC))
test/WriteLog.py:        Log_Template_file.write(str("CPU vs GPU X New vs Ref Global: "+ OKGREEN + "PASS" + ENDC))
test/WriteLog.py:        Log_Template_file.write(str("CPU vs GPU X New vs Ref Global: "+ FAIL + "FAIL" + ENDC))
test/Setup_Examples.sh:mkdir integration/new_gpu
test/Setup_Examples.sh:cp -frd GOMC_Examples integration/new_gpu
test/Setup_Examples.sh:mkdir integration/ref_gpu
test/Setup_Examples.sh:cp -frd GOMC_Examples integration/ref_gpu
test/BuildGPUTests.cmake:# Find CUDA is enabled, set it up
test/BuildGPUTests.cmake:	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G --keep -lineinfo")
test/BuildGPUTests.cmake:set(GEN_COMP_flag "-DGOMC_CUDA -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT ")
test/BuildGPUTests.cmake:	message("-- Enabling profiling with NVTX for GPU")
test/BuildGPUTests.cmake:include_directories(src/GPU)
test/BuildGPUTests.cmake:set(GPU_NPT_flags "-DENSEMBLE=4 ${GEN_COMP_flag}")
test/BuildGPUTests.cmake:set(GPU_NPT_name "GOMC_GPU_NPT")
test/BuildGPUTests.cmake:set(GPU_GC_flags "-DENSEMBLE=3 ${GEN_COMP_flag}")
test/BuildGPUTests.cmake:set(GPU_GC_name "GOMC_GPU_GCMC")
test/BuildGPUTests.cmake:set(GPU_GE_flags "-DENSEMBLE=2 ${GEN_COMP_flag}")
test/BuildGPUTests.cmake:set(GPU_GE_name "GOMC_GPU_GEMC")
test/BuildGPUTests.cmake:set(GPU_NVT_flags "-DENSEMBLE=1 ${GEN_COMP_flag}")
test/BuildGPUTests.cmake:set(GPU_NVT_name "GOMC_GPU_NVT")
test/BuildGPUTests.cmake:set(CMAKE_CUDA_STANDARD 14)
test/BuildGPUTests.cmake:set(CMAKE_CUDA_STANDARD_REQUIRED true)
test/BuildGPUTests.cmake:set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CCBIN} -Wno-deprecated-gpu-targets")
test/BuildGPUTests.cmake:include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
test/BuildGPUTests.cmake:function(add_GPU_NVT_test name)
test/BuildGPUTests.cmake:    add_executable(${name} ${cudaSources} ${cudaHeaders} ${GOMCHeaders} ${GOMCSources} ${libHeaders} ${libSources}
test/BuildGPUTests.cmake:        CUDA_SEPARABLE_COMPILATION ON
test/BuildGPUTests.cmake:        CUDA_ARCHITECTURES 35 60 70
test/BuildGPUTests.cmake:        COMPILE_FLAGS "${GPU_NVT_flags}")
test/BuildGPUTests.cmake:		message("-- Debug build type detected, ${name} setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
test/BuildGPUTests.cmake:    	set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
test/BuildGPUTests.cmake:endfunction(add_GPU_NVT_test)
test/BuildGPUTests.cmake:function(add_GPU_NPT_test name)
test/BuildGPUTests.cmake:    add_executable(${name} ${cudaSources} ${cudaHeaders} ${GOMCHeaders} ${GOMCSources} ${libHeaders} ${libSources}
test/BuildGPUTests.cmake:        CUDA_SEPARABLE_COMPILATION ON
test/BuildGPUTests.cmake:        CUDA_ARCHITECTURES 35 60 70
test/BuildGPUTests.cmake:        COMPILE_FLAGS "${GPU_NPT_flags}")
test/BuildGPUTests.cmake:		message("-- Debug build type detected, ${name} setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
test/BuildGPUTests.cmake:    	set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
test/BuildGPUTests.cmake:endfunction(add_GPU_NPT_test)
test/BuildGPUTests.cmake:function(add_GPU_GCMC_test name)
test/BuildGPUTests.cmake:    add_executable(${name} ${cudaSources} ${cudaHeaders} ${GOMCHeaders} ${GOMCSources} ${libHeaders} ${libSources}
test/BuildGPUTests.cmake:        CUDA_SEPARABLE_COMPILATION ON
test/BuildGPUTests.cmake:        CUDA_ARCHITECTURES 35 60 70
test/BuildGPUTests.cmake:        COMPILE_FLAGS "${GPU_GC_flags}")
test/BuildGPUTests.cmake:		message("-- Debug build type detected, ${name} setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
test/BuildGPUTests.cmake:    	set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
test/BuildGPUTests.cmake:endfunction(add_GPU_GCMC_test)
test/BuildGPUTests.cmake:function(add_GPU_GEMC_test name)
test/BuildGPUTests.cmake:    add_executable(${name} ${cudaSources} ${cudaHeaders} ${GOMCHeaders} ${GOMCSources} ${libHeaders} ${libSources}
test/BuildGPUTests.cmake:        CUDA_SEPARABLE_COMPILATION ON
test/BuildGPUTests.cmake:        CUDA_ARCHITECTURES 35 60 70
test/BuildGPUTests.cmake:        COMPILE_FLAGS "${GPU_GE_flags}")
test/BuildGPUTests.cmake:		message("-- Debug build type detected, ${name} setting CUDA_RESOLVE_DEVICE_SYMBOLS ON")
test/BuildGPUTests.cmake:    	set_property(TARGET ${name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
test/BuildGPUTests.cmake:endfunction(add_GPU_GEMC_test)
test/BuildGPUTests.cmake:add_GPU_NVT_test(GOMC_GPU_NVT_Test)
test/BuildGPUTests.cmake:add_GPU_NPT_test(GOMC_GPU_NPT_Test)
test/BuildGPUTests.cmake:add_GPU_GCMC_test(GOMC_GPU_GCMC_Test)
test/BuildGPUTests.cmake:add_GPU_GEMC_test(GOMC_GPU_GEMC_Test)
test/FileList.cmake:set(cudaHeaders
test/FileList.cmake:    src/GPU/ConstantDefinitionsCUDAKernel.cuh
test/FileList.cmake:    src/GPU/CalculateMinImageCUDAKernel.cuh
test/FileList.cmake:    src/GPU/CalculateEnergyCUDAKernel.cuh
test/FileList.cmake:    src/GPU/CalculateForceCUDAKernel.cuh
test/FileList.cmake:    src/GPU/CalculateEwaldCUDAKernel.cuh
test/FileList.cmake:    src/GPU/CUDAMemoryManager.cuh
test/FileList.cmake:    src/GPU/TransformParticlesCUDAKernel.cuh
test/FileList.cmake:    src/GPU/VariablesCUDA.cuh)
test/FileList.cmake:set(cudaSources
test/FileList.cmake:    src/GPU/CalculateEnergyCUDAKernel.cu
test/FileList.cmake:    src/GPU/CalculateForceCUDAKernel.cu
test/FileList.cmake:    src/GPU/CalculateEwaldCUDAKernel.cu
test/FileList.cmake:    src/GPU/ConstantDefinitionsCUDAKernel.cu
test/FileList.cmake:    src/GPU/CUDAMemoryManager.cu
test/FileList.cmake:    src/GPU/TransformParticlesCUDAKernel.cu)
test/FileList.cmake:source_group("CUDA Header Files" FILES ${cudaHeaders})
test/FileList.cmake:source_group("CUDA Source Files" FILES ${cudaSources})
test/src/CheckpointTest.cpp:#if !GOMC_CUDA
test/src/CheckpointTest.cpp:#if !GOMC_CUDA
test/src/CheckpointTest.cpp:#if !GOMC_CUDA
test/src/CheckpointTest.cpp:#if !GOMC_CUDA
test/src/CheckpointTest.cpp:#if !GOMC_CUDA
test/GoogleTest.cmake:# Find if CUDA exists and what is the version number
test/GoogleTest.cmake:check_language(CUDA)
test/GoogleTest.cmake:  if (CMAKE_CUDA_COMPILER)
test/GoogleTest.cmake:    include(${PROJECT_SOURCE_DIR}/test/BuildGPUTests.cmake)
CHANGELOG.md:+ This release forcues on GPU performance improvements
CHANGELOG.md:+ Using Random123 to be able to generate the same random number on GPU and port Translation and Rotation of molecules to the GPU
CHANGELOG.md:+ Upgraded our CMake to 3.8 and use built-in CUDA support
CHANGELOG.md:+ Changed the way we calculate pair interactions. The new approach allows us to optimize GPU by reducing the amount of cudaMemcpy needed.
CHANGELOG.md:+ GPU memory management class to watch the allocation and deallocation of GPU memories.
CHANGELOG.md:+ Bug fixes related to lambda functionalities on GPU
CHANGELOG.md:+ Added support for new GPU architectures 7.0 and 7.5
CHANGELOG.md:+ Added an error message when GPU version was used but there was no GPU present (#126)
CHANGELOG.md:+ Fixed to the bug in GPU pressure calculation for NPT simulation.
CHANGELOG.md:+ Removed compute_20 and compute_21 since they are depricated and CUDA 9 will generate fatal error. 
CHANGELOG.md:+ Fixed the bug where GPU and CPU total energy calculations where slighty different.
CHANGELOG.md:+ GPU implementation has been added to the project
CHANGELOG.md:+ Added Cell list optimization to the GPU and the Serial code.
CHANGELOG.md:+ Fixes for PDB output for NVT in the GPU code.
README.md:# GOMC - GPU Optimized Monte Carlo
README.md:  ./GOMC_<CPU|GPU>_XXXX +p4 in.conf
README.md:  If you wish to utilize NVIDIA graphic cards you will need to install NVIDIA toolkit before compiling. The metamake file will automatically detect the location of CUDA installation. (More info in Manual)
CMakeLists.txt:include_directories(src/GPU)
CMakeLists.txt:set(ENSEMBLE_GPU_NVT ON CACHE BOOL "Build GPU NVT version")
CMakeLists.txt:set(ENSEMBLE_GPU_GEMC ON CACHE BOOL "Build GPU GEMC version")
CMakeLists.txt:set(ENSEMBLE_GPU_GCMC ON CACHE BOOL "Build GPU GCMC version")
CMakeLists.txt:set(ENSEMBLE_GPU_NPT ON CACHE BOOL "Build GPU NPT version")
CMakeLists.txt:# Find if CUDA exists and what is the version number
CMakeLists.txt:check_language(CUDA)
CMakeLists.txt:if (CMAKE_CUDA_COMPILER)
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    include(${PROJECT_SOURCE_DIR}/CMake/GOMCCUDASetup.cmake)
metamakeMPI.sh:use_cuda=0
metamakeMPI.sh:        nvcc_version=($(python scripts/get_cuda_version.py))
metamakeMPI.sh:	use_cuda=1
metamakeMPI.sh:        # Check cuda version, if less than 11 then download CUB, otherwise skip
metamakeMPI.sh:                echo "CUDA version is 11.0 or higher, no need to download CUB library! Skipping..."
metamakeMPI.sh:    if (( $use_cuda )); then
metamakeMPI.sh:      	echo "Enabling NVTX profiling for CUDA "
metamakeMPI.sh:      	echo "Warning: Cannot enable NVTX profiling without CUDA enabled."
metamake.sh:use_cuda=0
metamake.sh:	use_cuda=1
metamake.sh:	nvcc_version=($(python scripts/get_cuda_version.py))
metamake.sh:		nvcc_version=($(python3 scripts/get_cuda_version.py))
metamake.sh:	# Check cuda version, if less than 11 then download CUB, otherwise skip
metamake.sh:		echo "CUDA version is 11.0 or higher, no need to download CUB library! Skipping..."
metamake.sh:        NVT|NPT|GCMC|GEMC|GPU_NVT|GPU_NPT|GPU_GCMC|GPU_GEMC)                   # or just:  -t|--t*)
metamake.sh:                echo 'Valid Options: {NVT|NPT|GCMC|GEMC|GPU_NVT|GPU_NPT|GPU_GCMC|GPU_GEMC}' >&2
metamake.sh:		if(( use_cuda ))
metamake.sh:        	ENSEMBLES+="GOMC_GPU_NVT_MPI_Test "
metamake.sh:        	ENSEMBLES+="GOMC_GPU_NPT_MPI_Test "
metamake.sh:        	ENSEMBLES+="GOMC_GPU_GCMC_MPI_Test "
metamake.sh:        	ENSEMBLES+="GOMC_GPU_GEMC_MPI_Test "
metamake.sh:		if(( use_cuda ))
metamake.sh:        	ENSEMBLES+="GOMC_GPU_NVT_Test "
metamake.sh:        	ENSEMBLES+="GOMC_GPU_NPT_Test "
metamake.sh:        	ENSEMBLES+="GOMC_GPU_GCMC_Test "
metamake.sh:        	ENSEMBLES+="GOMC_GPU_GEMC_Test "
metamake.sh:    if (( use_cuda )); then
metamake.sh:      	echo "Enabling NVTX profiling for CUDA "
metamake.sh:      	echo "Warning: Cannot enable NVTX profiling without CUDA enabled."
lib/Random123/features/iccfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/iccfeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/iccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/iccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/iccfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/iccfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
lib/Random123/features/sunprofeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/sunprofeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/sunprofeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/sunprofeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/sunprofeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/sunprofeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
lib/Random123/features/nvccfeatures.h:#if !defined(CUDART_VERSION)
lib/Random123/features/nvccfeatures.h:#error "why are we in nvccfeatures.h if CUDART_VERSION is not defined"
lib/Random123/features/nvccfeatures.h:#if CUDART_VERSION < 4010
lib/Random123/features/nvccfeatures.h:#error "CUDA versions earlier than 4.1 produce incorrect results for some templated functions in namespaces.  Random123 isunsupported.  See comments in nvccfeatures.h"
lib/Random123/features/nvccfeatures.h:// T=uint64_t in examples/uniform.hpp produces -1 for CUDA4.0 and
lib/Random123/features/nvccfeatures.h:// Thus, we no longer trust CUDA versions earlier than 4.1 even though
lib/Random123/features/nvccfeatures.h:// we had previously tested and timed Random123 with CUDA 3.x and 4.0.
lib/Random123/features/nvccfeatures.h://#ifdef  __CUDA_ARCH__ allows Philox32 and Philox64 to be compiled
lib/Random123/features/nvccfeatures.h://for both device and host functions in CUDA by setting compiler flags
lib/Random123/features/nvccfeatures.h:#ifdef  __CUDA_ARCH__
lib/Random123/features/nvccfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/nvccfeatures.h:#define R123_CUDA_DEVICE __device__
lib/Random123/features/nvccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/nvccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 1
lib/Random123/features/nvccfeatures.h:// No exceptions in CUDA, at least upto 4.0
lib/Random123/features/nvccfeatures.h:#else // ! __CUDA_ARCH__
lib/Random123/features/nvccfeatures.h:// If we're using nvcc not compiling for the CUDA architecture,
lib/Random123/features/nvccfeatures.h:#endif // __CUDA_ARCH__
lib/Random123/features/compilerfeatures.h:The Random123 library is portable across C, C++, CUDA, OpenCL environments,
lib/Random123/features/compilerfeatures.h:         MULHILO64_CUDA_INTRIN
lib/Random123/features/compilerfeatures.h:         MULHILO64_OPENCL_INTRIN
lib/Random123/features/compilerfeatures.h:(i.e. u01_*_53()), e.g. on OpenCL without the cl_khr_fp64 extension.
lib/Random123/features/compilerfeatures.h:<li>R123_CUDA_DEVICE - which expands to __device__ (or something else with
lib/Random123/features/compilerfeatures.h:  sufficiently similar semantics) when CUDA is in use, and expands
lib/Random123/features/compilerfeatures.h:  call assert (I'm looking at you, CUDA and OpenCL), or even include
lib/Random123/features/compilerfeatures.h:  assert.h safely (OpenCL).
lib/Random123/features/compilerfeatures.h:  is not available, e.g., MSVC and OpenCL.
lib/Random123/features/compilerfeatures.h:#elif defined(__OPENCL_VERSION__) && __OPENCL_VERSION__ > 0
lib/Random123/features/compilerfeatures.h:#include "openclfeatures.h"
lib/Random123/features/compilerfeatures.h:#elif defined(__CUDACC__)
lib/Random123/features/compilerfeatures.h:#define R123_USE_PHILOX_64BIT (R123_USE_64BIT && (R123_USE_MULHILO64_ASM || R123_USE_MULHILO64_MSVC_INTRIN || R123_USE_MULHILO64_CUDA_INTRIN || R123_USE_GNU_UINT128 || R123_USE_MULHILO64_C99 || R123_USE_MULHILO64_OPENCL_INTRIN || R123_USE_MULHILO64_MULHI_INTRIN))
lib/Random123/features/pgccfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/pgccfeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/pgccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/pgccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/pgccfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/pgccfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
lib/Random123/features/msvcfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/msvcfeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/msvcfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/msvcfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/msvcfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/msvcfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
lib/Random123/features/openclfeatures.h:#ifndef __openclfeatures_dot_hpp
lib/Random123/features/openclfeatures.h:#define __openclfeatures_dot_hpp
lib/Random123/features/openclfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/openclfeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/openclfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/openclfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/openclfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/openclfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 1
lib/Random123/features/xlcfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/xlcfeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/xlcfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/xlcfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/xlcfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/xlcfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
lib/Random123/features/gccfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/gccfeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/gccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/gccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/gccfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/gccfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
lib/Random123/features/metalfeatures.h:#ifndef R123_CUDA_DEVICE
lib/Random123/features/metalfeatures.h:#define R123_CUDA_DEVICE
lib/Random123/features/metalfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/features/metalfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
lib/Random123/features/metalfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/features/metalfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
lib/Random123/uniform.hpp:#if defined(__CUDACC__) || defined(_LIBCPP_HAS_NO_CONSTEXPR)
lib/Random123/uniform.hpp:// Amazing! cuda thinks numeric_limits::max() is a __host__ function, so
lib/Random123/uniform.hpp:R123_CONSTEXPR R123_STATIC_INLINE R123_CUDA_DEVICE T maxTvalue(){
lib/Random123/uniform.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE Ftype u01(Itype in){
lib/Random123/uniform.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE Ftype uneg11(Itype in){
lib/Random123/uniform.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE Ftype u01fixedpt(Itype in){
lib/Random123/u01fixedpt.h:    artifacts may exist on some GPU hardware.  The tests in
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE float u01fixedpt_closed_closed_32_float(uint32_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE float u01fixedpt_closed_open_32_float(uint32_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE float u01fixedpt_open_closed_32_float(uint32_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE float u01fixedpt_open_open_32_float(uint32_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_closed_closed_64_double(uint64_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_closed_open_64_double(uint64_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_open_closed_64_double(uint64_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_open_open_64_double(uint64_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_closed_closed_32_double(uint32_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_closed_open_32_double(uint32_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_open_closed_32_double(uint32_t i){
lib/Random123/u01fixedpt.h:R123_CUDA_DEVICE R123_STATIC_INLINE double u01fixedpt_open_open_32_double(uint32_t i){
lib/Random123/boxmuller.hpp:// on GPUs, these functions are remarkably fast, which makes
lib/Random123/boxmuller.hpp:// Box-Muller the fastest GRV generator we know of on GPUs.
lib/Random123/boxmuller.hpp:// namespace structures in CUDA.
lib/Random123/boxmuller.hpp:#if !defined(__CUDACC__)
lib/Random123/boxmuller.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE void sincosf(float x, float *s, float *c) {
lib/Random123/boxmuller.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE void sincos(double x, double *s, double *c) {
lib/Random123/boxmuller.hpp:#if !defined(CUDART_VERSION) || CUDART_VERSION < 5000 /* enabled if sincospi and sincospif are not in math lib */
lib/Random123/boxmuller.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE void sincospif(float x, float *s, float *c){
lib/Random123/boxmuller.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE void sincospi(double x, double *s, double *c) {
lib/Random123/boxmuller.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE float2 boxmuller(uint32_t u0, uint32_t u1) {
lib/Random123/boxmuller.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE double2 boxmuller(uint64_t u0, uint64_t u1) {
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){ \
lib/Random123/philox.h:   happens that CUDA was the first time we used the idiom. */
lib/Random123/philox.h:#define _mulhilo_cuda_intrin_tpl(W, Word, INTRIN)                       \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, R123_METAL_THREAD_ADDRESS_SPACE Word* hip){ \
lib/Random123/philox.h:_mulhilo_cuda_intrin_tpl(32, uint32_t, R123_MULHILO32_MULHI_INTRIN)
lib/Random123/philox.h:#elif R123_USE_MULHILO64_CUDA_INTRIN
lib/Random123/philox.h:_mulhilo_cuda_intrin_tpl(64, uint64_t, __umul64hi)
lib/Random123/philox.h:#elif R123_USE_MULHILO64_OPENCL_INTRIN
lib/Random123/philox.h:_mulhilo_cuda_intrin_tpl(64, uint64_t, mul_hi)
lib/Random123/philox.h:_mulhilo_cuda_intrin_tpl(64, uint64_t, R123_MULHILO64_MULHI_INTRIN)
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key)); \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key){ \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array1x##W _philox2x##W##bumpkey( struct r123array1x##W key) { \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key)); \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key){ \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox4x##W##bumpkey( struct r123array2x##W key) { \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_key_t philox##N##x##W##keyinit(philox##N##x##W##_ukey_t uk) { return uk; } \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key)); \
lib/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key) { \
lib/Random123/philox.h:    inline R123_CUDA_DEVICE R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key) const){ \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(uint64_t RotL_64(uint64_t x, unsigned int N));
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE uint64_t RotL_64(uint64_t x, unsigned int N)
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(uint32_t RotL_32(uint32_t x, unsigned int N));
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE uint32_t RotL_32(uint32_t x, unsigned int N)
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE threefry2x##W##_key_t threefry2x##W##keyinit(threefry2x##W##_ukey_t uk) { return uk; } \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry2x##W##_ctr_t threefry2x##W##_R(unsigned int Nrounds, threefry2x##W##_ctr_t in, threefry2x##W##_key_t k)); \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                          \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry2x##W##_ctr_t threefry2x##W(threefry2x##W##_ctr_t in, threefry2x##W##_key_t k)); \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                     \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE threefry4x##W##_key_t threefry4x##W##keyinit(threefry4x##W##_ukey_t uk) { return uk; } \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry4x##W##_ctr_t threefry4x##W##_R(unsigned int Nrounds, threefry4x##W##_ctr_t in, threefry4x##W##_key_t k)); \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                          \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry4x##W##_ctr_t threefry4x##W(threefry4x##W##_ctr_t in, threefry4x##W##_key_t k)); \
lib/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                     \
lib/Random123/threefry.h:   inline R123_CUDA_DEVICE R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key)){ \
lib/Random123/array.h:    When compiling with __CUDA_ARCH__ defined, the reverse iterator
lib/Random123/array.h:    CUDA does not support std::reverse_iterator.
lib/Random123/array.h:inline R123_CUDA_DEVICE value_type assemble_from_u32(uint32_t *p32){
lib/Random123/array.h:#ifdef __CUDA_ARCH__
lib/Random123/array.h:/* CUDA can't handle std::reverse_iterator.  We *could* implement it
lib/Random123/array.h:    R123_CUDA_DEVICE reverse_iterator rbegin(){ return reverse_iterator(end()); }                         \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); } \
lib/Random123/array.h:    R123_CUDA_DEVICE reverse_iterator rend(){ return reverse_iterator(begin()); }        \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator rend() const{ return const_reverse_iterator(begin()); } \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator crbegin() const{ return const_reverse_iterator(cend()); } \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator crend() const{ return const_reverse_iterator(cbegin()); } 
lib/Random123/array.h:    R123_CUDA_DEVICE reference operator[](size_type i){return v[i];}                     \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reference operator[](size_type i) const {return v[i];}        \
lib/Random123/array.h:    R123_CUDA_DEVICE reference at(size_type i){ if(i >=  _N) R123_THROW(std::out_of_range("array index out of range")); return (*this)[i]; } \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reference at(size_type i) const { if(i >=  _N) R123_THROW(std::out_of_range("array index out of range")); return (*this)[i]; } \
lib/Random123/array.h:    R123_CUDA_DEVICE size_type size() const { return  _N; }                              \
lib/Random123/array.h:    R123_CUDA_DEVICE size_type max_size() const { return _N; }                           \
lib/Random123/array.h:    R123_CUDA_DEVICE bool empty() const { return _N==0; };                               \
lib/Random123/array.h:    R123_CUDA_DEVICE iterator begin() { return &v[0]; }                                  \
lib/Random123/array.h:    R123_CUDA_DEVICE iterator end() { return &v[_N]; }                                   \
lib/Random123/array.h:    R123_CUDA_DEVICE const_iterator begin() const { return &v[0]; }                      \
lib/Random123/array.h:    R123_CUDA_DEVICE const_iterator end() const { return &v[_N]; }                       \
lib/Random123/array.h:    R123_CUDA_DEVICE const_iterator cbegin() const { return &v[0]; }                     \
lib/Random123/array.h:    R123_CUDA_DEVICE const_iterator cend() const { return &v[_N]; }                      \
lib/Random123/array.h:    R123_CUDA_DEVICE pointer data(){ return &v[0]; }                                     \
lib/Random123/array.h:    R123_CUDA_DEVICE const_pointer data() const{ return &v[0]; }                         \
lib/Random123/array.h:    R123_CUDA_DEVICE reference front(){ return v[0]; }                                   \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reference front() const{ return v[0]; }                       \
lib/Random123/array.h:    R123_CUDA_DEVICE reference back(){ return v[_N-1]; }                                 \
lib/Random123/array.h:    R123_CUDA_DEVICE const_reference back() const{ return v[_N-1]; }                     \
lib/Random123/array.h:    R123_CUDA_DEVICE bool operator==(const r123array##_N##x##W& rhs) const{ \
lib/Random123/array.h:	/* CUDA3 does not have std::equal */ \
lib/Random123/array.h:    R123_CUDA_DEVICE bool operator!=(const r123array##_N##x##W& rhs) const{ return !(*this == rhs); } \
lib/Random123/array.h:    /* CUDA3 does not have std::fill_n */ \
lib/Random123/array.h:    R123_CUDA_DEVICE void fill(const value_type& val){ for (size_t i = 0; i < _N; ++i) v[i] = val; } \
lib/Random123/array.h:    R123_CUDA_DEVICE void swap(r123array##_N##x##W& rhs){ \
lib/Random123/array.h:	/* CUDA3 does not have std::swap_ranges */ \
lib/Random123/array.h:    R123_CUDA_DEVICE r123array##_N##x##W& incr(R123_ULONG_LONG n=1){                         \
lib/Random123/array.h:    R123_CUDA_DEVICE static r123array##_N##x##W seed(SeedSeq &ss){      \
lib/Random123/array.h:    R123_CUDA_DEVICE r123array##_N##x##W& incr_carefully(R123_ULONG_LONG n){ \
lib/NumLib.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/FloydWarshallCycle.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/VectorLib.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/StrStrmLib.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/AlphaNum.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/Lambda.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/Lambda.h:#ifdef GOMC_CUDA
lib/Lambda.h:#include <cuda.h>
lib/Lambda.h:#include <cuda_runtime.h>
lib/Lambda.h:#include "VariablesCUDA.cuh"
lib/Lambda.h:#include "ConstantDefinitionsCUDAKernel.cuh"
lib/Lambda.h:  #ifdef GOMC_CUDA 
lib/Lambda.h:    VariablesCUDA *refVarCUDA
lib/Lambda.h:#ifdef GOMC_CUDA
lib/Lambda.h:    varCUDA = refVarCUDA;
lib/Lambda.h:    // Update Lambda on GPU
lib/Lambda.h:    UpdateGPULambda(varCUDA, molIndex, lambdaVDW,
lib/Lambda.h:#ifdef GOMC_CUDA
lib/Lambda.h:  VariablesCUDA *varCUDA;
lib/Lambda.h:#ifdef GOMC_CUDA
lib/Lambda.h:  // Update Lambda on GPU
lib/Lambda.h:  UpdateGPULambda(varCUDA, molIndex, lambdaVDW,
lib/Lambda.h:#ifdef GOMC_CUDA
lib/Lambda.h:  // Update Lambda on GPU
lib/Lambda.h:  UpdateGPULambda(varCUDA, molIndex, lambdaVDW,
lib/StrLib.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/GeomLib.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/FloydWarshallCycle.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/BasicTypes.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
lib/BasicTypes.h:#ifdef GOMC_CUDA
lib/BasicTypes.h:#define RECORD_DEBUG_FILE_NAME "gpu.debug"
lib/BitLib.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/HistOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CPUSide.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Reader.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CBMC.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ParallelTemperingUtilities.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.51
src/MolSetup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/OutputVars.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PDBOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Coordinates.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoleculeKind.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCLinkedHedron.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCHedronCycle.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFactory.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCHedronCycle.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCGraph.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/TrialMol.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCLinkedCycle.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeCycle.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCRotateCOM.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCHedron.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCLinear.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCRotateOnAtom.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/TrialMol.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCRotateOnAtom.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCRotateCOM.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCCyclic.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCCrankShaftAng.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCSingle.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeHedron.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeHedron.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCComponent.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeCycleSeed.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeHedronSeed.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCCrankShaftDih.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCCyclic.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCData.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeHedronSeed.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCLinkedHedron.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCGraph.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCCrankShaftDih.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCCrankShaftAng.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCHedron.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCOnSphere.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCLinear.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCLinkedCycle.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCSingle.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCOnSphere.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeCycle.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/cbmc/DCFreeCycleSeed.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ExtendedSystem.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ConsoleOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/InputFileReader.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Molecules.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ConfigSetup.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Writer.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoleculeKind.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Simulation.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Simulation.cpp:#include "CUDAMemoryManager.cuh"
src/Simulation.cpp:#ifdef GOMC_CUDA
src/Simulation.cpp:  CUDAMemoryManager::isFreed();
src/BlockOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CheckpointSetup.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFSetup.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/TransformParticlesCUDAKernel.cu:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/TransformParticlesCUDAKernel.cu:#ifdef GOMC_CUDA
src/GPU/TransformParticlesCUDAKernel.cu:#include "TransformParticlesCUDAKernel.cuh"
src/GPU/TransformParticlesCUDAKernel.cu:#include "CalculateMinImageCUDAKernel.cuh"
src/GPU/TransformParticlesCUDAKernel.cu:#include "CUDAMemoryManager.cuh"
src/GPU/TransformParticlesCUDAKernel.cu:__device__ inline double randomGPU(unsigned int counter, ulong step, ulong seed)
src/GPU/TransformParticlesCUDAKernel.cu:__device__ inline double3 randomCoordsGPU(unsigned int counter, unsigned int key, ulong step, ulong seed)
src/GPU/TransformParticlesCUDAKernel.cu:__device__ inline double randomGaussianGPU(unsigned int counter, ulong step,
src/GPU/TransformParticlesCUDAKernel.cu:__device__ inline double3 randomGaussianCoordsGPU(unsigned int counter, unsigned int key, ulong step,
src/GPU/TransformParticlesCUDAKernel.cu:__device__ inline double SymRandomGPU(unsigned int counter, unsigned int key, ulong step, ulong seed)
src/GPU/TransformParticlesCUDAKernel.cu:__device__ inline double3 SymRandomCoordsGPU(unsigned int counter, unsigned int key,
src/GPU/TransformParticlesCUDAKernel.cu:__device__ inline double3 RandomCoordsOnSphereGPU(unsigned int counter, unsigned int key,
src/GPU/TransformParticlesCUDAKernel.cu:                                     double axx, double axy, double axz, int gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cu:                                     double *gpu_cell_x, double *gpu_cell_y, double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                                     double *gpu_Invcell_x, double *gpu_Invcell_y, double *gpu_Invcell_z)
src/GPU/TransformParticlesCUDAKernel.cu:  if (gpu_nonOrth)
src/GPU/TransformParticlesCUDAKernel.cu:    UnwrapPBCNonOrth3(coor, com, axes, halfAx, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                      gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:  if (gpu_nonOrth)
src/GPU/TransformParticlesCUDAKernel.cu:    WrapPBCNonOrth3(coor, axes, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                    gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:void CallTranslateParticlesGPU(VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cu:  int8_t *gpu_isMoleculeInvolved;
src/GPU/TransformParticlesCUDAKernel.cu:  int *gpu_particleMol;
src/GPU/TransformParticlesCUDAKernel.cu:  CUMALLOC((void **) &gpu_isMoleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleMol, particleMol.size() * sizeof(int));
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcex, mForcex, molCount * sizeof(double),
src/GPU/TransformParticlesCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcey, mForcey, molCount * sizeof(double),
src/GPU/TransformParticlesCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcez, mForcez, molCount * sizeof(double),
src/GPU/TransformParticlesCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecx, molForceRecRef.x, molCount * sizeof(double),
src/GPU/TransformParticlesCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecy, molForceRecRef.y, molCount * sizeof(double),
src/GPU/TransformParticlesCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecz, molForceRecRef.z, molCount * sizeof(double),
src/GPU/TransformParticlesCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(gpu_particleMol, &particleMol[0], particleMol.size() * sizeof(int),
src/GPU/TransformParticlesCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(gpu_isMoleculeInvolved, &isMoleculeInvolved[0],
src/GPU/TransformParticlesCUDAKernel.cu:             isMoleculeInvolved.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, newMolPos.x, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, newMolPos.y, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, newMolPos.z, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comx, newCOMs.x, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comy, newCOMs.y, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comz, newCOMs.z, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcex,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcey,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcez,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_inForceRange,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_particleMol,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_isMoleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecz);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.x, vars->gpu_x, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.y, vars->gpu_y, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.z, vars->gpu_z, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newCOMs.x, vars->gpu_comx, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newCOMs.y, vars->gpu_comy, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newCOMs.z, vars->gpu_comz, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(t_k.x, vars->gpu_t_k_x, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(t_k.y, vars->gpu_t_k_y, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(t_k.z, vars->gpu_t_k_z, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(&inForceRange[0], vars->gpu_inForceRange, molCount * sizeof(int), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  CUFREE(gpu_isMoleculeInvolved);
src/GPU/TransformParticlesCUDAKernel.cu:  CUFREE(gpu_particleMol);
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:void CallRotateParticlesGPU(VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cu:  int8_t *gpu_isMoleculeInvolved;
src/GPU/TransformParticlesCUDAKernel.cu:  int *gpu_particleMol;
src/GPU/TransformParticlesCUDAKernel.cu:  CUMALLOC((void **) &gpu_isMoleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleMol, particleMol.size() * sizeof(int));
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mTorquex, mTorquex, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mTorquey, mTorquey, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mTorquez, mTorquez, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(gpu_particleMol, &particleMol[0], particleMol.size() * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, newMolPos.x, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, newMolPos.y, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, newMolPos.z, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comx, newCOMs.x, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comy, newCOMs.y, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comz, newCOMs.z, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(gpu_isMoleculeInvolved, &isMoleculeInvolved[0],
src/GPU/TransformParticlesCUDAKernel.cu:             isMoleculeInvolved.size() * sizeof(int8_t), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquex,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquey,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquez,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_inForceRange,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_particleMol,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_isMoleculeInvolved);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.x, vars->gpu_x, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.y, vars->gpu_y, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.z, vars->gpu_z, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(r_k.x, vars->gpu_r_k_x, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(r_k.y, vars->gpu_r_k_y, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(r_k.z, vars->gpu_r_k_z, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(&inForceRange[0], vars->gpu_inForceRange, molCount * sizeof(int), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  CUFREE(gpu_isMoleculeInvolved);
src/GPU/TransformParticlesCUDAKernel.cu:  CUFREE(gpu_particleMol);
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:    int *gpu_inForceRange,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:    int *gpu_particleMol,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_Invcell_z,
src/GPU/TransformParticlesCUDAKernel.cu:    int *gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_t_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_t_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_t_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:    int8_t *gpu_isMoleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_mForceRecx,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_mForceRecy,
src/GPU/TransformParticlesCUDAKernel.cu:    double *gpu_mForceRecz)
src/GPU/TransformParticlesCUDAKernel.cu:  int molIndex = gpu_particleMol[atomNumber];
src/GPU/TransformParticlesCUDAKernel.cu:  if(!gpu_isMoleculeInvolved[molIndex]) return;
src/GPU/TransformParticlesCUDAKernel.cu:  bool updateMol = atomNumber == 0 || (gpu_particleMol[atomNumber] != gpu_particleMol[atomNumber - 1]);
src/GPU/TransformParticlesCUDAKernel.cu:  double lbfx = (molForcex[molIndex] + gpu_mForceRecx[molIndex]) * lambdaBETA;
src/GPU/TransformParticlesCUDAKernel.cu:  double lbfy = (molForcey[molIndex] + gpu_mForceRecy[molIndex]) * lambdaBETA;
src/GPU/TransformParticlesCUDAKernel.cu:  double lbfz = (molForcez[molIndex] + gpu_mForceRecz[molIndex]) * lambdaBETA;
src/GPU/TransformParticlesCUDAKernel.cu:    double3 randnums = randomCoordsGPU(molIndex, key, step, seed);
src/GPU/TransformParticlesCUDAKernel.cu:    double3 randnums = SymRandomCoordsGPU(molIndex, key, step, seed);
src/GPU/TransformParticlesCUDAKernel.cu:  double3 coor = make_double3(gpu_x[atomNumber] + shiftx, gpu_y[atomNumber] + shifty,
src/GPU/TransformParticlesCUDAKernel.cu:                              gpu_z[atomNumber] + shiftz);
src/GPU/TransformParticlesCUDAKernel.cu:  if (gpu_nonOrth[0])
src/GPU/TransformParticlesCUDAKernel.cu:    WrapPBCNonOrth3(coor, axes, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                    gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:  gpu_x[atomNumber] = coor.x;
src/GPU/TransformParticlesCUDAKernel.cu:  gpu_y[atomNumber] = coor.y;
src/GPU/TransformParticlesCUDAKernel.cu:  gpu_z[atomNumber] = coor.z;
src/GPU/TransformParticlesCUDAKernel.cu:    double3 com = make_double3(gpu_comx[molIndex] + shiftx, gpu_comy[molIndex] + shifty,
src/GPU/TransformParticlesCUDAKernel.cu:                               gpu_comz[molIndex] + shiftz);
src/GPU/TransformParticlesCUDAKernel.cu:    if (gpu_nonOrth[0])
src/GPU/TransformParticlesCUDAKernel.cu:      WrapPBCNonOrth3(com, axes, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                      gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_comx[molIndex] = com.x;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_comy[molIndex] = com.y;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_comz[molIndex] = com.z;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_t_k_x[molIndex] = shiftx;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_t_k_y[molIndex] = shifty;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_t_k_z[molIndex] = shiftz;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_inForceRange[molIndex] = forceInRange;
src/GPU/TransformParticlesCUDAKernel.cu:                                      int *gpu_inForceRange,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:                                      int *gpu_particleMol,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_Invcell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                                      int *gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_r_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_r_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:                                      double *gpu_r_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:                                      int8_t *gpu_isMoleculeInvolved)
src/GPU/TransformParticlesCUDAKernel.cu:  int molIndex = gpu_particleMol[atomNumber];
src/GPU/TransformParticlesCUDAKernel.cu:  if(!gpu_isMoleculeInvolved[molIndex]) return;
src/GPU/TransformParticlesCUDAKernel.cu:  bool updateMol = atomNumber == 0 || (gpu_particleMol[atomNumber] != gpu_particleMol[atomNumber - 1]);
src/GPU/TransformParticlesCUDAKernel.cu:    double3 randnums = randomCoordsGPU(molIndex, key, step, seed);
src/GPU/TransformParticlesCUDAKernel.cu:    double3 randnums = RandomCoordsOnSphereGPU(molIndex, key, step, seed);
src/GPU/TransformParticlesCUDAKernel.cu:    theta = r_max * SymRandomGPU(molIndex, key, step, seed);
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_r_k_x[molIndex] = rotx;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_r_k_y[molIndex] = roty;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_r_k_z[molIndex] = rotz;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_inForceRange[molIndex] = forceInRange;
src/GPU/TransformParticlesCUDAKernel.cu:  ApplyRotation(gpu_x[atomNumber], gpu_y[atomNumber], gpu_z[atomNumber],
src/GPU/TransformParticlesCUDAKernel.cu:                gpu_comx[molIndex], gpu_comy[molIndex], gpu_comz[molIndex],
src/GPU/TransformParticlesCUDAKernel.cu:                theta, rotvec, xAxes, yAxes, zAxes, *gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cu:                gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:// CUDA implementation of MultiParticle Brownian transformation 
src/GPU/TransformParticlesCUDAKernel.cu:void BrownianMotionRotateParticlesGPU(
src/GPU/TransformParticlesCUDAKernel.cu:  VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cu:  int *gpu_moleculeInvolved;
src/GPU/TransformParticlesCUDAKernel.cu:  CUMALLOC((void **) &gpu_moleculeInvolved, molCountInBox * sizeof(int));
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mTorquex, mTorque.x, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mTorquey, mTorque.y, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mTorquez, mTorque.z, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, newMolPos.x, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, newMolPos.y, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, newMolPos.z, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comx, newCOMs.x, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comy, newCOMs.y, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comz, newCOMs.z, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(gpu_moleculeInvolved, &moleculeInvolved[0], molCountInBox * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_startAtomIdx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquex,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquey,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquez,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_moleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_startAtomIdx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquex,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquey,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mTorquez,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_r_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_moleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.x, vars->gpu_x, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.y, vars->gpu_y, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.z, vars->gpu_z, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(r_k.x, vars->gpu_r_k_x, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(r_k.y, vars->gpu_r_k_y, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(r_k.z, vars->gpu_r_k_z, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  CUFREE(gpu_moleculeInvolved);
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_r_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_r_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_r_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_Invcell_z,
src/GPU/TransformParticlesCUDAKernel.cu:  // thread 0 will set up the matrix and update the gpu_r_k
src/GPU/TransformParticlesCUDAKernel.cu:    com = make_double3(gpu_comx[molIndex], gpu_comy[molIndex], gpu_comz[molIndex]);
src/GPU/TransformParticlesCUDAKernel.cu:    double3 randnums = randomGaussianCoordsGPU(molIndex, key, step, seed, 0.0, stdDev);
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_r_k_x[molIndex] = rot_x;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_r_k_y[molIndex] = rot_y;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_r_k_z[molIndex] = rot_z;
src/GPU/TransformParticlesCUDAKernel.cu:    double3 coor = make_double3(gpu_x[atomIdx], gpu_y[atomIdx], gpu_z[atomIdx]);
src/GPU/TransformParticlesCUDAKernel.cu:      UnwrapPBCNonOrth3(coor, com, axis, halfAx, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                        gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:      WrapPBCNonOrth3(coor, axis, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                      gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_x[atomIdx] = coor.x;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_y[atomIdx] = coor.y;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_z[atomIdx] = coor.z;
src/GPU/TransformParticlesCUDAKernel.cu:void BrownianMotionTranslateParticlesGPU(
src/GPU/TransformParticlesCUDAKernel.cu:  VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cu:  int *gpu_moleculeInvolved;
src/GPU/TransformParticlesCUDAKernel.cu:  CUMALLOC((void **) &gpu_moleculeInvolved, molCountInBox * sizeof(int));
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcex, mForce.x, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcey, mForce.y, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcez, mForce.z, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecx, mForceRec.x, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecy, mForceRec.y, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecz, mForceRec.z, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, newMolPos.x, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, newMolPos.y, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, newMolPos.z, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comx, newCOMs.x, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comy, newCOMs.y, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(vars->gpu_comz, newCOMs.z, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(gpu_moleculeInvolved, &moleculeInvolved[0], molCountInBox * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_startAtomIdx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcex,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcey,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcez,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_moleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_startAtomIdx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcex,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcey,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForcez,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_mForceRecz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_t_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:      gpu_moleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/TransformParticlesCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/TransformParticlesCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.x, vars->gpu_x, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.y, vars->gpu_y, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newMolPos.z, vars->gpu_z, atomCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newCOMs.x, vars->gpu_comx, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newCOMs.y, vars->gpu_comy, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(newCOMs.z, vars->gpu_comz, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(t_k.x, vars->gpu_t_k_x, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(t_k.y, vars->gpu_t_k_y, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  cudaMemcpy(t_k.z, vars->gpu_t_k_z, molCount * sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/TransformParticlesCUDAKernel.cu:  CUFREE(gpu_moleculeInvolved);
src/GPU/TransformParticlesCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_t_k_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_t_k_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_t_k_z,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cu:  double *gpu_Invcell_z,
src/GPU/TransformParticlesCUDAKernel.cu:  // thread 0 will calculate the shift vector and update COM and gpu_t_k
src/GPU/TransformParticlesCUDAKernel.cu:    double3 com = make_double3(gpu_comx[molIndex], gpu_comy[molIndex], gpu_comz[molIndex]);
src/GPU/TransformParticlesCUDAKernel.cu:    double3 randnums = randomGaussianCoordsGPU(molIndex, key, step, seed, 0.0, stdDev);
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_t_k_x[molIndex] = shift.x;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_t_k_y[molIndex] = shift.y;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_t_k_z[molIndex] = shift.z;
src/GPU/TransformParticlesCUDAKernel.cu:      WrapPBCNonOrth3(com, axis, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                      gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_comx[molIndex] = com.x;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_comy[molIndex] = com.y;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_comz[molIndex] = com.z;
src/GPU/TransformParticlesCUDAKernel.cu:    double3 coor = make_double3(gpu_x[atomIdx], gpu_y[atomIdx], gpu_z[atomIdx]);
src/GPU/TransformParticlesCUDAKernel.cu:      WrapPBCNonOrth3(coor, axis, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cu:                      gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_x[atomIdx] = coor.x;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_y[atomIdx] = coor.y;
src/GPU/TransformParticlesCUDAKernel.cu:    gpu_z[atomIdx] = coor.z;
src/GPU/CalculateMinImageCUDAKernel.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CalculateMinImageCUDAKernel.cuh:#ifdef GOMC_CUDA
src/GPU/CalculateMinImageCUDAKernel.cuh:#include <cuda.h>
src/GPU/CalculateMinImageCUDAKernel.cuh:#include <cuda_runtime.h>
src/GPU/CalculateMinImageCUDAKernel.cuh:#include "ConstantDefinitionsCUDAKernel.cuh"
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline void TransformSlantGPU(double3 &dist, const double3 &slant,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                         const double *gpu_cell_x, const double *gpu_cell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                         const double *gpu_cell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.x = slant.x * gpu_cell_x[0] + slant.y * gpu_cell_x[1] + slant.z * gpu_cell_x[2];
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.y = slant.x * gpu_cell_y[0] + slant.y * gpu_cell_y[1] + slant.z * gpu_cell_y[2];
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.z = slant.x * gpu_cell_z[0] + slant.y * gpu_cell_z[1] + slant.z * gpu_cell_z[2];
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline void TransformUnSlantGPU(double3 &dist, const double3 &slant,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                           const double *gpu_Invcell_x, const double *gpu_Invcell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                           const double *gpu_Invcell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.x = slant.x * gpu_Invcell_x[0] + slant.y * gpu_Invcell_x[1] + slant.z * gpu_Invcell_x[2];
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.y = slant.x * gpu_Invcell_y[0] + slant.y * gpu_Invcell_y[1] + slant.z * gpu_Invcell_y[2];
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.z = slant.x * gpu_Invcell_z[0] + slant.y * gpu_Invcell_z[1] + slant.z * gpu_Invcell_z[2];
src/GPU/CalculateMinImageCUDAKernel.cuh:                                       const double *gpu_cell_x, const double *gpu_cell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                       const double *gpu_cell_z, const double *gpu_Invcell_x, 
src/GPU/CalculateMinImageCUDAKernel.cuh:                                       const double *gpu_Invcell_y, const double *gpu_Invcell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  TransformUnSlantGPU(t, v, gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:  TransformSlantGPU(v, t, gpu_cell_x, gpu_cell_y, gpu_cell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:                                         const double *gpu_cell_x, const double *gpu_cell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                         const double *gpu_cell_z, const double *gpu_Invcell_x, 
src/GPU/CalculateMinImageCUDAKernel.cuh:                                         const double *gpu_Invcell_y, const double *gpu_Invcell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  TransformUnSlantGPU(t, v, gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:  TransformUnSlantGPU(tref, ref, gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:  TransformSlantGPU(v, t, gpu_cell_x, gpu_cell_y, gpu_cell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline double MinImageSignedGPU(double raw, const double ax, const double halfAx)
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline double3 MinImageGPU(double3 rawVec, const double3 axis, const double3 halfAx)
src/GPU/CalculateMinImageCUDAKernel.cuh:  rawVec.x = MinImageSignedGPU(rawVec.x, axis.x, halfAx.x);
src/GPU/CalculateMinImageCUDAKernel.cuh:  rawVec.y = MinImageSignedGPU(rawVec.y, axis.y, halfAx.y);
src/GPU/CalculateMinImageCUDAKernel.cuh:  rawVec.z = MinImageSignedGPU(rawVec.z, axis.z, halfAx.z);
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline double3 MinImageNonOrthGPU(double3 rawVec, const double3 &axis, const double3 &halfAx,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                             const double *gpu_cell_x, const double *gpu_cell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                             const double *gpu_cell_z, const double *gpu_Invcell_x, 
src/GPU/CalculateMinImageCUDAKernel.cuh:                                             const double *gpu_Invcell_y, const double *gpu_Invcell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  TransformUnSlantGPU(t, rawVec, gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:  t = MinImageGPU(t, axis, halfAx);
src/GPU/CalculateMinImageCUDAKernel.cuh:  TransformSlantGPU(rawVec, t, gpu_cell_x, gpu_cell_y, gpu_cell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline void DeviceInRcut(double &distSq, double3 &dist, const double *gpu_x,
src/GPU/CalculateMinImageCUDAKernel.cuh:    const double *gpu_y, const double *gpu_z, int particleID, int otherParticle, double axx,
src/GPU/CalculateMinImageCUDAKernel.cuh:    double axy, double axz, int gpu_nonOrth, double *gpu_cell_x, double *gpu_cell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:    double *gpu_cell_z, double *gpu_Invcell_x, double *gpu_Invcell_y, double *gpu_Invcell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.x = gpu_x[particleID] - gpu_x[otherParticle];
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.y = gpu_y[particleID] - gpu_y[otherParticle];
src/GPU/CalculateMinImageCUDAKernel.cuh:  dist.z = gpu_z[particleID] - gpu_z[otherParticle];
src/GPU/CalculateMinImageCUDAKernel.cuh:  if(gpu_nonOrth) {
src/GPU/CalculateMinImageCUDAKernel.cuh:    dist = MinImageNonOrthGPU(dist, axes, halfAx, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/CalculateMinImageCUDAKernel.cuh:                              gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:    dist = MinImageGPU(dist, axes, halfAx);
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline bool InRcutGPU(double &distSq, const double *x, const double *y, const double *z,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 double gpu_rCut, int gpu_nonOrth,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 const double *gpu_cell_x, const double *gpu_cell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 const double *gpu_cell_z, const double *gpu_Invcell_x,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 const double *gpu_Invcell_y, const double *gpu_Invcell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  if(gpu_nonOrth) {
src/GPU/CalculateMinImageCUDAKernel.cuh:    dist = MinImageNonOrthGPU(dist, axis, halfAx, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/CalculateMinImageCUDAKernel.cuh:                              gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:    dist = MinImageGPU(dist, axis, halfAx);
src/GPU/CalculateMinImageCUDAKernel.cuh:  return ((gpu_rCut * gpu_rCut) > distSq);
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline bool InRcutGPU(double &distSq, double3 &dist,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 double gpu_rCut, int gpu_nonOrth,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 const double *gpu_cell_x, const double *gpu_cell_y,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 const double *gpu_cell_z, const double *gpu_Invcell_x,
src/GPU/CalculateMinImageCUDAKernel.cuh:                                 const double *gpu_Invcell_y, const double *gpu_Invcell_z)
src/GPU/CalculateMinImageCUDAKernel.cuh:  if(gpu_nonOrth) {
src/GPU/CalculateMinImageCUDAKernel.cuh:    dist = MinImageNonOrthGPU(dist, axis, halfAx, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/CalculateMinImageCUDAKernel.cuh:                              gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateMinImageCUDAKernel.cuh:    dist = MinImageGPU(dist, axis, halfAx);
src/GPU/CalculateMinImageCUDAKernel.cuh:  return ((gpu_rCut * gpu_rCut) > distSq);
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline int FlatIndexGPU(int i, int j, int gpu_count)
src/GPU/CalculateMinImageCUDAKernel.cuh:  return i + j * gpu_count;
src/GPU/CalculateMinImageCUDAKernel.cuh:__device__ inline double DotProductGPU(double kx, double ky, double kz,
src/GPU/CalculateMinImageCUDAKernel.cuh:    int box, const bool *gpu_isFraction, const int *gpu_molIndex,
src/GPU/CalculateMinImageCUDAKernel.cuh:    const double *gpu_lambdaVDW)
src/GPU/CalculateMinImageCUDAKernel.cuh:  if(gpu_isFraction[box]) {
src/GPU/CalculateMinImageCUDAKernel.cuh:    if(gpu_molIndex[box] == molA) {
src/GPU/CalculateMinImageCUDAKernel.cuh:      lambda *= gpu_lambdaVDW[box];
src/GPU/CalculateMinImageCUDAKernel.cuh:    if(gpu_molIndex[box] == molB) {
src/GPU/CalculateMinImageCUDAKernel.cuh:      lambda *= gpu_lambdaVDW[box];
src/GPU/CalculateMinImageCUDAKernel.cuh:    int box, const bool *gpu_isFraction, const int *gpu_molIndex,
src/GPU/CalculateMinImageCUDAKernel.cuh:    const double *gpu_lambdaCoulomb)
src/GPU/CalculateMinImageCUDAKernel.cuh:  if(gpu_isFraction[box]) {
src/GPU/CalculateMinImageCUDAKernel.cuh:    if(gpu_molIndex[box] == molA) {
src/GPU/CalculateMinImageCUDAKernel.cuh:      lambda *= gpu_lambdaCoulomb[box];
src/GPU/CalculateMinImageCUDAKernel.cuh:    if(gpu_molIndex[box] == molB) {
src/GPU/CalculateMinImageCUDAKernel.cuh:      lambda *= gpu_lambdaCoulomb[box];
src/GPU/CalculateMinImageCUDAKernel.cuh:    const bool *gpu_isFraction, const int *gpu_molIndex,
src/GPU/CalculateMinImageCUDAKernel.cuh:    const double *gpu_lambdaCoulomb)
src/GPU/CalculateMinImageCUDAKernel.cuh:  if(gpu_isFraction[box]) {
src/GPU/CalculateMinImageCUDAKernel.cuh:    if(gpu_molIndex[box] == mol) {
src/GPU/CalculateMinImageCUDAKernel.cuh:      lambda = gpu_lambdaCoulomb[box];
src/GPU/CalculateMinImageCUDAKernel.cuh:// Add atomic operations for GPUs that do not support it
src/GPU/CalculateMinImageCUDAKernel.cuh:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
src/GPU/CalculateMinImageCUDAKernel.cuh:#endif /*GOMC_CUDA*/
src/GPU/CalculateEnergyCUDAKernel.cu:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CalculateEnergyCUDAKernel.cu:#ifdef GOMC_CUDA
src/GPU/CalculateEnergyCUDAKernel.cu:#include <cuda.h>
src/GPU/CalculateEnergyCUDAKernel.cu:#include "ConstantDefinitionsCUDAKernel.cuh"
src/GPU/CalculateEnergyCUDAKernel.cu:#include "CalculateMinImageCUDAKernel.cuh"
src/GPU/CalculateEnergyCUDAKernel.cu:#include "CalculateForceCUDAKernel.cuh"
src/GPU/CalculateEnergyCUDAKernel.cu:#include "CalculateEnergyCUDAKernel.cuh"
src/GPU/CalculateEnergyCUDAKernel.cu:#include "CUDAMemoryManager.cuh"
src/GPU/CalculateEnergyCUDAKernel.cu:void CallBoxInterGPU(VariablesCUDA *vars,
src/GPU/CalculateEnergyCUDAKernel.cu:  int *gpu_particleKind, *gpu_particleMol;
src/GPU/CalculateEnergyCUDAKernel.cu:  int *gpu_neighborList, *gpu_cellStartIndex;
src/GPU/CalculateEnergyCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateEnergyCUDAKernel.cu:  double *gpu_REn, *gpu_LJEn;
src/GPU/CalculateEnergyCUDAKernel.cu:  double *gpu_final_REn, *gpu_final_LJEn;
src/GPU/CalculateEnergyCUDAKernel.cu:  CUMALLOC((void**) &gpu_neighborList, neighborListCount * sizeof(int));
src/GPU/CalculateEnergyCUDAKernel.cu:  CUMALLOC((void**) &gpu_cellStartIndex, cellStartIndex.size() * sizeof(int));
src/GPU/CalculateEnergyCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge, particleCharge.size() * sizeof(double));
src/GPU/CalculateEnergyCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleKind, particleKind.size() * sizeof(int));
src/GPU/CalculateEnergyCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleMol, particleMol.size() * sizeof(int));
src/GPU/CalculateEnergyCUDAKernel.cu:  CUMALLOC((void**) &gpu_LJEn, energyVectorLen * sizeof(double));
src/GPU/CalculateEnergyCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_LJEn, sizeof(double));
src/GPU/CalculateEnergyCUDAKernel.cu:    CUMALLOC((void**) &gpu_REn, energyVectorLen * sizeof(double));
src/GPU/CalculateEnergyCUDAKernel.cu:    CUMALLOC((void**) &gpu_final_REn, sizeof(double));
src/GPU/CalculateEnergyCUDAKernel.cu:  // Copy necessary data to GPU
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(gpu_neighborList, &neighborlist1D[0], neighborListCount * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(gpu_cellStartIndex, &cellStartIndex[0], cellStartIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(vars->gpu_cellVector, &cellVector[0], atomNumber * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0], particleCharge.size() * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(gpu_particleKind, &particleKind[0], particleKind.size() * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(gpu_particleMol, &particleMol[0], particleMol.size() * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateEnergyCUDAKernel.cu:  BoxInterGPU <<< blocksPerGrid, threadsPerBlock>>>(gpu_cellStartIndex,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_cellVector,
src/GPU/CalculateEnergyCUDAKernel.cu:      gpu_neighborList,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_x,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_y,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_z,
src/GPU/CalculateEnergyCUDAKernel.cu:      gpu_particleCharge,
src/GPU/CalculateEnergyCUDAKernel.cu:      gpu_particleKind,
src/GPU/CalculateEnergyCUDAKernel.cu:      gpu_particleMol,
src/GPU/CalculateEnergyCUDAKernel.cu:      gpu_REn,
src/GPU/CalculateEnergyCUDAKernel.cu:      gpu_LJEn,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_VDW_Kind,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_isMartini,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_count,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_rCutCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_rCutLow,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_nonOrth,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_rMin,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_rMaxSq,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_expConst,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_molIndex,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:      vars->gpu_isFraction,
src/GPU/CalculateEnergyCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEnergyCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEnergyCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_LJEn,
src/GPU/CalculateEnergyCUDAKernel.cu:                    gpu_final_LJEn, energyVectorLen);
src/GPU/CalculateEnergyCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_LJEn,
src/GPU/CalculateEnergyCUDAKernel.cu:                    gpu_final_LJEn, energyVectorLen);
src/GPU/CalculateEnergyCUDAKernel.cu:  CubDebugExit(cudaMemcpy(&LJEn, gpu_final_LJEn, sizeof(double),
src/GPU/CalculateEnergyCUDAKernel.cu:                          cudaMemcpyDeviceToHost));
src/GPU/CalculateEnergyCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_REn,
src/GPU/CalculateEnergyCUDAKernel.cu:                      gpu_final_REn, energyVectorLen);
src/GPU/CalculateEnergyCUDAKernel.cu:    CubDebugExit(cudaMemcpy(&REn, gpu_final_REn, sizeof(double),
src/GPU/CalculateEnergyCUDAKernel.cu:                            cudaMemcpyDeviceToHost));
src/GPU/CalculateEnergyCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateEnergyCUDAKernel.cu:  CUFREE(gpu_particleKind);
src/GPU/CalculateEnergyCUDAKernel.cu:  CUFREE(gpu_particleMol);
src/GPU/CalculateEnergyCUDAKernel.cu:  CUFREE(gpu_LJEn);
src/GPU/CalculateEnergyCUDAKernel.cu:  CUFREE(gpu_final_LJEn);
src/GPU/CalculateEnergyCUDAKernel.cu:    CUFREE(gpu_REn);
src/GPU/CalculateEnergyCUDAKernel.cu:    CUFREE(gpu_final_REn);
src/GPU/CalculateEnergyCUDAKernel.cu:  CUFREE(gpu_neighborList);
src/GPU/CalculateEnergyCUDAKernel.cu:  CUFREE(gpu_cellStartIndex);
src/GPU/CalculateEnergyCUDAKernel.cu:__global__ void BoxInterGPU(int *gpu_cellStartIndex,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_cellVector,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_neighborList,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_x,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_y,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_z,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_particleCharge,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_particleKind,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_particleMol,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_REn,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_LJEn,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_VDW_Kind,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_isMartini,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_count,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_rCutCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_rCutLow,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_nonOrth,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_cell_x,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_cell_y,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_cell_z,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_Invcell_x,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_Invcell_y,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_Invcell_z,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_rMin,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_rMaxSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_expConst,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int *gpu_molIndex,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:                            bool *gpu_isFraction,
src/GPU/CalculateEnergyCUDAKernel.cu:  double cutoff = fmax(gpu_rCut[0], gpu_rCutCoulomb[box]);
src/GPU/CalculateEnergyCUDAKernel.cu:  int neighborCell = gpu_neighborList[nCellIndex];
src/GPU/CalculateEnergyCUDAKernel.cu:  int endIndex = gpu_cellStartIndex[neighborCell + 1];
src/GPU/CalculateEnergyCUDAKernel.cu:  particlesInsideNeighboringCells = endIndex - gpu_cellStartIndex[neighborCell];
src/GPU/CalculateEnergyCUDAKernel.cu:  endIndex = gpu_cellStartIndex[currentCell + 1];
src/GPU/CalculateEnergyCUDAKernel.cu:  particlesInsideCurrentCell = endIndex - gpu_cellStartIndex[currentCell];
src/GPU/CalculateEnergyCUDAKernel.cu:    int currentParticle = gpu_cellVector[gpu_cellStartIndex[currentCell] + currentParticleIndex];
src/GPU/CalculateEnergyCUDAKernel.cu:    int neighborParticle = gpu_cellVector[gpu_cellStartIndex[neighborCell] + neighborParticleIndex];
src/GPU/CalculateEnergyCUDAKernel.cu:    if(currentParticle < neighborParticle && gpu_particleMol[currentParticle] != gpu_particleMol[neighborParticle]) {
src/GPU/CalculateEnergyCUDAKernel.cu:      if(InRcutGPU(distSq, gpu_x, gpu_y, gpu_z,
src/GPU/CalculateEnergyCUDAKernel.cu:                   axis, halfAx, cutoff, gpu_nonOrth[0], gpu_cell_x,
src/GPU/CalculateEnergyCUDAKernel.cu:                   gpu_cell_y, gpu_cell_z, gpu_Invcell_x, gpu_Invcell_y,
src/GPU/CalculateEnergyCUDAKernel.cu:                   gpu_Invcell_z)) {
src/GPU/CalculateEnergyCUDAKernel.cu:        int kA = gpu_particleKind[currentParticle];
src/GPU/CalculateEnergyCUDAKernel.cu:        int kB = gpu_particleKind[neighborParticle];
src/GPU/CalculateEnergyCUDAKernel.cu:        int mA = gpu_particleMol[currentParticle];
src/GPU/CalculateEnergyCUDAKernel.cu:        int mB = gpu_particleMol[neighborParticle];
src/GPU/CalculateEnergyCUDAKernel.cu:        double lambdaVDW = DeviceGetLambdaVDW(mA, mB, box, gpu_isFraction,
src/GPU/CalculateEnergyCUDAKernel.cu:                                              gpu_molIndex, gpu_lambdaVDW);
src/GPU/CalculateEnergyCUDAKernel.cu:        LJEn += CalcEnGPU(distSq, kA, kB, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                          gpu_VDW_Kind[0], gpu_isMartini[0], gpu_rCut[0],
src/GPU/CalculateEnergyCUDAKernel.cu:                          gpu_rOn[0], gpu_count[0], lambdaVDW, sc_sigma_6,
src/GPU/CalculateEnergyCUDAKernel.cu:                          sc_alpha, sc_power, gpu_rMin, gpu_rMaxSq, gpu_expConst);
src/GPU/CalculateEnergyCUDAKernel.cu:          double qi_qj_fact = gpu_particleCharge[currentParticle] * gpu_particleCharge[neighborParticle];
src/GPU/CalculateEnergyCUDAKernel.cu:            qi_qj_fact *= qqFactGPU;
src/GPU/CalculateEnergyCUDAKernel.cu:                                   gpu_isFraction, gpu_molIndex,
src/GPU/CalculateEnergyCUDAKernel.cu:                                   gpu_lambdaCoulomb);
src/GPU/CalculateEnergyCUDAKernel.cu:            REn += CalcCoulombGPU(distSq, kA, kB, qi_qj_fact, gpu_rCutLow[0],
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_ewald[0], gpu_VDW_Kind[0], gpu_alpha[box],
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_rCutCoulomb[box], gpu_isMartini[0],
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_diElectric_1[0], lambdaCoulomb, sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  sc_sigma_6, sc_alpha, sc_power, gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_count[0]);
src/GPU/CalculateEnergyCUDAKernel.cu:    gpu_REn[threadID] = REn;
src/GPU/CalculateEnergyCUDAKernel.cu:  gpu_LJEn[threadID] = LJEn;
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombGPU(double distSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double gpu_rCutLow,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 int gpu_VDW_Kind,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double gpu_rCutCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 int gpu_isMartini,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 int gpu_count)
src/GPU/CalculateEnergyCUDAKernel.cu:  if((gpu_rCutCoulomb * gpu_rCutCoulomb) < distSq) {
src/GPU/CalculateEnergyCUDAKernel.cu:  int index = FlatIndexGPU(kind1, kind2, gpu_count);
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_VDW_Kind == GPU_VDW_STD_KIND) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombParticleGPU(distSq, index, qi_qj_fact, gpu_ewald, gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_lambdaCoulomb, sc_coul, sc_sigma_6,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  sc_alpha, sc_power, gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_SHIFT_KIND) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombShiftGPU(distSq, index, qi_qj_fact, gpu_ewald, gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                               gpu_rCutCoulomb, gpu_lambdaCoulomb, sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cu:                               sc_sigma_6, sc_alpha, sc_power, gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_EXP6_KIND) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombExp6GPU(distSq, index, qi_qj_fact, gpu_ewald, gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                              gpu_lambdaCoulomb, sc_coul, sc_sigma_6, sc_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                              sc_power, gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_SWITCH_KIND && gpu_isMartini) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombSwitchMartiniGPU(distSq, index, qi_qj_fact, gpu_ewald, gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       gpu_rCutCoulomb, gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       gpu_lambdaCoulomb, sc_coul, sc_sigma_6,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       sc_alpha, sc_power, gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombSwitchGPU(distSq, index, qi_qj_fact, gpu_alpha, gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cu:                                gpu_rCutCoulomb, gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:                                gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnGPU(double distSq, int kind1, int kind2,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_epsilon_Cn, int gpu_VDW_Kind,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int gpu_isMartini, double gpu_rCut, double gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cu:                            int gpu_count, double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_rMin, double *gpu_rMaxSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                            double *gpu_expConst)
src/GPU/CalculateEnergyCUDAKernel.cu:  if((gpu_rCut * gpu_rCut) < distSq) {
src/GPU/CalculateEnergyCUDAKernel.cu:  int index = FlatIndexGPU(kind1, kind2, gpu_count);
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_VDW_Kind == GPU_VDW_STD_KIND) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnParticleGPU(distSq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                             gpu_lambdaVDW, sc_sigma_6, sc_alpha, sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_SHIFT_KIND) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnShiftGPU(distSq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                          gpu_rCut, gpu_lambdaVDW, sc_sigma_6, sc_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_EXP6_KIND) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnExp6GPU(distSq, index, gpu_sigmaSq, gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                         gpu_lambdaVDW, sc_sigma_6,
src/GPU/CalculateEnergyCUDAKernel.cu:                         sc_alpha, sc_power, gpu_rMin,
src/GPU/CalculateEnergyCUDAKernel.cu:                         gpu_rMaxSq, gpu_expConst);
src/GPU/CalculateEnergyCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_SWITCH_KIND && gpu_isMartini) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnSwitchMartiniGPU(distSq, index, gpu_sigmaSq, gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_epsilon_Cn, gpu_rCut, gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_lambdaVDW, sc_sigma_6, sc_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnSwitchGPU(distSq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                           gpu_rCut, gpu_rOn, gpu_lambdaVDW, sc_sigma_6,
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombParticleGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_lambdaCoulomb, bool sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cu:    uint sc_power, double *gpu_sigmaSq)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombParticleGPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cu:    double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombParticleGPUNoLambda(softRsq, qi_qj_fact, gpu_ewald, gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombParticleGPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombParticleGPUNoLambda(double distSq,
src/GPU/CalculateEnergyCUDAKernel.cu:    int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_alpha)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateEnergyCUDAKernel.cu:    value = erfc(gpu_alpha * dist);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombShiftGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:                                      int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                                      double gpu_rCut, double gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:                                      double *gpu_sigmaSq)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombShiftGPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:    double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombShiftGPUNoLambda(softRsq, qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:           gpu_ewald, gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:           gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombShiftGPUNoLambda(distSq, qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:           gpu_ewald, gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:           gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombShiftGPUNoLambda(double distSq, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rCut)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateEnergyCUDAKernel.cu:    double value = gpu_alpha * dist;
src/GPU/CalculateEnergyCUDAKernel.cu:    return qi_qj_fact * (1.0 / dist - 1.0 / gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombExp6GPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:                                     int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:                                     double gpu_lambdaCoulomb, bool sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cu:                                     uint sc_power, double *gpu_sigmaSq)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombExp6GPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cu:    double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombExp6GPUNoLambda(softRsq, qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:           gpu_ewald, gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombExp6GPUNoLambda(distSq, qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:           gpu_ewald, gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombExp6GPUNoLambda(double distSq, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:    int gpu_ewald, double gpu_alpha)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateEnergyCUDAKernel.cu:    value = erfc(gpu_alpha * dist);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombSwitchMartiniGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_sigmaSq)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombSwitchMartiniGPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha, gpu_rCut, gpu_diElectric_1);
src/GPU/CalculateEnergyCUDAKernel.cu:    double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombSwitchMartiniGPUNoLambda(softRsq, qi_qj_fact, gpu_ewald, gpu_alpha, gpu_rCut, gpu_diElectric_1);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombSwitchMartiniGPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha, gpu_rCut, gpu_diElectric_1);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombSwitchMartiniGPUNoLambda(double distSq,
src/GPU/CalculateEnergyCUDAKernel.cu:    int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_diElectric_1)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateEnergyCUDAKernel.cu:    double value = gpu_alpha * dist;
src/GPU/CalculateEnergyCUDAKernel.cu:    // double A1 = 1.0 * (-(1.0 + 4) * gpu_rCut) / (pow(gpu_rCut, 1.0 + 2) *
src/GPU/CalculateEnergyCUDAKernel.cu:                // pow(gpu_rCut, 2));
src/GPU/CalculateEnergyCUDAKernel.cu:    // double B1 = -1.0 * (-(1.0 + 3) * gpu_rCut) / (pow(gpu_rCut, 1.0 + 2) *
src/GPU/CalculateEnergyCUDAKernel.cu:                // pow(gpu_rCut, 3));
src/GPU/CalculateEnergyCUDAKernel.cu:    // double C1 = 1.0 / pow(gpu_rCut, 1.0) - A1 / 3.0 * pow(gpu_rCut, 3) -
src/GPU/CalculateEnergyCUDAKernel.cu:                // B1 / 4.0 * pow(gpu_rCut, 4);
src/GPU/CalculateEnergyCUDAKernel.cu:    double A1 = -5.0 / (gpu_rCut * gpu_rCut * gpu_rCut * gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:    double B1 = 4.0 / (gpu_rCut * gpu_rCut * gpu_rCut * gpu_rCut * gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:    double C1 = 1.0 / gpu_rCut - A1 / 3.0 * gpu_rCut * gpu_rCut * gpu_rCut -
src/GPU/CalculateEnergyCUDAKernel.cu:                B1 / 4.0 * gpu_rCut * gpu_rCut * gpu_rCut * gpu_rCut;
src/GPU/CalculateEnergyCUDAKernel.cu:    return qi_qj_fact * gpu_diElectric_1 * (1.0 / dist + coul);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombSwitchGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       double gpu_alpha, int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       double gpu_lambdaCoulomb, bool sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cu:                                       uint sc_power, double *gpu_sigmaSq)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcCoulombSwitchGPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha, gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:    double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombSwitchGPUNoLambda(softRsq, qi_qj_fact, gpu_ewald, gpu_alpha, gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:    return gpu_lambdaCoulomb * CalcCoulombSwitchGPUNoLambda(distSq, qi_qj_fact, gpu_ewald, gpu_alpha, gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcCoulombSwitchGPUNoLambda(double distSq, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cu:    int gpu_ewald, double gpu_alpha, double gpu_rCut)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateEnergyCUDAKernel.cu:    double value = gpu_alpha * dist;
src/GPU/CalculateEnergyCUDAKernel.cu:    double rCutSq = gpu_rCut * gpu_rCut;
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnParticleGPU(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cu:                                    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                                    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                                    double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnParticleGPUNoLambda(distSq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn);
src/GPU/CalculateEnergyCUDAKernel.cu:  double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:  return gpu_lambdaVDW * CalcEnParticleGPUNoLambda(softRsq, index, gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_n, gpu_epsilon_Cn);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnParticleGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_epsilon_Cn)
src/GPU/CalculateEnergyCUDAKernel.cu:  double rRat2 = gpu_sigmaSq[index] / distSq;
src/GPU/CalculateEnergyCUDAKernel.cu:  double repulse = pow(rRat2, gpu_n[index] * 0.5);
src/GPU/CalculateEnergyCUDAKernel.cu:  return gpu_epsilon_Cn[index] * (repulse - attract);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnShiftGPU(double distSq, int index, double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double *gpu_n, double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cu:                                 double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnShiftGPUNoLambda(distSq, index, gpu_sigmaSq, gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  gpu_epsilon_Cn, gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:  double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:  return gpu_lambdaVDW * CalcEnShiftGPUNoLambda(softRsq, index, gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnShiftGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_n, double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rCut)
src/GPU/CalculateEnergyCUDAKernel.cu:  double rRat2 = gpu_sigmaSq[index] / distSq;
src/GPU/CalculateEnergyCUDAKernel.cu:  double repulse = pow(rRat2, gpu_n[index] * 0.5);
src/GPU/CalculateEnergyCUDAKernel.cu:  double shiftRRat2 = gpu_sigmaSq[index] / (gpu_rCut * gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cu:  double shiftRepulse = pow(shiftRRat2, gpu_n[index] * 0.5);
src/GPU/CalculateEnergyCUDAKernel.cu:  double shiftConst = gpu_epsilon_Cn[index] * (shiftRepulse - shiftAttract);
src/GPU/CalculateEnergyCUDAKernel.cu:  return (gpu_epsilon_Cn[index] * (repulse - attract) - shiftConst);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnExp6GPU(double distSq, int index, double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                                double *gpu_n, double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cu:                                uint sc_power, double *gpu_rMin,
src/GPU/CalculateEnergyCUDAKernel.cu:                                double *gpu_rMaxSq, double *gpu_expConst)
src/GPU/CalculateEnergyCUDAKernel.cu:  if(distSq < gpu_rMaxSq[index]) {
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnExp6GPUNoLambda(distSq, index, gpu_n, gpu_rMin, gpu_expConst);
src/GPU/CalculateEnergyCUDAKernel.cu:  double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:  return gpu_lambdaVDW * CalcEnExp6GPUNoLambda(softRsq, index, gpu_n, gpu_rMin,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_expConst);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnExp6GPUNoLambda(double distSq, int index, double* gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                                        double* gpu_rMin, double* gpu_expConst)
src/GPU/CalculateEnergyCUDAKernel.cu:  double rRat = gpu_rMin[index] / dist;
src/GPU/CalculateEnergyCUDAKernel.cu:  uint alph_ij = gpu_n[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  return gpu_expConst[index] * (repulse - attract);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnSwitchMartiniGPU(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rCut, double gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnSwitchMartiniGPUNoLambda(distSq, index, gpu_sigmaSq, gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                                          gpu_epsilon_Cn, gpu_rCut, gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cu:  double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double) sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:  return gpu_lambdaVDW * CalcEnSwitchMartiniGPUNoLambda(softRsq, index,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_sigmaSq, gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_rCut, gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnSwitchMartiniGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rOn)
src/GPU/CalculateEnergyCUDAKernel.cu:  double r_n = pow(r_2, gpu_n[index] * 0.5);
src/GPU/CalculateEnergyCUDAKernel.cu:  double rij_ron = sqrt(distSq) - gpu_rOn;
src/GPU/CalculateEnergyCUDAKernel.cu:  double pn = gpu_n[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  // double An = pn * ((pn + 1) * gpu_rOn - (pn + 4) * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              // (pow(gpu_rCut, pn + 2) * pow(gpu_rCut - gpu_rOn, 2));
src/GPU/CalculateEnergyCUDAKernel.cu:  // double Bn = -pn * ((pn + 1) * gpu_rOn - (pn + 3) * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              // (pow(gpu_rCut, pn + 2) * pow(gpu_rCut - gpu_rOn, 3));
src/GPU/CalculateEnergyCUDAKernel.cu:  // double Cn = 1.0 / pow(gpu_rCut, pn) - An / 3.0 * pow(gpu_rCut - gpu_rOn, 3) -
src/GPU/CalculateEnergyCUDAKernel.cu:              // Bn / 4.0 * pow(gpu_rCut - gpu_rOn, 4);
src/GPU/CalculateEnergyCUDAKernel.cu:  // double A6 = 6.0 * ((6.0 + 1) * gpu_rOn - (6.0 + 4) * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              // (pow(gpu_rCut, 6.0 + 2) * pow(gpu_rCut - gpu_rOn, 2));
src/GPU/CalculateEnergyCUDAKernel.cu:  // double B6 = -6.0 * ((6.0 + 1) * gpu_rOn - (6.0 + 3) * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              // (pow(gpu_rCut, 6.0 + 2) * pow(gpu_rCut - gpu_rOn, 3));
src/GPU/CalculateEnergyCUDAKernel.cu:  // double C6 = 1.0 / pow(gpu_rCut, 6.0) - A6 / 3.0 * pow(gpu_rCut - gpu_rOn, 3) -
src/GPU/CalculateEnergyCUDAKernel.cu:              // B6 / 4.0 * pow(gpu_rCut - gpu_rOn, 4);
src/GPU/CalculateEnergyCUDAKernel.cu:  double An = pn * ((pn + 1.0) * gpu_rOn - (pn + 4.0) * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              (pow(gpu_rCut, pn + 2.0) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn));
src/GPU/CalculateEnergyCUDAKernel.cu:  double Bn = -pn * ((pn + 1.0) * gpu_rOn - (pn + 3.0) * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              (pow(gpu_rCut, pn + 2.0) * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn));
src/GPU/CalculateEnergyCUDAKernel.cu:  double Cn = 1.0 / pow(gpu_rCut, pn) - An / 3.0 * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) - Bn / 4.0 *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cu:  double A6 = 6.0 * (7.0 * gpu_rOn - 10.0 * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              (pow(gpu_rCut, 8.0) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn));
src/GPU/CalculateEnergyCUDAKernel.cu:  double B6 = -6.0 * (7.0 * gpu_rOn - 9.0 * gpu_rCut) /
src/GPU/CalculateEnergyCUDAKernel.cu:              (pow(gpu_rCut, 8.0) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn));
src/GPU/CalculateEnergyCUDAKernel.cu:  double C6 = pow(gpu_rCut, -6.0) - A6 / 3.0 * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) - B6 / 4.0 *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateEnergyCUDAKernel.cu:              (gpu_rCut - gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cu:  const double shiftRep = ( distSq > gpu_rOn * gpu_rOn ? shifttempRep : -Cn);
src/GPU/CalculateEnergyCUDAKernel.cu:  const double shiftAtt = ( distSq > gpu_rOn * gpu_rOn ? shifttempAtt : -C6);
src/GPU/CalculateEnergyCUDAKernel.cu:  double sig6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  double sign = pow(gpu_sigmaSq[index], pn * 0.5);
src/GPU/CalculateEnergyCUDAKernel.cu:  double Eij = gpu_epsilon_Cn[index] * (sign * (r_n + shiftRep) -
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnSwitchGPU(double distSq, int index, double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  double *gpu_n, double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  double gpu_rCut, double gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cu:                                  double gpu_lambdaVDW, double sc_sigma_6,
src/GPU/CalculateEnergyCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateEnergyCUDAKernel.cu:    return CalcEnSwitchGPUNoLambda(distSq, index, gpu_sigmaSq, gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:                                   gpu_epsilon_Cn, gpu_rCut, gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cu:  double sigma6 = gpu_sigmaSq[index] * gpu_sigmaSq[index] * gpu_sigmaSq[index];
src/GPU/CalculateEnergyCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateEnergyCUDAKernel.cu:  return gpu_lambdaVDW * CalcEnSwitchGPUNoLambda(softRsq, index, gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:         gpu_rCut, gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cu:__device__ double CalcEnSwitchGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cu:    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cu:    double gpu_rCut, double gpu_rOn)
src/GPU/CalculateEnergyCUDAKernel.cu:  double rCutSq = gpu_rCut * gpu_rCut;
src/GPU/CalculateEnergyCUDAKernel.cu:  double rOnSq = gpu_rOn * gpu_rOn;
src/GPU/CalculateEnergyCUDAKernel.cu:  double rRat2 = gpu_sigmaSq[index] / distSq;
src/GPU/CalculateEnergyCUDAKernel.cu:  double repulse = pow(rRat2, gpu_n[index] * 0.5);
src/GPU/CalculateEnergyCUDAKernel.cu:  return (gpu_epsilon_Cn[index] * (repulse - attract)) * factE;
src/GPU/ConstantDefinitionsCUDAKernel.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#ifndef CONSTANT_DEFINITIONS_CUDA_KERNEL
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#define CONSTANT_DEFINITIONS_CUDA_KERNEL
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#ifdef GOMC_CUDA
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#include <cuda.h>
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#include <cuda_runtime.h>
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#include "VariablesCUDA.cuh"
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#define GPU_VDW_STD_KIND 0
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#define GPU_VDW_SHIFT_KIND 1
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#define GPU_VDW_SWITCH_KIND 2
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#define GPU_VDW_EXP6_KIND 3
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void UpdateGPULambda(VariablesCUDA *vars, int *molIndex, double *lambdaVDW,
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void InitGPUForceField(VariablesCUDA &vars, double const *sigmaSq,
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void InitCoordinatesCUDA(VariablesCUDA *vars, uint atomNumber,
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void InitEwaldVariablesCUDA(VariablesCUDA *vars, uint imageTotal);
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void CopyCurrentToRefCUDA(VariablesCUDA *vars, uint box, uint imageTotal);
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void CopyRefToNewCUDA(VariablesCUDA *vars, uint box, uint imageTotal);
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void UpdateRecipVecCUDA(VariablesCUDA *vars, uint box);
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void UpdateRecipCUDA(VariablesCUDA *vars, uint box);
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void UpdateCellBasisCUDA(VariablesCUDA *vars, uint box, double *cellBasis_x,
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void UpdateInvCellBasisCUDA(VariablesCUDA *vars, uint box,
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void DestroyEwaldCUDAVars(VariablesCUDA *vars);
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void DestroyCUDAVars(VariablesCUDA *vars);
src/GPU/ConstantDefinitionsCUDAKernel.cuh:void InitExp6Variables(VariablesCUDA *vars, double *rMin, double *expConst,
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#endif /*GOMC_CUDA*/
src/GPU/ConstantDefinitionsCUDAKernel.cuh:#endif /*CONSTANT_DEFINITIONS_CUDA_KERNEL*/
src/GPU/CalculateEwaldCUDAKernel.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CalculateEwaldCUDAKernel.cuh:#ifndef CALCULATE_EWALD_CUDA_KERNEL
src/GPU/CalculateEwaldCUDAKernel.cuh:#define CALCULATE_EWALD_CUDA_KERNEL
src/GPU/CalculateEwaldCUDAKernel.cuh:#ifdef GOMC_CUDA
src/GPU/CalculateEwaldCUDAKernel.cuh:#include <cuda.h>
src/GPU/CalculateEwaldCUDAKernel.cuh:#include <cuda_runtime.h>
src/GPU/CalculateEwaldCUDAKernel.cuh:#include "VariablesCUDA.cuh"
src/GPU/CalculateEwaldCUDAKernel.cuh:void CallBoxForceReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cuh:void CallBoxReciprocalSetupGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cuh:void CallBoxReciprocalSumsGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cuh:void CallMolReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cuh:void CallChangeLambdaMolReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cuh:void CallSwapReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cuh:void CallMolExchangeReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cuh:__global__ void BoxForceReciprocalGPU(double *gpu_aForceRecx,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_aForceRecy,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_aForceRecz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_mForceRecx,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_mForceRecy,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_mForceRecz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      int *gpu_particleMol,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      bool *gpu_particleHasNoCharge,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                       bool *gpu_particleUsed,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     int *gpu_startMol,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      int *gpu_lengthMol,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_kx,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_ky,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_x,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_y,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_prefact,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      bool *gpu_isFraction,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      int *gpu_molIndex,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_lambdaCoulomb,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_cell_x,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_cell_y,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_cell_z,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_Invcell_x,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_Invcell_y,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      double *gpu_Invcell_z,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                      int *gpu_nonOrth,
src/GPU/CalculateEwaldCUDAKernel.cuh:__global__ void BoxReciprocalSumsGPU(double * gpu_x,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_y,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_kx,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_ky,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                     double * gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cuh:__global__ void MolReciprocalGPU(double *gpu_cx, double *gpu_cy, double *gpu_cz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_nx, double *gpu_ny, double *gpu_nz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_kx, double *gpu_ky, double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_sumRref,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_sumIref,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_prefactRef,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cuh:__global__ void ChangeLambdaMolReciprocalGPU(double *gpu_x, double *gpu_y, double *gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_kx, double *gpu_ky, double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_sumRref,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_sumIref,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_prefactRef,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                            double *gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cuh:__global__ void SwapReciprocalGPU(double *gpu_x, double *gpu_y, double *gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_kx, double *gpu_ky, double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_sumRref,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_sumIref,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_prefactRef,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                  double *gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cuh:__global__ void BoxReciprocalGPU(double *gpu_prefact,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cuh:                                 double *gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cuh:#endif /*GOMC_CUDA*/
src/GPU/CalculateEwaldCUDAKernel.cuh:#endif /*CALCULATE_EWALD_CUDA_KERNEL*/
src/GPU/CalculateEnergyCUDAKernel.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CalculateEnergyCUDAKernel.cuh:#ifdef GOMC_CUDA
src/GPU/CalculateEnergyCUDAKernel.cuh:#include <cuda.h>
src/GPU/CalculateEnergyCUDAKernel.cuh:#include <cuda_runtime.h>
src/GPU/CalculateEnergyCUDAKernel.cuh:#include "VariablesCUDA.cuh"
src/GPU/CalculateEnergyCUDAKernel.cuh:void CallBoxInterGPU(VariablesCUDA *vars,
src/GPU/CalculateEnergyCUDAKernel.cuh:__global__ void BoxInterGPU(int *gpu_cellStartIndex,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_cellVector,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_neighborList,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_x,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_y,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_z,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_particleCharge,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_particleKind,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_particleMol,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_REn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_LJEn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_VDW_Kind,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_isMartini,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_count,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_rCutCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_rCutLow,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_nonOrth,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_cell_x,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_cell_y,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_cell_z,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_Invcell_x,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_Invcell_y,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_Invcell_z,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_rMin,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_rMaxSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_expConst,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int *gpu_molIndex,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            bool *gpu_isFraction,
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombGPU(double distSq, int kind1, int kind2,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 double qi_qj_fact, double gpu_rCutLow,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 int gpu_ewald, int gpu_VDW_Kind,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 double gpu_alpha, double gpu_rCutCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 int gpu_isMartini, double gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 double gpu_lambdaCoulomb, bool sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 uint sc_power, double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 int gpu_count);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombVirGPU(double distSq, double qi_qj,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                    double gpu_rCutCoulomb, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                    int gpu_VDW_Kind, int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                    double gpu_diElectric_1, int gpu_isMartini);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnGPU(double distSq, int kind1, int kind2,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_epsilon_Cn, int gpu_VDW_Kind,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int gpu_isMartini, double gpu_rCut, double gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            int gpu_count, double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_rMin, double *gpu_rMaxSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:                            double *gpu_expConst);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombParticleGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_lambdaCoulomb, bool sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cuh:    uint sc_power, double *gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombParticleGPUNoLambda(double distSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:    int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombShiftGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                      int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                      double gpu_rCut, double gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                      double *gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombShiftGPUNoLambda(double distSq, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombExp6GPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                     int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                     double gpu_lambdaCoulomb, bool sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                     uint sc_power, double *gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombExp6GPUNoLambda(double distSq, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombSwitchMartiniGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_diElectric_1,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_lambdaCoulomb,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombSwitchMartiniGPUNoLambda(double distSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:    int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_alpha,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_diElectric_1);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombSwitchGPU(double distSq, int index, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                       double gpu_alpha, int gpu_ewald,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                       double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                       double gpu_lambdaCoulomb, bool sc_coul,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                       uint sc_power, double *gpu_sigmaSq);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcCoulombSwitchGPUNoLambda(double distSq, double qi_qj_fact,
src/GPU/CalculateEnergyCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha, double gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnParticleGPU(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                    double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnParticleGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_epsilon_Cn);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnShiftGPU(double distSq, int index, double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 double *gpu_n, double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                 double gpu_rCut, double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnShiftGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_n, double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rCut);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnExp6GPU(double distSq, int index, double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                double *gpu_n, double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                uint sc_power, double *gpu_rMin,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                double *gpu_rMaxSq, double *gpu_expConst);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnExp6GPUNoLambda(double distSq, int index, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                        double *gpu_rMin, double *gpu_expConst);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnSwitchMartiniGPU(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rCut, double gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_lambdaVDW,
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnSwitchMartiniGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rCut,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnSwitchGPU(double distSq, int index, double *gpu_sigmaSq,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                  double *gpu_n, double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                  double gpu_rCut, double gpu_rOn,
src/GPU/CalculateEnergyCUDAKernel.cuh:                                  double gpu_lambdaVDW, double sc_sigma_6,
src/GPU/CalculateEnergyCUDAKernel.cuh:__device__ double CalcEnSwitchGPUNoLambda(double distSq, int index,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double *gpu_epsilon_Cn,
src/GPU/CalculateEnergyCUDAKernel.cuh:    double gpu_rCut, double gpu_rOn);
src/GPU/CalculateEnergyCUDAKernel.cuh:#endif /*GOMC_CUDA*/
src/GPU/TransformParticlesCUDAKernel.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/TransformParticlesCUDAKernel.cuh:#ifdef GOMC_CUDA
src/GPU/TransformParticlesCUDAKernel.cuh:#include <cuda.h>
src/GPU/TransformParticlesCUDAKernel.cuh:#include <cuda_runtime.h>
src/GPU/TransformParticlesCUDAKernel.cuh:#include "VariablesCUDA.cuh"
src/GPU/TransformParticlesCUDAKernel.cuh:void CallTranslateParticlesGPU(VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cuh:void CallRotateParticlesGPU(VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cuh:    int *gpu_particleMol,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_Invcell_z,
src/GPU/TransformParticlesCUDAKernel.cuh:    int *gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_t_k_x,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_t_k_y,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_t_k_z,
src/GPU/TransformParticlesCUDAKernel.cuh:    int8_t *gpu_isMoleculeInvolved,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_mForceRecx,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_mForceRecy,
src/GPU/TransformParticlesCUDAKernel.cuh:    double *gpu_mForceRecz);
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      int *gpu_particleMol,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_Invcell_z,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      int *gpu_nonOrth,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_r_k_x,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_r_k_y,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      double *gpu_r_k_z,
src/GPU/TransformParticlesCUDAKernel.cuh:                                      int8_t *gpu_isMoleculeInvolved);
src/GPU/TransformParticlesCUDAKernel.cuh:void BrownianMotionRotateParticlesGPU(
src/GPU/TransformParticlesCUDAKernel.cuh:  VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cuh:void BrownianMotionTranslateParticlesGPU(
src/GPU/TransformParticlesCUDAKernel.cuh:  VariablesCUDA *vars,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_r_k_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_r_k_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_r_k_z,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_Invcell_z,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_z,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_comx,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_comy,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_comz,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_t_k_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_t_k_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_t_k_z,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_cell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_cell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_cell_z,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_Invcell_x,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_Invcell_y,
src/GPU/TransformParticlesCUDAKernel.cuh:  double *gpu_Invcell_z,
src/GPU/CalculateForceCUDAKernel.cu:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CalculateForceCUDAKernel.cu:#ifdef GOMC_CUDA
src/GPU/CalculateForceCUDAKernel.cu:#include <cuda.h>
src/GPU/CalculateForceCUDAKernel.cu:#include "CalculateForceCUDAKernel.cuh"
src/GPU/CalculateForceCUDAKernel.cu:#include "CalculateEnergyCUDAKernel.cuh"
src/GPU/CalculateForceCUDAKernel.cu:#include "ConstantDefinitionsCUDAKernel.cuh"
src/GPU/CalculateForceCUDAKernel.cu:#include "CalculateMinImageCUDAKernel.cuh"
src/GPU/CalculateForceCUDAKernel.cu:#include "CUDAMemoryManager.cuh"
src/GPU/CalculateForceCUDAKernel.cu:void CallBoxInterForceGPU(VariablesCUDA *vars,
src/GPU/CalculateForceCUDAKernel.cu:  int *gpu_particleKind;
src/GPU/CalculateForceCUDAKernel.cu:  int *gpu_particleMol;
src/GPU/CalculateForceCUDAKernel.cu:  int *gpu_neighborList, *gpu_cellStartIndex;
src/GPU/CalculateForceCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateForceCUDAKernel.cu:  double *gpu_final_value;
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_neighborList, numberOfCellPairs * sizeof(int));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_cellStartIndex,
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleKind, particleKind.size() * sizeof(int));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleMol, particleMol.size() * sizeof(int));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_value, sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_rT11, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_rT12, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_rT13, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_rT22, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_rT23, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_rT33, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_vT11, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_vT12, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_vT13, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_vT22, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_vT23, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_vT33, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_mapParticleToCell, &mapParticleToCell[0],
src/GPU/CalculateForceCUDAKernel.cu:             atomNumber * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_neighborList, &neighborlist1D[0],
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_cellStartIndex, &cellStartIndex[0],
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_cellVector, &cellVector[0],
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, currentCoords.x, atomNumber * sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, currentCoords.y, atomNumber * sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, currentCoords.z, atomNumber * sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_comx, currentCOM.x, molNumber * sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_comy, currentCOM.y, molNumber * sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_comz, currentCOM.z, molNumber * sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_particleKind, &particleKind[0],
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_particleMol, &particleMol[0],
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  BoxInterForceGPU <<< blocksPerGrid, threadsPerBlock>>>(gpu_cellStartIndex,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cellVector,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_neighborList,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_mapParticleToCell,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_x,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_y,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_comx,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_comy,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_comz,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_particleKind,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_particleMol,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_vT11,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_vT12,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_vT13,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_vT22,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_vT23,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_vT33,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_count,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rCutCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rCutLow,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_nonOrth,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_lambdaCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateForceCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT11,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT11,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&vT11, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT12,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&vT12, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT13,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&vT13, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT22,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&vT22, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT23,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&vT23, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_vT33,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&vT33, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cu:                      gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:    cudaMemcpy(&rT11, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:               cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cu:                      gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:    cudaMemcpy(&rT12, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:               cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cu:                      gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:    cudaMemcpy(&rT13, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:               cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cu:                      gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:    cudaMemcpy(&rT22, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:               cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cu:                      gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:    cudaMemcpy(&rT23, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:               cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cu:                      gpu_final_value, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:    cudaMemcpy(&rT33, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:               cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(vars->gpu_rT11);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(vars->gpu_rT12);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(vars->gpu_rT13);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(vars->gpu_rT22);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(vars->gpu_rT23);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(vars->gpu_rT33);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_vT11);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_vT12);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_vT13);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_vT22);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_vT23);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_vT33);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_particleKind);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_particleMol);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_final_value);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_neighborList);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_cellStartIndex);
src/GPU/CalculateForceCUDAKernel.cu:void CallBoxForceGPU(VariablesCUDA *vars,
src/GPU/CalculateForceCUDAKernel.cu:  int *gpu_particleKind, *gpu_particleMol;
src/GPU/CalculateForceCUDAKernel.cu:  int *gpu_neighborList, *gpu_cellStartIndex;
src/GPU/CalculateForceCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateForceCUDAKernel.cu:  double *gpu_REn, *gpu_LJEn;
src/GPU/CalculateForceCUDAKernel.cu:  double *gpu_final_REn, *gpu_final_LJEn;
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge, particleCharge.size() * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_neighborList, numberOfCellPairs * sizeof(int));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_cellStartIndex, cellStartIndex.size() * sizeof(int));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleKind, particleKind.size() * sizeof(int));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleMol, particleMol.size() * sizeof(int));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_LJEn, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_LJEn, sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &gpu_REn, energyVectorLen * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:    CUMALLOC((void**) &gpu_final_REn, sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  // Copy necessary data to GPU
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_aForcex, aForcex, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_aForcey, aForcey, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_aForcez, aForcez, atomCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcex, mForcex, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcey, mForcey, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForcez, mForcez, molCount * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_mapParticleToCell, &mapParticleToCell[0], atomNumber * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_neighborList, &neighborlist1D[0], numberOfCellPairs * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_cellStartIndex, &cellStartIndex[0], cellStartIndex.size() * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_cellVector, &cellVector[0], atomNumber * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0], particleCharge.size() * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_particleKind, &particleKind[0], particleKind.size() * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_particleMol, &particleMol[0], particleMol.size() * sizeof(int), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  BoxForceGPU <<< blocksPerGrid, threadsPerBlock>>>(gpu_cellStartIndex,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cellVector,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_neighborList,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_mapParticleToCell,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_x,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_y,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_particleKind,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_particleMol,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_REn,
src/GPU/CalculateForceCUDAKernel.cu:      gpu_LJEn,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_count,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rCutCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rCutLow,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_nonOrth,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cell_x[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cell_y[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_cell_z[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_Invcell_x[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_Invcell_y[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_Invcell_z[box],
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_aForcex,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_aForcey,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_aForcez,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_mForcex,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_mForcey,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_mForcez,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_lambdaCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:      vars->gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateForceCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_LJEn,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_LJEn, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_LJEn,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_LJEn, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:  CubDebugExit(cudaMemcpy(&cpu_final_LJEn, gpu_final_LJEn, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:                          cudaMemcpyDeviceToHost));
src/GPU/CalculateForceCUDAKernel.cu:    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_REn,
src/GPU/CalculateForceCUDAKernel.cu:                      gpu_final_REn, energyVectorLen);
src/GPU/CalculateForceCUDAKernel.cu:    CubDebugExit(cudaMemcpy(&cpu_final_REn, gpu_final_REn, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:                            cudaMemcpyDeviceToHost));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(aForcex, vars->gpu_aForcex, sizeof(double) * atomCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(aForcey, vars->gpu_aForcey, sizeof(double) * atomCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(aForcez, vars->gpu_aForcez, sizeof(double) * atomCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(mForcex, vars->gpu_mForcex, sizeof(double) * molCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(mForcey, vars->gpu_mForcey, sizeof(double) * molCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(mForcez, vars->gpu_mForcez, sizeof(double) * molCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_particleKind);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_particleMol);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_LJEn);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_final_LJEn);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(gpu_REn);
src/GPU/CalculateForceCUDAKernel.cu:    CUFREE(gpu_final_REn);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_neighborList);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_cellStartIndex);
src/GPU/CalculateForceCUDAKernel.cu:void CallVirialReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateForceCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateForceCUDAKernel.cu:  double *gpu_final_value;
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_value, sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, currentCoords.x, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, currentCoords.y, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, currentCoords.z, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_dx, currentCOMDiff.x, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_dy, currentCOMDiff.y, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(vars->gpu_dz, currentCOMDiff.z, atomNumber * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rT11, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemset(vars->gpu_rT11, 0, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rT12, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemset(vars->gpu_rT12, 0, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rT13, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemset(vars->gpu_rT13, 0, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rT22, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemset(vars->gpu_rT22, 0, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rT23, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemset(vars->gpu_rT23, 0, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rT33, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemset(vars->gpu_rT33, 0, imageSize * sizeof(double));
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0], particleCharge.size() * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/CalculateForceCUDAKernel.cu:  VirialReciprocalGPU <<< blocksPerGrid,
src/GPU/CalculateForceCUDAKernel.cu:                      threadsPerBlock>>>(vars->gpu_x,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_y,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_dx,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_dy,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_dz,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_kxRef[box],
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_kyRef[box],
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_kzRef[box],
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_prefactRef[box],
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_hsqrRef[box],
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_sumRref[box],
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_sumIref[box],
src/GPU/CalculateForceCUDAKernel.cu:                                         gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cu:                                         vars->gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateForceCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, imageSize);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, imageSize);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&rT11, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, imageSize);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&rT12, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, imageSize);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&rT13, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, imageSize);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&rT22, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, imageSize);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&rT23, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, vars->gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cu:                    gpu_final_value, imageSize);
src/GPU/CalculateForceCUDAKernel.cu:  cudaMemcpy(&rT33, gpu_final_value, sizeof(double),
src/GPU/CalculateForceCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_rT11);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_rT12);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_rT13);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_rT22);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_rT23);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(vars->gpu_rT33);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateForceCUDAKernel.cu:  CUFREE(gpu_final_value);
src/GPU/CalculateForceCUDAKernel.cu:__global__ void BoxInterForceGPU(int *gpu_cellStartIndex,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_cellVector,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_neighborList,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_mapParticleToCell,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_x,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_y,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_comx,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_comy,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_comz,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_particleKind,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_particleMol,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_vT11,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_vT12,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_vT13,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_vT22,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_vT23,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_vT33,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_count,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rCutCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rCutLow,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_cell_x,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_cell_y,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_cell_z,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_Invcell_x,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_Invcell_y,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_Invcell_z,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_nonOrth,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cu:                                 int *gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_lambdaCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:                                 bool *gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cu:  gpu_vT11[threadID] = 0.0, gpu_vT22[threadID] = 0.0, gpu_vT33[threadID] = 0.0;
src/GPU/CalculateForceCUDAKernel.cu:  gpu_vT12[threadID] = 0.0, gpu_vT13[threadID] = 0.0, gpu_vT23[threadID] = 0.0;
src/GPU/CalculateForceCUDAKernel.cu:    gpu_rT11[threadID] = 0.0, gpu_rT22[threadID] = 0.0, gpu_rT33[threadID] = 0.0;
src/GPU/CalculateForceCUDAKernel.cu:    gpu_rT12[threadID] = 0.0, gpu_rT13[threadID] = 0.0, gpu_rT23[threadID] = 0.0;
src/GPU/CalculateForceCUDAKernel.cu:  double cutoff = fmax(gpu_rCut[0], gpu_rCutCoulomb[box]);
src/GPU/CalculateForceCUDAKernel.cu:  int neighborCell = gpu_neighborList[nCellIndex];
src/GPU/CalculateForceCUDAKernel.cu:  int endIndex = gpu_cellStartIndex[neighborCell + 1];
src/GPU/CalculateForceCUDAKernel.cu:  particlesInsideNeighboringCell = endIndex - gpu_cellStartIndex[neighborCell];
src/GPU/CalculateForceCUDAKernel.cu:  endIndex = gpu_cellStartIndex[currentCell + 1];
src/GPU/CalculateForceCUDAKernel.cu:  particlesInsideCurrentCell = endIndex - gpu_cellStartIndex[currentCell];
src/GPU/CalculateForceCUDAKernel.cu:    int currentParticle = gpu_cellVector[gpu_cellStartIndex[currentCell] + currentParticleIndex];
src/GPU/CalculateForceCUDAKernel.cu:    int neighborParticle = gpu_cellVector[gpu_cellStartIndex[neighborCell] + neighborParticleIndex];
src/GPU/CalculateForceCUDAKernel.cu:    if(currentParticle < neighborParticle && gpu_particleMol[currentParticle] != gpu_particleMol[neighborParticle]) {
src/GPU/CalculateForceCUDAKernel.cu:      if(InRcutGPU(distSq, virComponents, gpu_x, gpu_y, gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:                   axis, halfAx, cutoff, gpu_nonOrth[0], gpu_cell_x,
src/GPU/CalculateForceCUDAKernel.cu:                   gpu_cell_y, gpu_cell_z, gpu_Invcell_x, gpu_Invcell_y,
src/GPU/CalculateForceCUDAKernel.cu:                   gpu_Invcell_z)) {
src/GPU/CalculateForceCUDAKernel.cu:        int kA = gpu_particleKind[currentParticle];
src/GPU/CalculateForceCUDAKernel.cu:        int kB = gpu_particleKind[neighborParticle];
src/GPU/CalculateForceCUDAKernel.cu:        int mA = gpu_particleMol[currentParticle];
src/GPU/CalculateForceCUDAKernel.cu:        int mB = gpu_particleMol[neighborParticle];
src/GPU/CalculateForceCUDAKernel.cu:        double lambdaVDW = DeviceGetLambdaVDW(mA, mB, box, gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cu:                                              gpu_molIndex, gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cu:        diff_com = Difference(gpu_comx, gpu_comy, gpu_comz, mA, mB);
src/GPU/CalculateForceCUDAKernel.cu:        if (gpu_nonOrth[0])
src/GPU/CalculateForceCUDAKernel.cu:          diff_com = MinImageNonOrthGPU(diff_com, axis, halfAx, gpu_cell_x, gpu_cell_y, gpu_cell_z,
src/GPU/CalculateForceCUDAKernel.cu:                                        gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateForceCUDAKernel.cu:          diff_com = MinImageGPU(diff_com, axis, halfAx);
src/GPU/CalculateForceCUDAKernel.cu:        double pVF = CalcEnForceGPU(distSq, kA, kB,
src/GPU/CalculateForceCUDAKernel.cu:                                    gpu_sigmaSq, gpu_n, gpu_epsilon_Cn, gpu_rCut[0],
src/GPU/CalculateForceCUDAKernel.cu:                                    gpu_rOn[0], gpu_isMartini[0], gpu_VDW_Kind[0],
src/GPU/CalculateForceCUDAKernel.cu:                                    gpu_count[0], lambdaVDW, sc_sigma_6, sc_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                    sc_power, gpu_rMin, gpu_rMaxSq, gpu_expConst);
src/GPU/CalculateForceCUDAKernel.cu:        gpu_vT11[threadID] += pVF * (virComponents.x * diff_com.x);
src/GPU/CalculateForceCUDAKernel.cu:        gpu_vT22[threadID] += pVF * (virComponents.y * diff_com.y);
src/GPU/CalculateForceCUDAKernel.cu:        gpu_vT33[threadID] += pVF * (virComponents.z * diff_com.z);
src/GPU/CalculateForceCUDAKernel.cu:        gpu_vT12[threadID] += pVF * (0.5 * (virComponents.x * diff_com.y + virComponents.y * diff_com.x));
src/GPU/CalculateForceCUDAKernel.cu:        gpu_vT13[threadID] += pVF * (0.5 * (virComponents.x * diff_com.z + virComponents.z * diff_com.x));
src/GPU/CalculateForceCUDAKernel.cu:        gpu_vT23[threadID] += pVF * (0.5 * (virComponents.y * diff_com.z + virComponents.z * diff_com.y));
src/GPU/CalculateForceCUDAKernel.cu:          double qi_qj = gpu_particleCharge[currentParticle] * gpu_particleCharge[neighborParticle];
src/GPU/CalculateForceCUDAKernel.cu:                                   gpu_isFraction, gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cu:                                   gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cu:            double pRF = CalcCoulombForceGPU(distSq, qi_qj, gpu_VDW_Kind[0], gpu_ewald[0],
src/GPU/CalculateForceCUDAKernel.cu:                                             gpu_isMartini[0], gpu_alpha[box],
src/GPU/CalculateForceCUDAKernel.cu:                                             gpu_rCutCoulomb[box], gpu_diElectric_1[0],
src/GPU/CalculateForceCUDAKernel.cu:                                             gpu_sigmaSq, sc_coul, sc_sigma_6, sc_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                             sc_power, lambdaCoulomb, gpu_count[0],
src/GPU/CalculateForceCUDAKernel.cu:            gpu_rT11[threadID] += pRF * (virComponents.x * diff_com.x);
src/GPU/CalculateForceCUDAKernel.cu:            gpu_rT22[threadID] += pRF * (virComponents.y * diff_com.y);
src/GPU/CalculateForceCUDAKernel.cu:            gpu_rT33[threadID] += pRF * (virComponents.z * diff_com.z);
src/GPU/CalculateForceCUDAKernel.cu:            gpu_rT12[threadID] += pRF * (0.5 * (virComponents.x * diff_com.y + virComponents.y * diff_com.x));
src/GPU/CalculateForceCUDAKernel.cu:            gpu_rT13[threadID] += pRF * (0.5 * (virComponents.x * diff_com.z + virComponents.z * diff_com.x));
src/GPU/CalculateForceCUDAKernel.cu:            gpu_rT23[threadID] += pRF * (0.5 * (virComponents.y * diff_com.z + virComponents.z * diff_com.y));
src/GPU/CalculateForceCUDAKernel.cu:__global__ void BoxForceGPU(int *gpu_cellStartIndex,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_cellVector,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_neighborList,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_mapParticleToCell,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_x,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_y,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_particleKind,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_particleMol,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_REn,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_LJEn,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_count,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_rCutCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_rCutLow,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_nonOrth,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_cell_x,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_cell_y,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_cell_z,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_Invcell_x,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_Invcell_y,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_Invcell_z,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_aForcex,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_aForcey,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_aForcez,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_mForcex,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_mForcey,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_mForcez,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cu:                            int *gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cu:                            double *gpu_lambdaCoulomb,
src/GPU/CalculateForceCUDAKernel.cu:                            bool *gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cu:  double cutoff = fmax(gpu_rCut[0], gpu_rCutCoulomb[box]);
src/GPU/CalculateForceCUDAKernel.cu:  int neighborCell = gpu_neighborList[nCellIndex];
src/GPU/CalculateForceCUDAKernel.cu:  int endIndex = gpu_cellStartIndex[neighborCell + 1];
src/GPU/CalculateForceCUDAKernel.cu:  particlesInsideNeighboringCell = endIndex - gpu_cellStartIndex[neighborCell];
src/GPU/CalculateForceCUDAKernel.cu:  endIndex = gpu_cellStartIndex[currentCell + 1];
src/GPU/CalculateForceCUDAKernel.cu:  particlesInsideCurrentCell = endIndex - gpu_cellStartIndex[currentCell];
src/GPU/CalculateForceCUDAKernel.cu:    int currentParticle = gpu_cellVector[gpu_cellStartIndex[currentCell] + currentParticleIndex];
src/GPU/CalculateForceCUDAKernel.cu:    int neighborParticle = gpu_cellVector[gpu_cellStartIndex[neighborCell] + neighborParticleIndex];
src/GPU/CalculateForceCUDAKernel.cu:    if(currentParticle < neighborParticle && gpu_particleMol[currentParticle] != gpu_particleMol[neighborParticle]) {
src/GPU/CalculateForceCUDAKernel.cu:      if(InRcutGPU(distSq, virComponents, gpu_x, gpu_y, gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:                   axis, halfAx, cutoff, gpu_nonOrth[0], gpu_cell_x,
src/GPU/CalculateForceCUDAKernel.cu:                   gpu_cell_y, gpu_cell_z, gpu_Invcell_x, gpu_Invcell_y,
src/GPU/CalculateForceCUDAKernel.cu:                   gpu_Invcell_z)) {
src/GPU/CalculateForceCUDAKernel.cu:        int kA = gpu_particleKind[currentParticle];
src/GPU/CalculateForceCUDAKernel.cu:        int kB = gpu_particleKind[neighborParticle];
src/GPU/CalculateForceCUDAKernel.cu:        int mA = gpu_particleMol[currentParticle];
src/GPU/CalculateForceCUDAKernel.cu:        int mB = gpu_particleMol[neighborParticle];
src/GPU/CalculateForceCUDAKernel.cu:        double lambdaVDW = DeviceGetLambdaVDW(mA, mB, box, gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cu:                                              gpu_molIndex, gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cu:        LJEn += CalcEnGPU(distSq, kA, kB, gpu_sigmaSq, gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                          gpu_epsilon_Cn, gpu_VDW_Kind[0],
src/GPU/CalculateForceCUDAKernel.cu:                          gpu_isMartini[0], gpu_rCut[0],
src/GPU/CalculateForceCUDAKernel.cu:                          gpu_rOn[0], gpu_count[0], lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cu:                          gpu_rMin, gpu_rMaxSq, gpu_expConst);
src/GPU/CalculateForceCUDAKernel.cu:        double pVF = CalcEnForceGPU(distSq, kA, kB, gpu_sigmaSq, gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                    gpu_epsilon_Cn, gpu_rCut[0], gpu_rOn[0],
src/GPU/CalculateForceCUDAKernel.cu:                                    gpu_isMartini[0], gpu_VDW_Kind[0],
src/GPU/CalculateForceCUDAKernel.cu:                                    gpu_count[0], lambdaVDW, sc_sigma_6,
src/GPU/CalculateForceCUDAKernel.cu:                                    sc_alpha, sc_power, gpu_rMin, gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cu:                                    gpu_expConst);
src/GPU/CalculateForceCUDAKernel.cu:          double qi_qj_fact = gpu_particleCharge[currentParticle] * gpu_particleCharge[neighborParticle];
src/GPU/CalculateForceCUDAKernel.cu:            qi_qj_fact *= qqFactGPU;
src/GPU/CalculateForceCUDAKernel.cu:                                   gpu_isFraction, gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cu:                                   gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cu:            REn += CalcCoulombGPU(distSq, kA, kB,
src/GPU/CalculateForceCUDAKernel.cu:                                  qi_qj_fact, gpu_rCutLow[0],
src/GPU/CalculateForceCUDAKernel.cu:                                  gpu_ewald[0], gpu_VDW_Kind[0],
src/GPU/CalculateForceCUDAKernel.cu:                                  gpu_alpha[box],
src/GPU/CalculateForceCUDAKernel.cu:                                  gpu_rCutCoulomb[box],
src/GPU/CalculateForceCUDAKernel.cu:                                  gpu_isMartini[0],
src/GPU/CalculateForceCUDAKernel.cu:                                  gpu_diElectric_1[0],
src/GPU/CalculateForceCUDAKernel.cu:                                  gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                                  gpu_count[0]);
src/GPU/CalculateForceCUDAKernel.cu:            double coulombVir = CalcCoulombForceGPU(distSq, qi_qj_fact,
src/GPU/CalculateForceCUDAKernel.cu:                                                    gpu_VDW_Kind[0], gpu_ewald[0],
src/GPU/CalculateForceCUDAKernel.cu:                                                    gpu_isMartini[0],
src/GPU/CalculateForceCUDAKernel.cu:                                                    gpu_alpha[box],
src/GPU/CalculateForceCUDAKernel.cu:                                                    gpu_rCutCoulomb[box],
src/GPU/CalculateForceCUDAKernel.cu:                                                    gpu_diElectric_1[0],
src/GPU/CalculateForceCUDAKernel.cu:                                                    gpu_sigmaSq, sc_coul,
src/GPU/CalculateForceCUDAKernel.cu:                                                    gpu_count[0], kA, kB);
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_aForcex[currentParticle], forceReal.x + forceLJ.x);
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_aForcey[currentParticle], forceReal.y + forceLJ.y);
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_aForcez[currentParticle], forceReal.z + forceLJ.z);
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_aForcex[neighborParticle], -1.0 * (forceReal.x + forceLJ.x));
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_aForcey[neighborParticle], -1.0 * (forceReal.y + forceLJ.y));
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_aForcez[neighborParticle], -1.0 * (forceReal.z + forceLJ.z));
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_mForcex[mA], forceReal.x + forceLJ.x);
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_mForcey[mA], forceReal.y + forceLJ.y);
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_mForcez[mA], forceReal.z + forceLJ.z);
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_mForcex[mB], -1.0 * (forceReal.x + forceLJ.x));
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_mForcey[mB], -1.0 * (forceReal.y + forceLJ.y));
src/GPU/CalculateForceCUDAKernel.cu:        atomicAdd(&gpu_mForcez[mB], -1.0 * (forceReal.z + forceLJ.z));
src/GPU/CalculateForceCUDAKernel.cu:  gpu_LJEn[threadID] = LJEn;
src/GPU/CalculateForceCUDAKernel.cu:    gpu_REn[threadID] = REn;
src/GPU/CalculateForceCUDAKernel.cu:__global__ void VirialReciprocalGPU(double *gpu_x,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_y,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_z,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_comDx,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_comDy,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_comDz,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_kxRef,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_kyRef,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_kzRef,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_prefactRef,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_hsqrRef,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_sumRref,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_sumIref,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cu:                                    double *gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cu:    shared_coords[threadIdx.x * 7    ] = gpu_x[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateForceCUDAKernel.cu:    shared_coords[threadIdx.x * 7 + 1] = gpu_y[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateForceCUDAKernel.cu:    shared_coords[threadIdx.x * 7 + 2] = gpu_z[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateForceCUDAKernel.cu:    shared_coords[threadIdx.x * 7 + 3] = gpu_comDx[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateForceCUDAKernel.cu:    shared_coords[threadIdx.x * 7 + 4] = gpu_comDy[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateForceCUDAKernel.cu:    shared_coords[threadIdx.x * 7 + 5] = gpu_comDz[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateForceCUDAKernel.cu:    shared_coords[threadIdx.x * 7 + 6] = gpu_particleCharge[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateForceCUDAKernel.cu:    double constant_part = constVal + 1.0 / gpu_hsqrRef[imageID];
src/GPU/CalculateForceCUDAKernel.cu:    factor = gpu_prefactRef[imageID] * (gpu_sumRref[imageID] * gpu_sumRref[imageID] +
src/GPU/CalculateForceCUDAKernel.cu:                                        gpu_sumIref[imageID] * gpu_sumIref[imageID]);
src/GPU/CalculateForceCUDAKernel.cu:    rT11 = factor * (1.0 - 2.0 * constant_part * gpu_kxRef[imageID] * gpu_kxRef[imageID]);
src/GPU/CalculateForceCUDAKernel.cu:    rT12 = factor * (-2.0 * constant_part * gpu_kxRef[imageID] * gpu_kyRef[imageID]);
src/GPU/CalculateForceCUDAKernel.cu:    rT13 = factor * (-2.0 * constant_part * gpu_kxRef[imageID] * gpu_kzRef[imageID]);
src/GPU/CalculateForceCUDAKernel.cu:    rT22 = factor * (1.0 - 2.0 * constant_part * gpu_kyRef[imageID] * gpu_kyRef[imageID]);
src/GPU/CalculateForceCUDAKernel.cu:    rT23 = factor * (-2.0 * constant_part * gpu_kyRef[imageID] * gpu_kzRef[imageID]);
src/GPU/CalculateForceCUDAKernel.cu:    rT33 = factor * (1.0 - 2.0 * constant_part * gpu_kzRef[imageID] * gpu_kzRef[imageID]);
src/GPU/CalculateForceCUDAKernel.cu:    dot = DotProductGPU(gpu_kxRef[imageID], gpu_kyRef[imageID],
src/GPU/CalculateForceCUDAKernel.cu:                        gpu_kzRef[imageID], shared_coords[particleID * 7],
src/GPU/CalculateForceCUDAKernel.cu:    factor = gpu_prefactRef[imageID] * 2.0 * shared_coords[particleID * 7 + 6] *
src/GPU/CalculateForceCUDAKernel.cu:             (gpu_sumIref[imageID] * dotcos - gpu_sumRref[imageID] * dotsin);
src/GPU/CalculateForceCUDAKernel.cu:    rT11 += factor * (gpu_kxRef[imageID] * shared_coords[particleID * 7 + 3]);
src/GPU/CalculateForceCUDAKernel.cu:    rT12 += factor * 0.5 * (gpu_kxRef[imageID] * shared_coords[particleID * 7 + 4] + gpu_kyRef[imageID] * shared_coords[particleID * 7 + 3]);
src/GPU/CalculateForceCUDAKernel.cu:    rT13 += factor * 0.5 * (gpu_kxRef[imageID] * shared_coords[particleID * 7 + 5] + gpu_kzRef[imageID] * shared_coords[particleID * 7 + 3]);
src/GPU/CalculateForceCUDAKernel.cu:    rT22 += factor * (gpu_kyRef[imageID] * shared_coords[particleID * 7 + 4]);
src/GPU/CalculateForceCUDAKernel.cu:    rT23 += factor * 0.5 * (gpu_kyRef[imageID] * shared_coords[particleID * 7 + 5] + gpu_kzRef[imageID] * shared_coords[particleID * 7 + 4]);
src/GPU/CalculateForceCUDAKernel.cu:    rT33 += factor * (gpu_kzRef[imageID] * shared_coords[particleID * 7 + 5]);
src/GPU/CalculateForceCUDAKernel.cu:  atomicAdd(&gpu_rT11[imageID], rT11);
src/GPU/CalculateForceCUDAKernel.cu:  atomicAdd(&gpu_rT12[imageID], rT12);
src/GPU/CalculateForceCUDAKernel.cu:  atomicAdd(&gpu_rT13[imageID], rT13);
src/GPU/CalculateForceCUDAKernel.cu:  atomicAdd(&gpu_rT22[imageID], rT22);
src/GPU/CalculateForceCUDAKernel.cu:  atomicAdd(&gpu_rT23[imageID], rT23);
src/GPU/CalculateForceCUDAKernel.cu:  atomicAdd(&gpu_rT33[imageID], rT33);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcEnForceGPU(double distSq, int kind1, int kind2,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                                 double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:                                 double gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cu:                                 int gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cu:                                 int gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cu:                                 int gpu_count,
src/GPU/CalculateForceCUDAKernel.cu:                                 double gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_expConst)
src/GPU/CalculateForceCUDAKernel.cu:  if((gpu_rCut * gpu_rCut) < distSq) {
src/GPU/CalculateForceCUDAKernel.cu:  int index = FlatIndexGPU(kind1, kind2, gpu_count);
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_VDW_Kind == GPU_VDW_STD_KIND) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirParticleGPU(distSq, index, gpu_sigmaSq[index], gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                              gpu_epsilon_Cn, sc_sigma_6, sc_alpha, sc_power, gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_SHIFT_KIND) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirShiftGPU(distSq, index, gpu_sigmaSq[index], gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                           sc_sigma_6, sc_alpha, sc_power, gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_EXP6_KIND) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirExp6GPU(distSq, index, gpu_sigmaSq[index], gpu_n, gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cu:                          gpu_rMaxSq, gpu_expConst, sc_sigma_6,
src/GPU/CalculateForceCUDAKernel.cu:                          sc_alpha, sc_power, gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cu:  } else if(gpu_VDW_Kind == GPU_VDW_SWITCH_KIND && gpu_isMartini) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirSwitchMartiniGPU(distSq, index, gpu_sigmaSq[index], gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                   gpu_epsilon_Cn, gpu_rCut, gpu_rOn, sc_sigma_6, sc_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                   sc_power, gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirSwitchGPU(distSq, index, gpu_sigmaSq[index], gpu_epsilon_Cn, gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                            gpu_rCut, gpu_rOn);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirParticleGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:    int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_lambdaCoulomb)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcCoulombVirParticleGPU(distSq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:    double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirParticleGPU(softRsq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirParticleGPU(distSq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirParticleGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald, double gpu_alpha)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateForceCUDAKernel.cu:    double constValue = gpu_alpha * M_2_SQRTPI;
src/GPU/CalculateForceCUDAKernel.cu:    double expConstValue = exp(-1.0 * gpu_alpha * gpu_alpha * distSq);
src/GPU/CalculateForceCUDAKernel.cu:    double temp = 1.0 - erf(gpu_alpha * dist);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirShiftGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:    int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_lambdaCoulomb)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcCoulombVirShiftGPU(distSq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:    double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirShiftGPU(softRsq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirShiftGPU(distSq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirShiftGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald, double gpu_alpha)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateForceCUDAKernel.cu:    double constValue = gpu_alpha * M_2_SQRTPI;
src/GPU/CalculateForceCUDAKernel.cu:    double expConstValue = exp(-1.0 * gpu_alpha * gpu_alpha * distSq);
src/GPU/CalculateForceCUDAKernel.cu:    double temp = 1.0 - erf(gpu_alpha * dist);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirExp6GPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:                                        int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                        int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                                        double gpu_lambdaCoulomb)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcCoulombVirExp6GPU(distSq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:    double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirExp6GPU(softRsq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirExp6GPU(distSq, qi_qj, gpu_ewald, gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirExp6GPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:                                        int gpu_ewald, double gpu_alpha)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateForceCUDAKernel.cu:    double constValue = gpu_alpha * M_2_SQRTPI;
src/GPU/CalculateForceCUDAKernel.cu:    double expConstValue = exp(-1.0 * gpu_alpha * gpu_alpha * distSq);
src/GPU/CalculateForceCUDAKernel.cu:    double temp = erfc(gpu_alpha * dist);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirSwitchMartiniGPU(double distSq,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_lambdaCoulomb)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcCoulombVirSwitchMartiniGPU(distSq, qi_qj, gpu_ewald, gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                          gpu_rCut, gpu_diElectric_1);
src/GPU/CalculateForceCUDAKernel.cu:    double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirSwitchMartiniGPU(softRsq, qi_qj, gpu_ewald, gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                          gpu_rCut, gpu_diElectric_1);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirSwitchMartiniGPU(distSq, qi_qj, gpu_ewald, gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                          gpu_rCut, gpu_diElectric_1);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirSwitchMartiniGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_diElectric_1)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateForceCUDAKernel.cu:    double constValue = gpu_alpha * M_2_SQRTPI;
src/GPU/CalculateForceCUDAKernel.cu:    double expConstValue = exp(-1.0 * gpu_alpha * gpu_alpha * distSq);
src/GPU/CalculateForceCUDAKernel.cu:    double temp = 1.0 - erf(gpu_alpha * dist);
src/GPU/CalculateForceCUDAKernel.cu:    // double A1 = 1.0 * (-(1.0 + 4) * gpu_rCut) / (pow(gpu_rCut, 1.0 + 2) *
src/GPU/CalculateForceCUDAKernel.cu:                // pow(gpu_rCut, 2));
src/GPU/CalculateForceCUDAKernel.cu:    // double B1 = -1.0 * (-(1.0 + 3) * gpu_rCut) / (pow(gpu_rCut, 1.0 + 2) *
src/GPU/CalculateForceCUDAKernel.cu:                // pow(gpu_rCut, 3));
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_invrCut = 1.0/gpu_rCut;
src/GPU/CalculateForceCUDAKernel.cu:    double A1 = -5.0 * gpu_invrCut * gpu_invrCut * gpu_invrCut * gpu_invrCut;
src/GPU/CalculateForceCUDAKernel.cu:    double B1 = 4.0 * gpu_invrCut * gpu_invrCut * gpu_invrCut * gpu_invrCut * gpu_invrCut;
src/GPU/CalculateForceCUDAKernel.cu:    return qi_qj * gpu_diElectric_1 * (rij_ronCoul_3 + virCoul / dist);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirSwitchGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_rCut, int index,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_sigmaSq, bool sc_coul,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_lambdaCoulomb)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaCoulomb >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcCoulombVirSwitchGPU(distSq, qi_qj, gpu_ewald, gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:                                   gpu_rCut);
src/GPU/CalculateForceCUDAKernel.cu:    double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:    double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaCoulomb), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirSwitchGPU(softRsq, qi_qj, gpu_ewald, gpu_alpha, gpu_rCut);
src/GPU/CalculateForceCUDAKernel.cu:    return gpu_lambdaCoulomb *
src/GPU/CalculateForceCUDAKernel.cu:           CalcCoulombVirSwitchGPU(distSq, qi_qj, gpu_ewald, gpu_alpha, gpu_rCut);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcCoulombVirSwitchGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cu:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_rCut)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_ewald) {
src/GPU/CalculateForceCUDAKernel.cu:    double constValue = gpu_alpha * M_2_SQRTPI;
src/GPU/CalculateForceCUDAKernel.cu:    double expConstValue = exp(-1.0 * gpu_alpha * gpu_alpha * distSq);
src/GPU/CalculateForceCUDAKernel.cu:    double temp = 1.0 - erf(gpu_alpha * dist);
src/GPU/CalculateForceCUDAKernel.cu:    double rCutSq = gpu_rCut * gpu_rCut;
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirParticleGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cu:                                     double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                     double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                                     double gpu_lambdaVDW)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirParticleGPU(distSq, index, gpu_sigmaSq, gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                              gpu_epsilon_Cn);
src/GPU/CalculateForceCUDAKernel.cu:  double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:  return gpu_lambdaVDW * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:         CalcVirParticleGPU(softRsq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirParticleGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cu:                                     double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                     double *gpu_epsilon_Cn)
src/GPU/CalculateForceCUDAKernel.cu:  double rRat2 = gpu_sigmaSq * rNeg2;
src/GPU/CalculateForceCUDAKernel.cu:  double repulse = pow(rRat2, gpu_n[index] * 0.5);
src/GPU/CalculateForceCUDAKernel.cu:  // return gpu_epsilon_Cn[index] * 6.0 *
src/GPU/CalculateForceCUDAKernel.cu:         // ((gpu_n[index] / 6.0) * repulse - attract) * rNeg2;
src/GPU/CalculateForceCUDAKernel.cu:  return gpu_epsilon_Cn[index] * rNeg2 * (gpu_n[index] * repulse - 6.0 * attract);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirShiftGPU(double distSq, int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                                  double *gpu_n, double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                                  uint sc_power, double gpu_lambdaVDW)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirShiftGPU(distSq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn);
src/GPU/CalculateForceCUDAKernel.cu:  double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:  return gpu_lambdaVDW * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:         CalcVirShiftGPU(softRsq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirShiftGPU(double distSq, int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                                  double *gpu_n, double *gpu_epsilon_Cn)
src/GPU/CalculateForceCUDAKernel.cu:  double rRat2 = gpu_sigmaSq * rNeg2;
src/GPU/CalculateForceCUDAKernel.cu:  double repulse = pow(rRat2, gpu_n[index] * 0.5);
src/GPU/CalculateForceCUDAKernel.cu:  // return gpu_epsilon_Cn[index] * 6.0 *
src/GPU/CalculateForceCUDAKernel.cu:         // ((gpu_n[index] / 6.0) * repulse - attract) * rNeg2;
src/GPU/CalculateForceCUDAKernel.cu:  return gpu_epsilon_Cn[index]  * rNeg2 * (gpu_n[index] * repulse - 6.0 * attract);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirExp6GPU(double distSq, int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_n, double *gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rMaxSq, double *gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cu:                                 double gpu_lambdaVDW)
src/GPU/CalculateForceCUDAKernel.cu:  if(distSq < gpu_rMaxSq[index]) {
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirExp6GPU(distSq, index, gpu_n, gpu_rMin, gpu_expConst);
src/GPU/CalculateForceCUDAKernel.cu:  double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:  return gpu_lambdaVDW * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:         CalcVirExp6GPU(softRsq, index, gpu_n, gpu_rMin, gpu_expConst);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirExp6GPU(double distSq, int index, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                 double *gpu_rMin, double *gpu_expConst)
src/GPU/CalculateForceCUDAKernel.cu:  double rRat = gpu_rMin[index] / dist;
src/GPU/CalculateForceCUDAKernel.cu:  uint alpha_ij = gpu_n[index];
src/GPU/CalculateForceCUDAKernel.cu:  double repulse = (dist / gpu_rMin[index]) * exp(alpha_ij *
src/GPU/CalculateForceCUDAKernel.cu:                   (1.0 - dist / gpu_rMin[index]));
src/GPU/CalculateForceCUDAKernel.cu:  return 6.0 * gpu_expConst[index] * (repulse - attract) / distSq;
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirSwitchMartiniGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:    double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_rCut, double gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_lambdaVDW)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirSwitchMartiniGPU(distSq, index, gpu_sigmaSq, gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                                   gpu_epsilon_Cn, gpu_rCut, gpu_rOn);
src/GPU/CalculateForceCUDAKernel.cu:  double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:  return gpu_lambdaVDW * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:         CalcVirSwitchMartiniGPU(softRsq, index, gpu_sigmaSq, gpu_n, gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                                 gpu_rCut, gpu_rOn);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirSwitchMartiniGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:    double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:    double gpu_rCut, double gpu_rOn)
src/GPU/CalculateForceCUDAKernel.cu:  double r_n2 = pow(r_1, gpu_n[index] + 2.0);
src/GPU/CalculateForceCUDAKernel.cu:  double rij_ron = sqrt(distSq) - gpu_rOn;
src/GPU/CalculateForceCUDAKernel.cu:  double pn = gpu_n[index];
src/GPU/CalculateForceCUDAKernel.cu:  double An = pn * ((pn + 1.0) * gpu_rOn - (pn + 4.0) * gpu_rCut) /
src/GPU/CalculateForceCUDAKernel.cu:              (pow(gpu_rCut, pn + 2.0) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn));
src/GPU/CalculateForceCUDAKernel.cu:  double Bn = -pn * ((pn + 1.0) * gpu_rOn - (pn + 3.0) * gpu_rCut) /
src/GPU/CalculateForceCUDAKernel.cu:              (pow(gpu_rCut, pn + 2.0) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateForceCUDAKernel.cu:              (gpu_rCut - gpu_rOn));
src/GPU/CalculateForceCUDAKernel.cu:  double sig6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:  double sign = pow(gpu_sigmaSq, pn * 0.5);
src/GPU/CalculateForceCUDAKernel.cu:  double A6 = 6.0 * (7.0 * gpu_rOn - 10.0 * gpu_rCut) /
src/GPU/CalculateForceCUDAKernel.cu:              (pow(gpu_rCut, 8.0) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn));
src/GPU/CalculateForceCUDAKernel.cu:  double B6 = -6.0 * (7.0 * gpu_rOn - 9.0 * gpu_rCut) /
src/GPU/CalculateForceCUDAKernel.cu:              (pow(gpu_rCut, 8.0) * (gpu_rCut - gpu_rOn) * (gpu_rCut - gpu_rOn) *
src/GPU/CalculateForceCUDAKernel.cu:              (gpu_rCut - gpu_rOn));
src/GPU/CalculateForceCUDAKernel.cu:  const double dshiftRep = ( distSq > gpu_rOn * gpu_rOn ?
src/GPU/CalculateForceCUDAKernel.cu:  const double dshiftAtt = ( distSq > gpu_rOn * gpu_rOn ?
src/GPU/CalculateForceCUDAKernel.cu:  double Wij = gpu_epsilon_Cn[index] * (sign * (pn * r_n2 + dshiftRep) -
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirSwitchGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cu:                                   double gpu_sigmaSq, double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                                   double *gpu_n, double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:                                   double gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cu:                                   double gpu_lambdaVDW)
src/GPU/CalculateForceCUDAKernel.cu:  if(gpu_lambdaVDW >= 0.999999) {
src/GPU/CalculateForceCUDAKernel.cu:    return CalcVirSwitchGPU(distSq, index, gpu_sigmaSq, gpu_epsilon_Cn, gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                            gpu_rCut, gpu_rOn);
src/GPU/CalculateForceCUDAKernel.cu:  double sigma6 = gpu_sigmaSq * gpu_sigmaSq * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:  double lambdaCoef = sc_alpha * pow((1.0 - gpu_lambdaVDW), (double)sc_power);
src/GPU/CalculateForceCUDAKernel.cu:  return gpu_lambdaVDW * correction * correction *
src/GPU/CalculateForceCUDAKernel.cu:         CalcVirSwitchGPU(softRsq, index, gpu_sigmaSq, gpu_epsilon_Cn, gpu_n,
src/GPU/CalculateForceCUDAKernel.cu:                          gpu_rCut, gpu_rOn);
src/GPU/CalculateForceCUDAKernel.cu:__device__ double CalcVirSwitchGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cu:                                   double gpu_sigmaSq, double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cu:                                   double *gpu_n, double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cu:                                   double gpu_rOn)
src/GPU/CalculateForceCUDAKernel.cu:  double rCutSq = gpu_rCut * gpu_rCut;
src/GPU/CalculateForceCUDAKernel.cu:  double rOnSq = gpu_rOn * gpu_rOn;
src/GPU/CalculateForceCUDAKernel.cu:  double rRat2 = rNeg2 * gpu_sigmaSq;
src/GPU/CalculateForceCUDAKernel.cu:  double repulse = pow(rRat2, gpu_n[index] * 0.5);
src/GPU/CalculateForceCUDAKernel.cu:  double Wij = gpu_epsilon_Cn[index] * rNeg2 * (gpu_n[index] * repulse - 6.0 * attract);
src/GPU/CalculateForceCUDAKernel.cu:  double Eij = gpu_epsilon_Cn[index] * (repulse - attract);
src/GPU/CalculateForceCUDAKernel.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CalculateForceCUDAKernel.cuh:#ifndef CALCULATE_FORCE_CUDA_KERNEL
src/GPU/CalculateForceCUDAKernel.cuh:#define CALCULATE_FORCE_CUDA_KERNEL
src/GPU/CalculateForceCUDAKernel.cuh:#ifdef GOMC_CUDA
src/GPU/CalculateForceCUDAKernel.cuh:#include "VariablesCUDA.cuh"
src/GPU/CalculateForceCUDAKernel.cuh:#include "ConstantDefinitionsCUDAKernel.cuh"
src/GPU/CalculateForceCUDAKernel.cuh:#include "CalculateMinImageCUDAKernel.cuh"
src/GPU/CalculateForceCUDAKernel.cuh:void CallBoxForceGPU(VariablesCUDA *vars,
src/GPU/CalculateForceCUDAKernel.cuh:void CallBoxInterForceGPU(VariablesCUDA *vars,
src/GPU/CalculateForceCUDAKernel.cuh:void CallVirialReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateForceCUDAKernel.cuh:__global__ void BoxForceGPU(int *gpu_cellStartIndex,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_cellVector,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_neighborList,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_mapParticleToCell,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_x,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_y,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_z,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_particleKind,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_particleMol,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_REn,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_LJEn,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_count,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_rCutCoulomb,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_rCutLow,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_nonOrth,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_cell_x,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_cell_y,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_cell_z,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_Invcell_x,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_Invcell_y,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_Invcell_z,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_aForcex,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_aForcey,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_aForcez,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_mForcex,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_mForcey,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_mForcez,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cuh:                            int *gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cuh:                            double *gpu_lambdaCoulomb,
src/GPU/CalculateForceCUDAKernel.cuh:                            bool *gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cuh:__global__ void BoxInterForceGPU(int *gpu_cellStartIndex,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_cellVector,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_neighborList,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_mapParticleToCell,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_x,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_y,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_z,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_comx,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_comy,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_comz,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_particleKind,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_particleMol,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_vT11,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_vT12,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_vT13,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_vT22,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_vT23,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_vT33,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_count,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rCutCoulomb,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rCutLow,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_cell_x,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_cell_y,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_cell_z,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_Invcell_x,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_Invcell_y,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_Invcell_z,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_nonOrth,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int *gpu_molIndex,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_lambdaCoulomb,
src/GPU/CalculateForceCUDAKernel.cuh:                                 bool *gpu_isFraction,
src/GPU/CalculateForceCUDAKernel.cuh:__global__ void VirialReciprocalGPU(double *gpu_x,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_y,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_z,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_comDx,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_comDy,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_comDz,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_kxRef,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_kyRef,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_kzRef,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_prefactRef,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_hsqrRef,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_sumRref,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_sumIref,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_particleCharge,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_rT11,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_rT12,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_rT13,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_rT22,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_rT23,
src/GPU/CalculateForceCUDAKernel.cuh:                                    double *gpu_rT33,
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcEnForceGPU(double distSq, int kind1, int kind2,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int gpu_VDW_Kind,
src/GPU/CalculateForceCUDAKernel.cuh:                                 int gpu_count,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double gpu_lambdaVDW,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rMin,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_expConst);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirParticleGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:    int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirParticleGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirShiftGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:    int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirShiftGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirExp6GPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:                                        int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                                        int index, double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:                                        double gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirExp6GPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:                                        int gpu_ewald, double gpu_alpha);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirSwitchMartiniGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirSwitchMartiniGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_diElectric_1);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirSwitchGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_rCut, int index,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_sigmaSq, bool sc_coul,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcCoulombVirSwitchGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_ewald, double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_rCut);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirParticleGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:                                     double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                     double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:                                     double gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirParticleGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:                                     double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                     double *gpu_epsilon_Cn);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirShiftGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:                                  double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                  double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:                                  double gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirShiftGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:                                  double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                  double *gpu_epsilon_Cn);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirExp6GPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rMin, double *gpu_rMaxSq,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_expConst,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirExp6GPU(double distSq, int index, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:                                 double *gpu_rMin, double *gpu_expConst);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirSwitchMartiniGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:    double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_rCut, double rOn,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirSwitchMartiniGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_sigmaSq, double *gpu_n,
src/GPU/CalculateForceCUDAKernel.cuh:    double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_rCut, double rOn);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirSwitchGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:                                   double gpu_sigmaSq, double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:                                   double *gpu_n, double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cuh:                                   double gpu_rOn,
src/GPU/CalculateForceCUDAKernel.cuh:                                   uint sc_power, double gpu_lambdaVDW);
src/GPU/CalculateForceCUDAKernel.cuh:__device__ double CalcVirSwitchGPU(double distSq, int index,
src/GPU/CalculateForceCUDAKernel.cuh:                                   double gpu_sigmaSq, double *gpu_epsilon_Cn,
src/GPU/CalculateForceCUDAKernel.cuh:                                   double *gpu_n, double gpu_rCut,
src/GPU/CalculateForceCUDAKernel.cuh:                                   double gpu_rOn);
src/GPU/CalculateForceCUDAKernel.cuh:// since CUDA doesn't allow __global__ to call __device__
src/GPU/CalculateForceCUDAKernel.cuh:// Wanted to call CalcCoulombForceGPU() from CalculateEnergyCUDAKernel.cu file
src/GPU/CalculateForceCUDAKernel.cuh:__device__ inline double CalcCoulombForceGPU(double distSq, double qi_qj,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_VDW_Kind, int gpu_ewald,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_isMartini,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_rCutCoulomb,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cuh:    double *gpu_sigmaSq,
src/GPU/CalculateForceCUDAKernel.cuh:    double gpu_lambdaCoulomb,
src/GPU/CalculateForceCUDAKernel.cuh:    int gpu_count, int kind1,
src/GPU/CalculateForceCUDAKernel.cuh:  if((gpu_rCutCoulomb * gpu_rCutCoulomb) < distSq) {
src/GPU/CalculateForceCUDAKernel.cuh:  int index = FlatIndexGPU(kind1, kind2, gpu_count);
src/GPU/CalculateForceCUDAKernel.cuh:  if(gpu_VDW_Kind == GPU_VDW_STD_KIND) {
src/GPU/CalculateForceCUDAKernel.cuh:    return CalcCoulombVirParticleGPU(distSq, qi_qj, gpu_ewald, gpu_alpha, index,
src/GPU/CalculateForceCUDAKernel.cuh:                                     gpu_sigmaSq[index], sc_coul, sc_sigma_6, sc_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                                     sc_power, gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:  } else if(gpu_VDW_Kind == GPU_VDW_SHIFT_KIND) {
src/GPU/CalculateForceCUDAKernel.cuh:    return CalcCoulombVirShiftGPU(distSq, qi_qj, gpu_ewald, gpu_alpha, index,
src/GPU/CalculateForceCUDAKernel.cuh:                                  gpu_sigmaSq[index], sc_coul, sc_sigma_6, sc_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                                  sc_power, gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:  } else if(gpu_VDW_Kind == GPU_VDW_EXP6_KIND) {
src/GPU/CalculateForceCUDAKernel.cuh:    return CalcCoulombVirExp6GPU(distSq, qi_qj, gpu_ewald, gpu_alpha, index,
src/GPU/CalculateForceCUDAKernel.cuh:                                 gpu_sigmaSq[index], sc_coul, sc_sigma_6, sc_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                                 sc_power, gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:  } else if(gpu_VDW_Kind == GPU_VDW_SWITCH_KIND && gpu_isMartini) {
src/GPU/CalculateForceCUDAKernel.cuh:    return CalcCoulombVirSwitchMartiniGPU(distSq, qi_qj, gpu_ewald, gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                                          gpu_rCutCoulomb, gpu_diElectric_1,
src/GPU/CalculateForceCUDAKernel.cuh:                                          index, gpu_sigmaSq[index], sc_coul,
src/GPU/CalculateForceCUDAKernel.cuh:                                          gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:    return CalcCoulombVirSwitchGPU(distSq, qi_qj, gpu_ewald, gpu_alpha,
src/GPU/CalculateForceCUDAKernel.cuh:                                   gpu_rCutCoulomb, index, gpu_sigmaSq[index], sc_coul,
src/GPU/CalculateForceCUDAKernel.cuh:                                   gpu_lambdaCoulomb);
src/GPU/CalculateForceCUDAKernel.cuh:#endif /*GOMC_CUDA*/
src/GPU/CalculateForceCUDAKernel.cuh:#endif /*CALCULATE_FORCE_CUDA_KERNEL*/
src/GPU/ConstantDefinitionsCUDAKernel.cu:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/ConstantDefinitionsCUDAKernel.cu:#ifdef GOMC_CUDA
src/GPU/ConstantDefinitionsCUDAKernel.cu:#include <cuda.h>
src/GPU/ConstantDefinitionsCUDAKernel.cu:#include <cuda_runtime.h>
src/GPU/ConstantDefinitionsCUDAKernel.cu:#include "ConstantDefinitionsCUDAKernel.cuh"
src/GPU/ConstantDefinitionsCUDAKernel.cu:#include "CUDAMemoryManager.cuh"
src/GPU/ConstantDefinitionsCUDAKernel.cu:void UpdateGPULambda(VariablesCUDA *vars, int *molIndex, double *lambdaVDW,
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_molIndex, molIndex, BOX_TOTAL * sizeof(int),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_lambdaVDW, lambdaVDW, BOX_TOTAL * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_lambdaCoulomb, lambdaCoulomb, BOX_TOTAL * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_isFraction, isFraction, BOX_TOTAL * sizeof(bool),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void InitGPUForceField(VariablesCUDA &vars, double const *sigmaSq,
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_sigmaSq, countSq * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_epsilon_Cn, countSq * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_n, countSq * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_VDW_Kind, sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_isMartini, sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_count, sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_rCut, sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_rCutCoulomb, BOX_TOTAL * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_rCutLow, sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_rOn, sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_alpha, BOX_TOTAL * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_ewald, sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_diElectric_1, sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  // allocate gpu memory for lambda variables
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_molIndex, (int)BOX_TOTAL * sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_lambdaVDW, (int)BOX_TOTAL * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_lambdaCoulomb, (int)BOX_TOTAL * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars.gpu_isFraction, (int)BOX_TOTAL * sizeof(bool));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_sigmaSq, sigmaSq, countSq * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_epsilon_Cn, epsilon_Cn, countSq * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_n, n, countSq * sizeof(double), cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_VDW_Kind, &VDW_Kind, sizeof(int),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_isMartini, &isMartini, sizeof(int),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_count, &count, sizeof(int), cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_rCut, &Rcut, sizeof(double), cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_rCutCoulomb, rCutCoulomb, BOX_TOTAL * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_rCutLow, &RcutLow, sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_rOn, &Ron, sizeof(double), cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_alpha, alpha, BOX_TOTAL * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_ewald, &ewald, sizeof(int), cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars.gpu_diElectric_1, &diElectric_1, sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void InitCoordinatesCUDA(VariablesCUDA *vars, uint atomNumber,
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_x, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_y, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_z, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_dx, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_dy, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_dz, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_nx, maxAtomsInMol * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_ny, maxAtomsInMol * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_nz, maxAtomsInMol * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_comx, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_comy, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_comz, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_r_k_x, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_r_k_y, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_r_k_z, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_t_k_x, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_t_k_y, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_t_k_z, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_nonOrth, sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_cell_x = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_cell_y = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_cell_z = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_Invcell_x = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_Invcell_y = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_Invcell_z = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_cell_x[b], 3 * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_cell_y[b], 3 * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_cell_z[b], 3 * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_Invcell_x[b], 3 * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_Invcell_y[b], 3 * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_Invcell_z[b], 3 * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_aForcex, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_aForcey, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_aForcez, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mForcex, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mForcey, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mForcez, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mTorquex, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mTorquey, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mTorquez, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_inForceRange, maxMolNumber * sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_aForceRecx, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_aForceRecy, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_aForceRecz, atomNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mForceRecx, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mForceRecy, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mForceRecz, maxMolNumber * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_cellVector, atomNumber * sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_mapParticleToCell, atomNumber * sizeof(int));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void InitExp6Variables(VariablesCUDA *vars, double *rMin, double *expConst,
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rMin, size * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_rMaxSq, size * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUMALLOC((void**) &vars->gpu_expConst, size * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_rMin, rMin, size * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_rMaxSq, rMaxSq, size * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_expConst, expConst, size * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void InitEwaldVariablesCUDA(VariablesCUDA *vars, uint imageTotal)
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kx = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_ky = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kz = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kxRef = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kyRef = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kzRef = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumRnew = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumRref = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumInew = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumIref = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_prefact = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_prefactRef = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_hsqr = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_hsqrRef = new double *[BOX_TOTAL];
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_kx[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_ky[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_kz[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_kxRef[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_kyRef[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_kzRef[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_sumRnew[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_sumRref[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_sumInew[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_sumIref[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_prefact[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_prefactRef[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_hsqr[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUMALLOC((void**) &vars->gpu_hsqrRef[b], imageTotal * sizeof(double));
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void CopyCurrentToRefCUDA(VariablesCUDA *vars, uint box, uint imageTotal)
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_sumRref[box], vars->gpu_sumRnew[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_sumIref[box], vars->gpu_sumInew[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_prefactRef[box], vars->gpu_prefact[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_hsqrRef[box], vars->gpu_hsqr[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_kxRef[box], vars->gpu_kx[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_kyRef[box], vars->gpu_ky[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_kzRef[box], vars->gpu_kz[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void CopyRefToNewCUDA(VariablesCUDA *vars, uint box, uint imageTotal)
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_sumRnew[box], vars->gpu_sumRref[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_sumInew[box], vars->gpu_sumIref[box],
src/GPU/ConstantDefinitionsCUDAKernel.cu:             imageTotal * sizeof(double), cudaMemcpyDeviceToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void UpdateRecipVecCUDA(VariablesCUDA *vars, uint box)
src/GPU/ConstantDefinitionsCUDAKernel.cu:  tempKx = vars->gpu_kxRef[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  tempKy = vars->gpu_kyRef[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  tempKz = vars->gpu_kzRef[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  tempHsqr = vars->gpu_hsqrRef[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  tempPrefact = vars->gpu_prefactRef[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kxRef[box] = vars->gpu_kx[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kyRef[box] = vars->gpu_ky[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kzRef[box] = vars->gpu_kz[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_hsqrRef[box] = vars->gpu_hsqr[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_prefactRef[box] = vars->gpu_prefact[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kx[box] = tempKx;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_ky[box] = tempKy;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_kz[box] = tempKz;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_hsqr[box] = tempHsqr;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_prefact[box] = tempPrefact;
src/GPU/ConstantDefinitionsCUDAKernel.cu:void UpdateRecipCUDA(VariablesCUDA *vars, uint box)
src/GPU/ConstantDefinitionsCUDAKernel.cu:  tempR = vars->gpu_sumRref[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  tempI = vars->gpu_sumIref[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumRref[box] = vars->gpu_sumRnew[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumIref[box] = vars->gpu_sumInew[box];
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumRnew[box] = tempR;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  vars->gpu_sumInew[box] = tempI;
src/GPU/ConstantDefinitionsCUDAKernel.cu:void UpdateCellBasisCUDA(VariablesCUDA *vars, uint box, double *cellBasis_x,
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_cell_x[box], cellBasis_x, 3 * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_cell_y[box], cellBasis_y, 3 * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_cell_z[box], cellBasis_z, 3 * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_nonOrth, &nonOrth, sizeof(int), cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void UpdateInvCellBasisCUDA(VariablesCUDA *vars, uint box,
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_Invcell_x[box], invCellBasis_x, 3 * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_Invcell_y[box], invCellBasis_y, 3 * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_Invcell_z[box], invCellBasis_z, 3 * sizeof(double),
src/GPU/ConstantDefinitionsCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  cudaMemcpy(vars->gpu_nonOrth, &nonOrth, sizeof(int), cudaMemcpyHostToDevice);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/ConstantDefinitionsCUDAKernel.cu:void DestroyEwaldCUDAVars(VariablesCUDA *vars)
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_kx[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_ky[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_kz[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_kxRef[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_kyRef[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_kzRef[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_sumRnew[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_sumRref[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_sumInew[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_sumIref[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_prefact[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_prefactRef[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_hsqr[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_hsqrRef[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_kx;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_ky;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_kz;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_kxRef;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_kyRef;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_kzRef;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_sumRnew;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_sumRref;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_sumInew;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_sumIref;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_prefact;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_prefactRef;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_hsqr;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars->gpu_hsqrRef;
src/GPU/ConstantDefinitionsCUDAKernel.cu:void DestroyCUDAVars(VariablesCUDA *vars)
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_sigmaSq);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_epsilon_Cn);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_n);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_VDW_Kind);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_isMartini);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_count);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_rCut);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_rCutCoulomb);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_rCutLow);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_rOn);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_alpha);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_ewald);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_diElectric_1);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_x);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_y);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_z);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_dx);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_dy);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_dz);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_nx);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_ny);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_nz);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_comx);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_comy);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_comz);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_r_k_x);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_r_k_y);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_r_k_z);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_t_k_x);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_t_k_y);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_t_k_z);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_aForcex);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_aForcey);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_aForcez);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mForcex);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mForcey);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mForcez);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mTorquex);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mTorquey);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mTorquez);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_inForceRange);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_aForceRecx);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_aForceRecy);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_aForceRecz);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mForceRecx);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mForceRecy);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mForceRecz);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_cellVector);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_mapParticleToCell);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_nonOrth);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_startAtomIdx);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_cell_x[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_cell_y[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_cell_z[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_Invcell_x[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_Invcell_y[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:    CUFREE(vars->gpu_Invcell_z[b]);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  // delete gpu memory for lambda variables
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_molIndex);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_lambdaVDW);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_lambdaCoulomb);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  CUFREE(vars->gpu_isFraction);
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars-> gpu_cell_x;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars-> gpu_cell_y;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars-> gpu_cell_z;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars-> gpu_Invcell_x;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars-> gpu_Invcell_y;
src/GPU/ConstantDefinitionsCUDAKernel.cu:  delete [] vars-> gpu_Invcell_z;
src/GPU/ConstantDefinitionsCUDAKernel.cu:#endif /*GOMC_CUDA*/
src/GPU/CUDAMemoryManager.cu:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CUDAMemoryManager.cu:#include "CUDAMemoryManager.cuh"
src/GPU/CUDAMemoryManager.cu:#ifdef GOMC_CUDA
src/GPU/CUDAMemoryManager.cu:long long CUDAMemoryManager::totalAllocatedBytes = 0;
src/GPU/CUDAMemoryManager.cu:std::unordered_map<void *, std::pair<unsigned int, std::string> > CUDAMemoryManager::allocatedPointers;
src/GPU/CUDAMemoryManager.cu:cudaError_t CUDAMemoryManager::mallocMemory(void **address, unsigned int size, std::string var_name)
src/GPU/CUDAMemoryManager.cu:  cudaError_t ret = cudaMalloc(address, size);
src/GPU/CUDAMemoryManager.cu:cudaError_t CUDAMemoryManager::freeMemory(void *address, std::string var_name)
src/GPU/CUDAMemoryManager.cu:  return cudaFree(address);
src/GPU/CUDAMemoryManager.cu:bool CUDAMemoryManager::isFreed()
src/GPU/CalculateEwaldCUDAKernel.cu:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CalculateEwaldCUDAKernel.cu:#ifdef GOMC_CUDA
src/GPU/CalculateEwaldCUDAKernel.cu:#include <cuda.h>
src/GPU/CalculateEwaldCUDAKernel.cu:#include <cuda_runtime.h>
src/GPU/CalculateEwaldCUDAKernel.cu:#include "CalculateEwaldCUDAKernel.cuh"
src/GPU/CalculateEwaldCUDAKernel.cu:#include "ConstantDefinitionsCUDAKernel.cuh"
src/GPU/CalculateEwaldCUDAKernel.cu:#include "CalculateMinImageCUDAKernel.cuh"
src/GPU/CalculateEwaldCUDAKernel.cu:#include "CUDAMemoryManager.cuh"
src/GPU/CalculateEwaldCUDAKernel.cu:void CallBoxReciprocalSetupGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_energyRecip;
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_final_energyRecip;
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_energyRecip, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_energyRecip, sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_kx[box], kx, imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_ky[box], ky, imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_kz[box], kz, imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_prefact[box], prefact, imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_hsqr[box], hsqr, imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemset(vars->gpu_sumRnew[box], 0, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemset(vars->gpu_sumInew[box], 0, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  BoxReciprocalSumsGPU <<< blocksPerGrid, threadsPerBlock>>>(
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_x,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_y,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kx[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_ky[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kz[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  BoxReciprocalGPU <<< blocksPerGrid, threadsPerBlock>>>(
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_prefact[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumRnew, vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumInew, vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecip, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecip, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(&energyRecip, gpu_final_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:             sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_energyRecip);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_final_energyRecip);
src/GPU/CalculateEwaldCUDAKernel.cu:void CallBoxReciprocalSumsGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_energyRecip;
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_final_energyRecip;
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_energyRecip, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_energyRecip, sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemset(vars->gpu_sumRnew[box], 0, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemset(vars->gpu_sumInew[box], 0, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  BoxReciprocalSumsGPU <<< blocksPerGrid, threadsPerBlock>>>(
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_x,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_y,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kxRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kyRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kzRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  BoxReciprocalGPU <<< blocksPerGrid, threadsPerBlock>>>(
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_prefactRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumRnew, vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumInew, vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecip, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecip, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(&energyRecip, gpu_final_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:             sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_energyRecip);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_final_energyRecip);
src/GPU/CalculateEwaldCUDAKernel.cu:__global__ void BoxReciprocalSumsGPU(double *gpu_x,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_y,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_kx,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_ky,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cu:                                     double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cu:    shared_coords[threadIdx.x * 3    ] = gpu_x[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateEwaldCUDAKernel.cu:    shared_coords[threadIdx.x * 3 + 1] = gpu_y[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateEwaldCUDAKernel.cu:    shared_coords[threadIdx.x * 3 + 2] = gpu_z[offset_coordinates_index + threadIdx.x];
src/GPU/CalculateEwaldCUDAKernel.cu:    double dot = DotProductGPU(gpu_kx[imageID], gpu_ky[imageID], gpu_kz[imageID],
src/GPU/CalculateEwaldCUDAKernel.cu:    sumR += gpu_particleCharge[offset_coordinates_index + particleID] * dotcos;
src/GPU/CalculateEwaldCUDAKernel.cu:    sumI += gpu_particleCharge[offset_coordinates_index + particleID] * dotsin;
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_sumRnew[imageID], sumR);
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_sumInew[imageID], sumI);
src/GPU/CalculateEwaldCUDAKernel.cu:__global__ void BoxReciprocalGPU(double *gpu_prefact,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_energyRecip,
src/GPU/CalculateEwaldCUDAKernel.cu:  gpu_energyRecip[threadID] = ((gpu_sumRnew[threadID] * gpu_sumRnew[threadID] +
src/GPU/CalculateEwaldCUDAKernel.cu:                                gpu_sumInew[threadID] * gpu_sumInew[threadID]) *
src/GPU/CalculateEwaldCUDAKernel.cu:                               gpu_prefact[threadID]);
src/GPU/CalculateEwaldCUDAKernel.cu:void CallMolReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_energyRecipNew, *gpu_final_energyRecipNew;
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_energyRecipNew, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_energyRecipNew, sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, currentCoords.x, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, currentCoords.y, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, currentCoords.z, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_nx, newCoords.x, newCoordsNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_ny, newCoords.y, newCoordsNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_nz, newCoords.z, newCoordsNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  MolReciprocalGPU <<< blocksPerGrid,
src/GPU/CalculateEwaldCUDAKernel.cu:                   threadsPerBlock>>>(vars->gpu_x, vars->gpu_y, vars->gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_nx, vars->gpu_ny, vars->gpu_nz,
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_kxRef[box], vars->gpu_kyRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_kzRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_sumRref[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_sumIref[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      vars->gpu_prefactRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumRnew, vars->gpu_sumRnew[box], imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumInew, vars->gpu_sumInew[box], imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecipNew, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecipNew, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(&energyRecipNew, gpu_final_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:             sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_energyRecipNew);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_final_energyRecipNew);
src/GPU/CalculateEwaldCUDAKernel.cu:void CallChangeLambdaMolReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cu:double *gpu_particleCharge;
src/GPU/CalculateEwaldCUDAKernel.cu:double *gpu_energyRecipNew, *gpu_final_energyRecipNew;
src/GPU/CalculateEwaldCUDAKernel.cu:CUMALLOC((void**) &gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:CUMALLOC((void**) &gpu_energyRecipNew, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:CUMALLOC((void**) &gpu_final_energyRecipNew, sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpy(gpu_particleCharge, &particleCharge[0],
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:ChangeLambdaMolReciprocalGPU <<< blocksPerGrid, threadsPerBlock>>>(
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_x, vars->gpu_y, vars->gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_kxRef[box], vars->gpu_kyRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_kzRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_sumRref[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_sumIref[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  vars->gpu_prefactRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:  gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpy(sumRnew, vars->gpu_sumRnew[box], imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpy(sumInew, vars->gpu_sumInew[box], imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:gpu_final_energyRecipNew, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:gpu_final_energyRecipNew, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:cudaMemcpy(&energyRecipNew, gpu_final_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:CUFREE(gpu_particleCharge);
src/GPU/CalculateEwaldCUDAKernel.cu:CUFREE(gpu_energyRecipNew);
src/GPU/CalculateEwaldCUDAKernel.cu:CUFREE(gpu_final_energyRecipNew);
src/GPU/CalculateEwaldCUDAKernel.cu:void CallSwapReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_energyRecipNew, *gpu_final_energyRecipNew;
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_energyRecipNew, imageSize * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void**) &gpu_final_energyRecipNew, sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0],
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, coords.x, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, coords.y, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, coords.z, atomNumber * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  SwapReciprocalGPU <<< blocksPerGrid,
src/GPU/CalculateEwaldCUDAKernel.cu:                    threadsPerBlock>>>(vars->gpu_x, vars->gpu_y, vars->gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:                                       vars->gpu_kxRef[box], vars->gpu_kyRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                       vars->gpu_kzRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                       gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:                                       vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                       vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                       vars->gpu_sumRref[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                       vars->gpu_sumIref[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                       vars->gpu_prefactRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:                                       gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumRnew, vars->gpu_sumRnew[box], imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(sumInew, vars->gpu_sumInew[box], imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecipNew, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_final_energyRecipNew, imageSize);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(&energyRecipNew, gpu_final_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:             sizeof(double), cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_energyRecipNew);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_final_energyRecipNew);
src/GPU/CalculateEwaldCUDAKernel.cu:void CallMolExchangeReciprocalGPU(VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_sumRnew[box], sumRnew, imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_sumInew[box], sumInew, imageSize * sizeof(double),
src/GPU/CalculateEwaldCUDAKernel.cu:             cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:void CallBoxForceReciprocalGPU(
src/GPU/CalculateEwaldCUDAKernel.cu:  VariablesCUDA *vars,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_particleCharge;
src/GPU/CalculateEwaldCUDAKernel.cu:  int *gpu_particleMol;
src/GPU/CalculateEwaldCUDAKernel.cu:  bool *gpu_particleHasNoCharge, *gpu_particleUsed;
src/GPU/CalculateEwaldCUDAKernel.cu:  int *gpu_startMol, *gpu_lengthMol;
src/GPU/CalculateEwaldCUDAKernel.cu:  // particleHasNoCharge is stored in vector<bool>, so in order to copy it to GPU
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void **) &gpu_particleCharge, particleCharge.size() * sizeof(double));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void **) &gpu_particleHasNoCharge, particleHasNoCharge.size() * sizeof(bool));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void **) &gpu_particleUsed, atomCount * sizeof(bool));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void **) &gpu_startMol, startMol.size() * sizeof(int));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void **) &gpu_lengthMol, lengthMol.size() * sizeof(int));
src/GPU/CalculateEwaldCUDAKernel.cu:  CUMALLOC((void **) &gpu_particleMol, particleMol.size() * sizeof(int));
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_aForceRecx, atomForceRec.x, sizeof(double) * atomCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_aForceRecy, atomForceRec.y, sizeof(double) * atomCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_aForceRecz, atomForceRec.z, sizeof(double) * atomCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecx, molForceRec.x, sizeof(double) * molCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecy, molForceRec.y, sizeof(double) * molCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_mForceRecz, molForceRec.z, sizeof(double) * molCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleCharge, &particleCharge[0], sizeof(double) * particleCharge.size(), cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleMol, &particleMol[0], sizeof(int) * particleMol.size(), cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleHasNoCharge, arr_particleHasNoCharge, sizeof(bool) * particleHasNoCharge.size(), cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_particleUsed, particleUsed, sizeof(bool) * atomCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_x, molCoords.x, sizeof(double) * atomCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_y, molCoords.y, sizeof(double) * atomCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(vars->gpu_z, molCoords.z, sizeof(double) * atomCount, cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_startMol, &startMol[0], sizeof(int) * startMol.size(), cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(gpu_lengthMol, &lengthMol[0], sizeof(int) * lengthMol.size(), cudaMemcpyHostToDevice);
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  BoxForceReciprocalGPU <<< blocksPerGrid, threadsPerBlock>>>(
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_aForceRecx,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_aForceRecy,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_aForceRecz,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_mForceRecx,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_mForceRecy,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_mForceRecz,
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_particleMol,
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_particleHasNoCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_particleUsed,
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_startMol,
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_lengthMol,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kxRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kyRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_kzRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_x,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_y,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_prefactRef[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumRnew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_sumInew[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_isFraction,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_molIndex,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_lambdaCoulomb,
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_cell_x[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_cell_y[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_cell_z[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_Invcell_x[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_Invcell_y[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_Invcell_z[box],
src/GPU/CalculateEwaldCUDAKernel.cu:    vars->gpu_nonOrth,
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  checkLastErrorCUDA(__FILE__, __LINE__);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(atomForceRec.x, vars->gpu_aForceRecx, sizeof(double) * atomCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(atomForceRec.y, vars->gpu_aForceRecy, sizeof(double) * atomCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(atomForceRec.z, vars->gpu_aForceRecz, sizeof(double) * atomCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(molForceRec.x, vars->gpu_mForceRecx, sizeof(double) * molCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(molForceRec.y, vars->gpu_mForceRecy, sizeof(double) * molCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaMemcpy(molForceRec.z, vars->gpu_mForceRecz, sizeof(double) * molCount, cudaMemcpyDeviceToHost);
src/GPU/CalculateEwaldCUDAKernel.cu:  cudaDeviceSynchronize();
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleCharge);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleHasNoCharge);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleUsed);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_startMol);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_lengthMol);
src/GPU/CalculateEwaldCUDAKernel.cu:  CUFREE(gpu_particleMol);
src/GPU/CalculateEwaldCUDAKernel.cu:__global__ void BoxForceReciprocalGPU(
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_aForceRecx,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_aForceRecy,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_aForceRecz,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_mForceRecx,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_mForceRecy,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_mForceRecz,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  int *gpu_particleMol,
src/GPU/CalculateEwaldCUDAKernel.cu:  bool *gpu_particleHasNoCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  bool *gpu_particleUsed,
src/GPU/CalculateEwaldCUDAKernel.cu:  int *gpu_startMol,
src/GPU/CalculateEwaldCUDAKernel.cu:  int *gpu_lengthMol,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_kx,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_ky,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_x,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_y,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_prefact,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cu:  bool *gpu_isFraction,
src/GPU/CalculateEwaldCUDAKernel.cu:  int *gpu_molIndex,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_lambdaCoulomb,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_cell_x,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_cell_y,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_cell_z,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_Invcell_x,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_Invcell_y,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_Invcell_z,
src/GPU/CalculateEwaldCUDAKernel.cu:  int *gpu_nonOrth,
src/GPU/CalculateEwaldCUDAKernel.cu:    shared_kvector[threadIdx.x * 3] = gpu_kx[offset_vector_index + threadIdx.x];
src/GPU/CalculateEwaldCUDAKernel.cu:    shared_kvector[threadIdx.x * 3 + 1] = gpu_ky[offset_vector_index + threadIdx.x];
src/GPU/CalculateEwaldCUDAKernel.cu:    shared_kvector[threadIdx.x * 3 + 2] = gpu_kz[offset_vector_index + threadIdx.x];
src/GPU/CalculateEwaldCUDAKernel.cu:  if (particleID >= atomCount || !gpu_particleUsed[particleID]) return;
src/GPU/CalculateEwaldCUDAKernel.cu:  int moleculeID = gpu_particleMol[particleID];
src/GPU/CalculateEwaldCUDAKernel.cu:  if(gpu_particleHasNoCharge[particleID])
src/GPU/CalculateEwaldCUDAKernel.cu:  double x = gpu_x[particleID];
src/GPU/CalculateEwaldCUDAKernel.cu:  double y = gpu_y[particleID];
src/GPU/CalculateEwaldCUDAKernel.cu:  double z = gpu_z[particleID];
src/GPU/CalculateEwaldCUDAKernel.cu:  double lambdaCoef = DeviceGetLambdaCoulomb(moleculeID, box, gpu_isFraction, gpu_molIndex, gpu_lambdaCoulomb);
src/GPU/CalculateEwaldCUDAKernel.cu:    double factor = 2.0 * gpu_particleCharge[particleID] *
src/GPU/CalculateEwaldCUDAKernel.cu:                    gpu_prefact[offset_vector_index + vectorIndex] * lambdaCoef *
src/GPU/CalculateEwaldCUDAKernel.cu:                    (dotsin * gpu_sumRnew[offset_vector_index + vectorIndex] -
src/GPU/CalculateEwaldCUDAKernel.cu:                     dotcos * gpu_sumInew[offset_vector_index + vectorIndex]);
src/GPU/CalculateEwaldCUDAKernel.cu:    int lastParticleWithinSameMolecule = gpu_startMol[particleID] + gpu_lengthMol[particleID];
src/GPU/CalculateEwaldCUDAKernel.cu:    for(int otherParticle = gpu_startMol[particleID];
src/GPU/CalculateEwaldCUDAKernel.cu:        DeviceInRcut(distSq, distVect, gpu_x, gpu_y, gpu_z, particleID, otherParticle,
src/GPU/CalculateEwaldCUDAKernel.cu:                     axx, axy, axz, *gpu_nonOrth, gpu_cell_x, gpu_cell_y,
src/GPU/CalculateEwaldCUDAKernel.cu:                     gpu_cell_z, gpu_Invcell_x, gpu_Invcell_y, gpu_Invcell_z);
src/GPU/CalculateEwaldCUDAKernel.cu:        double qiqj = gpu_particleCharge[particleID] * gpu_particleCharge[otherParticle] * qqFactGPU;
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_aForceRecx[particleID], forceX);
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_aForceRecy[particleID], forceY);
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_aForceRecz[particleID], forceZ);
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_mForceRecx[moleculeID], forceX);
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_mForceRecy[moleculeID], forceY);
src/GPU/CalculateEwaldCUDAKernel.cu:  atomicAdd(&gpu_mForceRecz[moleculeID], forceZ);
src/GPU/CalculateEwaldCUDAKernel.cu:__global__ void SwapReciprocalGPU(double *gpu_x, double *gpu_y, double *gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_kx, double *gpu_ky, double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_sumRref,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_sumIref,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_prefactRef,
src/GPU/CalculateEwaldCUDAKernel.cu:                                  double *gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:    double dotProduct = DotProductGPU(gpu_kx[threadID], gpu_ky[threadID],
src/GPU/CalculateEwaldCUDAKernel.cu:                                      gpu_kz[threadID], gpu_x[p], gpu_y[p], gpu_z[p]);
src/GPU/CalculateEwaldCUDAKernel.cu:    sumReal += (gpu_particleCharge[p] * dotcos);
src/GPU/CalculateEwaldCUDAKernel.cu:    sumImaginary += (gpu_particleCharge[p] * dotsin);
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_sumRnew[threadID] = gpu_sumRref[threadID] + sumReal;
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_sumInew[threadID] = gpu_sumIref[threadID] + sumImaginary;
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_sumRnew[threadID] = gpu_sumRref[threadID] - sumReal;
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_sumInew[threadID] = gpu_sumIref[threadID] - sumImaginary;
src/GPU/CalculateEwaldCUDAKernel.cu:  gpu_energyRecipNew[threadID] = ((gpu_sumRnew[threadID] *
src/GPU/CalculateEwaldCUDAKernel.cu:                                   gpu_sumRnew[threadID] +
src/GPU/CalculateEwaldCUDAKernel.cu:                                   gpu_sumInew[threadID] *
src/GPU/CalculateEwaldCUDAKernel.cu:                                   gpu_sumInew[threadID]) *
src/GPU/CalculateEwaldCUDAKernel.cu:                                  gpu_prefactRef[threadID]);
src/GPU/CalculateEwaldCUDAKernel.cu:__global__ void MolReciprocalGPU(double *gpu_cx, double *gpu_cy, double *gpu_cz,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_nx, double *gpu_ny, double *gpu_nz,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_kx, double *gpu_ky, double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_sumRref,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_sumIref,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_prefactRef,
src/GPU/CalculateEwaldCUDAKernel.cu:                                 double *gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:    double dotProductOld = DotProductGPU(gpu_kx[threadID], gpu_ky[threadID],
src/GPU/CalculateEwaldCUDAKernel.cu:                                         gpu_kz[threadID],
src/GPU/CalculateEwaldCUDAKernel.cu:                                         gpu_cx[p], gpu_cy[p], gpu_cz[p]);
src/GPU/CalculateEwaldCUDAKernel.cu:    double dotProductNew = DotProductGPU(gpu_kx[threadID], gpu_ky[threadID],
src/GPU/CalculateEwaldCUDAKernel.cu:                                         gpu_kz[threadID],
src/GPU/CalculateEwaldCUDAKernel.cu:                                         gpu_nx[p], gpu_ny[p], gpu_nz[p]);
src/GPU/CalculateEwaldCUDAKernel.cu:    sumRealOld += (gpu_particleCharge[p] * oldcos);
src/GPU/CalculateEwaldCUDAKernel.cu:    sumImaginaryOld += (gpu_particleCharge[p] * oldsin);
src/GPU/CalculateEwaldCUDAKernel.cu:    sumRealNew += (gpu_particleCharge[p] * newcos);
src/GPU/CalculateEwaldCUDAKernel.cu:    sumImaginaryNew += (gpu_particleCharge[p] * newsin);
src/GPU/CalculateEwaldCUDAKernel.cu:  gpu_sumRnew[threadID] = gpu_sumRref[threadID] - sumRealOld + sumRealNew;
src/GPU/CalculateEwaldCUDAKernel.cu:  gpu_sumInew[threadID] = gpu_sumIref[threadID] - sumImaginaryOld +
src/GPU/CalculateEwaldCUDAKernel.cu:  gpu_energyRecipNew[threadID] = ((gpu_sumRnew[threadID] *
src/GPU/CalculateEwaldCUDAKernel.cu:                                   gpu_sumRnew[threadID] +
src/GPU/CalculateEwaldCUDAKernel.cu:                                   gpu_sumInew[threadID] *
src/GPU/CalculateEwaldCUDAKernel.cu:                                   gpu_sumInew[threadID]) *
src/GPU/CalculateEwaldCUDAKernel.cu:                                  gpu_prefactRef[threadID]);
src/GPU/CalculateEwaldCUDAKernel.cu:__global__ void ChangeLambdaMolReciprocalGPU(
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_x, double *gpu_y, double *gpu_z,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_kx, double *gpu_ky, double *gpu_kz,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_particleCharge,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_sumRnew,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_sumInew,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_sumRref,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_sumIref,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_prefactRef,
src/GPU/CalculateEwaldCUDAKernel.cu:  double *gpu_energyRecipNew,
src/GPU/CalculateEwaldCUDAKernel.cu:double dotProductNew = DotProductGPU(gpu_kx[threadID], gpu_ky[threadID],
src/GPU/CalculateEwaldCUDAKernel.cu:          gpu_kz[threadID], gpu_x[p], gpu_y[p], gpu_z[p]);
src/GPU/CalculateEwaldCUDAKernel.cu:sumRealNew += (gpu_particleCharge[p] * newcos);
src/GPU/CalculateEwaldCUDAKernel.cu:sumImaginaryNew += (gpu_particleCharge[p] * newsin);
src/GPU/CalculateEwaldCUDAKernel.cu:gpu_sumRnew[threadID] = gpu_sumRref[threadID] + lambdaCoef * sumRealNew;
src/GPU/CalculateEwaldCUDAKernel.cu:gpu_sumInew[threadID] = gpu_sumIref[threadID] + lambdaCoef * sumImaginaryNew;
src/GPU/CalculateEwaldCUDAKernel.cu:gpu_energyRecipNew[threadID] = ((gpu_sumRnew[threadID] *
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_sumRnew[threadID] +
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_sumInew[threadID] *
src/GPU/CalculateEwaldCUDAKernel.cu:    gpu_sumInew[threadID]) *
src/GPU/CalculateEwaldCUDAKernel.cu:   gpu_prefactRef[threadID]);
src/GPU/CUDAMemoryManager.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/CUDAMemoryManager.cuh:#ifdef GOMC_CUDA
src/GPU/CUDAMemoryManager.cuh:#include <cuda.h>
src/GPU/CUDAMemoryManager.cuh:#include <cuda_runtime.h>
src/GPU/CUDAMemoryManager.cuh:#define CUMALLOC(address,size) CUDAMemoryManager::mallocMemory(address,size,#address)
src/GPU/CUDAMemoryManager.cuh:#define CUFREE(address) CUDAMemoryManager::freeMemory(address,#address)
src/GPU/CUDAMemoryManager.cuh:class CUDAMemoryManager
src/GPU/CUDAMemoryManager.cuh:  static cudaError_t mallocMemory(void **address, unsigned int size, std::string var_name);
src/GPU/CUDAMemoryManager.cuh:  static cudaError_t freeMemory(void *address, std::string var_name);
src/GPU/VariablesCUDA.cuh:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GPU/VariablesCUDA.cuh:#ifdef GOMC_CUDA
src/GPU/VariablesCUDA.cuh:#include <cuda.h>
src/GPU/VariablesCUDA.cuh:#include <cuda_runtime.h>
src/GPU/VariablesCUDA.cuh://See CUDA Programming Guide section I.4.13 for details 
src/GPU/VariablesCUDA.cuh:static const __device__ double qqFactGPU = num::qqFact;
src/GPU/VariablesCUDA.cuh:#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
src/GPU/VariablesCUDA.cuh:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
src/GPU/VariablesCUDA.cuh:  if (code != cudaSuccess) {
src/GPU/VariablesCUDA.cuh:    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
src/GPU/VariablesCUDA.cuh:inline void checkLastErrorCUDA(const char *file, int line)
src/GPU/VariablesCUDA.cuh:  cudaError_t code = cudaGetLastError();
src/GPU/VariablesCUDA.cuh:  if (code != cudaSuccess) {
src/GPU/VariablesCUDA.cuh:    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
src/GPU/VariablesCUDA.cuh:  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
src/GPU/VariablesCUDA.cuh:  if ( cudaSuccess != cuda_status ) {
src/GPU/VariablesCUDA.cuh:    printf("Error: cudaMemGetInfo fails, %s \n",
src/GPU/VariablesCUDA.cuh:           cudaGetErrorString(cuda_status) );
src/GPU/VariablesCUDA.cuh:  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
src/GPU/VariablesCUDA.cuh:class VariablesCUDA
src/GPU/VariablesCUDA.cuh:  VariablesCUDA()
src/GPU/VariablesCUDA.cuh:    gpu_sigmaSq = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_epsilon_Cn = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_n = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_VDW_Kind = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_isMartini = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_count = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_rCut = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_rCutLow = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_rOn = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_alpha = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_rCutCoulomb = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_ewald = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_diElectric_1 = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_aForcex = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_aForcey = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_aForcez = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_mForcex = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_mForcey = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_mForcez = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_startAtomIdx = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_molIndex = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_lambdaVDW = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_lambdaCoulomb = NULL;
src/GPU/VariablesCUDA.cuh:    gpu_isFraction = NULL;
src/GPU/VariablesCUDA.cuh:  double *gpu_sigmaSq;
src/GPU/VariablesCUDA.cuh:  double *gpu_epsilon_Cn;
src/GPU/VariablesCUDA.cuh:  double *gpu_n;
src/GPU/VariablesCUDA.cuh:  int *gpu_VDW_Kind;
src/GPU/VariablesCUDA.cuh:  int *gpu_isMartini;
src/GPU/VariablesCUDA.cuh:  int *gpu_count;
src/GPU/VariablesCUDA.cuh:  int *gpu_startAtomIdx; //start atom index of the molecule
src/GPU/VariablesCUDA.cuh:  double *gpu_rCut;
src/GPU/VariablesCUDA.cuh:  double *gpu_rCutCoulomb;
src/GPU/VariablesCUDA.cuh:  double *gpu_rCutLow;
src/GPU/VariablesCUDA.cuh:  double *gpu_rOn;
src/GPU/VariablesCUDA.cuh:  double *gpu_alpha;
src/GPU/VariablesCUDA.cuh:  int *gpu_ewald;
src/GPU/VariablesCUDA.cuh:  double *gpu_diElectric_1;
src/GPU/VariablesCUDA.cuh:  double *gpu_x, *gpu_y, *gpu_z;
src/GPU/VariablesCUDA.cuh:  double *gpu_nx, *gpu_ny, *gpu_nz;
src/GPU/VariablesCUDA.cuh:  double *gpu_dx, *gpu_dy, *gpu_dz;
src/GPU/VariablesCUDA.cuh:  double **gpu_kx, **gpu_ky, **gpu_kz;
src/GPU/VariablesCUDA.cuh:  double **gpu_kxRef, **gpu_kyRef, **gpu_kzRef;
src/GPU/VariablesCUDA.cuh:  double **gpu_sumRnew, **gpu_sumInew, **gpu_sumRref, **gpu_sumIref;
src/GPU/VariablesCUDA.cuh:  double **gpu_prefact, **gpu_prefactRef;
src/GPU/VariablesCUDA.cuh:  double **gpu_hsqr, **gpu_hsqrRef;
src/GPU/VariablesCUDA.cuh:  double *gpu_comx, *gpu_comy, *gpu_comz;
src/GPU/VariablesCUDA.cuh:  double *gpu_rT11, *gpu_rT12, *gpu_rT13;
src/GPU/VariablesCUDA.cuh:  double *gpu_rT22, *gpu_rT23, *gpu_rT33;
src/GPU/VariablesCUDA.cuh:  double *gpu_vT11, *gpu_vT12, *gpu_vT13;
src/GPU/VariablesCUDA.cuh:  double *gpu_vT22, *gpu_vT23, *gpu_vT33;
src/GPU/VariablesCUDA.cuh:  double **gpu_cell_x, **gpu_cell_y, **gpu_cell_z;
src/GPU/VariablesCUDA.cuh:  double **gpu_Invcell_x, **gpu_Invcell_y, **gpu_Invcell_z;
src/GPU/VariablesCUDA.cuh:  int *gpu_nonOrth;
src/GPU/VariablesCUDA.cuh:  double *gpu_aForcex, *gpu_aForcey, *gpu_aForcez;
src/GPU/VariablesCUDA.cuh:  double *gpu_mForcex, *gpu_mForcey, *gpu_mForcez;
src/GPU/VariablesCUDA.cuh:  double *gpu_mTorquex, *gpu_mTorquey, *gpu_mTorquez;
src/GPU/VariablesCUDA.cuh:  int *gpu_inForceRange;
src/GPU/VariablesCUDA.cuh:  double *gpu_aForceRecx, *gpu_aForceRecy, *gpu_aForceRecz;
src/GPU/VariablesCUDA.cuh:  double *gpu_mForceRecx, *gpu_mForceRecy, *gpu_mForceRecz;
src/GPU/VariablesCUDA.cuh:  double *gpu_rMin, *gpu_expConst, *gpu_rMaxSq;
src/GPU/VariablesCUDA.cuh:  double *gpu_r_k_x, *gpu_r_k_y, *gpu_r_k_z;
src/GPU/VariablesCUDA.cuh:  double *gpu_t_k_x, *gpu_t_k_y, *gpu_t_k_z;
src/GPU/VariablesCUDA.cuh:  int *gpu_molIndex;
src/GPU/VariablesCUDA.cuh:  double *gpu_lambdaVDW, *gpu_lambdaCoulomb;
src/GPU/VariablesCUDA.cuh:  bool *gpu_isFraction;
src/GPU/VariablesCUDA.cuh:  // new pair interaction calculation done on GPU
src/GPU/VariablesCUDA.cuh:  int *gpu_cellVector, *gpu_mapParticleToCell;
src/ConstField.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CBMC.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ParallelTemperingPreprocessor.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MolPick.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/OutputAbstracts.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoveSettings.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoveConst.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/GOMCEventsProfile.h:// GOMC_CUDA build by defining GOMC_NVTX_ENABLED in CMake
src/GOMCEventsProfile.h:#if defined(GOMC_CUDA) && defined(GOMC_NVTX_ENABLED)
src/GOMCEventsProfile.h:#if CUDART_VERSION >= 10000
src/GOMCEventsProfile.h://#include </opt/nvidia/nsight-systems/2020.4.3/target-linux-x64/nvtx/include/nvtx3/nvToolsExt.h>  // CUDA >= 10 has NVTX V3+
src/GOMCEventsProfile.h:#include <nvtx3/nvToolsExt.h>  // CUDA >= 10 has NVTX V3+
src/GOMCEventsProfile.h:#error NVTXv3 requires CUDA 10.0 or greater
src/GOMCEventsProfile.h://#include <nvToolsExt.h>        // CUDA < 10 has NVTX V2
src/GOMCEventsProfile.h:#include <cuda_profiler_api.h>
src/GOMCEventsProfile.h:    cudaProfilerStart(); \
src/GOMCEventsProfile.h:    cudaProfilerStop(); \
src/GOMCEventsProfile.h:#endif // GOMC_CUDA && GOMC_NVTX_ENABLED
src/Main.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Main.cpp:#ifdef GOMC_CUDA
src/Main.cpp:#include "cuda.h"
src/Main.cpp:#include <cuda_runtime_api.h>
src/Main.cpp:#ifdef GOMC_CUDA
src/Main.cpp:void PrintGPUHardwareInfo();
src/Main.cpp:#ifdef GOMC_CUDA
src/Main.cpp:  remove("gpu.debug");
src/Main.cpp:#ifdef GOMC_CUDA
src/Main.cpp:  PrintGPUHardwareInfo();
src/Main.cpp:#ifdef GOMC_CUDA
src/Main.cpp:void PrintGPUHardwareInfo()
src/Main.cpp:  cudaGetDeviceCount(&nDevices);
src/Main.cpp:    printf("There are no available device(s) that support CUDA\n");
src/Main.cpp:    printf("GPU information:\n");
src/Main.cpp:      cudaDeviceProp prop;
src/Main.cpp:      cudaGetDeviceProperties(&prop, i);
src/Main.cpp:  cudaSetDevice(fastIndex);
src/CheckpointSetup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CheckpointSetup.h:  // on the GPU, when InitStep is set to 0
src/Checkpoint.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/BondAdjacencyList.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/SubdividedArray.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Molecules.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFParticle.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFParticle.h:#ifdef GOMC_CUDA
src/FFParticle.h:#include "VariablesCUDA.cuh"
src/FFParticle.h:#ifdef GOMC_CUDA
src/FFParticle.h:  VariablesCUDA *getCUDAVars()
src/FFParticle.h:    return varCUDA;
src/FFParticle.h:#ifdef GOMC_CUDA
src/FFParticle.h:  VariablesCUDA *varCUDA;
src/ParallelTemperingPreprocessor.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/System.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CheckpointOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CheckpointOutput.h:  // on the GPU, when InitStep is set to 0
src/CellList.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/InputFileReader.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/EnergyTypes.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Forcefield.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFSwitch.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PSFOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Clock.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/System.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/System.cpp:#ifdef GOMC_CUDA
src/System.cpp:  //Set the pointer to cudaVar and initialize the values
src/System.cpp:  #ifdef GOMC_CUDA 
src/System.cpp:    statV.forcefield.particles->getCUDAVars()
src/PRNG.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFParticle.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFParticle.cpp:#ifdef GOMC_CUDA
src/FFParticle.cpp:#include "ConstantDefinitionsCUDAKernel.cuh"
src/FFParticle.cpp:#ifdef GOMC_CUDA
src/FFParticle.cpp:  , varCUDA(NULL)
src/FFParticle.cpp:#ifdef GOMC_CUDA
src/FFParticle.cpp:  DestroyCUDAVars(varCUDA);
src/FFParticle.cpp:  delete varCUDA;
src/FFParticle.cpp:#ifdef GOMC_CUDA
src/FFParticle.cpp:  // Variables for GPU stored in here
src/FFParticle.cpp:  varCUDA = new VariablesCUDA();
src/FFParticle.cpp:#ifdef GOMC_CUDA
src/FFParticle.cpp:  InitGPUForceField(*varCUDA, sigmaSq, epsilon_cn, n, forcefield.vdwKind,
src/COM.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFAngles.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFDihedrals.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PDBSetup.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFDihedrals.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PRNGSetup.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/TransformMatrix.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoleculeLookup.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoleculeLookup.cpp:#ifdef GOMC_CUDA
src/MoleculeLookup.cpp:#include <cuda.h>
src/MoleculeLookup.cpp:#include <cuda_runtime.h>
src/MoleculeLookup.cpp:#include "CUDAMemoryManager.cuh"
src/MoleculeLookup.cpp:#include "VariablesCUDA.cuh"
src/MoleculeLookup.cpp:// allocate and set gpu variables
src/MoleculeLookup.cpp:#ifdef GOMC_CUDA
src/MoleculeLookup.cpp:  VariablesCUDA *cudaVars = ff.particles->getCUDAVars();
src/MoleculeLookup.cpp:  CUMALLOC((void**) &cudaVars->gpu_startAtomIdx, numMol * sizeof(int));
src/MoleculeLookup.cpp:  cudaMemcpy(cudaVars->gpu_startAtomIdx, mols.start, numMol * sizeof(int), cudaMemcpyHostToDevice);
src/CheckpointOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/BlockOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PRNGSetup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/BoxDimensionsNonOrth.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CalculateEnergy.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CPUSide.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/EnsemblePreprocessor.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/HistOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFConst.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/UnitConst.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoveSettings.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MolSetup.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ConfigSetup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/EwaldCached.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/SeedReader.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ExtendedSystemOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFShift.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FreeEnergyOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ExtendedSystem.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/IntraMoleculeExchange2.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/IntraMoleculeExchange1.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/TargetedSwap.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/IntraSwap.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/MoleculeExchange3.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/MultiParticle.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/MultiParticle.h:#ifdef GOMC_CUDA
src/moves/MultiParticle.h:#include "TransformParticlesCUDAKernel.cuh"
src/moves/MultiParticle.h:#include "VariablesCUDA.cuh"
src/moves/MultiParticle.h:#ifdef GOMC_CUDA
src/moves/MultiParticle.h:  VariablesCUDA *cudaVars;
src/moves/MultiParticle.h:#ifdef GOMC_CUDA
src/moves/MultiParticle.h:  cudaVars = sys.statV.forcefield.particles->getCUDAVars();
src/moves/MultiParticle.h:#ifdef GOMC_CUDA
src/moves/MultiParticle.h:    CallRotateParticlesGPU(cudaVars, isMoleculeInvolved, bPick, r_max,
src/moves/MultiParticle.h:    CallTranslateParticlesGPU(cudaVars, isMoleculeInvolved, bPick, t_max,
src/moves/MoveBase.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/CrankShaft.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/MoleculeExchange1.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/Rotation.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/MoleculeTransfer.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/Regrowth.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/IntraMoleculeExchange3.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/MoleculeExchange2.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/VolumeTransfer.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/VolumeTransfer.h:#ifdef GOMC_CUDA
src/moves/VolumeTransfer.h:#include "ConstantDefinitionsCUDAKernel.cuh"
src/moves/VolumeTransfer.h:#ifdef GOMC_CUDA
src/moves/VolumeTransfer.h:      //update unitcell to the original in GPU
src/moves/VolumeTransfer.h:        UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), bPick[b],
src/moves/VolumeTransfer.h:          // so cast and copy the additional data to the GPU
src/moves/VolumeTransfer.h:          UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), bPick[b],
src/moves/VolumeTransfer.h:      UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/moves/VolumeTransfer.h:        // so cast and copy the additional data to the GPU
src/moves/VolumeTransfer.h:        UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/moves/NeMTMC.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/MultiParticleBrownianMotion.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.50
src/moves/MultiParticleBrownianMotion.h:#ifdef GOMC_CUDA
src/moves/MultiParticleBrownianMotion.h:#include <cuda.h>
src/moves/MultiParticleBrownianMotion.h:#include <cuda_runtime.h>
src/moves/MultiParticleBrownianMotion.h:#include "CUDAMemoryManager.cuh"
src/moves/MultiParticleBrownianMotion.h:#include "TransformParticlesCUDAKernel.cuh"
src/moves/MultiParticleBrownianMotion.h:#include "VariablesCUDA.cuh"
src/moves/MultiParticleBrownianMotion.h:#ifdef GOMC_CUDA
src/moves/MultiParticleBrownianMotion.h:    cudaVars = NULL;
src/moves/MultiParticleBrownianMotion.h:    cudaFreeHost(kill);
src/moves/MultiParticleBrownianMotion.h:#ifdef GOMC_CUDA
src/moves/MultiParticleBrownianMotion.h:  VariablesCUDA *cudaVars;
src/moves/MultiParticleBrownianMotion.h:#ifdef GOMC_CUDA
src/moves/MultiParticleBrownianMotion.h:  cudaVars = sys.statV.forcefield.particles->getCUDAVars();
src/moves/MultiParticleBrownianMotion.h:  cudaMallocHost((void**) &kill, sizeof(int));
src/moves/MultiParticleBrownianMotion.h:  checkLastErrorCUDA(__FILE__, __LINE__);
src/moves/MultiParticleBrownianMotion.h:#ifdef GOMC_CUDA
src/moves/MultiParticleBrownianMotion.h:    BrownianMotionRotateParticlesGPU(
src/moves/MultiParticleBrownianMotion.h:      cudaVars,
src/moves/MultiParticleBrownianMotion.h:    BrownianMotionTranslateParticlesGPU(
src/moves/MultiParticleBrownianMotion.h:      cudaVars,
src/moves/Translate.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/moves/IntraTargetedSwap.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Ewald.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Checkpoint.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Checkpoint.h:        // on the GPU, when InitStep is set to 0
src/PDBOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/SimEventFrequency.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFSetup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PDBConst.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/OutConst.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CalculateEnergy.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CalculateEnergy.cpp:#ifdef GOMC_CUDA
src/CalculateEnergy.cpp:#include "CalculateEnergyCUDAKernel.cuh"
src/CalculateEnergy.cpp:#include "CalculateForceCUDAKernel.cuh"
src/CalculateEnergy.cpp:#include "ConstantDefinitionsCUDAKernel.cuh"
src/CalculateEnergy.cpp:#ifdef GOMC_CUDA
src/CalculateEnergy.cpp:  InitCoordinatesCUDA(forcefield.particles->getCUDAVars(),
src/CalculateEnergy.cpp:// this function. Need to implement the GPU function
src/CalculateEnergy.cpp:#ifdef GOMC_CUDA
src/CalculateEnergy.cpp:  //update unitcell in GPU
src/CalculateEnergy.cpp:  UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/CalculateEnergy.cpp:    // so cast and copy the additional data to the GPU
src/CalculateEnergy.cpp:    UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/CalculateEnergy.cpp:  CallBoxInterGPU(forcefield.particles->getCUDAVars(), cellVector, cellStartIndex,
src/CalculateEnergy.cpp:#ifdef GOMC_CUDA
src/CalculateEnergy.cpp:  //update unitcell in GPU
src/CalculateEnergy.cpp:  UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/CalculateEnergy.cpp:    // so cast and copy the additional data to the GPU
src/CalculateEnergy.cpp:    UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/CalculateEnergy.cpp:  CallBoxForceGPU(forcefield.particles->getCUDAVars(), cellVector,
src/CalculateEnergy.cpp:#ifdef GOMC_CUDA
src/CalculateEnergy.cpp:  //update unitcell in GPU
src/CalculateEnergy.cpp:  UpdateCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/CalculateEnergy.cpp:    // so cast and copy the additional data to the GPU
src/CalculateEnergy.cpp:    UpdateInvCellBasisCUDA(forcefield.particles->getCUDAVars(), box,
src/CalculateEnergy.cpp:  CallBoxInterForceGPU(forcefield.particles->getCUDAVars(),
src/BoxDimensions.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Ewald.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:#include "CalculateEwaldCUDAKernel.cuh"
src/Ewald.cpp:#include "CalculateForceCUDAKernel.cuh"
src/Ewald.cpp:#include "ConstantDefinitionsCUDAKernel.cuh"
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    DestroyEwaldCUDAVars(ff.particles->getCUDAVars());
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:  InitEwaldVariablesCUDA(ff.particles->getCUDAVars(), imageTotal);
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    CallBoxReciprocalSetupGPU(ff.particles->getCUDAVars(),
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    CallBoxReciprocalSumsGPU(ff.particles->getCUDAVars(), thisBoxCoords,
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    CallMolReciprocalGPU(ff.particles->getCUDAVars(),
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    CallSwapReciprocalGPU(ff.particles->getCUDAVars(),
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    CallChangeLambdaMolReciprocalGPU(ff.particles->getCUDAVars(),
src/Ewald.cpp:  //Need to implement GPU
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    CallSwapReciprocalGPU(ff.particles->getCUDAVars(),
src/Ewald.cpp://Because MolExchangeReciprocal does not have a matching GPU function, this is
src/Ewald.cpp://a stub function to copy the CPU sumRnew and sumInew vectors to the GPU in case
src/Ewald.cpp://the move is accepted. If this function is ported to the GPU, this call should
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:  CallMolExchangeReciprocalGPU(ff.particles->getCUDAVars(), imageSizeRef[box],
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:  CopyCurrentToRefCUDA(ff.particles->getCUDAVars(), box, imageSize[box]);
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:  CallVirialReciprocalGPU(ff.particles->getCUDAVars(), thisBoxCoords,
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:  UpdateRecipCUDA(ff.particles->getCUDAVars(), box);
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:  CopyRefToNewCUDA(ff.particles->getCUDAVars(), box, imageSizeRef[box]);
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:  UpdateRecipVecCUDA(ff.particles->getCUDAVars(), box);
src/Ewald.cpp:#ifdef GOMC_CUDA
src/Ewald.cpp:    CallBoxForceReciprocalGPU(
src/Ewald.cpp:      ff.particles->getCUDAVars(),
src/Forcefield.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Velocity.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FxdWidthWrtr.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ExtendedSystemOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFConst.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/OutputVars.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/StaticVals.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/InputAbstracts.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ParallelTemperingUtilities.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.51
src/Geometry.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/NoEwald.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/OutConst.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/StaticVals.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/EnPartCntSampleOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/EwaldCached.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/EwaldCached.cpp:  //Need to implement GPU
src/Simulation.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CellList.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PSFOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFSwitchMartini.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/PDBSetup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/BoxDimensions.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/BoxDimensionsNonOrth.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/XYZArray.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFExp6.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFExp6.h:#ifdef GOMC_CUDA
src/FFExp6.h:#include "ConstantDefinitionsCUDAKernel.cuh"
src/FFExp6.h:#ifdef GOMC_CUDA
src/FFExp6.h:  InitExp6Variables(varCUDA, rMin, expConst, rMaxSq, size);
src/BondAdjacencyList.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Reader.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/ConsoleOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/EnPartCntSampleOutput.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Geometry.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/NoEwald.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FixedWidthReader.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/MoleculeLookup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Coordinates.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/Setup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/CoordinateSetup.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FreeEnergyOutput.cpp:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75
src/FFBonds.h:GPU OPTIMIZED MONTE CARLO (GOMC) 2.75

```
