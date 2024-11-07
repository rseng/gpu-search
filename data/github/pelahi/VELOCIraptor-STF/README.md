# https://github.com/pelahi/VELOCIraptor-STF

```console
CMakeLists.txt:vr_option(ENABLE_OPENMP_GPU "Attempt to include OpenMP GPU support in VELOCIraptor" OFF)
CMakeLists.txt:vr_option(OPENACC          "Attempt to include OpenACC support in VELOCIraptor" OFF)
CMakeLists.txt:if (VR_ENABLE_OPENMP_GPU)
CMakeLists.txt:	set(NBODY_ENABLE_OPENMP_GPU ON)
CMakeLists.txt:vr_option_defines(OPENMPGPU                 USEOPENMPTARGET)
cmake/VRCompilationMessages.cmake:            "OpenMP GPU offloading support" OPENMPGPU
cmake/VRCMakeUtils.cmake:	set(VR_HAS_OPENMPGPU ${NBODYLIB_HAS_OPENMPGPU})
src/utilities.cxx:#ifdef _GPU
src/utilities.cxx:    pu_gpuErrorCheck(pu_gpuGetDeviceCount(&nDevices));
src/utilities.cxx:            pu_gpuDeviceProp_t prop;
src/utilities.cxx:            // pu_gpuErrorCheck(pu_gpuSetDevice(i));
src/utilities.cxx:            pu_gpuErrorCheck(pu_gpuGetDeviceProperties(&prop, i));
src/utilities.cxx:            // Get the PCIBusId for each GPU and use it to query for UUID
src/utilities.cxx:            pu_gpuErrorCheck(pu_gpuDeviceGetPCIBusId(busid, 64, i));
src/utilities.cxx:            s += "GPU device " + std::to_string(i);

```
