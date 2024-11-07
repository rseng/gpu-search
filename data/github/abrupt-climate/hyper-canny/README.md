# https://github.com/abrupt-climate/hyper-canny

```console
Dockerfile:    libpng12-dev mesa-opencl-icd ocl-icd-opencl-dev opencl-headers \
test/gtest/include/gtest/internal/gtest-port.h:// with a TR1 tuple implementation.  NVIDIA's CUDA NVCC compiler
test/gtest/include/gtest/internal/gtest-port.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
include/CL/cl2.hpp: *   \brief C++ bindings for OpenCL 1.0 (rev 48), OpenCL 1.1 (rev 33),
include/CL/cl2.hpp: *       OpenCL 1.2 (rev 15) and OpenCL 2.0 (rev 29)
include/CL/cl2.hpp: *   Derived from the OpenCL 1.x C++ bindings written by
include/CL/cl2.hpp: *       http://khronosgroup.github.io/OpenCL-CLHPP/
include/CL/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP/releases
include/CL/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP
include/CL/cl2.hpp: * reasonable to define C++ bindings for OpenCL.
include/CL/cl2.hpp: * fixes in the new header as well as additional OpenCL 2.0 features.
include/CL/cl2.hpp: * Due to the evolution of the underlying OpenCL API the 2.0 C++ bindings
include/CL/cl2.hpp: * and the range of valid underlying OpenCL runtime versions supported.
include/CL/cl2.hpp: * The combination of preprocessor macros CL_HPP_TARGET_OPENCL_VERSION and
include/CL/cl2.hpp: * CL_HPP_MINIMUM_OPENCL_VERSION control this range. These are three digit
include/CL/cl2.hpp: * decimal values representing OpenCL runime versions. The default for
include/CL/cl2.hpp: * the target is 200, representing OpenCL 2.0 and the minimum is also
include/CL/cl2.hpp: * The OpenCL 1.x versions of the C++ bindings included a size_t wrapper
include/CL/cl2.hpp: * In OpenCL 2.0 OpenCL C is not entirely backward compatibility with
include/CL/cl2.hpp: * earlier versions. As a result a flag must be passed to the OpenCL C
include/CL/cl2.hpp: * compiled to request OpenCL 2.0 compilation of kernels with 1.2 as
include/CL/cl2.hpp: * For those cases the compilation defaults to OpenCL C 2.0.
include/CL/cl2.hpp: * - CL_HPP_TARGET_OPENCL_VERSION
include/CL/cl2.hpp: *   Defines the target OpenCL runtime version to build the header
include/CL/cl2.hpp: *   against. Defaults to 200, representing OpenCL 2.0.
include/CL/cl2.hpp: *   Enables device fission for OpenCL 1.2 platforms.
include/CL/cl2.hpp: *   Default to OpenCL C 1.2 compilation rather than OpenCL C 2.0
include/CL/cl2.hpp:    #define CL_HPP_TARGET_OPENCL_VERSION 200
include/CL/cl2.hpp:            if (platver.find("OpenCL 2.") != std::string::npos) {
include/CL/cl2.hpp:            std::cout << "No OpenCL 2.0 platform found.";
include/CL/cl2.hpp:#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
include/CL/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 200 (OpenCL 2.0)")
include/CL/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION != 100 && CL_HPP_TARGET_OPENCL_VERSION != 110 && CL_HPP_TARGET_OPENCL_VERSION != 120 && CL_HPP_TARGET_OPENCL_VERSION != 200
include/CL/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 200")
include/CL/cl2.hpp:# undef CL_HPP_TARGET_OPENCL_VERSION
include/CL/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
include/CL/cl2.hpp:#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
include/CL/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 200
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && CL_HPP_MINIMUM_OPENCL_VERSION != 110 && CL_HPP_MINIMUM_OPENCL_VERSION != 120 && CL_HPP_MINIMUM_OPENCL_VERSION != 200
include/CL/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 100")
include/CL/cl2.hpp:# undef CL_HPP_MINIMUM_OPENCL_VERSION
include/CL/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 100
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
include/CL/cl2.hpp:# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
include/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
include/CL/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
include/CL/cl2.hpp:#include <OpenCL/opencl.h>
include/CL/cl2.hpp:#include <CL/opencl.h>
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:        *  OpenCL C calls that require arrays of size_t values, whose
include/CL/cl2.hpp: * \brief The OpenCL C++ bindings are defined within this namespace.
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:// Flags deprecated in OpenCL 2.0
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
include/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
include/CL/cl2.hpp:#ifdef CL_DEVICE_GPU_OVERLAP_NV
include/CL/cl2.hpp:CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp: * OpenCL 1.2 devices do have retain/release.
include/CL/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp: * OpenCL 1.1 devices do not have retain/release.
include/CL/cl2.hpp:#endif // ! (CL_HPP_TARGET_OPENCL_VERSION >= 120)
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#else // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:     *  \param devices returns a vector of OpenCL D3D10 devices found. The cl::Device
include/CL/cl2.hpp:     *  values returned in devices can be used to identify a specific OpenCL
include/CL/cl2.hpp:     *  The application can query specific capabilities of the OpenCL device(s)
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp: * Unload the OpenCL compiler.
include/CL/cl2.hpp: * \note Deprecated for OpenCL 1.2. Use Platform::unloadCompiler instead.
include/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:/*! \brief Class interface for creating OpenCL buffers from ID3D10Buffer's.
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
include/CL/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:    *              The channel order may differ as described in the OpenCL
include/CL/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp: *  \note Deprecated for OpenCL 1.2. Please use ImageGL instead.
include/CL/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
include/CL/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif  // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp: * was performed by OpenCL anyway.
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:     * \param context A valid OpenCL context in which to construct the program.
include/CL/cl2.hpp:     * \param devices A vector of OpenCL device objects for which the program will be created.
include/CL/cl2.hpp:     *   CL_INVALID_DEVICE if OpenCL devices listed in devices are not in the list of devices associated with context.
include/CL/cl2.hpp:     *   CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:     *     The pattern type must be an accepted OpenCL data type.
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
include/CL/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
include/CL/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/CL/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#endif // defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
include/CL/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp: * SVM buffer back to the OpenCL runtime.
include/CL/cl2.hpp: * SVM buffer back to the OpenCL runtime.
include/CL/cl2.hpp: * SVM buffer back to the OpenCL runtime.
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
include/CL/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
include/CL/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
meson.build:opencl_dep = dependency('OpenCL')
meson.build:    dependencies: opencl_dep,
meson.build:    dependencies : opencl_dep,
meson.build:    dependencies : [opencl_dep, fftw_dep, fftwf_dep, netcdf_dep],
meson.build:    # dependencies: [opencl_dep, netcdf_dep],
src/cl-util/info.cc:/*! \brief Prints some information on the available OpenCL runtime.
src/cl-util/info.cc:void HyperCanny::print_opencl_info(
src/cl-util/info.cc:				.msg("OpenCL version:   ", device.getInfo<CL_DEVICE_OPENCL_C_VERSION>())
src/cl-util/get_gpu_context.hh:/*! \file cl-util/get_gpu_context.hh
src/cl-util/get_gpu_context.hh: *  \brief Get the OpenCL context for the local GPU.
src/cl-util/get_gpu_context.hh:    extern std::tuple<std::vector<cl::Device>, cl::Context> get_default_gpu_context();
src/cl-util/base.hh:#define CL_HPP_MINIMUM_OPENCL_VERSION 100
src/cl-util/base.hh:#define CL_HPP_TARGET_OPENCL_VERSION  120
src/cl-util/compile.hh: *  \brief Short-hands for compiling OpenCL kernels.
src/cl-util/info.hh: *  \brief Print information on OpenCL runtime.
src/cl-util/info.hh:    extern void print_opencl_info(std::vector<cl::Platform> const &platform_list);
src/cl-util/get_gpu_context.cc:#include "get_gpu_context.hh"
src/cl-util/get_gpu_context.cc:    HyperCanny::get_default_gpu_context()
src/cl-util/get_gpu_context.cc:        "\033[32m☷ \033[m OpenCL initialisation ...",
src/cl-util/get_gpu_context.cc:        "  is OpenCL correctly installed?"
src/cl-util/get_gpu_context.cc:    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &device_list);
src/cl-util/get_gpu_context.cc:        " cl::Platform::getDevices failed to find a GPU device;\n"
src/cl-util/meson.build:src_cl_util_files = files('./compile.cc','./get_gpu_context.cc','./info.cc','./timing.cc')
src/cl-info/main.cc:    print_opencl_info(platform_list);

```
