# https://github.com/mjuric/galfast

```console
tests/os_photometry/go.sh:#/opt/cuda//bin/nvcc -o /dev/null -cuda -arch sm_13 --ptxas-options="-v" -I.. -I/home/kreso/projects/galaxy/src/common -I/usr/include/sqlplus/ -I/usr/include/mysql/ -DDATADIR="\"/home/kreso/projects/galaxy/workspace/staging/share/galaxy\"" -DDEBUGMODE=1 -I/home/kreso/projects/libpeyton/include -I/opt/cuda//include /home/kreso/projects/galaxy/src/simulate_gpu.cu
CMakeLists.txt:### CUDA
CMakeLists.txt:find_package(CUDA 2.3 REQUIRED)
CMakeLists.txt:set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_13)
CMakeLists.txt:	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-ccbin=${GCC_ROOT}/bin")
CMakeLists.txt:	message(STATUS "Note: CUDA will use host compiler from ${GCC_ROOT}")
CMakeLists.txt:if( "x${CUDA_BUILD_EMULATION}" STREQUAL xON )
CMakeLists.txt:	set(CUDA_DEVEMU 1)
CMakeLists.txt:	message(STATUS "Note: Targeting CUDA code for device emulation")
CMakeLists.txt:endif( "x${CUDA_BUILD_EMULATION}" STREQUAL xON )
CMakeLists.txt:cuda_add_executable(galfast
CMakeLists.txt:  src/kernels_gpu.cu
configure:	   --with-cuda=<cuda_root>      # use CUDA toolkit from <cuda_root>
configure:	   --devemu			# target CUDA code for device emulation
configure:TEMP=`getopt -o hp: -l help,prefix:,debug,optimized,devemu,env:,with-gcc:,with-cuda:,with-boost:,with-libpeyton: -- "$@"` || { xecho "Try $0 --help"; exit $?; }
configure:			xecho "Targeting CUDA code for device emulation"
configure:			CMAKE="$CMAKE -DCUDA_BUILD_EMULATION=ON"
configure:		--with-cuda)
configure:			xecho "Using CUDA Toolkit from '$2'"
configure:			CMAKE="$CMAKE -DCUDA_TOOLKIT_ROOT_DIR='$2'"
galfast_config.h.in:/* Targeting CUDA device emulation */
galfast_config.h.in:#cmakedefine CUDA_DEVEMU	1
galfast_config.h.in:/* NVIDIA CUDA (GPU) support (this is now required) */
galfast_config.h.in:#define HAVE_CUDA 1
cmake/modules/FindCUDA.cmake:# - Tools for building CUDA C files: libraries and build dependencies.
cmake/modules/FindCUDA.cmake:# This script locates the NVIDIA CUDA C tools. It should work on linux, windows,
cmake/modules/FindCUDA.cmake:# and mac and should be reasonably up to date with CUDA C releases.
cmake/modules/FindCUDA.cmake:# REQUIRED and QUIET.  CUDA_FOUND will report if an acceptable version of CUDA
cmake/modules/FindCUDA.cmake:# The script will prompt the user to specify CUDA_TOOLKIT_ROOT_DIR if the prefix
cmake/modules/FindCUDA.cmake:# toolkit set the environment variable CUDA_BIN_PATH before running cmake
cmake/modules/FindCUDA.cmake:# (e.g. CUDA_BIN_PATH=/usr/local/cuda1.0 instead of the default /usr/local/cuda)
cmake/modules/FindCUDA.cmake:# or set CUDA_TOOLKIT_ROOT_DIR after configuring.  If you change the value of
cmake/modules/FindCUDA.cmake:# CUDA_TOOLKIT_ROOT_DIR, various components that depend on the path will be
cmake/modules/FindCUDA.cmake:# It might be necessary to set CUDA_TOOLKIT_ROOT_DIR manually on certain
cmake/modules/FindCUDA.cmake:# platforms, or to use a cuda runtime not installed in the default location. In
cmake/modules/FindCUDA.cmake:# newer versions of the toolkit the cuda library is included with the graphics
cmake/modules/FindCUDA.cmake:# driver- be sure that the driver version matches what is needed by the cuda
cmake/modules/FindCUDA.cmake:# times in the same directory before calling CUDA_ADD_EXECUTABLE,
cmake/modules/FindCUDA.cmake:# CUDA_ADD_LIBRARY, CUDA_COMPILE, CUDA_COMPILE_PTX or CUDA_WRAP_SRCS.
cmake/modules/FindCUDA.cmake:#  CUDA_64_BIT_DEVICE_CODE (Default matches host bit size)
cmake/modules/FindCUDA.cmake:#     or C files from CUDA code just won't work, because size_t gets defined by
cmake/modules/FindCUDA.cmake:#  CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE (Default ON)
cmake/modules/FindCUDA.cmake:#     file in Visual Studio.  Turn OFF if you add the same cuda file to multiple
cmake/modules/FindCUDA.cmake:#     This allows the user to build the target from the CUDA file; however, bad
cmake/modules/FindCUDA.cmake:#     things can happen if the CUDA source file is added to multiple targets.
cmake/modules/FindCUDA.cmake:#  CUDA_BUILD_CUBIN (Default OFF)
cmake/modules/FindCUDA.cmake:#  CUDA_BUILD_EMULATION (Default OFF for device mode)
cmake/modules/FindCUDA.cmake:#  -- Set to ON for Emulation mode. -D_DEVICEEMU is defined for CUDA C files
cmake/modules/FindCUDA.cmake:#     when CUDA_BUILD_EMULATION is TRUE.
cmake/modules/FindCUDA.cmake:#  CUDA_GENERATED_OUTPUT_DIR (Default CMAKE_CURRENT_BINARY_DIR)
cmake/modules/FindCUDA.cmake:#  CUDA_HOST_COMPILATION_CPP (Default ON)
cmake/modules/FindCUDA.cmake:#  CUDA_NVCC_FLAGS
cmake/modules/FindCUDA.cmake:#  CUDA_NVCC_FLAGS_<CONFIG>
cmake/modules/FindCUDA.cmake:#  CUDA_PROPAGATE_HOST_FLAGS (Default ON)
cmake/modules/FindCUDA.cmake:#     via CUDA_NVCC_FLAGS or through the OPTION flags specified through
cmake/modules/FindCUDA.cmake:#     CUDA_ADD_LIBRARY, CUDA_ADD_EXECUTABLE, or CUDA_WRAP_SRCS.  Flags used for
cmake/modules/FindCUDA.cmake:#  CUDA_VERBOSE_BUILD (Default OFF)
cmake/modules/FindCUDA.cmake:#  -- Set to ON to see all the commands used when building the CUDA file.  When
cmake/modules/FindCUDA.cmake:#     VERBOSE=1 to see output), although setting CUDA_VERBOSE_BUILD to ON will
cmake/modules/FindCUDA.cmake:#  CUDA_ADD_CUFFT_TO_TARGET( cuda_target )
cmake/modules/FindCUDA.cmake:#  CUDA_ADD_CUBLAS_TO_TARGET( cuda_target )
cmake/modules/FindCUDA.cmake:#  CUDA_ADD_EXECUTABLE( cuda_target file0 file1 ...
cmake/modules/FindCUDA.cmake:#  -- Creates an executable "cuda_target" which is made up of the files
cmake/modules/FindCUDA.cmake:#     specified.  All of the non CUDA C files are compiled using the standard
cmake/modules/FindCUDA.cmake:#     build rules specified by CMAKE and the cuda files are compiled to object
cmake/modules/FindCUDA.cmake:#     files using nvcc and the host compiler.  In addition CUDA_INCLUDE_DIRS is
cmake/modules/FindCUDA.cmake:#     nvcc.  Such flags should be modified before calling CUDA_ADD_EXECUTABLE,
cmake/modules/FindCUDA.cmake:#     CUDA_ADD_LIBRARY or CUDA_WRAP_SRCS.
cmake/modules/FindCUDA.cmake:#  CUDA_ADD_LIBRARY( cuda_target file0 file1 ...
cmake/modules/FindCUDA.cmake:#  -- Same as CUDA_ADD_EXECUTABLE except that a library is created.
cmake/modules/FindCUDA.cmake:#  CUDA_BUILD_CLEAN_TARGET()
cmake/modules/FindCUDA.cmake:#  CUDA_COMPILE( generated_files file0 file1 ... [STATIC | SHARED | MODULE]
cmake/modules/FindCUDA.cmake:#  CUDA_COMPILE_PTX( generated_files file0 file1 ... [OPTIONS ...] )
cmake/modules/FindCUDA.cmake:#  CUDA_INCLUDE_DIRECTORIES( path0 path1 ... )
cmake/modules/FindCUDA.cmake:#  CUDA_WRAP_SRCS ( cuda_target format generated_files file0 file1 ...
cmake/modules/FindCUDA.cmake:#  -- This is where all the magic happens.  CUDA_ADD_EXECUTABLE,
cmake/modules/FindCUDA.cmake:#     CUDA_ADD_LIBRARY, CUDA_COMPILE, and CUDA_COMPILE_PTX all call this
cmake/modules/FindCUDA.cmake:#     CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE set to ON), no generated file will
cmake/modules/FindCUDA.cmake:#     be produced for the given cuda file.  This is because when you add the
cmake/modules/FindCUDA.cmake:#     cuda file to Visual Studio it knows that this file produces an object file
cmake/modules/FindCUDA.cmake:#     BUILD_SHARED_LIBS is ignored in CUDA_WRAP_SRCS, but it is respected in
cmake/modules/FindCUDA.cmake:#     CUDA_ADD_LIBRARY.  On some systems special flags are added for building
cmake/modules/FindCUDA.cmake:#  CUDA_VERSION_MAJOR    -- The major version of cuda as reported by nvcc.
cmake/modules/FindCUDA.cmake:#  CUDA_VERSION_MINOR    -- The minor version.
cmake/modules/FindCUDA.cmake:#  CUDA_VERSION
cmake/modules/FindCUDA.cmake:#  CUDA_VERSION_STRING   -- CUDA_VERSION_MAJOR.CUDA_VERSION_MINOR
cmake/modules/FindCUDA.cmake:#  CUDA_TOOLKIT_ROOT_DIR -- Path to the CUDA Toolkit (defined if not set).
cmake/modules/FindCUDA.cmake:#  CUDA_SDK_ROOT_DIR     -- Path to the CUDA SDK.  Use this to find files in the
cmake/modules/FindCUDA.cmake:#                           supported by NVIDIA.  If you want to change
cmake/modules/FindCUDA.cmake:#                           FindCUDA.cmake script for an example of how to clear
cmake/modules/FindCUDA.cmake:#                           use the CUDA_SDK_ROOT_DIR to locate headers or
cmake/modules/FindCUDA.cmake:#  CUDA_INCLUDE_DIRS     -- Include directory for cuda headers.  Added automatically
cmake/modules/FindCUDA.cmake:#                           for CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY.
cmake/modules/FindCUDA.cmake:#  CUDA_LIBRARIES        -- Cuda RT library.
cmake/modules/FindCUDA.cmake:#  CUDA_CUFFT_LIBRARIES  -- Device or emulation library for the Cuda FFT
cmake/modules/FindCUDA.cmake:#                           CUDA_ADD_CUFFT_TO_TARGET macro)
cmake/modules/FindCUDA.cmake:#  CUDA_CUBLAS_LIBRARIES -- Device or emulation library for the Cuda BLAS
cmake/modules/FindCUDA.cmake:#                           CUDA_ADD_CUBLAS_TO_TARGET macro).
cmake/modules/FindCUDA.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
cmake/modules/FindCUDA.cmake:#  Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
cmake/modules/FindCUDA.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
cmake/modules/FindCUDA.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
cmake/modules/FindCUDA.cmake:# FindCUDA.cmake
cmake/modules/FindCUDA.cmake:macro(CUDA_FIND_HELPER_FILE _name _extension)
cmake/modules/FindCUDA.cmake:  find_file(CUDA_${_name} ${_full_name} PATHS ${CMAKE_CURRENT_LIST_DIR}/FindCUDA NO_DEFAULT_PATH)
cmake/modules/FindCUDA.cmake:  if(NOT CUDA_${_name})
cmake/modules/FindCUDA.cmake:    if(CUDA_FIND_REQUIRED)
cmake/modules/FindCUDA.cmake:    else(CUDA_FIND_REQUIRED)
cmake/modules/FindCUDA.cmake:      if(NOT CUDA_FIND_QUIETLY)
cmake/modules/FindCUDA.cmake:      endif(NOT CUDA_FIND_QUIETLY)
cmake/modules/FindCUDA.cmake:    endif(CUDA_FIND_REQUIRED)
cmake/modules/FindCUDA.cmake:  endif(NOT CUDA_${_name})
cmake/modules/FindCUDA.cmake:  set(CUDA_${_name} ${CUDA_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
cmake/modules/FindCUDA.cmake:endmacro(CUDA_FIND_HELPER_FILE)
cmake/modules/FindCUDA.cmake:## CUDA_INCLUDE_NVCC_DEPENDENCIES
cmake/modules/FindCUDA.cmake:macro(CUDA_INCLUDE_NVCC_DEPENDENCIES dependency_file)
cmake/modules/FindCUDA.cmake:  set(CUDA_NVCC_DEPEND)
cmake/modules/FindCUDA.cmake:  set(CUDA_NVCC_DEPEND_REGENERATE FALSE)
cmake/modules/FindCUDA.cmake:    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
cmake/modules/FindCUDA.cmake:#   if(DEFINED CUDA_NVCC_DEPEND)
cmake/modules/FindCUDA.cmake:#     message("CUDA_NVCC_DEPEND set")
cmake/modules/FindCUDA.cmake:#     message("CUDA_NVCC_DEPEND NOT set")
cmake/modules/FindCUDA.cmake:  if(CUDA_NVCC_DEPEND)
cmake/modules/FindCUDA.cmake:    #message("CUDA_NVCC_DEPEND true")
cmake/modules/FindCUDA.cmake:    foreach(f ${CUDA_NVCC_DEPEND})
cmake/modules/FindCUDA.cmake:        set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
cmake/modules/FindCUDA.cmake:  else(CUDA_NVCC_DEPEND)
cmake/modules/FindCUDA.cmake:    #message("CUDA_NVCC_DEPEND false")
cmake/modules/FindCUDA.cmake:    set(CUDA_NVCC_DEPEND_REGENERATE TRUE)
cmake/modules/FindCUDA.cmake:  endif(CUDA_NVCC_DEPEND)
cmake/modules/FindCUDA.cmake:  #message("CUDA_NVCC_DEPEND_REGENERATE = ${CUDA_NVCC_DEPEND_REGENERATE}")
cmake/modules/FindCUDA.cmake:  if(CUDA_NVCC_DEPEND_REGENERATE)
cmake/modules/FindCUDA.cmake:    file(WRITE ${dependency_file} "#FindCUDA.cmake generated file.  Do not edit.\n")
cmake/modules/FindCUDA.cmake:  endif(CUDA_NVCC_DEPEND_REGENERATE)
cmake/modules/FindCUDA.cmake:endmacro(CUDA_INCLUDE_NVCC_DEPENDENCIES)
cmake/modules/FindCUDA.cmake:  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT ON)
cmake/modules/FindCUDA.cmake:  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT OFF)
cmake/modules/FindCUDA.cmake:option(CUDA_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${CUDA_64_BIT_DEVICE_CODE_DEFAULT})
cmake/modules/FindCUDA.cmake:option(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE "Attach the build rule to the CUDA source file.  Enable only when the CUDA source file is added to at most one target." ON)
cmake/modules/FindCUDA.cmake:# Prints out extra information about the cuda file during compilation
cmake/modules/FindCUDA.cmake:option(CUDA_BUILD_CUBIN "Generate and parse .cubin files in Device mode." OFF)
cmake/modules/FindCUDA.cmake:option(CUDA_BUILD_EMULATION "Build in Emulation mode" OFF)
cmake/modules/FindCUDA.cmake:set(CUDA_GENERATED_OUTPUT_DIR "" CACHE PATH "Directory to put all the output files.  If blank it will default to the CMAKE_CURRENT_BINARY_DIR")
cmake/modules/FindCUDA.cmake:option(CUDA_HOST_COMPILATION_CPP "Generated file extension" ON)
cmake/modules/FindCUDA.cmake:set(CUDA_NVCC_FLAGS "" CACHE STRING "Semi-colon delimit multiple arguments.")
cmake/modules/FindCUDA.cmake:option(CUDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" ON)
cmake/modules/FindCUDA.cmake:option(CUDA_VERBOSE_BUILD "Print out the commands run while compiling the CUDA source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)
cmake/modules/FindCUDA.cmake:  CUDA_64_BIT_DEVICE_CODE
cmake/modules/FindCUDA.cmake:  CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
cmake/modules/FindCUDA.cmake:  CUDA_GENERATED_OUTPUT_DIR
cmake/modules/FindCUDA.cmake:  CUDA_HOST_COMPILATION_CPP
cmake/modules/FindCUDA.cmake:  CUDA_NVCC_FLAGS
cmake/modules/FindCUDA.cmake:  CUDA_PROPAGATE_HOST_FLAGS
cmake/modules/FindCUDA.cmake:set(CUDA_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
cmake/modules/FindCUDA.cmake:list(REMOVE_DUPLICATES CUDA_configuration_types)
cmake/modules/FindCUDA.cmake:foreach(config ${CUDA_configuration_types})
cmake/modules/FindCUDA.cmake:    set(CUDA_NVCC_FLAGS_${config_upper} "" CACHE STRING "Semi-colon delimit multiple arguments.")
cmake/modules/FindCUDA.cmake:    mark_as_advanced(CUDA_NVCC_FLAGS_${config_upper})
cmake/modules/FindCUDA.cmake:# Locate CUDA, Set Build Type, etc.
cmake/modules/FindCUDA.cmake:# Check to see if the CUDA_TOOLKIT_ROOT_DIR and CUDA_SDK_ROOT_DIR have changed,
cmake/modules/FindCUDA.cmake:if(NOT "${CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "${CUDA_TOOLKIT_ROOT_DIR_INTERNAL}")
cmake/modules/FindCUDA.cmake:  unset(CUDA_NVCC_EXECUTABLE CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_VERSION CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_TOOLKIT_INCLUDE CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_CUDART_LIBRARY CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_CUDA_LIBRARY CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_cublas_LIBRARY CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_cublasemu_LIBRARY CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_cufft_LIBRARY CACHE)
cmake/modules/FindCUDA.cmake:  unset(CUDA_cufftemu_LIBRARY CACHE)
cmake/modules/FindCUDA.cmake:if(NOT "${CUDA_SDK_ROOT_DIR}" STREQUAL "${CUDA_SDK_ROOT_DIR_INTERNAL}")
cmake/modules/FindCUDA.cmake:  # find_package(CUDA) to clean up any variables that may depend on this path.
cmake/modules/FindCUDA.cmake:  #   unset(MY_SPECIAL_CUDA_SDK_INCLUDE_DIR CACHE)
cmake/modules/FindCUDA.cmake:  #   unset(MY_SPECIAL_CUDA_SDK_LIBRARY CACHE)
cmake/modules/FindCUDA.cmake:# Search for the cuda distribution.
cmake/modules/FindCUDA.cmake:if(NOT CUDA_TOOLKIT_ROOT_DIR)
cmake/modules/FindCUDA.cmake:  # Search in the CUDA_BIN_PATH first.
cmake/modules/FindCUDA.cmake:  find_path(CUDA_TOOLKIT_ROOT_DIR
cmake/modules/FindCUDA.cmake:    PATHS ENV CUDA_BIN_PATH
cmake/modules/FindCUDA.cmake:  find_path(CUDA_TOOLKIT_ROOT_DIR
cmake/modules/FindCUDA.cmake:          /usr/local/cuda/bin
cmake/modules/FindCUDA.cmake:  if (CUDA_TOOLKIT_ROOT_DIR)
cmake/modules/FindCUDA.cmake:    string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
cmake/modules/FindCUDA.cmake:    set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR} CACHE PATH "Toolkit location." FORCE)
cmake/modules/FindCUDA.cmake:  endif(CUDA_TOOLKIT_ROOT_DIR)
cmake/modules/FindCUDA.cmake:  if (NOT EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
cmake/modules/FindCUDA.cmake:    if(CUDA_FIND_REQUIRED)
cmake/modules/FindCUDA.cmake:      message(FATAL_ERROR "Specify CUDA_TOOLKIT_ROOT_DIR")
cmake/modules/FindCUDA.cmake:    elseif(NOT CUDA_FIND_QUIETLY)
cmake/modules/FindCUDA.cmake:      message("CUDA_TOOLKIT_ROOT_DIR not found or specified")
cmake/modules/FindCUDA.cmake:  endif (NOT EXISTS ${CUDA_TOOLKIT_ROOT_DIR})
cmake/modules/FindCUDA.cmake:endif (NOT CUDA_TOOLKIT_ROOT_DIR)
cmake/modules/FindCUDA.cmake:# CUDA_NVCC_EXECUTABLE
cmake/modules/FindCUDA.cmake:find_program(CUDA_NVCC_EXECUTABLE
cmake/modules/FindCUDA.cmake:  PATHS "${CUDA_TOOLKIT_ROOT_DIR}/bin"
cmake/modules/FindCUDA.cmake:        "${CUDA_TOOLKIT_ROOT_DIR}/bin64"
cmake/modules/FindCUDA.cmake:  ENV CUDA_BIN_PATH
cmake/modules/FindCUDA.cmake:find_program(CUDA_NVCC_EXECUTABLE nvcc)
cmake/modules/FindCUDA.cmake:mark_as_advanced(CUDA_NVCC_EXECUTABLE)
cmake/modules/FindCUDA.cmake:if(CUDA_NVCC_EXECUTABLE AND NOT CUDA_VERSION)
cmake/modules/FindCUDA.cmake:  execute_process (COMMAND ${CUDA_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
cmake/modules/FindCUDA.cmake:  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${NVCC_OUT})
cmake/modules/FindCUDA.cmake:  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${NVCC_OUT})
cmake/modules/FindCUDA.cmake:  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
cmake/modules/FindCUDA.cmake:  mark_as_advanced(CUDA_VERSION)
cmake/modules/FindCUDA.cmake:set(CUDA_VERSION_STRING "${CUDA_VERSION}")
cmake/modules/FindCUDA.cmake:# assume that is unless CUDA_FIND_VERSION_EXACT or CUDA_FIND_VERSION is
cmake/modules/FindCUDA.cmake:set(_cuda_version_acceptable TRUE)
cmake/modules/FindCUDA.cmake:if(CUDA_FIND_VERSION_EXACT AND NOT CUDA_VERSION VERSION_EQUAL CUDA_FIND_VERSION)
cmake/modules/FindCUDA.cmake:  set(_cuda_version_acceptable FALSE)
cmake/modules/FindCUDA.cmake:if(CUDA_FIND_VERSION       AND     CUDA_VERSION VERSION_LESS  CUDA_FIND_VERSION)
cmake/modules/FindCUDA.cmake:  set(_cuda_version_acceptable FALSE)
cmake/modules/FindCUDA.cmake:if(NOT _cuda_version_acceptable)
cmake/modules/FindCUDA.cmake:  set(_cuda_error_message "Requested CUDA version ${CUDA_FIND_VERSION}, but found unacceptable version ${CUDA_VERSION}")
cmake/modules/FindCUDA.cmake:  if(CUDA_FIND_REQUIRED)
cmake/modules/FindCUDA.cmake:    message("${_cuda_error_message}")
cmake/modules/FindCUDA.cmake:  elseif(NOT CUDA_FIND_QUIETLY)
cmake/modules/FindCUDA.cmake:    message("${_cuda_error_message}")
cmake/modules/FindCUDA.cmake:# CUDA_TOOLKIT_INCLUDE
cmake/modules/FindCUDA.cmake:find_path(CUDA_TOOLKIT_INCLUDE
cmake/modules/FindCUDA.cmake:  PATHS "${CUDA_TOOLKIT_ROOT_DIR}/include"
cmake/modules/FindCUDA.cmake:  ENV CUDA_INC_PATH
cmake/modules/FindCUDA.cmake:find_path(CUDA_TOOLKIT_INCLUDE device_functions.h)
cmake/modules/FindCUDA.cmake:mark_as_advanced(CUDA_TOOLKIT_INCLUDE)
cmake/modules/FindCUDA.cmake:set (CUDA_NVCC_INCLUDE_ARGS_USER "")
cmake/modules/FindCUDA.cmake:set (CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_INCLUDE})
cmake/modules/FindCUDA.cmake:    set(_cuda_64bit_lib_dir "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
cmake/modules/FindCUDA.cmake:    PATHS ${_cuda_64bit_lib_dir}
cmake/modules/FindCUDA.cmake:          "${CUDA_TOOLKIT_ROOT_DIR}/lib"
cmake/modules/FindCUDA.cmake:    ENV CUDA_LIB_PATH
cmake/modules/FindCUDA.cmake:# CUDA_LIBRARIES
cmake/modules/FindCUDA.cmake:find_library_local_first(CUDA_CUDART_LIBRARY cudart "\"cudart\" library")
cmake/modules/FindCUDA.cmake:set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
cmake/modules/FindCUDA.cmake:  # We need to add the path to cudart to the linker using rpath, since the
cmake/modules/FindCUDA.cmake:  # library name for the cuda libraries is prepended with @rpath.
cmake/modules/FindCUDA.cmake:  get_filename_component(_cuda_path_to_cudart "${CUDA_CUDART_LIBRARY}" PATH)
cmake/modules/FindCUDA.cmake:  if(_cuda_path_to_cudart)
cmake/modules/FindCUDA.cmake:    list(APPEND CUDA_LIBRARIES -Wl,-rpath "-Wl,${_cuda_path_to_cudart}")
cmake/modules/FindCUDA.cmake:find_library_local_first(CUDA_CUDA_LIBRARY cuda "\"cuda\" library (older versions only).")
cmake/modules/FindCUDA.cmake:# Add cuda library to the link line only if it is found.
cmake/modules/FindCUDA.cmake:if (CUDA_CUDA_LIBRARY)
cmake/modules/FindCUDA.cmake:  set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
cmake/modules/FindCUDA.cmake:endif(CUDA_CUDA_LIBRARY)
cmake/modules/FindCUDA.cmake:  CUDA_CUDA_LIBRARY
cmake/modules/FindCUDA.cmake:  CUDA_CUDART_LIBRARY
cmake/modules/FindCUDA.cmake:macro(FIND_CUDA_HELPER_LIBS _name)
cmake/modules/FindCUDA.cmake:  find_library_local_first(CUDA_${_name}_LIBRARY ${_name} "\"${_name}\" library")
cmake/modules/FindCUDA.cmake:  mark_as_advanced(CUDA_${_name}_LIBRARY)
cmake/modules/FindCUDA.cmake:endmacro(FIND_CUDA_HELPER_LIBS)
cmake/modules/FindCUDA.cmake:find_cuda_helper_libs(cufftemu)
cmake/modules/FindCUDA.cmake:find_cuda_helper_libs(cublasemu)
cmake/modules/FindCUDA.cmake:find_cuda_helper_libs(cufft)
cmake/modules/FindCUDA.cmake:find_cuda_helper_libs(cublas)
cmake/modules/FindCUDA.cmake:if (CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:  set(CUDA_CUFFT_LIBRARIES ${CUDA_cufftemu_LIBRARY})
cmake/modules/FindCUDA.cmake:  set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublasemu_LIBRARY})
cmake/modules/FindCUDA.cmake:  set(CUDA_CUFFT_LIBRARIES ${CUDA_cufft_LIBRARY})
cmake/modules/FindCUDA.cmake:  set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublas_LIBRARY})
cmake/modules/FindCUDA.cmake:find_path(CUDA_SDK_ROOT_DIR common/inc/cutil.h
cmake/modules/FindCUDA.cmake:  "$ENV{NVSDKCUDA_ROOT}"
cmake/modules/FindCUDA.cmake:  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]"
cmake/modules/FindCUDA.cmake:  "/Developer/GPU\ Computing/C"
cmake/modules/FindCUDA.cmake:# Keep the CUDA_SDK_ROOT_DIR first in order to be able to override the
cmake/modules/FindCUDA.cmake:set(CUDA_SDK_SEARCH_PATH
cmake/modules/FindCUDA.cmake:  "${CUDA_SDK_ROOT_DIR}"
cmake/modules/FindCUDA.cmake:  "${CUDA_TOOLKIT_ROOT_DIR}/local/NVSDK0.2"
cmake/modules/FindCUDA.cmake:  "${CUDA_TOOLKIT_ROOT_DIR}/NVSDK0.2"
cmake/modules/FindCUDA.cmake:  "${CUDA_TOOLKIT_ROOT_DIR}/NV_CUDA_SDK"
cmake/modules/FindCUDA.cmake:  "$ENV{HOME}/NVIDIA_CUDA_SDK"
cmake/modules/FindCUDA.cmake:  "$ENV{HOME}/NVIDIA_CUDA_SDK_MACOSX"
cmake/modules/FindCUDA.cmake:  "/Developer/CUDA"
cmake/modules/FindCUDA.cmake:# Example of how to find an include file from the CUDA_SDK_ROOT_DIR
cmake/modules/FindCUDA.cmake:# find_path(CUDA_CUT_INCLUDE_DIR
cmake/modules/FindCUDA.cmake:#   PATHS ${CUDA_SDK_SEARCH_PATH}
cmake/modules/FindCUDA.cmake:# find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")
cmake/modules/FindCUDA.cmake:# mark_as_advanced(CUDA_CUT_INCLUDE_DIR)
cmake/modules/FindCUDA.cmake:# Example of how to find a library in the CUDA_SDK_ROOT_DIR
cmake/modules/FindCUDA.cmake:#   set(cuda_cutil_name cutil64)
cmake/modules/FindCUDA.cmake:#   set(cuda_cutil_name cutil32)
cmake/modules/FindCUDA.cmake:# find_library(CUDA_CUT_LIBRARY
cmake/modules/FindCUDA.cmake:#   NAMES cutil ${cuda_cutil_name}
cmake/modules/FindCUDA.cmake:#   PATHS ${CUDA_SDK_SEARCH_PATH}
cmake/modules/FindCUDA.cmake:# find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
cmake/modules/FindCUDA.cmake:# mark_as_advanced(CUDA_CUT_LIBRARY)
cmake/modules/FindCUDA.cmake:# set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
cmake/modules/FindCUDA.cmake:set(CUDA_FOUND TRUE)
cmake/modules/FindCUDA.cmake:set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL "${CUDA_TOOLKIT_ROOT_DIR}" CACHE INTERNAL
cmake/modules/FindCUDA.cmake:  "This is the value of the last time CUDA_TOOLKIT_ROOT_DIR was set successfully." FORCE)
cmake/modules/FindCUDA.cmake:set(CUDA_SDK_ROOT_DIR_INTERNAL "${CUDA_SDK_ROOT_DIR}" CACHE INTERNAL
cmake/modules/FindCUDA.cmake:  "This is the value of the last time CUDA_SDK_ROOT_DIR was set successfully." FORCE)
cmake/modules/FindCUDA.cmake:find_package_handle_standard_args(CUDA DEFAULT_MSG
cmake/modules/FindCUDA.cmake:  CUDA_TOOLKIT_ROOT_DIR
cmake/modules/FindCUDA.cmake:  CUDA_NVCC_EXECUTABLE
cmake/modules/FindCUDA.cmake:  CUDA_INCLUDE_DIRS
cmake/modules/FindCUDA.cmake:  CUDA_CUDART_LIBRARY
cmake/modules/FindCUDA.cmake:  _cuda_version_acceptable
cmake/modules/FindCUDA.cmake:macro(CUDA_INCLUDE_DIRECTORIES)
cmake/modules/FindCUDA.cmake:    list(APPEND CUDA_NVCC_INCLUDE_ARGS_USER "-I${dir}")
cmake/modules/FindCUDA.cmake:endmacro(CUDA_INCLUDE_DIRECTORIES)
cmake/modules/FindCUDA.cmake:cuda_find_helper_file(parse_cubin cmake)
cmake/modules/FindCUDA.cmake:cuda_find_helper_file(make2cmake cmake)
cmake/modules/FindCUDA.cmake:cuda_find_helper_file(run_nvcc cmake)
cmake/modules/FindCUDA.cmake:macro(CUDA_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
cmake/modules/FindCUDA.cmake:macro(CUDA_PARSE_NVCC_OPTIONS _option_prefix)
cmake/modules/FindCUDA.cmake:    foreach(config ${CUDA_configuration_types})
cmake/modules/FindCUDA.cmake:# Helper to add the include directory for CUDA only once
cmake/modules/FindCUDA.cmake:function(CUDA_ADD_CUDA_INCLUDE_ONCE)
cmake/modules/FindCUDA.cmake:      if("${dir}" STREQUAL "${CUDA_INCLUDE_DIRS}")
cmake/modules/FindCUDA.cmake:    include_directories(${CUDA_INCLUDE_DIRS})
cmake/modules/FindCUDA.cmake:function(CUDA_BUILD_SHARED_LIBRARY shared_flag)
cmake/modules/FindCUDA.cmake:  list(FIND cmake_args SHARED _cuda_found_SHARED)
cmake/modules/FindCUDA.cmake:  list(FIND cmake_args MODULE _cuda_found_MODULE)
cmake/modules/FindCUDA.cmake:  list(FIND cmake_args STATIC _cuda_found_STATIC)
cmake/modules/FindCUDA.cmake:  if( _cuda_found_SHARED GREATER -1 OR
cmake/modules/FindCUDA.cmake:      _cuda_found_MODULE GREATER -1 OR
cmake/modules/FindCUDA.cmake:      _cuda_found_STATIC GREATER -1)
cmake/modules/FindCUDA.cmake:    set(_cuda_build_shared_libs)
cmake/modules/FindCUDA.cmake:      set(_cuda_build_shared_libs SHARED)
cmake/modules/FindCUDA.cmake:      set(_cuda_build_shared_libs STATIC)
cmake/modules/FindCUDA.cmake:  set(${shared_flag} ${_cuda_build_shared_libs} PARENT_SCOPE)
cmake/modules/FindCUDA.cmake:# to generate a dependency file and a second time with -cuda or -ptx to generate
cmake/modules/FindCUDA.cmake:#   cuda_target         - Target name
cmake/modules/FindCUDA.cmake:macro(CUDA_WRAP_SRCS cuda_target format generated_files)
cmake/modules/FindCUDA.cmake:    message( FATAL_ERROR "Invalid format flag passed to CUDA_WRAP_SRCS: '${format}'.  Use OBJ or PTX.")
cmake/modules/FindCUDA.cmake:  if (CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:  else(CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:  endif(CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:  if(CUDA_HOST_COMPILATION_CPP)
cmake/modules/FindCUDA.cmake:    set(CUDA_C_OR_CXX CXX)
cmake/modules/FindCUDA.cmake:  else(CUDA_HOST_COMPILATION_CPP)
cmake/modules/FindCUDA.cmake:    if(CUDA_VERSION VERSION_LESS "3.0")
cmake/modules/FindCUDA.cmake:      message(WARNING "--host-compilation flag is deprecated in CUDA version >= 3.0.  Removing --host-compilation C flag" )
cmake/modules/FindCUDA.cmake:    set(CUDA_C_OR_CXX C)
cmake/modules/FindCUDA.cmake:  endif(CUDA_HOST_COMPILATION_CPP)
cmake/modules/FindCUDA.cmake:  set(generated_extension ${CMAKE_${CUDA_C_OR_CXX}_OUTPUT_EXTENSION})
cmake/modules/FindCUDA.cmake:  if(CUDA_64_BIT_DEVICE_CODE)
cmake/modules/FindCUDA.cmake:    set( CUDA_build_configuration "$(ConfigurationName)" )
cmake/modules/FindCUDA.cmake:    set( CUDA_build_configuration "${CMAKE_BUILD_TYPE}")
cmake/modules/FindCUDA.cmake:  # Initialize our list of includes with the user ones followed by the CUDA system ones.
cmake/modules/FindCUDA.cmake:  set(CUDA_NVCC_INCLUDE_ARGS ${CUDA_NVCC_INCLUDE_ARGS_USER} "-I${CUDA_INCLUDE_DIRS}")
cmake/modules/FindCUDA.cmake:  get_directory_property(CUDA_NVCC_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
cmake/modules/FindCUDA.cmake:  if(CUDA_NVCC_INCLUDE_DIRECTORIES)
cmake/modules/FindCUDA.cmake:    foreach(dir ${CUDA_NVCC_INCLUDE_DIRECTORIES})
cmake/modules/FindCUDA.cmake:      list(APPEND CUDA_NVCC_INCLUDE_ARGS "-I${dir}")
cmake/modules/FindCUDA.cmake:  set(CUDA_WRAP_OPTION_NVCC_FLAGS)
cmake/modules/FindCUDA.cmake:  foreach(config ${CUDA_configuration_types})
cmake/modules/FindCUDA.cmake:    set(CUDA_WRAP_OPTION_NVCC_FLAGS_${config_upper})
cmake/modules/FindCUDA.cmake:  CUDA_GET_SOURCES_AND_OPTIONS(_cuda_wrap_sources _cuda_wrap_cmake_options _cuda_wrap_options ${ARGN})
cmake/modules/FindCUDA.cmake:  CUDA_PARSE_NVCC_OPTIONS(CUDA_WRAP_OPTION_NVCC_FLAGS ${_cuda_wrap_options})
cmake/modules/FindCUDA.cmake:  # respected in CUDA_ADD_LIBRARY.
cmake/modules/FindCUDA.cmake:  set(_cuda_build_shared_libs FALSE)
cmake/modules/FindCUDA.cmake:  list(FIND _cuda_wrap_cmake_options SHARED _cuda_found_SHARED)
cmake/modules/FindCUDA.cmake:  list(FIND _cuda_wrap_cmake_options MODULE _cuda_found_MODULE)
cmake/modules/FindCUDA.cmake:  if(_cuda_found_SHARED GREATER -1 OR _cuda_found_MODULE GREATER -1)
cmake/modules/FindCUDA.cmake:    set(_cuda_build_shared_libs TRUE)
cmake/modules/FindCUDA.cmake:  list(FIND _cuda_wrap_cmake_options STATIC _cuda_found_STATIC)
cmake/modules/FindCUDA.cmake:  if(_cuda_found_STATIC GREATER -1)
cmake/modules/FindCUDA.cmake:    set(_cuda_build_shared_libs FALSE)
cmake/modules/FindCUDA.cmake:  # CUDA_HOST_FLAGS
cmake/modules/FindCUDA.cmake:  if(_cuda_build_shared_libs)
cmake/modules/FindCUDA.cmake:    set(CUDA_HOST_SHARED_FLAGS ${CMAKE_SHARED_LIBRARY_${CUDA_C_OR_CXX}_FLAGS})
cmake/modules/FindCUDA.cmake:    set(CUDA_HOST_SHARED_FLAGS)
cmake/modules/FindCUDA.cmake:  if(CUDA_PROPAGATE_HOST_FLAGS)
cmake/modules/FindCUDA.cmake:    set(CUDA_HOST_FLAGS "set(CMAKE_HOST_FLAGS ${CMAKE_${CUDA_C_OR_CXX}_FLAGS} ${CUDA_HOST_SHARED_FLAGS})")
cmake/modules/FindCUDA.cmake:    set(CUDA_HOST_FLAGS "set(CMAKE_HOST_FLAGS ${CUDA_HOST_SHARED_FLAGS})")
cmake/modules/FindCUDA.cmake:  set(CUDA_NVCC_FLAGS_CONFIG "# Build specific configuration flags")
cmake/modules/FindCUDA.cmake:  foreach(config ${CUDA_configuration_types})
cmake/modules/FindCUDA.cmake:    if(CUDA_PROPAGATE_HOST_FLAGS)
cmake/modules/FindCUDA.cmake:        string(REPLACE "-g3" "-g" _cuda_C_FLAGS "${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}}")
cmake/modules/FindCUDA.cmake:        set(_cuda_C_FLAGS "${CMAKE_${CUDA_C_OR_CXX}_FLAGS_${config_upper}}")
cmake/modules/FindCUDA.cmake:      set(CUDA_HOST_FLAGS "${CUDA_HOST_FLAGS}\nset(CMAKE_HOST_FLAGS_${config_upper} ${_cuda_C_FLAGS})")
cmake/modules/FindCUDA.cmake:    # Note that if we ever want CUDA_NVCC_FLAGS_<CONFIG> to be string (instead of a list
cmake/modules/FindCUDA.cmake:    # ${CUDA_NVCC_FLAGS_${config_upper}} variable like the CMAKE_HOST_FLAGS_<CONFIG> variable.
cmake/modules/FindCUDA.cmake:    set(CUDA_NVCC_FLAGS_CONFIG "${CUDA_NVCC_FLAGS_CONFIG}\nset(CUDA_NVCC_FLAGS_${config_upper} \"${CUDA_NVCC_FLAGS_${config_upper}};;${CUDA_WRAP_OPTION_NVCC_FLAGS_${config_upper}}\")")
cmake/modules/FindCUDA.cmake:    set(CUDA_HOST_FLAGS)
cmake/modules/FindCUDA.cmake:    set(CUDA_NVCC_FLAGS_CONFIG)
cmake/modules/FindCUDA.cmake:  get_directory_property(CUDA_NVCC_DEFINITIONS COMPILE_DEFINITIONS)
cmake/modules/FindCUDA.cmake:  if(CUDA_NVCC_DEFINITIONS)
cmake/modules/FindCUDA.cmake:    foreach(_definition ${CUDA_NVCC_DEFINITIONS})
cmake/modules/FindCUDA.cmake:  if(_cuda_build_shared_libs)
cmake/modules/FindCUDA.cmake:    list(APPEND nvcc_flags "-D${cuda_target}_EXPORTS")
cmake/modules/FindCUDA.cmake:  if(CUDA_GENERATED_OUTPUT_DIR)
cmake/modules/FindCUDA.cmake:    set(cuda_compile_output_dir "${CUDA_GENERATED_OUTPUT_DIR}")
cmake/modules/FindCUDA.cmake:    set(cuda_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
cmake/modules/FindCUDA.cmake:  set(_cuda_wrap_generated_files "")
cmake/modules/FindCUDA.cmake:        set(generated_file_path "${cuda_compile_output_dir}")
cmake/modules/FindCUDA.cmake:        set(generated_file_basename "${cuda_target}_generated_${basename}.ptx")
cmake/modules/FindCUDA.cmake:        file(MAKE_DIRECTORY "${cuda_compile_output_dir}")
cmake/modules/FindCUDA.cmake:        set(generated_file_path "${cuda_compile_output_dir}/${CMAKE_CFG_INTDIR}")
cmake/modules/FindCUDA.cmake:        set(generated_file_basename "${cuda_target}_generated_${basename}${generated_extension}")
cmake/modules/FindCUDA.cmake:      # Bring in the dependencies.  Creates a variable CUDA_NVCC_DEPEND #######
cmake/modules/FindCUDA.cmake:      cuda_include_nvcc_dependencies(${cmake_dependency_file})
cmake/modules/FindCUDA.cmake:      if(CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:        set(cuda_build_type "Emulation")
cmake/modules/FindCUDA.cmake:      else(CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:        set(cuda_build_type "Device")
cmake/modules/FindCUDA.cmake:      endif(CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:      if ( NOT CUDA_BUILD_EMULATION AND CUDA_BUILD_CUBIN )
cmake/modules/FindCUDA.cmake:      endif( NOT CUDA_BUILD_EMULATION AND CUDA_BUILD_CUBIN )
cmake/modules/FindCUDA.cmake:      configure_file("${CUDA_run_nvcc}" "${custom_target_script}" @ONLY)
cmake/modules/FindCUDA.cmake:      # So if a user specifies the same cuda file as input more than once, you
cmake/modules/FindCUDA.cmake:      if(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE)
cmake/modules/FindCUDA.cmake:      if(CUDA_VERBOSE_BUILD)
cmake/modules/FindCUDA.cmake:        set(cuda_build_comment_string "Building NVCC ptx file ${generated_file_relative_path}")
cmake/modules/FindCUDA.cmake:        set(cuda_build_comment_string "Building NVCC (${cuda_build_type}) object ${generated_file_relative_path}")
cmake/modules/FindCUDA.cmake:        DEPENDS ${CUDA_NVCC_DEPEND}
cmake/modules/FindCUDA.cmake:          -D build_configuration:STRING=${CUDA_build_configuration}
cmake/modules/FindCUDA.cmake:        COMMENT "${cuda_build_comment_string}"
cmake/modules/FindCUDA.cmake:      # visual studio and we are attaching the build rule to the cuda file.  VS
cmake/modules/FindCUDA.cmake:      set(cuda_add_generated_file TRUE)
cmake/modules/FindCUDA.cmake:      if(NOT compile_to_ptx AND CMAKE_GENERATOR MATCHES "Visual Studio" AND CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE)
cmake/modules/FindCUDA.cmake:          set(cuda_add_generated_file FALSE)
cmake/modules/FindCUDA.cmake:      if(cuda_add_generated_file)
cmake/modules/FindCUDA.cmake:        list(APPEND _cuda_wrap_generated_files ${generated_file})
cmake/modules/FindCUDA.cmake:      list(APPEND CUDA_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
cmake/modules/FindCUDA.cmake:      list(REMOVE_DUPLICATES CUDA_ADDITIONAL_CLEAN_FILES)
cmake/modules/FindCUDA.cmake:      set(CUDA_ADDITIONAL_CLEAN_FILES ${CUDA_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the cuda dependency scanning.")
cmake/modules/FindCUDA.cmake:  set(${generated_files} ${_cuda_wrap_generated_files})
cmake/modules/FindCUDA.cmake:endmacro(CUDA_WRAP_SRCS)
cmake/modules/FindCUDA.cmake:macro(CUDA_ADD_LIBRARY cuda_target)
cmake/modules/FindCUDA.cmake:  CUDA_ADD_CUDA_INCLUDE_ONCE()
cmake/modules/FindCUDA.cmake:  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
cmake/modules/FindCUDA.cmake:  CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
cmake/modules/FindCUDA.cmake:  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
cmake/modules/FindCUDA.cmake:    ${_cmake_options} ${_cuda_shared_flag}
cmake/modules/FindCUDA.cmake:  add_library(${cuda_target} ${_cmake_options}
cmake/modules/FindCUDA.cmake:  target_link_libraries(${cuda_target}
cmake/modules/FindCUDA.cmake:    ${CUDA_LIBRARIES}
cmake/modules/FindCUDA.cmake:  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
cmake/modules/FindCUDA.cmake:  set_target_properties(${cuda_target}
cmake/modules/FindCUDA.cmake:    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
cmake/modules/FindCUDA.cmake:endmacro(CUDA_ADD_LIBRARY cuda_target)
cmake/modules/FindCUDA.cmake:macro(CUDA_ADD_EXECUTABLE cuda_target)
cmake/modules/FindCUDA.cmake:  CUDA_ADD_CUDA_INCLUDE_ONCE()
cmake/modules/FindCUDA.cmake:  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
cmake/modules/FindCUDA.cmake:  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )
cmake/modules/FindCUDA.cmake:  add_executable(${cuda_target} ${_cmake_options}
cmake/modules/FindCUDA.cmake:  target_link_libraries(${cuda_target}
cmake/modules/FindCUDA.cmake:    ${CUDA_LIBRARIES}
cmake/modules/FindCUDA.cmake:  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
cmake/modules/FindCUDA.cmake:  set_target_properties(${cuda_target}
cmake/modules/FindCUDA.cmake:    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
cmake/modules/FindCUDA.cmake:endmacro(CUDA_ADD_EXECUTABLE cuda_target)
cmake/modules/FindCUDA.cmake:# CUDA COMPILE
cmake/modules/FindCUDA.cmake:macro(CUDA_COMPILE generated_files)
cmake/modules/FindCUDA.cmake:  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
cmake/modules/FindCUDA.cmake:  CUDA_WRAP_SRCS( cuda_compile OBJ _generated_files ${_sources} ${_cmake_options}
cmake/modules/FindCUDA.cmake:endmacro(CUDA_COMPILE)
cmake/modules/FindCUDA.cmake:# CUDA COMPILE PTX
cmake/modules/FindCUDA.cmake:macro(CUDA_COMPILE_PTX generated_files)
cmake/modules/FindCUDA.cmake:  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
cmake/modules/FindCUDA.cmake:  CUDA_WRAP_SRCS( cuda_compile_ptx PTX _generated_files ${_sources} ${_cmake_options}
cmake/modules/FindCUDA.cmake:endmacro(CUDA_COMPILE_PTX)
cmake/modules/FindCUDA.cmake:# CUDA ADD CUFFT TO TARGET
cmake/modules/FindCUDA.cmake:macro(CUDA_ADD_CUFFT_TO_TARGET target)
cmake/modules/FindCUDA.cmake:  if (CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:    target_link_libraries(${target} ${CUDA_cufftemu_LIBRARY})
cmake/modules/FindCUDA.cmake:    target_link_libraries(${target} ${CUDA_cufft_LIBRARY})
cmake/modules/FindCUDA.cmake:# CUDA ADD CUBLAS TO TARGET
cmake/modules/FindCUDA.cmake:macro(CUDA_ADD_CUBLAS_TO_TARGET target)
cmake/modules/FindCUDA.cmake:  if (CUDA_BUILD_EMULATION)
cmake/modules/FindCUDA.cmake:    target_link_libraries(${target} ${CUDA_cublasemu_LIBRARY})
cmake/modules/FindCUDA.cmake:    target_link_libraries(${target} ${CUDA_cublas_LIBRARY})
cmake/modules/FindCUDA.cmake:# CUDA BUILD CLEAN TARGET
cmake/modules/FindCUDA.cmake:macro(CUDA_BUILD_CLEAN_TARGET)
cmake/modules/FindCUDA.cmake:  # Call this after you add all your CUDA targets, and you will get a convience
cmake/modules/FindCUDA.cmake:  set(cuda_clean_target_name clean_cuda_depends)
cmake/modules/FindCUDA.cmake:    string(TOUPPER ${cuda_clean_target_name} cuda_clean_target_name)
cmake/modules/FindCUDA.cmake:  add_custom_target(${cuda_clean_target_name}
cmake/modules/FindCUDA.cmake:    COMMAND ${CMAKE_COMMAND} -E remove ${CUDA_ADDITIONAL_CLEAN_FILES})
cmake/modules/FindCUDA.cmake:  set(CUDA_ADDITIONAL_CLEAN_FILES "" CACHE INTERNAL "List of intermediate files that are part of the cuda dependency scanning.")
cmake/modules/FindCUDA.cmake:endmacro(CUDA_BUILD_CLEAN_TARGET)
cmake/modules/FindCUDA/parse_cubin.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
cmake/modules/FindCUDA/parse_cubin.cmake:#  Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
cmake/modules/FindCUDA/parse_cubin.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
cmake/modules/FindCUDA/parse_cubin.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
cmake/modules/FindCUDA/run_nvcc.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
cmake/modules/FindCUDA/run_nvcc.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
cmake/modules/FindCUDA/run_nvcc.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
cmake/modules/FindCUDA/run_nvcc.cmake:#                               entries in CUDA_HOST_FLAGS. This is the build
cmake/modules/FindCUDA/run_nvcc.cmake:set(CUDA_make2cmake "@CUDA_make2cmake@")
cmake/modules/FindCUDA/run_nvcc.cmake:set(CUDA_parse_cubin "@CUDA_parse_cubin@")
cmake/modules/FindCUDA/run_nvcc.cmake:set(CUDA_NVCC_EXECUTABLE "@CUDA_NVCC_EXECUTABLE@")
cmake/modules/FindCUDA/run_nvcc.cmake:set(CUDA_NVCC_FLAGS "@CUDA_NVCC_FLAGS@;;@CUDA_WRAP_OPTION_NVCC_FLAGS@")
cmake/modules/FindCUDA/run_nvcc.cmake:@CUDA_NVCC_FLAGS_CONFIG@
cmake/modules/FindCUDA/run_nvcc.cmake:set(CUDA_NVCC_INCLUDE_ARGS "@CUDA_NVCC_INCLUDE_ARGS@")
cmake/modules/FindCUDA/run_nvcc.cmake:# been chosen by FindCUDA.cmake.
cmake/modules/FindCUDA/run_nvcc.cmake:@CUDA_HOST_FLAGS@
cmake/modules/FindCUDA/run_nvcc.cmake:#message("CUDA_NVCC_HOST_COMPILER_FLAGS = ${CUDA_NVCC_HOST_COMPILER_FLAGS}")
cmake/modules/FindCUDA/run_nvcc.cmake:list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${build_configuration}})
cmake/modules/FindCUDA/run_nvcc.cmake:# cuda_execute_process - Executes a command with optional command echo and status message.
cmake/modules/FindCUDA/run_nvcc.cmake:#   CUDA_result - return value from running the command
cmake/modules/FindCUDA/run_nvcc.cmake:macro(cuda_execute_process status command)
cmake/modules/FindCUDA/run_nvcc.cmake:    message(FATAL_ERROR "Malformed call to cuda_execute_process.  Missing COMMAND as second argument. (command = ${command})")
cmake/modules/FindCUDA/run_nvcc.cmake:    set(cuda_execute_process_string)
cmake/modules/FindCUDA/run_nvcc.cmake:        list(APPEND cuda_execute_process_string "\"${arg}\"")
cmake/modules/FindCUDA/run_nvcc.cmake:        list(APPEND cuda_execute_process_string ${arg})
cmake/modules/FindCUDA/run_nvcc.cmake:    execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${cuda_execute_process_string})
cmake/modules/FindCUDA/run_nvcc.cmake:  execute_process(COMMAND ${ARGN} RESULT_VARIABLE CUDA_result )
cmake/modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:# For CUDA 2.3 and below, -G -M doesn't work, so remove the -G flag
cmake/modules/FindCUDA/run_nvcc.cmake:set(depends_CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
cmake/modules/FindCUDA/run_nvcc.cmake:set(CUDA_VERSION @CUDA_VERSION@)
cmake/modules/FindCUDA/run_nvcc.cmake:if(CUDA_VERSION VERSION_LESS "3.0")
cmake/modules/FindCUDA/run_nvcc.cmake:  list(REMOVE_ITEM depends_CUDA_NVCC_FLAGS "-G")
cmake/modules/FindCUDA/run_nvcc.cmake:# nvcc doesn't define __CUDACC__ for some reason when generating dependency files.  This
cmake/modules/FindCUDA/run_nvcc.cmake:set(CUDACC_DEFINE -D__CUDACC__)
cmake/modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
cmake/modules/FindCUDA/run_nvcc.cmake:  ${CUDACC_DEFINE}
cmake/modules/FindCUDA/run_nvcc.cmake:  ${depends_CUDA_NVCC_FLAGS}
cmake/modules/FindCUDA/run_nvcc.cmake:  ${CUDA_NVCC_INCLUDE_ARGS}
cmake/modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
cmake/modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:  -P "${CUDA_make2cmake}"
cmake/modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
cmake/modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
cmake/modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
cmake/modules/FindCUDA/run_nvcc.cmake:cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:  COMMAND "${CUDA_NVCC_EXECUTABLE}"
cmake/modules/FindCUDA/run_nvcc.cmake:  ${CUDA_NVCC_FLAGS}
cmake/modules/FindCUDA/run_nvcc.cmake:  ${CUDA_NVCC_INCLUDE_ARGS}
cmake/modules/FindCUDA/run_nvcc.cmake:if(CUDA_result)
cmake/modules/FindCUDA/run_nvcc.cmake:  cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:  cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:    COMMAND "${CUDA_NVCC_EXECUTABLE}"
cmake/modules/FindCUDA/run_nvcc.cmake:    ${CUDA_NVCC_FLAGS}
cmake/modules/FindCUDA/run_nvcc.cmake:    ${CUDA_NVCC_INCLUDE_ARGS}
cmake/modules/FindCUDA/run_nvcc.cmake:  cuda_execute_process(
cmake/modules/FindCUDA/run_nvcc.cmake:    -P "${CUDA_parse_cubin}"
cmake/modules/FindCUDA/make2cmake.cmake:#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
cmake/modules/FindCUDA/make2cmake.cmake:#  Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindCuda.html
cmake/modules/FindCUDA/make2cmake.cmake:#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
cmake/modules/FindCUDA/make2cmake.cmake:#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
cmake/modules/FindCUDA/make2cmake.cmake:  set(cuda_nvcc_depend "${cuda_nvcc_depend} \"${file}\"\n")
cmake/modules/FindCUDA/make2cmake.cmake:file(WRITE ${output_file} "# Generated by: make2cmake.cmake\nSET(CUDA_NVCC_DEPEND\n ${cuda_nvcc_depend})\n\n")
src/simulate_gpu.cu.h:#include "modules/FeH_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/GaussianFeH_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/fixedFeH_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/unresolvedMultiples_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/photometry_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/kinTMIII_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/Bond2010_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/vel2pm_gpu.cu.h"
src/simulate_gpu.cu.h:#include "modules/gal2other_gpu.cu.h"
src/modules/unresolvedMultiples_host.cpp:#include "unresolvedMultiples_gpu.cu.h"
src/modules/unresolvedMultiples_host.cpp:DECLARE_TEXTURE(secProb,  float, 1, cudaReadModeElementType);
src/modules/unresolvedMultiples_host.cpp:DECLARE_TEXTURE(cumLF,    float, 1, cudaReadModeElementType);
src/modules/unresolvedMultiples_host.cpp:DECLARE_TEXTURE(invCumLF, float, 1, cudaReadModeElementType);
src/modules/fixedFeH_host.cpp:#include "fixedFeH_gpu.cu.h"
src/modules/FeH_gpu.cu.h:#ifndef FEH_gpu_cu_h__
src/modules/FeH_gpu.cu.h:#define FEH_gpu_cu_h__
src/modules/FeH_gpu.cu.h:#if (__CUDACC__ || BUILD_FOR_CPU)
src/modules/FeH_gpu.cu.h:		otable_ks ks, os_FeH_data par, gpu_rng_t rng, 
src/modules/FeH_gpu.cu.h:		cint_t::gpu_t comp,
src/modules/FeH_gpu.cu.h:		cint_t::gpu_t hidden,
src/modules/FeH_gpu.cu.h:		cfloat_t::gpu_t XYZ,
src/modules/FeH_gpu.cu.h:		cfloat_t::gpu_t FeH),
src/modules/FeH_gpu.cu.h:#endif // (__CUDACC__ || BUILD_FOR_CPU)
src/modules/FeH_gpu.cu.h:#endif // FEH_gpu_cu_h__
src/modules/module_lib.h:	#include "gpu.h"
src/modules/module_lib.h:// 	#ifdef __CUDACC__
src/modules/module_lib.h:// 		__device__ __constant__ float Rg_gpu;
src/modules/module_lib.h:// 		__device__ inline float Rg() { return Rg_gpu; }
src/modules/module_lib.h:	namespace cudacc
src/modules/vel2pm_host.cpp:#include "vel2pm_gpu.cu.h"
src/modules/Bond2010_host.cpp:#include "Bond2010_gpu.cu.h"
src/modules/Bond2010_host.cpp:	cuxUploadConst("os_Bond2010_par", static_cast<os_Bond2010_data&>(*this));	// for GPU execution
src/modules/photometricErrors_host.cpp:#include "FeH_gpu.cu.h"
src/modules/gal2other_gpu.cu.h:#ifndef gal2other_gpu_cu_h__
src/modules/gal2other_gpu.cu.h:#define gal2other_gpu_cu_h__
src/modules/gal2other_gpu.cu.h:#if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/gal2other_gpu.cu.h:	DECLARE_KERNEL(os_gal2other_kernel(otable_ks ks, int coordsys, cint_t::gpu_t hidden, cdouble_t::gpu_t lb0, cdouble_t::gpu_t out));
src/modules/gal2other_gpu.cu.h:#else // #if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/gal2other_gpu.cu.h:		os_gal2other_kernel(otable_ks ks, int coordsys, cint_t::gpu_t hidden, cdouble_t::gpu_t lb0, cdouble_t::gpu_t out),
src/modules/gal2other_gpu.cu.h:#endif // #else (!__CUDACC__ && !BUILD_FOR_CPU)
src/modules/gal2other_gpu.cu.h:#endif // gal2other_gpu_cu_h__
src/modules/photometricErrors_gpu.cu.h:#ifndef photometricErrors_gpu_cu_h__
src/modules/photometricErrors_gpu.cu.h:#define photometricErrors_gpu_cu_h__
src/modules/photometricErrors_gpu.cu.h:// -- none for now (this is a CPU module; TODO: port this module to GPU)
src/modules/photometricErrors_gpu.cu.h:#if (__CUDACC__ || BUILD_FOR_CPU)
src/modules/photometricErrors_gpu.cu.h:// -- no kernel yet (this is a CPU module; TODO: port this module to GPU)
src/modules/photometricErrors_gpu.cu.h:#endif // (__CUDACC__ || BUILD_FOR_CPU)
src/modules/photometricErrors_gpu.cu.h:#endif // photometricErrors_gpu_cu_h__
src/modules/Bond2010_gpu.cu.h:#ifndef Bond2010_gpu_cu_h__
src/modules/Bond2010_gpu.cu.h:#define Bond2010_gpu_cu_h__
src/modules/Bond2010_gpu.cu.h:#if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/Bond2010_gpu.cu.h:	DECLARE_KERNEL(os_Bond2010_kernel(otable_ks ks, gpu_rng_t rng, cint_t::gpu_t comp, cint_t::gpu_t hidden, cfloat_t::gpu_t XYZ, cfloat_t::gpu_t vcyl));
src/modules/Bond2010_gpu.cu.h:#else // #if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/Bond2010_gpu.cu.h:	void __device__ inline trivar_gaussK_draw(float y[3], float s11, float s12, float s13, float s22, float s23, float s33, gpu_rng_t &rng)
src/modules/Bond2010_gpu.cu.h:	__device__ inline void add_dispersion(K_IO float v[3], float Rsquared, float Z, const farray5 ellip[6], K_IO gpu_rng_t &rng)
src/modules/Bond2010_gpu.cu.h:	__device__ inline void get_disk_kinematics(K_OUT float v[3], float Rsquared, float Z, gpu_rng_t &rng, const os_Bond2010_data& par)
src/modules/Bond2010_gpu.cu.h:	__device__ inline void get_halo_kinematics(K_OUT float v[3], float Rcyl_squared, float Z, K_IO gpu_rng_t &rng, const os_Bond2010_data& par)
src/modules/Bond2010_gpu.cu.h:			otable_ks ks, gpu_rng_t rng, 
src/modules/Bond2010_gpu.cu.h:			cint_t::gpu_t comp,
src/modules/Bond2010_gpu.cu.h:			cint_t::gpu_t hidden,
src/modules/Bond2010_gpu.cu.h:			cfloat_t::gpu_t XYZ,
src/modules/Bond2010_gpu.cu.h:			cfloat_t::gpu_t vcyl),
src/modules/Bond2010_gpu.cu.h:#endif // #else (!__CUDACC__ && !BUILD_FOR_CPU)
src/modules/Bond2010_gpu.cu.h:#endif // Bond2010_gpu_cu_h__
src/modules/vel2pm_gpu.cu.h:#ifndef vel2pm_gpu_cu_h__
src/modules/vel2pm_gpu.cu.h:#define vel2pm_gpu_cu_h__
src/modules/vel2pm_gpu.cu.h:#if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/vel2pm_gpu.cu.h:			otable_ks ks, os_vel2pm_data par, gpu_rng_t rng, 
src/modules/vel2pm_gpu.cu.h:			cdouble_t::gpu_t lb0,
src/modules/vel2pm_gpu.cu.h:			cfloat_t::gpu_t XYZ,
src/modules/vel2pm_gpu.cu.h:			cfloat_t::gpu_t vcyl,
src/modules/vel2pm_gpu.cu.h:			cfloat_t::gpu_t pmlb,
src/modules/vel2pm_gpu.cu.h:			cint_t::gpu_t hidden
src/modules/vel2pm_gpu.cu.h:#else // #if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/vel2pm_gpu.cu.h:			otable_ks ks, os_vel2pm_data par, gpu_rng_t rng, 
src/modules/vel2pm_gpu.cu.h:			cdouble_t::gpu_t lb0,
src/modules/vel2pm_gpu.cu.h:			cfloat_t::gpu_t XYZ,
src/modules/vel2pm_gpu.cu.h:			cfloat_t::gpu_t vcyl,
src/modules/vel2pm_gpu.cu.h:			cfloat_t::gpu_t pmout,
src/modules/vel2pm_gpu.cu.h:			cint_t::gpu_t hidden),
src/modules/vel2pm_gpu.cu.h:#endif // #else (!__CUDACC__ && !BUILD_FOR_CPU)
src/modules/vel2pm_gpu.cu.h:#endif // vel2pm_gpu_cu_h__
src/modules/GaussianFeH_host.cpp:#include "GaussianFeH_gpu.cu.h"
src/modules/GaussianFeH_host.cpp:DECLARE_KERNEL(os_GaussianFeH_kernel(otable_ks ks, bit_map applyToComponents, os_GaussianFeH_data par, gpu_rng_t rng,
src/modules/GaussianFeH_host.cpp:		cint_t::gpu_t comp, cint_t::gpu_t hidden,
src/modules/GaussianFeH_host.cpp:		cfloat_t::gpu_t XYZ,
src/modules/GaussianFeH_host.cpp:		cint_t::gpu_t   FeH_comp,
src/modules/GaussianFeH_host.cpp:		cfloat_t::gpu_t FeH));
src/modules/photometry_host.cpp:		cuxUploadConst("os_photometry_params", static_cast<os_photometry_data&>(*this));	// for GPU execution
src/modules/gal2other_host.cpp:#include "gal2other_gpu.cu.h"
src/modules/kinTMIII_gpu.cu.h:#ifndef kinTMIII_gpu_cu_h__
src/modules/kinTMIII_gpu.cu.h:#define kinTMIII_gpu_cu_h__
src/modules/kinTMIII_gpu.cu.h:#if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/kinTMIII_gpu.cu.h:	DECLARE_KERNEL(os_kinTMIII_kernel(otable_ks ks, gpu_rng_t rng, cint_t::gpu_t comp, cint_t::gpu_t hidden, cfloat_t::gpu_t XYZ, cfloat_t::gpu_t vcyl));
src/modules/kinTMIII_gpu.cu.h:#else // #if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/kinTMIII_gpu.cu.h:	void __device__ inline trivar_gaussK_draw(float y[3], float s11, float s12, float s13, float s22, float s23, float s33, gpu_rng_t &rng)
src/modules/kinTMIII_gpu.cu.h:	__device__ inline void add_dispersion(K_IO float v[3], float Rsquared, float Z, farray5 ellip[6],K_IO gpu_rng_t &rng)
src/modules/kinTMIII_gpu.cu.h:	__device__ inline void get_disk_kinematics(K_OUT float v[3], float Rsquared, float Z, gpu_rng_t &rng,
src/modules/kinTMIII_gpu.cu.h:	__device__ inline void get_halo_kinematics(K_OUT float v[3], float Rsquared, float Z, K_IO gpu_rng_t &rng, farray5 haloMeans[3], farray5 haloEllip[6])
src/modules/kinTMIII_gpu.cu.h:			otable_ks ks, gpu_rng_t rng, 
src/modules/kinTMIII_gpu.cu.h:			cint_t::gpu_t comp, cint_t::gpu_t hidden,
src/modules/kinTMIII_gpu.cu.h:			cfloat_t::gpu_t XYZ,
src/modules/kinTMIII_gpu.cu.h:			cfloat_t::gpu_t vcyl),
src/modules/kinTMIII_gpu.cu.h:#endif // #else (!__CUDACC__ && !BUILD_FOR_CPU)
src/modules/kinTMIII_gpu.cu.h:#endif // kinTMIII_gpu_cu_h__
src/modules/GaussianFeH_gpu.cu.h:#ifndef GaussianFeH_gpu_cu_h__
src/modules/GaussianFeH_gpu.cu.h:#define GaussianFeH_gpu_cu_h__
src/modules/GaussianFeH_gpu.cu.h:#if (__CUDACC__ || BUILD_FOR_CPU)
src/modules/GaussianFeH_gpu.cu.h:		otable_ks ks, bit_map applyToComponents, os_GaussianFeH_data par, gpu_rng_t rng,
src/modules/GaussianFeH_gpu.cu.h:		cint_t::gpu_t comp, cint_t::gpu_t hidden,
src/modules/GaussianFeH_gpu.cu.h:		cfloat_t::gpu_t XYZ,
src/modules/GaussianFeH_gpu.cu.h:		cint_t::gpu_t FeH_comp,
src/modules/GaussianFeH_gpu.cu.h:		cfloat_t::gpu_t FeH),
src/modules/GaussianFeH_gpu.cu.h:#endif // (__CUDACC__ || BUILD_FOR_CPU)
src/modules/GaussianFeH_gpu.cu.h:#endif // GaussianFeH_gpu_cu_h__
src/modules/photometry.h:#if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/photometry.h:	DECLARE_TEXTURE(color0, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:	DECLARE_TEXTURE(color1, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:	DECLARE_TEXTURE(color2, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:	DECLARE_TEXTURE(color3, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:	DECLARE_TEXTURE(cflags0, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:	DECLARE_TEXTURE(cflags1, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:	DECLARE_TEXTURE(cflags2, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:	DECLARE_TEXTURE(cflags3, float4, 2, cudaReadModeElementType);
src/modules/photometry.h:#endif // !__CUDACC__ && !BUILD_FOR_CPU
src/modules/FeH_host.cpp:#include "FeH_gpu.cu.h"
src/modules/FeH_host.cpp:DECLARE_KERNEL(os_FeH_kernel(otable_ks ks, os_FeH_data par, gpu_rng_t rng, cint_t::gpu_t comp, cint_t::gpu_t hidden, cfloat_t::gpu_t XYZ, cfloat_t::gpu_t FeH))
src/modules/kinTMIII_host.cpp:#include "kinTMIII_gpu.cu.h"
src/modules/kinTMIII_host.cpp:	cuxUploadConst("os_kinTMIII_par", static_cast<os_kinTMIII_data&>(*this));	// for GPU execution
src/modules/fixedFeH_gpu.cu.h:#ifndef fixedFeH_gpu_cu_h__
src/modules/fixedFeH_gpu.cu.h:#define fixedFeH_gpu_cu_h__
src/modules/fixedFeH_gpu.cu.h:#if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/fixedFeH_gpu.cu.h:	DECLARE_KERNEL(os_fixedFeH_kernel(otable_ks ks, bit_map applyToComponents, float fixedFeH, cint_t::gpu_t comp, cint_t::gpu_t hidden, cfloat_t::gpu_t FeH));
src/modules/fixedFeH_gpu.cu.h:		os_fixedFeH_kernel(otable_ks ks, bit_map applyToComponents, float fixedFeH, cint_t::gpu_t comp, cint_t::gpu_t hidden, cfloat_t::gpu_t FeH),
src/modules/fixedFeH_gpu.cu.h:#endif // (__CUDACC__ || BUILD_FOR_CPU)
src/modules/fixedFeH_gpu.cu.h:#endif // fixedFeH_gpu_cu_h__
src/modules/unresolvedMultiples_gpu.cu.h:#ifndef unresolvedMultiples_gpu_cu_h__
src/modules/unresolvedMultiples_gpu.cu.h:#define unresolvedMultiples_gpu_cu_h__
src/modules/unresolvedMultiples_gpu.cu.h:#if !__CUDACC__ && !BUILD_FOR_CPU
src/modules/unresolvedMultiples_gpu.cu.h:	DECLARE_KERNEL(os_unresolvedMultiples_kernel(otable_ks ks, bit_map applyToComponents, gpu_rng_t rng, int nabsmag, cfloat_t::gpu_t M, cfloat_t::gpu_t Msys, cint_t::gpu_t ncomp, cint_t::gpu_t comp, cint_t::gpu_t hidden, multiplesAlgorithms::algo algo));
src/modules/unresolvedMultiples_gpu.cu.h:	DEFINE_TEXTURE( secProb, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/unresolvedMultiples_gpu.cu.h:	DEFINE_TEXTURE(   cumLF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/unresolvedMultiples_gpu.cu.h:	DEFINE_TEXTURE(invCumLF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/unresolvedMultiples_gpu.cu.h:	__device__ inline bool draw_companion(float &M2, float M1, multiplesAlgorithms::algo algo, gpu_rng_t &rng)
src/modules/unresolvedMultiples_gpu.cu.h:		os_unresolvedMultiples_kernel(otable_ks ks, bit_map applyToComponents, gpu_rng_t rng, int nabsmag, cfloat_t::gpu_t M, cfloat_t::gpu_t Msys, cint_t::gpu_t ncomp, cint_t::gpu_t comp, cint_t::gpu_t hidden, multiplesAlgorithms::algo algo),
src/modules/unresolvedMultiples_gpu.cu.h:#endif // (__CUDACC__ || BUILD_FOR_CPU)
src/modules/unresolvedMultiples_gpu.cu.h:#endif // unresolvedMultiples_gpu_cu_h__
src/modules/photometry_gpu.cu.h:#ifndef photometry_gpu_cu_h__
src/modules/photometry_gpu.cu.h:#define photometry_gpu_cu_h__
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(color0, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(color1, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(color2, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(color3, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(cflags0, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(cflags1, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(cflags2, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:DEFINE_TEXTURE(cflags3, float4, 2, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/modules/photometry_gpu.cu.h:#endif // photometry_gpu_cu_h__
src/skygen/model_powerLawEllipsoid.h:DEFINE_TEXTURE(powerLawEllipsoidLF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/model_powerLawEllipsoid.h:		using namespace cudacc;
src/skygen/model_LCBulge.h:DEFINE_TEXTURE(LCBulgeLF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/model_LCBulge.h:		using namespace cudacc;
src/skygen/skygen.h:#include <cuda_runtime_api.h>
src/skygen/skygen.h:#include "gpu.h"
src/skygen/skygen.h:typedef prngs::gpu::mwc gpuRng;
src/skygen/skygen.h:	struct state {};		// the state of the model that needs to be kept on the GPU
src/skygen/skygen.h:	cdouble_t::gpu_t	lb;
src/skygen/skygen.h:	cint_t::gpu_t		projIdx;
src/skygen/skygen.h:	cfloat_t::gpu_t		projXY, DM, M, XYZ, Am, AmInf;
src/skygen/skygen.h:	cint_t::gpu_t		comp;
src/skygen/skygen.h:	cint_t::gpu_t		hidden;
src/skygen/skygen.h:		cudaMemset(cont.ptr, 0, nthreads*4); // not continuing a previous run
src/skygen/skygen.h:	void constructor()	// as CUDA doesn't allow real constructors
src/skygen/skygen.h:// This piece gets uploaded as a __constant__ to the GPU. It differs from
src/skygen/skygen.h:struct ALIGN(16) skygenGPU : public skygenParams
src/skygen/skygen.h:// Host interface to the mock catalog generator. Derived from skygenGPU part (which
src/skygen/skygen.h:class ALIGN(16) skygenHost : public skygenGPU<Model>, public skygenInterface
src/skygen/skygen.h:	gpuRng *rng;
src/skygen/skygen.h:	dim3 gridDim, blockDim;		// CUDA grid dimension, block dimension
src/skygen/skygen.h:	virtual uint32_t component() const { return this->model.comp; }			// NOTE: this should actually point to model.component(), but it wouldn't compile on gcc 4.3.4 + CUDA 2.3
src/skygen/skygen.h:// Some generic stub code to instantiate the GPU kernels, the class,
src/skygen/skygen.h:#define SKYGEN_ON_GPU	1
src/skygen/skygen.h:#ifdef __CUDACC__
src/skygen/skygen.h:	#if SKYGEN_ON_GPU
src/skygen/skygen.h:			__device__ __constant__ skygenGPU<name> name##Sky; \
src/skygen/skygen.h:				cuxUploadConst(name##Sky, (skygenGPU<name>&)*this); \
src/skygen/skygen.h:	#if BUILD_FOR_CPU && !SKYGEN_ON_GPU
src/skygen/skygen.h:			__device__ __constant__ skygenGPU<name> name##Sky; \
src/skygen/skygen.h:				cuxUploadConst(name##Sky, (skygenGPU<name>&)*this); \
src/skygen/skygen.h:// Some globally used on-gpu static variables (constant memory)
src/skygen/skygen.h:#if __CUDACC__
src/skygen/skyconfig_impl.h:// Download skygen data to GPU. Flag 'draw' denotes if this call was preceeded
src/skygen/skyconfig_impl.h:#if !SKYGEN_ON_GPU
src/skygen/skyconfig_impl.h:extern gpuRng::constant rng;	// GPU RNG
src/skygen/skyconfig_impl.h:// Upload skygen data to GPU. Flag 'draw' denotes if this call will be followed
src/skygen/skyconfig_impl.h:		cudaMemset(this->rhoHistograms.ptr, 0, this->nthreads*this->nhistbins*4);
src/skygen/skyconfig_impl.h:		cudaMemset(this->maxCount.ptr, 0, this->nthreads*4);
src/skygen/skyconfig_impl.h:		cudaMemset(this->counts.ptr, 0, this->nthreads*4);
src/skygen/skyconfig_impl.h:		cudaMemset(this->countsCovered.ptr, 0, this->nthreads*4);
src/skygen/skyconfig_impl.h:#if SKYGEN_ON_GPU
src/skygen/skyconfig_impl.h:	// GPU kernel execution setup (TODO: should I load this through skygenParams? Or autodetect based on the GPU?)
src/skygen/skyconfig_impl.h:	shb = gpu_rng_t::state_bytes() * blockDim.x; // for RNG
src/skygen/skyconfig_impl.h:	rng = new gpu_rng_t(cpurng);
src/skygen/skygen.cu.h:#include "../gpulog/gpulog.h"
src/skygen/skygen.cu.h:#include "../gpulog/lprintf.h"
src/skygen/skygen.cu.h:#if __CUDACC__
src/skygen/skygen.cu.h:using namespace cudacc;
src/skygen/skygen.cu.h:__constant__ gpuRng::constant rng;	// GPU RNG
src/skygen/skygen.cu.h:#if __CUDACC__
src/skygen/skygen.cu.h:// The assumption is all CUDA code will be concatenated/included and compiled
src/skygen/skygen.cu.h:gpulog::host_log hlog;
src/skygen/skygen.cu.h:__constant__ gpulog::device_log dlog;
src/skygen/skygen.cu.h:extern "C" void flush_logs();	// GPU debug logs
src/skygen/skygen.cu.h:DEFINE_TEXTURE(ext_north, float, 3, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/skygen.cu.h:DEFINE_TEXTURE(ext_south, float, 3, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/skygen.cu.h:DEFINE_TEXTURE(ext_beam, float, 3, cudaReadModeElementType, false, cudaFilterModePoint, cudaAddressModeClamp);
src/skygen/skygen.cu.h:#if !__CUDACC__
src/skygen/skygen.cu.h:#define CPUGPU(name) cpu_##name
src/skygen/skygen.cu.h:#define CPUGPU(name) name
src/skygen/skygen.cu.h:cuxSmartPtr<float4> CPUGPU(resample_extinction_texture)(cuxTexture<float, 3> &tex, float2 crange[3], int npix[3], lambert *proj)
src/skygen/skygen.cu.h:__device__ float3 skygenGPU<T>::compute_pos(float &D, float &Am, float M, const int im, const pencilBeam &pix) const
src/skygen/skygen.cu.h:__device__ bool skygenGPU<T>::advance(int &ilb, int &i, int &j, pencilBeam &dir, const int x, const int y) const
src/skygen/skygen.cu.h:__device__ void skygenGPU<T>::draw_stars(int &ndraw, const float &M, const int &im, const pencilBeam &pix, float AmMin) const
src/skygen/skygen.cu.h:#if __CUDACC__
src/skygen/skygen.cu.h:__device__ void skygenGPU<T>::kernel() const
src/skygen/skygen.cu.h:// NOTE: CUDA compatibility -- in principle, we could only _declare_, but 
src/skygen/skygen.cu.h:	cuxErrCheck( cudaThreadSynchronize() );
src/skygen/skygen.cu.h:#if !__CUDACC__
src/skygen/skygen.cu.h:	cuxErrCheck( cudaThreadSynchronize() );
src/skygen/model_expDisk.h:DEFINE_TEXTURE(expDiskLF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/model_brokenPowerLaw.h://DEFINE_TEXTURE(powerLawEllipsoidLF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/model_brokenPowerLaw.h:		using namespace cudacc;
src/skygen/model_densityCube.h:DEFINE_TEXTURE(densityCubeLF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/model_densityCube.h:DEFINE_TEXTURE(densityCubeTex, float, 3, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/model_densityCube.h:	// uploaded to a GPU __constant__
src/skygen/os_skygen.cpp:	// We're packing this as a 3D texture, because CUDA can't take a 2D texture wider than 64k
src/skygen/os_skygen.cpp:DECLARE_TEXTURE(ext_north, float, 3, cudaReadModeElementType);
src/skygen/os_skygen.cpp:DECLARE_TEXTURE(ext_south, float, 3, cudaReadModeElementType);
src/skygen/os_skygen.cpp:DECLARE_TEXTURE(ext_beam, float, 3, cudaReadModeElementType);
src/skygen/os_skygen.cpp:#include "../gpulog/gpulog.h"
src/skygen/os_skygen.cpp:#include "../gpulog/lprintf.h"
src/skygen/os_skygen.cpp:extern gpulog::host_log hlog;
src/skygen/os_skygen.cpp:	copy(hlog, "dlog", gpulog::LOG_DEVCLEAR);
src/skygen/os_skygen.cpp:	gpulog::alloc_device_log("dlog", device_buffer_size);
src/skygen/model_J08.h:DEFINE_TEXTURE(J08LF, float, 1, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/skygen/model_J08.h:	__device__ int component(float x, float y, float z, float M, gpuRng::constant &rng) const
src/common/cux.h:#ifndef cuda_cux__
src/common/cux.h:#define cuda_cux__
src/common/cux.h:	cuxException -- thrown if a CUDA-related error is detected
src/common/cux.h:	cudaError err;
src/common/cux.h:	cuxException(cudaError err_) : err(err_) {}
src/common/cux.h:	const char *msg() const { return cudaGetErrorString(err); }
src/common/cux.h:	cuxErrCheck macro -- aborts with message if the enclosed call returns != cudaSuccess
src/common/cux.h:void cuxErrCheck_impl(cudaError err, const char *fun, const char *file, const int line);
src/common/cux.h:void cuxMallocErrCheck_impl(cudaError err, size_t msize, const char *fun, const char *file, const int line);
src/common/cux.h:	Hides the raw CUDA API.
src/common/cux.h:		cuxMallocErrCheck( cudaMalloc((void**)&devptr, size), size );
src/common/cux.h:	Hides the raw CUDA API.
src/common/cux.h:		cuxErrCheck( cudaFree(v) );
src/common/cux.h:	Hides the raw CUDA API.
src/common/cux.h:#ifdef __CUDACC__
src/common/cux.h:		cuxErrCheck( cudaMemcpyToSymbol(dest, &source, size) );
src/common/cux.h:		cuxErrCheck( cudaMemcpyToSymbol(symbol, &source, size) );
src/common/cux.h:	it from the CUDA APIs.
src/common/cux.h:			cuxErrCheck( cudaMemcpy(this->ptr, src, size, cudaMemcpyHostToDevice) );
src/common/cux.h:			cuxErrCheck( cudaMemcpy(dest, this->ptr, size, cudaMemcpyDeviceToHost) );
src/common/cux.h:	cudaArray* cuArray;		// CUDA array copy of the data
src/common/cux.h:	bool cleanCudaArray;		// true if the last access operation was obtaining a reference to cudaArray
src/common/cux.h:	// constructs a CUDA array. Used by bind_texture
src/common/cux.h:	cudaArray *getCUDAArray(cudaChannelFormatDesc &channelDesc);
src/common/cux.h:	Smart pointer to an array in GPU or CPU memory, with on-demand garbage collection, and auto-syncing of CPU/GPU copies
src/common/cux.h:		- At any given time, the block is either on GPU, CPU, or bound to texture (in a CUDA array)
src/common/cux.h:		operator gptr<T, dim>()			// request access to data on the GPU
src/common/cux.h:	Work around CUDA defficiency with some built-in struct alignments.
src/common/cux.h:	CUDA header files declare some structs (float2 being an example) with
src/common/cux.h:	CUDACC. This makes those structure's alignments different in nvcc compiled
src/common/cux.h:	the problematic CUDA type. It should be used instead of the CUDA type
src/common/cux.h:	Holds the pointer to CUDA texture reference, and implements host emulation
src/common/cux.h:	object and a CUDA texture refrence.
src/common/cux.h:template<typename T, int dim, enum cudaTextureReadMode mode>
src/common/cux.h:		textureReference &texref;	// the CUDA texture reference to which the current texture will be bound
src/common/cux.h:	tex1D/2D/3D -- Host implementation of texture sampling, source-compat w. CUDA
src/common/cux.h:	These are here primarely for source-compabitbility with CUDA tex?D calls,
src/common/cux.h:template<typename T, int dim, enum cudaTextureReadMode mode>
src/common/cux.h:template<typename T, int dim, enum cudaTextureReadMode mode>
src/common/cux.h:template<typename T, int dim, enum cudaTextureReadMode mode>
src/common/cux.h:	sample_impl -- GPU implementation of sampling with texture coordinates
src/common/cux.h:	Macros to declare, define and sample from CUDA texture references, on device or the host.
src/common/cux.h:	DEFINE_TEXTURE(name...) -- if compiled with nvcc, defines a CUDA textureReference,
src/common/cux.h:	binding/unbinding of textures to the CUDA texture reference.
src/common/cux.h:		DEFINE_TEXTURE(mytexture, float, 3, cudaReadModeElementType, false, cudaFilterModeLinear, cudaAddressModeClamp);
src/common/cux.h:		DECLARE_TEXTURE(mytexture, float, 3, cudaReadModeElementType);
src/common/cux.h:			... call CUDA kernel ...
src/common/cux.h:#if !__CUDACC__
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:	template<typename T, enum cudaTextureReadMode mode>
src/common/cux.h:#endif // cuda_cux__
src/common/cux_lowlevel.h:// CUDA (or emulation) APIs
src/common/cux_lowlevel.h:#if HAVE_CUDA
src/common/cux_lowlevel.h:	#include <cuda_runtime.h>
src/common/cux_lowlevel.h:	#include "cuda_emulation.h"
src/common/cux_lowlevel.h:// Struct alignment is handled differently between the CUDA compiler and other
src/common/cux_lowlevel.h:#ifdef __CUDACC__
src/common/cux_lowlevel.h:#if __CUDACC__
src/common/cux_lowlevel.h:	// For CPU versions of GPU algorithms
src/common/cux_lowlevel.h:	namespace gpuemu // prevent collision with nvcc's symbols
src/common/cux_lowlevel.h:	using namespace gpuemu;
src/common/column.h:#include "gpu.h"
src/common/column.h:#ifndef __CUDACC__
src/common/column.h:#ifndef __CUDACC__
src/common/column.h:	typedef gptr<T, 2> gpu_t;
src/common/column.h:	// the array is stored such that nrows() is its width. This allows the GPU
src/common/column.h:typedef column<double>::gpu_t	gcdouble_t;
src/common/column.h:typedef column<int>::gpu_t	gcint_t;
src/common/column.h:typedef column<float>::gpu_t	gcfloat_t;
src/common/cuda_emulation.h:// Emulate CUDA types and keywords when no CUDA toolkit has been
src/common/cuda_emulation.h:#ifndef _cuda_emulation_h__
src/common/cuda_emulation.h:#define _cuda_emulation_h__
src/common/cuda_emulation.h:* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
src/common/cuda_emulation.h:* This source code is subject to NVIDIA ownership rights under U.S. and
src/common/cuda_emulation.h:* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
src/common/cuda_emulation.h:* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
src/common/cuda_emulation.h:* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
src/common/cuda_emulation.h:enum cudaError
src/common/cuda_emulation.h:cudaSuccess                           =      0,   ///< No errors
src/common/cuda_emulation.h:cudaErrorMissingConfiguration         =      1,   ///< Missing configuration error
src/common/cuda_emulation.h:cudaErrorMemoryAllocation             =      2,   ///< Memory allocation error
src/common/cuda_emulation.h:cudaErrorInitializationError          =      3,   ///< Initialization error
src/common/cuda_emulation.h:cudaErrorLaunchFailure                =      4,   ///< Launch failure
src/common/cuda_emulation.h:cudaErrorPriorLaunchFailure           =      5,   ///< Prior launch failure
src/common/cuda_emulation.h:cudaErrorLaunchTimeout                =      6,   ///< Launch timeout error
src/common/cuda_emulation.h:cudaErrorLaunchOutOfResources         =      7,   ///< Launch out of resources error
src/common/cuda_emulation.h:cudaErrorInvalidDeviceFunction        =      8,   ///< Invalid device function
src/common/cuda_emulation.h:cudaErrorInvalidConfiguration         =      9,   ///< Invalid configuration
src/common/cuda_emulation.h:cudaErrorInvalidDevice                =     10,   ///< Invalid device
src/common/cuda_emulation.h:cudaErrorInvalidValue                 =     11,   ///< Invalid value
src/common/cuda_emulation.h:cudaErrorInvalidPitchValue            =     12,   ///< Invalid pitch value
src/common/cuda_emulation.h:cudaErrorInvalidSymbol                =     13,   ///< Invalid symbol
src/common/cuda_emulation.h:cudaErrorMapBufferObjectFailed        =     14,   ///< Map buffer object failed
src/common/cuda_emulation.h:cudaErrorUnmapBufferObjectFailed      =     15,   ///< Unmap buffer object failed
src/common/cuda_emulation.h:cudaErrorInvalidHostPointer           =     16,   ///< Invalid host pointer
src/common/cuda_emulation.h:cudaErrorInvalidDevicePointer         =     17,   ///< Invalid device pointer
src/common/cuda_emulation.h:cudaErrorInvalidTexture               =     18,   ///< Invalid texture
src/common/cuda_emulation.h:cudaErrorInvalidTextureBinding        =     19,   ///< Invalid texture binding
src/common/cuda_emulation.h:cudaErrorInvalidChannelDescriptor     =     20,   ///< Invalid channel descriptor
src/common/cuda_emulation.h:cudaErrorInvalidMemcpyDirection       =     21,   ///< Invalid memcpy direction
src/common/cuda_emulation.h:cudaErrorAddressOfConstant            =     22,   ///< Address of constant error
src/common/cuda_emulation.h:cudaErrorTextureFetchFailed           =     23,   ///< Texture fetch failed
src/common/cuda_emulation.h:cudaErrorTextureNotBound              =     24,   ///< Texture not bound error
src/common/cuda_emulation.h:cudaErrorSynchronizationError         =     25,   ///< Synchronization error
src/common/cuda_emulation.h:cudaErrorInvalidFilterSetting         =     26,   ///< Invalid filter setting
src/common/cuda_emulation.h:cudaErrorInvalidNormSetting           =     27,   ///< Invalid norm setting
src/common/cuda_emulation.h:cudaErrorMixedDeviceExecution         =     28,   ///< Mixed device execution
src/common/cuda_emulation.h:cudaErrorCudartUnloading              =     29,   ///< CUDA runtime unloading
src/common/cuda_emulation.h:cudaErrorUnknown                      =     30,   ///< Unknown error condition
src/common/cuda_emulation.h:cudaErrorNotYetImplemented            =     31,   ///< Function not yet implemented
src/common/cuda_emulation.h:cudaErrorMemoryValueTooLarge          =     32,   ///< Memory value too large
src/common/cuda_emulation.h:cudaErrorInvalidResourceHandle        =     33,   ///< Invalid resource handle
src/common/cuda_emulation.h:cudaErrorNotReady                     =     34,   ///< Not ready error
src/common/cuda_emulation.h:cudaErrorInsufficientDriver           =     35,   ///< CUDA runtime is newer than driver
src/common/cuda_emulation.h:cudaErrorSetOnActiveProcess           =     36,   ///< Set on active process error
src/common/cuda_emulation.h:cudaErrorNoDevice                     =     38,   ///< No available CUDA device
src/common/cuda_emulation.h:cudaErrorStartupFailure               =   0x7f,   ///< Startup failure
src/common/cuda_emulation.h:cudaErrorApiFailureBase               =  10000    ///< API failure base
src/common/cuda_emulation.h:enum cudaMemcpyKind
src/common/cuda_emulation.h:cudaMemcpyHostToHost          =   0,      ///< Host   -> Host
src/common/cuda_emulation.h:cudaMemcpyHostToDevice        =   1,      ///< Host   -> Device
src/common/cuda_emulation.h:cudaMemcpyDeviceToHost        =   2,      ///< Device -> Host
src/common/cuda_emulation.h:cudaMemcpyDeviceToDevice      =   3       ///< Device -> Device
src/common/cuda_emulation.h:enum cudaChannelFormatKind
src/common/cuda_emulation.h:cudaChannelFormatKindSigned           =   0,      ///< Signed channel format
src/common/cuda_emulation.h:cudaChannelFormatKindUnsigned         =   1,      ///< Unsigned channel format
src/common/cuda_emulation.h:cudaChannelFormatKindFloat            =   2,      ///< Float channel format
src/common/cuda_emulation.h:cudaChannelFormatKindNone             =   3,      ///< No channel format
src/common/cuda_emulation.h:* CUDA Channel format descriptor
src/common/cuda_emulation.h:struct cudaChannelFormatDesc
src/common/cuda_emulation.h:enum cudaChannelFormatKind f; ///< Channel format kind
src/common/cuda_emulation.h:/*                       END NVIDIA CODE                           */
src/common/cuda_emulation.h:inline const char *cudaGetErrorString(cudaError err) { return "CUDA Error: CUDA Error when no CUDA used (?!)"; }
src/common/cuda_emulation.h:inline cudaError cudaMalloc(void** devPtr, size_t count)
src/common/cuda_emulation.h:	return cudaSuccess;
src/common/cuda_emulation.h:inline cudaError cudaFree(void* devPtr)
src/common/cuda_emulation.h:	return cudaSuccess;
src/common/cuda_emulation.h:inline cudaError cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
src/common/cuda_emulation.h:	return cudaSuccess;
src/common/cuda_emulation.h:* CUDA array
src/common/cuda_emulation.h:struct cudaArray;
src/common/cuda_emulation.h:inline cudaError cudaFreeArray(cudaArray* array ) { assert(0); }
src/common/cuda_emulation.h:inline cudaError cudaMallocArray(cudaArray** array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height ) { assert(0); }
src/common/cuda_emulation.h:inline cudaError cudaMemcpy2DToArray(cudaArray* dstArray, size_t dstX, size_t dstY, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) { assert(0); }
src/common/cuda_emulation.h:#endif // _cuda_emulation_h__
src/common/cuda_rng.h:#ifndef cuda_rng_h__
src/common/cuda_rng.h:#define cuda_rng_h__
src/common/cuda_rng.h:/*  A set of multithreaded GPU integrators.                                 */
src/common/cuda_rng.h:	typedef prngs::gpu::mwc cuda_rng;
src/common/cuda_rng.h:	cuda_rng rng = cuda_rng::create(seed, nstreams); // nstreams must be >= nthreads
src/common/cuda_rng.h:	template<uint32_t state_dim, bool on_gpu>
src/common/cuda_rng.h://		#ifdef __CUDACC__
src/common/cuda_rng.h:		// upload the state vector to the GPU
src/common/cuda_rng.h:			// copy rng stream state to GPU
src/common/cuda_rng.h:			if(on_gpu)
src/common/cuda_rng.h:				cuxErrCheck( cudaMemcpy(gstate, states, sizeof(uint32_t)*nstreams*statewidth, cudaMemcpyHostToDevice) );
src/common/cuda_rng.h:			if(on_gpu)
src/common/cuda_rng.h:				cuxErrCheck( cudaMemcpy(states, gstate, sizeof(uint32_t)*nstreams*statewidth, cudaMemcpyDeviceToHost) );
src/common/cuda_rng.h:		// free the GPU state vector
src/common/cuda_rng.h:			if(on_gpu)
src/common/cuda_rng.h:				cuxErrCheck( cudaFree(gstate) );
src/common/cuda_rng.h://		#ifdef __CUDACC__
src/common/cuda_rng.h:	template<bool on_gpu>
src/common/cuda_rng.h:	struct ran0_impl : public rng_base<1, on_gpu>
src/common/cuda_rng.h://		#ifdef __CUDACC__
src/common/cuda_rng.h:	template<bool on_gpu>
src/common/cuda_rng.h:	struct mwc_impl : public rng_base<3, on_gpu>
src/common/cuda_rng.h://		#ifdef __CUDACC__
src/common/cuda_rng.h:					http://www.ast.cam.ac.uk/~stg20/cuda/random/index.html
src/common/cuda_rng.h:	template<bool on_gpu>
src/common/cuda_rng.h:	struct taus2_impl : public rng_base<3, on_gpu>
src/common/cuda_rng.h:		#ifdef __CUDACC__
src/common/cuda_rng.h:	template<bool on_gpu>
src/common/cuda_rng.h:	struct rand48_impl : public rng_base<3, on_gpu>
src/common/cuda_rng.h:		// 	http://forums.nvidia.com/index.php?act=attach&type=post&id=9512
src/common/cuda_rng.h:		#ifdef __CUDACC__
src/common/cuda_rng.h:	namespace gpu
src/common/cuda_rng.h:#endif // cuda_rng_h__
src/common/cux.cpp:#include "gpu.h"
src/common/cux.cpp:#include <cuda.h>
src/common/cux.cpp:	cleanCudaArray = false;
src/common/cux.cpp:		cuxErrCheck( cudaFree(m_data.ptr) );
src/common/cux.cpp:			cuxErrCheck( cudaFree(slave) );
src/common/cux.cpp:	// if the cudaArray is dirty, or there are no textures bound to it
src/common/cux.cpp:	if(!cleanCudaArray || boundTextures.empty())
src/common/cux.cpp:			cuxErrCheck( cudaFreeArray(cuArray) );
src/common/cux.cpp:		cleanCudaArray = false;
src/common/cux.cpp:#if !CUDA_DEVEMU
src/common/cux.cpp:	cuxErrCheck( (cudaError)cuMemGetInfo(&free, &total) );
src/common/cux.cpp:void cuxMallocErrCheck_impl(cudaError err, size_t msize, const char *fun, const char *file, const int line)
src/common/cux.cpp:	VERIFY(err == cudaSuccess)
src/common/cux.cpp:		MLOG(verb1) << "CUDA ERROR: " << cudaGetErrorString(err);
src/common/cux.cpp:		MLOG(verb1) << "CUDA ERROR: Attempted to allocate " << msize / (1<<20) << "MB, " << cuxGetFreeMem() / (1<<20) << "MB free.";
src/common/cux.cpp:		MLOG(verb1) << "CUDA ERROR: In " << fun << " (" << file << ":" << line << ")\n";
src/common/cux.cpp:		cudaError err = (x); \
src/common/cux.cpp:		if(err == cudaErrorMemoryAllocation) \
src/common/cux.cpp:		if(device)	// syncing to GPU device
src/common/cux.cpp:			GC_AND_RETRY_IF_FAIL( cudaMalloc((void**)&slave, memsize()), memsize() );
src/common/cux.cpp:		cudaMemcpyKind dir = onDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
src/common/cux.cpp:		cuxErrCheck( cudaMemcpy(m_data.ptr, slave, memsize(), dir) );
src/common/cux.cpp:	cleanCudaArray = false;
src/common/cux.cpp:cudaArray *cuxSmartPtr_impl_t::getCUDAArray(cudaChannelFormatDesc &channelDesc)
src/common/cux.cpp:	// FIXME: This all seems to be majorly fu*ked up, as CUDA devemu
src/common/cux.cpp:	// has bugs with cudaMalloc3DArray that has any of the extent dimensions
src/common/cux.cpp:	// set to zero. Will have to be fixed by trial-and-error on the real GPU.
src/common/cux.cpp:	if(!cleanCudaArray)
src/common/cux.cpp:		cudaExtent ex = make_cudaExtent(m_width, m_data.extent[1], m_data.extent[2]);
src/common/cux.cpp:				GC_AND_RETRY_IF_FAIL( cudaMalloc3DArray(&cuArray, &channelDesc, ex), memsize() );
src/common/cux.cpp:				GC_AND_RETRY_IF_FAIL( cudaMallocArray(&cuArray, &channelDesc, ex.width, ex.height), memsize() );
src/common/cux.cpp:			cudaMemcpy3DParms par = { 0 };
src/common/cux.cpp:			par.srcPtr = make_cudaPitchedPtr(m_data.ptr, m_data.extent[0], ex.width, ex.height);
src/common/cux.cpp:			par.kind = cudaMemcpyHostToDevice;
src/common/cux.cpp:			cuxErrCheck( cudaMemcpy3D(&par) );
src/common/cux.cpp:			cuxErrCheck( cudaMemcpy2DToArray(cuArray, 0, 0, m_data.ptr, m_data.extent[0], ex.width*m_elementSize, ex.height, cudaMemcpyHostToDevice) );
src/common/cux.cpp:		cleanCudaArray = true;
src/common/cux.cpp:	cudaArray *cuArray = getCUDAArray(texref.channelDesc);
src/common/cux.cpp:	cuxErrCheck( cudaBindTextureToArray(&texref, cuArray, &texref.channelDesc) );
src/common/cux.cpp:	cuxErrCheck( cudaUnbindTexture(&texref) );
src/common/cux.cpp:gpu_rng_t::persistent_rng gpu_rng_t::gpuRNG;
src/common/cux.cpp:gpu_prng_impl &gpu_rng_t::persistent_rng::get(rng_t &seeder)
src/common/cux.cpp:		// initialize CPU and GPU RNGs
src/common/cux.cpp:	// GPU active
src/common/cux.cpp:	if(gpuGetActiveDevice() >= 0)
src/common/cux.cpp:			gpuRNG.upload(cpuRNG.gstate, cpuRNG.nstreams);
src/common/cux.cpp:			state = GPU;
src/common/cux.cpp:		return gpuRNG;
src/common/cux.cpp:	if(state == GPU)
src/common/cux.cpp:		gpuRNG.download(cpuRNG.gstate);
src/common/cux.cpp:	return (gpu_prng_impl&)cpuRNG;
src/common/cux.cpp:// CUDA emulation for the CPU
src/common/cux.cpp:// Used by CPU versions of CUDA kernels
src/common/cux.cpp:namespace gpuemu	// prevent collision with nvcc's symbols
src/common/cux.cpp:#if HAVE_CUDA || !ALIAS_GPU_RNG
src/common/cux.cpp:#if HAVE_CUDA
src/common/cux.cpp:	static uint32_t *gpu_streams;	// Pointer to streams state on the device (used in GPU mode)
src/common/cux.cpp:	static bool onDevice;		// Whether the master copy is on the device (GPU)
src/common/cux.cpp:		MLOG(verb1) << "ERROR: Must call rng_mwc::init before using GPU random number generator";
src/common/cux.cpp:	static uint32_t *gpuStreams()
src/common/cux.cpp:#if HAVE_CUDA
src/common/cux.cpp:			cudaError err;
src/common/cux.cpp:			if(gpu_streams == NULL)
src/common/cux.cpp:				err = cudaMalloc((void**)&gpu_streams, statebytes());
src/common/cux.cpp:				if(err != cudaSuccess) { MLOG(verb1) << "CUDA Error: " << cudaGetErrorString(err); abort(); }
src/common/cux.cpp:			err = cudaMemcpy(gpu_streams, cpu_streams, statebytes(), cudaMemcpyHostToDevice);
src/common/cux.cpp:			if(err != cudaSuccess) { MLOG(verb1) << "CUDA Error: " << cudaGetErrorString(err); abort(); }
src/common/cux.cpp:		return gpu_streams;
src/common/cux.cpp:		MLOG(verb1) << "ERROR: We should have never gotten here with CUDA support disabled!";
src/common/cux.cpp:#if HAVE_CUDA
src/common/cux.cpp:			cudaError err = cudaThreadSynchronize();
src/common/cux.cpp:			if(err != cudaSuccess) { MLOG(verb1) << "CUDA Error: " << cudaGetErrorString(err); abort(); }
src/common/cux.cpp:			err = cudaMemcpy(cpu_streams, gpu_streams, statebytes(), cudaMemcpyDeviceToHost);
src/common/cux.cpp:			if(err != cudaSuccess) { MLOG(verb1) << "CUDA Error: " << cudaGetErrorString(err); abort(); }
src/common/cux.cpp:#if HAVE_CUDA
src/common/cux.cpp:uint32_t *rng_mwc::gpu_streams = NULL;
src/common/cux.cpp:gpu_rng_t::gpu_rng_t(rng_t &rng)
src/common/cux.cpp:	streams = gpuGetActiveDevice() < 0 ? rng_mwc::cpuStreams() : rng_mwc::gpuStreams();
src/common/cux.cpp:	cudaMallocHost((void **)&base, memsize());
src/common/cux.cpp:		cudaFreeHost(base);
src/common/cux.cpp:// CUDA helpers
src/common/cux.cpp:void abort_on_cuda_error(cudaError err)
src/common/cux.cpp:	VERIFY(err == cudaSuccess)
src/common/cux.cpp:		MLOG(verb1) << "CUDA ERROR: " << cudaGetErrorString(err);
src/common/cux.cpp:void cuxErrCheck_impl(cudaError err, const char *fun, const char *file, const int line)
src/common/cux.cpp:	VERIFY(err == cudaSuccess)
src/common/cux.cpp:		MLOG(verb1) << "CUDA ERROR: " << cudaGetErrorString(err);
src/common/cux.cpp:		MLOG(verb1) << "CUDA ERROR: In " << fun << " (" << file << ":" << line << ")\n";
src/common/cux.cpp:#if HAVE_CUDA
src/common/cux.cpp:static int cuda_initialized = 0;
src/common/cux.cpp:static int cuda_enabled = 0;
src/common/cux.cpp:bool gpuExecutionEnabled(const char *kernel)
src/common/cux.cpp:	return cuda_enabled;
src/common/cux.cpp:	Initialize cux library and CUDA device.
src/common/cux.cpp:	if(cuda_initialized) { return true; }
src/common/cux.cpp:	const char *devStr = getenv("CUDA_DEVICE");
src/common/cux.cpp:		// disable GPU acceleration
src/common/cux.cpp:			cuda_initialized = 1;
src/common/cux.cpp:			cuda_enabled = 0;
src/common/cux.cpp:			MLOG(verb1) << "GPU accelerator: Using CPU: \"" << cpuinfo() << "\"";
src/common/cux.cpp:			cuxErrCheck( cudaSetDevice(dev) );
src/common/cux.cpp:#if !CUDA_DEVEMU
src/common/cux.cpp:	// ensure a CUDA context is created and fetch the active
src/common/cux.cpp:	cuxErrCheck( cudaFree(tmp) );
src/common/cux.cpp:	cuxErrCheck( cudaGetDevice(&dev) );
src/common/cux.cpp:#if !CUDA_DEVEMU
src/common/cux.cpp:	cudaDeviceProp deviceProp;
src/common/cux.cpp:	cuxErrCheck( cudaGetDeviceProperties(&deviceProp, dev) );
src/common/cux.cpp:	MLOG(verb1) << io::format("GPU accelerator: Using Device %d: \"%s\"%s") << dev << deviceProp.name << (autoselect ? " (autoselected)" : "");
src/common/cux.cpp:	MLOG(verb1) << "GPU accelerator: Using Device Emulation";
src/common/cux.cpp:#if !CUDA_DEVEMU
src/common/cux.cpp:	cuxErrCheck( (cudaError)cuMemGetInfo(&free, &total) );
src/common/cux.cpp:	cuda_initialized = 1;
src/common/cux.cpp:	cuda_enabled = 1;
src/common/cux.cpp:#endif // HAVE_CUDA
src/common/cux.cpp:#if HAVE_CUDA
src/common/cux.cpp:#if CUDART_VERSION < 2020
src/common/cux.cpp:cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
src/common/cux.cpp:cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset);
src/common/cux.cpp:cudaError_t cudaLaunch(const char *entry);
src/common/cux.cpp:struct cudaFuncAttributes
src/common/cux.cpp:typedef int cuda_stream_t;
src/common/cux.cpp:cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *attr, const char *func)
src/common/cux.cpp:	return cudaSuccess;
src/common/cux.cpp:#define CUDA_RETURN_ON_FAIL(x) \
src/common/cux.cpp:	{ cudaError_t ret_54e843 = (x); if(ret_54e843 != cudaSuccess) { return ret_54e843; } }
src/common/cux.cpp:		cudaStream_t stream;
src/common/cux.cpp:		cudaError_t err;
src/common/cux.cpp:		kernel(const char *kernel_name_, size_t nthreads_, size_t sharedMem_ = 0, cudaStream_t stream_ = (cudaStream_t)(-1) )
src/common/cux.cpp:			cudaGetDevice(&dev);
src/common/cux.cpp:			cudaDeviceProp dprop;
src/common/cux.cpp:			cudaGetDeviceProperties(&dprop, dev);
src/common/cux.cpp:			cudaFuncAttributes attr;
src/common/cux.cpp:			err = cudaFuncGetAttributes(&attr, name);
src/common/cux.cpp:			if(err != cudaSuccess) { return; }
src/common/cux.cpp:inline cudaError_t bindKernelParam(const T &p, size_t &offs)
src/common/cux.cpp:	cudaError_t ret = cudaSetupArgument(&p, sizeof(T), offs);
src/common/cux.cpp:inline cudaError_t  bindKernelParam(const emptyArg &p, size_t &offs)
src/common/cux.cpp:	return cudaSuccess;
src/common/cux.cpp:cudaError_t callKernel3(const nv::kernel &kc, const T1 &v1, const T2 &v2, const T3 &v3)
src/common/cux.cpp:	CUDA_RETURN_ON_FAIL( cudaConfigureCall(kc.gridDim, kc.blockDim, kc.sharedMem, kc.stream) );
src/common/cux.cpp:	return cudaLaunch(kc.name);
src/common/cux.cpp:void test_cuda_caller()
src/common/gpu.h:#ifndef __gpu_h
src/common/gpu.h:#define __gpu_h
src/common/gpu.h:// #define HAVE_CUDA 1 -- CUDA support is on
src/common/gpu.h:// Note: __CUDACC__ will never be defined when BUILD_FOR_CPU is defined
src/common/gpu.h:// Based on NVIDIA's LinuxStopWatch class
src/common/gpu.h:// Run-time selection of GPU/CPU execution
src/common/gpu.h:#if HAVE_CUDA
src/common/gpu.h:	// Return true if <kernel> should execute on GPU
src/common/gpu.h:	bool gpuExecutionEnabled(const char *kernel);
src/common/gpu.h:	inline int gpuGetActiveDevice() { return active_compute_device; }
src/common/gpu.h:	// Not compiling with CUDA -- always run on CPU
src/common/gpu.h:	inline bool gpuExecutionEnabled(const char *kernel) { return false; }
src/common/gpu.h:	inline int gpuGetActiveDevice() { return -1; }
src/common/gpu.h:// Building with CUDA support. Declarations and runtime stubs for
src/common/gpu.h:// calling the GPU or CPU kernel versions.
src/common/gpu.h:#if HAVE_CUDA
src/common/gpu.h:		void gpulaunch_##kDecl;
src/common/gpu.h:			activeDevice dev(gpuExecutionEnabled(#kName)? 0 : -1); \
src/common/gpu.h:			if(gpuGetActiveDevice() < 0) \
src/common/gpu.h:				gpulaunch_##kName(__VA_ARGS__); \
src/common/gpu.h:// No CUDA at all. Only declare, build and call host-based kernels
src/common/gpu.h:#if !HAVE_CUDA
src/common/gpu.h:// Have CUDA and building GPU kernel.
src/common/gpu.h:#if HAVE_CUDA && !BUILD_FOR_CPU
src/common/gpu.h:		__global__ void gpu_##kDecl; \
src/common/gpu.h:		void gpulaunch_##kDecl \
src/common/gpu.h:			gpu_##kName<<<gridDim, threadsPerBlock, threadsPerBlock*dynShmemPerThread>>>kArgs; \
src/common/gpu.h:			cuxErrCheck( cudaThreadSynchronize() );\
src/common/gpu.h:		__global__ void gpu_##kDecl
src/common/gpu.h:// No CUDA, or building the CPU version of the kernel.
src/common/gpu.h:#if !HAVE_CUDA || BUILD_FOR_CPU
src/common/gpu.h:// GPU random number generators
src/common/gpu.h:	// interface compatibility with gpu_rng_t
src/common/gpu.h:#define ALIAS_GPU_RNG 0
src/common/gpu.h:// GPU random number generator abstraction
src/common/gpu.h:#if !HAVE_CUDA && ALIAS_GPU_RNG
src/common/gpu.h:typedef rng_t &gpu_rng_t;
src/common/gpu.h:#if !__CUDACC__
src/common/gpu.h:#if HAVE_CUDA || !ALIAS_GPU_RNG
src/common/gpu.h:	#include "cuda_rng.h"
src/common/gpu.h:	typedef prngs::gpu::mwc gpu_prng_impl;
src/common/gpu.h:	// to GPU as needed. It is auto-initialized on first call, using the
src/common/gpu.h:	//	gpu_rng_t bla(seeder)
src/common/gpu.h:	struct gpu_rng_t : public gpu_prng_impl
src/common/gpu.h:			gpu_prng_impl gpuRNG;
src/common/gpu.h:			enum { EMPTY, GPU, CPU } state;
src/common/gpu.h:				gpuRNG.free();
src/common/gpu.h:			gpu_prng_impl &get(rng_t &seeder);
src/common/gpu.h:		static persistent_rng gpuRNG;
src/common/gpu.h:		gpu_rng_t(rng_t &seeder)
src/common/gpu.h:			(gpu_prng_impl&)*this = gpuRNG.get(seeder);
src/common/gpu.h:		gpu_rng_t() {}
src/pipeline.cpp:// convert a list of (closed!) ranges to a bitmap, and upload it to the GPU
src/pipeline.cpp:	MLOG(verb2) << "GPU kernels runtime: " << kernelRunSwatch.getTime();
src/galfast.cpp:#include "gpu.h"
src/galfast.cpp:#include "gpulog/gpulog.h"
src/galfast.cpp:		"  cudaquery - \tquery available cuda devices\n"
src/galfast.cpp:	uopts["cudaquery"].reset(new Options(argv0 + " util cudaquery", progdesc + " Query available CUDA devices.", version, Authorship::majuric));
src/galfast.cpp:		MLOG(verb1) << "Error initializing GPU acceleration. Aborting.";
src/galfast.cpp:	if(cmd == "util cudaquery")
src/galfast.cpp:		// The real test was that we successfully passed the cuda_init() step above.
src/kernels_cpu.cpp:#include "simulate_gpu.cu.h"
src/tests.cpp://		"  cudaquery - \tquery available cuda devices\n"
src/kernels_gpu.cu:#include "simulate_gpu.cu.h"
src/gpulog/gpulog.h:#ifndef gpulog_h__
src/gpulog/gpulog.h:#define gpulog_h__
src/gpulog/gpulog.h:#include "bits/gpulog_debug.h"
src/gpulog/gpulog.h:#include "bits/gpulog_align.h"
src/gpulog/gpulog.h:#include "bits/gpulog_types.h"
src/gpulog/gpulog.h:#include "bits/gpulog_ttraits.h"
src/gpulog/gpulog.h:#include "bits/gpulog_constants.h"
src/gpulog/gpulog.h:#include "bits/gpulog_msg_layout.h"
src/gpulog/gpulog.h:#include "bits/gpulog_log.h"
src/gpulog/gpulog.h:#include "bits/gpulog_logrecord.h"
src/gpulog/gpulog.h:#include "bits/gpulog_ilogstream.h"
src/gpulog/gpulog.h:#include "bits/gpulog_macro_cleanup.h"
src/gpulog/gpulog.h:namespace gpulog
src/gpulog/gpulog.h:	// into gpulog namespace
src/gpulog/gpulog.h:#endif // gpulog_h__
src/gpulog/lprintf.h:#include "gpulog.h"
src/gpulog/lprintf.h:	/* CUDA 2.3 compatible float->double converter */
src/gpulog/lprintf.h:	/* CUDA 2.2 compatible float->double converter hack */
src/gpulog/lprintf.h:		#include "bits/gpulog_printf.h"
src/gpulog/lprintf.h:	#if !__CUDACC__
src/gpulog/lprintf.h:	inline std::string run_printf(gpulog::logrecord &lr)
src/gpulog/lprintf.h:        	using namespace gpulog;
src/gpulog/lprintf.h:	inline int replay_printf(std::ostream &out, gpulog::ilogstream &ls)
src/gpulog/lprintf.h:		using namespace gpulog;
src/gpulog/lprintf.h:	inline int replay_printf(std::ostream &out, const gpulog::host_log &log)
src/gpulog/lprintf.h:		gpulog::ilogstream ls(log);
src/gpulog/bits/gpulog_constants.h:#ifndef bits_gpulog_constants_h__
src/gpulog/bits/gpulog_constants.h:#define bits_gpulog_constants_h__
src/gpulog/bits/gpulog_constants.h:namespace gpulog
src/gpulog/bits/gpulog_constants.h:	// flags for gpulog::copy() and related functions
src/gpulog/bits/gpulog_constants.h:#endif // bits_gpulog_constants_h__
src/gpulog/bits/gpulog_types.h:#ifndef bits_gpulog_types_h__
src/gpulog/bits/gpulog_types.h:#define bits_gpulog_types_h__
src/gpulog/bits/gpulog_types.h:#include <cuda_runtime.h>
src/gpulog/bits/gpulog_types.h:namespace gpulog
src/gpulog/bits/gpulog_types.h:#endif // bits_gpulog_types_h__
src/gpulog/bits/gpulog_ttraits.h:#ifndef bits_gpulog_ttraits_h__
src/gpulog/bits/gpulog_ttraits.h:#define bits_gpulog_ttraits_h__
src/gpulog/bits/gpulog_ttraits.h:#if GPULOG_DEBUG && !__CUDACC__
src/gpulog/bits/gpulog_ttraits.h:namespace gpulog
src/gpulog/bits/gpulog_ttraits.h:	/* See table B-1 in CUDA 2.2 Programming Guide for reference */
src/gpulog/bits/gpulog_ttraits.h:	#define SCALAR(T)	typename gpulog::internal::ttrait<T>::scalarT	/* The scalar of type T (extracts T out of array<T>, if it's an array) */
src/gpulog/bits/gpulog_ttraits.h:	#if GPULOG_DEBUG && !__CUDACC__
src/gpulog/bits/gpulog_ttraits.h:} // namespace gpulog
src/gpulog/bits/gpulog_ttraits.h:#endif // bits_gpulog_ttraits_h__
src/gpulog/bits/gpulog_write.h:// Generated using ../scripts/gen_gpulog_write.pl write
src/gpulog/bits/gpulog_msg_layout.h:#ifndef bits_gpulog_msg_layout_h__
src/gpulog/bits/gpulog_msg_layout.h:#define bits_gpulog_msg_layout_h__
src/gpulog/bits/gpulog_msg_layout.h:namespace gpulog
src/gpulog/bits/gpulog_msg_layout.h:	#if !__CUDACC__
src/gpulog/bits/gpulog_msg_layout.h:			DGPU( printf("Writing start=%d len=%d\n", start, datalen); );
src/gpulog/bits/gpulog_msg_layout.h:			DGPU( printf("Writing presized array start=%d len=%d\n", start, datalen); );
src/gpulog/bits/gpulog_msg_layout.h:			DGPU( printf("Allocating array start=%d element_size=%d\n", start, datalen); );
src/gpulog/bits/gpulog_msg_layout.h:	// CUDA 2.2/2.3 compilation speedup hack -- otherwise (if ASTART is called directly), nvcc
src/gpulog/bits/gpulog_msg_layout.h:#endif // bits_gpulog_msg_layout_h__
src/gpulog/bits/gpulog_macro_cleanup.h:#ifndef bits_gpulog_macro_cleanup_h__
src/gpulog/bits/gpulog_macro_cleanup.h:#define bits_gpulog_macro_cleanup_h__
src/gpulog/bits/gpulog_macro_cleanup.h:#endif // bits_gpulog_macro_cleanup_h__
src/gpulog/bits/gpulog_logrecord.h:#ifndef bits_gpulog_logrecord_h__
src/gpulog/bits/gpulog_logrecord.h:#define bits_gpulog_logrecord_h__
src/gpulog/bits/gpulog_logrecord.h:namespace gpulog
src/gpulog/bits/gpulog_logrecord.h:	/* workaround for CUDA 2.2 template parsing bug */
src/gpulog/bits/gpulog_logrecord.h:		#if ARGINFO && !__CUDACC__
src/gpulog/bits/gpulog_logrecord.h:				std::cerr << "Assertion failed GPUTypeTraits::" #a " != HostTypeTraits::" #b << " (" << a << " != " << TT::b << ")\n"; \
src/gpulog/bits/gpulog_logrecord.h:#endif // bits_gpulog_logrecord_h__
src/gpulog/bits/gpulog_align.h:#ifndef bits_gpulog_align_h__
src/gpulog/bits/gpulog_align.h:#define bits_gpulog_align_h__
src/gpulog/bits/gpulog_align.h:// Struct alignment is handled differently between the CUDA compiler and other
src/gpulog/bits/gpulog_align.h:	#ifdef __CUDACC__
src/gpulog/bits/gpulog_align.h:#endif // bits_gpulog_align_h__
src/gpulog/bits/gpulog_log.h:#ifndef bits_gpulog_log_h__
src/gpulog/bits/gpulog_log.h:#define bits_gpulog_log_h__
src/gpulog/bits/gpulog_log.h:namespace gpulog
src/gpulog/bits/gpulog_log.h:				cudaMalloc((void **)&ret, num*sizeof(T));
src/gpulog/bits/gpulog_log.h:				cudaMemcpy(&ret, ptr, sizeof(ret), cudaMemcpyDeviceToHost);
src/gpulog/bits/gpulog_log.h:				cudaMemcpy(ptr, &val, sizeof(*ptr), cudaMemcpyHostToDevice);
src/gpulog/bits/gpulog_log.h:				cudaFree(p);
src/gpulog/bits/gpulog_log.h:		#ifdef __CUDACC__
src/gpulog/bits/gpulog_log.h:		#ifdef __CUDACC__
src/gpulog/bits/gpulog_log.h:	   workaround for CUDA 2.2 template parsing bug -- CUDA 2.2 tries to compile a template
src/gpulog/bits/gpulog_log.h:	/* CUDA 2.2 compatible version */
src/gpulog/bits/gpulog_log.h:	#define PTR_T(T)  gpulog::internal::ptr_t<T>
src/gpulog/bits/gpulog_log.h:	/* CUDA 2.3 and beyond */
src/gpulog/bits/gpulog_log.h:		#include "gpulog_write.h"
src/gpulog/bits/gpulog_log.h:		cudaMemcpyFromSymbol(&log, name, sizeof(log), 0, cudaMemcpyDeviceToHost);
src/gpulog/bits/gpulog_log.h:		cudaMemcpyToSymbol(name, &log, sizeof(log), 0, cudaMemcpyHostToDevice);
src/gpulog/bits/gpulog_log.h:		cudaMemcpy(to.internal_buffer(), from.internal_buffer(), size, cudaMemcpyDeviceToHost);
src/gpulog/bits/gpulog_log.h:} // namespace gpulog
src/gpulog/bits/gpulog_log.h:#endif //  bits_gpulog_log_h__
src/gpulog/bits/gpulog_debug.h:#ifndef bits_gpulog_debug_h__
src/gpulog/bits/gpulog_debug.h:#define bits_gpulog_debug_h__
src/gpulog/bits/gpulog_debug.h:// Debugging macros. Define GPULOG_DEBUG=1 to turn on
src/gpulog/bits/gpulog_debug.h:#if GPULOG_DEBUG
src/gpulog/bits/gpulog_debug.h:#if !__CUDACC__
src/gpulog/bits/gpulog_debug.h:	#define DGPU(x)
src/gpulog/bits/gpulog_debug.h:		#define DGPU(x) DBG(x)
src/gpulog/bits/gpulog_debug.h:		// #define dev_assert(x) assert(x) /* has to be disabled on CUDA 2.3 or compilation fails */
src/gpulog/bits/gpulog_debug.h:		#define dev_assert(x) { if(!(x)) { printf("Assertion failed: " #x "\n"); exit(-1); } } /* This is a CUDA 2.3 compatible replacement for assert(); */
src/gpulog/bits/gpulog_debug.h:		#define DGPU(x)
src/gpulog/bits/gpulog_debug.h:#endif // bits_gpulog_debug_h
src/gpulog/bits/gpulog_printf.h:// Generated using ./gen_gpulog_write.pl printf
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt);
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2), f2d(T3, v3));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2), f2d(T3, v3), f2d(T4, v4));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2), f2d(T3, v3), f2d(T4, v4), f2d(T5, v5));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2), f2d(T3, v3), f2d(T4, v4), f2d(T5, v5), f2d(T6, v6));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2), f2d(T3, v3), f2d(T4, v4), f2d(T5, v5), f2d(T6, v6), f2d(T7, v7));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2), f2d(T3, v3), f2d(T4, v4), f2d(T5, v5), f2d(T6, v6), f2d(T7, v7), f2d(T8, v8));
src/gpulog/bits/gpulog_printf.h:		log.write(gpulog::MSG_PRINTF, fmt, f2d(T1, v1), f2d(T2, v2), f2d(T3, v3), f2d(T4, v4), f2d(T5, v5), f2d(T6, v6), f2d(T7, v7), f2d(T8, v8), f2d(T9, v9));
src/gpulog/bits/gpulog_ilogstream.h:#ifndef bits_gpulog_ilogstream_h__
src/gpulog/bits/gpulog_ilogstream.h:#define bits_gpulog_ilogstream_h__
src/gpulog/bits/gpulog_ilogstream.h:namespace gpulog

```
