# https://github.com/hypre-space/hypredrive

```console
configure.ac:dnl Check for CUDA support
configure.ac:AC_ARG_WITH([cuda],
configure.ac:            [AS_HELP_STRING([--with-cuda], [Enable CUDA support])],
configure.ac:            [with_cuda=yes],
configure.ac:            [with_cuda=no])
configure.ac:dnl Allow specifying CUDA home directory
configure.ac:AC_ARG_WITH([cuda-home],
configure.ac:            [AS_HELP_STRING([--with-cuda-home=DIR], [Specify CUDA home directory (default is taken from CUDA_HOME environment variable)])],
configure.ac:            [cuda_home="$withval"],
configure.ac:            [cuda_home="$CUDA_HOME"])
configure.ac:dnl Verify CUDA support
configure.ac:AS_IF([test "x$with_cuda" = "xyes"], [
configure.ac:    dnl Check for CUDA_HOME
configure.ac:    AS_IF([test -z "$cuda_home"], [
configure.ac:        AC_MSG_ERROR([CUDA home directory is not defined. Please define the CUDA_HOME environment variable or use --with-cuda-home=DIR.])
configure.ac:    LDFLAGS+=" -L$cuda_home/lib64 -Wl,-rpath,$cuda_home/lib64"
configure.ac:    LIBS+=" -lcudart -lcusparse -lcublas -lcurand -lcusolver -lstdc++"
configure.ac:    dnl Check for libcudart
configure.ac:    AC_CHECK_LIB([cudart],
configure.ac:                 [cudaMalloc],
configure.ac:                 [AC_MSG_NOTICE([libcudart found.])],
configure.ac:                 [AC_MSG_ERROR([libcudart not found. Please ensure CUDA is installed correctly.])])
configure.ac:    dnl Check for libcudart
configure.ac:                 [AC_MSG_ERROR([libcusparse not found. Please ensure CUDA is installed correctly.])])
configure.ac:                 [AC_MSG_ERROR([libcublas not found. Please ensure CUDA is installed correctly.])])
configure.ac:                 [AC_MSG_ERROR([libcusolver not found. Please ensure CUDA is installed correctly.])])
configure.ac:                 [AC_MSG_ERROR([libcurand not found. Please ensure CUDA is installed correctly.])])
configure.ac:dnl Allow specifying ROCM home directory
configure.ac:AC_ARG_WITH([rocm-path],
configure.ac:            [AS_HELP_STRING([--with-rocm-path=DIR], [Specify ROCm installation directory])],
configure.ac:            [rocm_path="$withval"],
configure.ac:            [rocm_path="$ROCM_PATH"])
configure.ac:dnl Verify ROCM_PATH is defined if HIP support is enabled
configure.ac:    dnl Check for ROCM_PATH
configure.ac:    AS_IF([test -z "$rocm_path"], [
configure.ac:        AC_MSG_ERROR([ROCM home directory is not defined. Please define the ROCM_PATH environment variable or use --with-rocm-path=DIR.])
configure.ac:    LDFLAGS+=" -L$rocm_path/lib -Wl,-rpath,$rocm_path/lib"
configure.ac:                 [AC_MSG_ERROR([libamdhip64 not found. Please ensure ROCm is installed correctly.])])
configure.ac:                 [AC_MSG_ERROR([librocsparse not found. Please ensure ROCm is installed correctly.])])
configure.ac:                 [AC_MSG_ERROR([librocblas not found. Please ensure ROCm is installed correctly.])])
configure.ac:                 [AC_MSG_ERROR([librocrand not found. Please ensure ROCm is installed correctly.])])
configure.ac:                 [AC_MSG_ERROR([librocsolver not found. Please ensure ROCm is installed correctly.])])
configure.ac:dnl Avoid using CUDA and HIP
configure.ac:AS_IF([test "x$with_cuda" = "xyes" -a "x$with_hip" = "xyes"], [
configure.ac:    AC_MSG_ERROR([--with-cuda and --with-hip options are mutually exclusive. Please choose one.])
docs/usrman-src/faq.rst:Can I use `hypredrive` on GPU-accelerated systems?
docs/usrman-src/faq.rst:Yes, `hypredrive` supports GPU acceleration. Note that `hypre` also needs to be compiled
docs/usrman-src/faq.rst:with GPU support and the keyword ``exec_policy`` under ``general`` must be set to
docs/usrman-src/input_file_structure.rst:  (CPU) or ``device`` (GPU). When hypre is built without GPU support, the default value
docs/usrman-src/input_file_structure.rst:    integer. Default value is `0` for CPU runs or `1` for GPU runs.
docs/usrman-src/input_file_structure.rst:    integer. Default value is `0` for CPU runs or `1` for GPU runs.
docs/usrman-src/input_file_structure.rst:          type: hmis # pmis for GPU runs
docs/usrman-src/input_file_structure.rst:          mod_rap2: off # on for GPU runs
docs/usrman-src/input_file_structure.rst:          keep_transpose: off # on for GPU runs
docs/usrman-src/input_file_structure.rst:  value is `1` (Adaptive) for CPUs and `3` (Static) for GPUs.
docs/usrman-src/installation.rst:   For GPU support, add `--with-cuda` in the case of NVIDIA GPUs or `--with-hip` in the
docs/usrman-src/installation.rst:   case of AMD GPUs to the `./configure` line.
README.md:3. For GPU support, add `--with-cuda` (NVIDIA GPUs) or `--with-hip` (AMD GPUs) to
src/linsys.c:#if defined (HYPRE_USING_GPU)
src/info.c:      char  gpuInfo[256] = "Unknown";
src/info.c:      if (strlen(gpuInfo) == 0)
src/info.c:         strncpy(gpuInfo, "Unknown", 8 * sizeof(char));
src/info.c:               strncpy(gpuInfo, start + strlen(controller_type), sizeof(gpuInfo) - 1);
src/info.c:               gpuInfo[sizeof(gpuInfo) - 1] = '\0';
src/info.c:               size_t len = strlen(gpuInfo);
src/info.c:               if (len > 0 && gpuInfo[len - 1] == '\n')
src/info.c:                  gpuInfo[len - 1] = '\0';
src/info.c:               printf("GPU Model #%d          : %s\n", gcount++, gpuInfo);
src/info.c:               strncpy(gpuInfo, buffer, sizeof(gpuInfo) - 1);
src/info.c:               gpuInfo[sizeof(gpuInfo) - 1] = '\0';
src/info.c:      else if (system("command -v nvidia-smi > /dev/null 2>&1") == 0)
src/info.c:         fp = popen("nvidia-smi --query-gpu=name --format=csv,noheader", "r");
src/info.c:               printf("GPU Model #%d          : %s", gcount++, buffer);
src/info.c:      /* NVIDIA GPU Memory Information */
src/info.c:      if (system("command -v nvidia-smi > /dev/null 2>&1") == 0)
src/info.c:         fp = popen("nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits", "r");
src/info.c:               printf("GPU RAM #%d            : %6.2f / %6.2f  (%5.2f %%) GB\n",
src/info.c:      /* AMD GPU Memory Information */
src/info.c:      if (system("command -v rocm-smi > /dev/null 2>&1") == 0)
src/info.c:         fp = popen("rocm-smi --showmeminfo vram --json", "r");
src/info.c:            printf("GPU RAM #%d            : %6.2f / %6.2f  (%5.2f %%) GB\n",
src/amg.c:#if defined (HYPRE_USING_GPU)
src/amg.c:#if defined (HYPRE_USING_GPU)

```
