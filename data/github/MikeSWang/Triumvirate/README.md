# https://github.com/MikeSWang/Triumvirate

```console
setup.py:CUDA_XCOMPILER_OPTS = (
setup.py:CUDA_XCOMPILER_OPTS_EXACT = [
setup.py:CUDA_XCOMPILER_OPTS_PARTIAL = [
setup.py:        if detect_cuda():
setup.py:            # Modify compiler options for CUDA for object compilation/linking.
setup.py:                    if opt.startswith(CUDA_XCOMPILER_OPTS):  # noqa: E231
setup.py:        if detect_cuda():
setup.py:            # Modify compiler options for CUDA for object linking.
setup.py:                if opt in CUDA_XCOMPILER_OPTS_EXACT:  # noqa: E231
setup.py:                elif any(map(opt.__contains__, CUDA_XCOMPILER_OPTS_PARTIAL)):
setup.py:        if detect_cuda():
setup.py:            # Modify compiler options for CUDA for object compilation.
setup.py:                if opt.startswith(CUDA_XCOMPILER_OPTS):
setup.py:        'PY_CUDA',  # enable CUDA
setup.py:# List CUDA compiler (NVCC assumed).
setup.py:CUDA_COMPILER = 'nvcc'
setup.py:def get_compiler(cuda=False):
setup.py:    cuda : bool, optional
setup.py:        If `True`, return the CUDA compiler as defined by
setup.py:        :py:const:`CUDA_COMPILER` (default is `False`).
setup.py:    _compiler = CUDA_COMPILER if cuda else COMPILERS[get_platform()]
setup.py:def set_cli_compiler(compiler=None, cuda=False):
setup.py:    cuda : bool, optional
setup.py:        If `True`, set the CUDA compiler (default is `False`).
setup.py:    _compiler = compiler or get_compiler(cuda=cuda)
setup.py:def get_pkg_libname(cuda=False):
setup.py:    cuda : bool, optional
setup.py:        If `True`, '_cuda' is appended to the package library name
setup.py:    return f"{PKG_LIB_NAME}{'_cuda' if cuda else ''}"
setup.py:def get_noncuda_dep_libs(core=False, cuda=False):
setup.py:    """Get non-CUDA dependency libraries.
setup.py:    cuda : bool, optional
setup.py:        If `True` (default is `False`), use CUDA libraries.
setup.py:    LIBS_NONCUDA = {
setup.py:        'core,cuda': ['gsl',],
setup.py:        'full,cuda': ['gsl', 'gslcblas', 'm',],
setup.py:    key = ('core' if core else 'full') + (',cuda' if cuda else '')
setup.py:    return LIBS_NONCUDA[key]
setup.py:def get_cuda_dep_libs():
setup.py:    """Get CUDA dependency libraries.
setup.py:        CUDA libraries.
setup.py:    LIBS_CUDA = ['cufft', 'cufftw',]  # noqa: E231
setup.py:    return LIBS_CUDA
setup.py:def parse_cli_cflags_omp(cuda=False):
setup.py:    cuda : bool, optional
setup.py:        for CUDA compilation where necessary.
setup.py:            if not cuda:
setup.py:def parse_cli_ldflags_omp(cuda=False):
setup.py:    cuda : bool, optional
setup.py:        for CUDA compilation where necessary.
setup.py:            if not cuda:
setup.py:                       cuda=False):
setup.py:    cuda : bool, optional
setup.py:        for CUDA.
setup.py:    libs_core = deepcopy(get_noncuda_dep_libs(core=True, cuda=cuda))
setup.py:    libs_full = deepcopy(get_noncuda_dep_libs(cuda=cuda))
setup.py:    if cuda:
setup.py:        libs += get_cuda_dep_libs()
setup.py:                    cuda=False):
setup.py:    cuda : bool, optional
setup.py:        for CUDA.
setup.py:    if not cuda:
setup.py:        macros_omp, cflags_omp = parse_cli_cflags_omp(cuda=cuda)
setup.py:        cflags_omp = ['-fopenmp',] if not cuda \
setup.py:    if not cuda:
setup.py:        ldflags_omp, libs_omp, lib_dirs_omp = parse_cli_ldflags_omp(cuda=cuda)
setup.py:        ldflags_omp = ['-fopenmp',] if not cuda  \
setup.py:        libs_omp = [OPENMP_LIBS[get_platform()],] if not cuda \
setup.py:                     cuda=False):
setup.py:    cuda : bool, optional
setup.py:        for CUDA.
setup.py:    if cuda:
setup.py:        MACROS_PKG.append(('TRV_USE_CUDA', None))
setup.py:        if not cuda:
setup.py:    libs = [get_pkg_libname(cuda=cuda),] + libs  # noqa: E231
setup.py:def detect_cuda():
setup.py:    """Detect CUDA support.
setup.py:        `True` if CUDA is detected, `False` otherwise.
setup.py:    usecuda = os.getenv('PY_CUDA')
setup.py:        return (usecuda.lower() in ['true', '1', ''])
setup.py:def define_pkg_library(cuda=False):
setup.py:    cuda : bool, optional
setup.py:        If `True`, append '_cuda' to the package library name
setup.py:    pkg_libname = get_pkg_libname(cuda=cuda)
setup.py:    usecuda = detect_cuda()
setup.py:    set_cli_compiler(cuda=usecuda)
setup.py:        macros, cflags, ldflags, libs, lib_dirs, include_dirs, cuda=usecuda
setup.py:        macros, cflags, ldflags, libs, lib_dirs, include_dirs, cuda=usecuda
setup.py:        macros, cflags, ldflags, libs, lib_dirs, include_dirs, cuda=usecuda
setup.py:    pkg_libraries = [define_pkg_library(cuda=usecuda),]  # noqa: E231
setup.py:        define_pkg_extension(ext, cuda=usecuda, **cfg)
publication/joss/paper.md:to graphic processing units (GPUs) can bring further parallelisation that
docs/source/installation.rst:CUDA support
docs/source/installation.rst:.. .. image:: https://img.shields.io/pypi/v/Triumvirate-CUDA?logo=PyPI&color=informational
docs/source/installation.rst:..     :target: https://pypi.org/project/Triumvirate-CUDA
docs/source/installation.rst:.. .. image:: https://img.shields.io/conda/v/msw/triumvirate-cuda?logo=Anaconda&color=informational
docs/source/installation.rst:..     :target: https://anaconda.org/msw/triumvirate-cuda
docs/source/installation.rst:CUDA-capable GPU using equivalent libraries. This requires a CUDA-capable
docs/source/installation.rst:GPU and the appropriate driver.
docs/source/installation.rst:The CUDA-enabled Python package is distributed through |PyPICUDARepo|_ and
docs/source/installation.rst:|CondaCUDARepo|_ as ``Triumvirate-CUDA`` and ``triumvirate-cuda``, and OpenMP
docs/source/installation.rst:To install from |PyPICUDARepo|_, execute in shell:
docs/source/installation.rst:    python -m pip install triumvirate-cuda
docs/source/installation.rst:To install using |CondaCUDARepo|_, execute in shell:
docs/source/installation.rst:    conda install -c msw triumvirate-cuda
docs/source/installation.rst:should be created for installing and using the CUDA variant package
docs/source/installation.rst:(e.g. a Conda environment created with ``conda create -n <cuda-env>`` and
docs/source/installation.rst:activated with ``conda activate <cuda-env>``).
docs/source/installation.rst:is optional (see '`OpenMP support`_') and to enable CUDA support, pass
docs/source/installation.rst:``usecuda=true`` or ``usecuda=1`` to `make`.
docs/source/installation.rst:The compiler defaults to ``nvcc`` mandatorily. If the CUDA Toolkit
docs/source/installation.rst:    # If ``CUDA_HOME`` is not set in the system's environment.
docs/source/installation.rst:    # The variable ``CUDA_PATH`` is a similar alternative.
docs/source/installation.rst:    export CUDA_HOME=/usr/local/cuda
docs/source/installation.rst:    export CXX=${CUDA_HOME}/bin/nvcc
docs/source/installation.rst:    # Set the path to the CUDA Toolkit libraries.
docs/source/installation.rst:    export INCLUDES="-I${CUDA_HOME}/include"
docs/source/installation.rst:    export LDFLAGS="-L${CUDA_HOME}/lib[64]"
docs/source/installation.rst:.. |PyPICUDARepo| replace:: PyPI
docs/source/installation.rst:.. _PyPICUDARepo: https://pypi.org/project/Triumvirate-CUDA
docs/source/installation.rst:.. |CondaCUDARepo| replace:: Conda
docs/source/installation.rst:.. _CondaCUDARepo: https://anaconda.org/msw/triumvirate-cuda
Makefile:# CUDA: enabled with ``useomp=(true|1)``; disabled otherwise
Makefile:ifdef usecuda
Makefile:ifeq ($(strip ${usecuda}), $(filter $(strip ${usecuda}), true 1))
Makefile:usecuda := true
Makefile:else   # usecuda != (true|1)
Makefile:unexport usecuda
Makefile:endif  # usecuda == (true|1)
Makefile:endif  # usecuda
Makefile:## If using CUDA/HIP, use CUDA/HIP compiler.
Makefile:	ifdef usecuda
Makefile:	else   # !usehip && !usecuda
Makefile:	endif  # !usehip && usecuda
Makefile:	ifdef usecuda
Makefile:	$(error "CUDA is not supported on macOS.")
Makefile:	endif  # usecuda
Makefile:## If using CUDA/HIP, use CUDA/HIP compiler.
Makefile:	ifdef usecuda
Makefile:	else   # !usehip && !usecuda
Makefile:	endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:CPPFLAGS += -D__HIP_PLATFORM_NVIDIA__
Makefile:else   # usehip && !usecuda
Makefile:endif  # usehip && usecuda
Makefile:ifdef usecuda
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:## NVIDIA HPC SDK
Makefile:	ifdef usecuda
Makefile:	endif  # usecuda
Makefile:	# 	ifdef usecuda
Makefile:	# 	else   # !usehip && !usecuda
Makefile:	# 	endif  # !usehip && usecuda
Makefile:	# ifndef usecuda
Makefile:	# endif  # !usecuda
Makefile:	ifdef usecuda
Makefile:	INCLUDES += -I${NVIDIA_PATH}/math_libs/include
Makefile:	LDFLAGS += -Xlinker -rpath,${NVIDIA_PATH}/math_libs/lib64 -L${NVIDIA_PATH}/math_libs/lib64
Makefile:	endif  # !usehip && usecuda
Makefile:	ifdef usecuda
Makefile:	else   # !usehip && !usecuda
Makefile:	endif  # !usehip && usecuda
Makefile:## NVIDIA HPC SDK
Makefile:	ifdef usecuda
Makefile:	endif  # usecuda
Makefile:## ROCm/HIP
Makefile:	# ROCm is managed by Conda.
Makefile:	CXXFLAGS += --rocm-device-lib-path=${CONDA_PREFIX}/lib/amdgcn/bitcode
Makefile:	ifdef usecuda
Makefile:	else   # !usehip && !usecuda
Makefile:	endif  # !usehip && usecuda
Makefile:### If using CUDA, add preprocessing flags.
Makefile:			ifdef usecuda
Makefile:			else   # !usecuda
Makefile:			endif  # usecuda
Makefile:			ifdef usecuda
Makefile:			else   # !usecuda
Makefile:			endif  # usecuda
Makefile:### If using CUDA, add preprocessing flags.
Makefile:		ifdef usecuda  # !usehip && usecuda
Makefile:		else   # !usehip && !usecuda
Makefile:		endif  # !usehip && usecuda
Makefile:	ifdef usecuda
Makefile:	else   # !usehip && !usecuda
Makefile:	endif  # !usehip && usecuda
Makefile:## If using CUDA FFT, do not include OpenMP FFTW dependency.
Makefile:	ifdef usecuda
Makefile:	else   # !usehip && !usecuda
Makefile:	endif  # !usehip && usecuda
Makefile:# CUDA/HIP
Makefile:ifdef usecuda
Makefile:CPPFLAGS += -DTRV_USE_CUDA
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:endif  # usecuda
Makefile:ifdef usecuda
Makefile:export PY_CUDA=true
Makefile:endif  # usecuda
Makefile:ifdef usecuda
Makefile:PKGSUFFIX := -HIPCUDA
Makefile:PROGSUFFIX := _hipcuda
Makefile:LIBSUFFIX := _hipcuda
Makefile:endif  # usecuda
Makefile:ifdef usecuda
Makefile:PKGSUFFIX := -CUDA
Makefile:PROGSUFFIX := _cuda
Makefile:LIBSUFFIX := _cuda
Makefile:endif  # usecuda
Makefile:ifdef usecuda
Makefile:OBJS := $(SRCS:${DIR_PKG_SRC}/%.cpp=${DIR_BUILDOBJ}/%_hipcuda.o)
Makefile:else   # usehip && !usecuda
Makefile:endif  # usehip && usecuda
Makefile:ifdef usecuda
Makefile:OBJS := $(SRCS:${DIR_PKG_SRC}/%.cpp=${DIR_BUILDOBJ}/%_cuda.o)
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:ifdef usecuda
Makefile:	@cp deploy/pkg/pyproject/.pyproject_hipcuda.toml pyproject.toml
Makefile:else   # usehip && !usecuda
Makefile:endif  # usehip && usecuda
Makefile:ifdef usecuda
Makefile:	@cp deploy/pkg/pyproject/.pyproject_cuda.toml pyproject.toml
Makefile:else  # !usehip && usecuda
Makefile:endif  # !usehip && !usecuda
Makefile:	@echo "Uninstalling Triumvirate(-CUDA/HIP/HIPCUDA) C++ library/program..."
Makefile:ifdef usecuda
Makefile:$(OBJS): ${DIR_BUILDOBJ}/%_hipcuda.o: ${DIR_PKG_SRC}/%.cpp | objects_
Makefile:else   # usehip && !usecuda
Makefile:endif  # usehip && usecuda
Makefile:ifdef usecuda
Makefile:$(OBJS): ${DIR_BUILDOBJ}/%_cuda.o: ${DIR_PKG_SRC}/%.cpp | objects_
Makefile:else   # !usehip && !usecuda
Makefile:endif  # !usehip && usecuda
Makefile:	@echo "Cleaning up Triumvirate(-CUDA/HIP/HIPCUDA) C++ build..."
Makefile:	@echo "Cleaning up Triumvirate(-CUDA/HIP/HIPCUDA) Python build..."
Makefile:	@echo "Cleaning up Triumvirate(-CUDA/HIP/HIPCUDA) tests..."
Makefile:	@echo "Cleaning up Triumvirate(-CUDA/HIP/HIPCUDA) distributions..."
Makefile:	@echo "Cleaning up Triumvirate(-CUDA/HIP/HIPCUDA) runs..."
README.md:<!-- [![PyPI](https://img.shields.io/pypi/v/Triumvirate-CUDA?logo=PyPI&color=informational)](https://pypi.org/project/Triumvirate-CUDA)
README.md:[![Conda](https://img.shields.io/conda/v/msw/triumvirate-cuda?logo=Anaconda&color=informational)](https://anaconda.org/msw/triumvirate-cuda) -->
README.md:> CUDA variants of the Python package are/will be made available as
README.md:> ``Triumvirate-CUDA`` on [PyPI](https://pypi.org/project/Triumvirate-CUDA)
README.md:> and ``triumvirate-cuda`` through
README.md:> [Conda](https://anaconda.org/msw/triumvirate-cuda).
README.md:make ([py|cpp]install)|(cpp[libinstall|appbuild]) [useomp=(true|1)] [usecuda=(true|1)]
README.md:CUDA support, append ``usecuda=true`` or ``usecuda=1`` to the end of the
README.md:> If enabling CUDA capability, ensure there is a CUDA-capable GPU with the
README.md:> appropriate driver installed. For atypical CUDA Toolkit paths, you may
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:{% set name = 'Triumvirate-CUDA' %}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:# {% set cuda_vers = environ.get('CUDA_VERSION', '12.0') %}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:# {% set cuda_vers_parts = cuda_vers.split('.') %}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:# {% set cuda_vers_major_minor = cuda_vers_parts[0] ~ cuda_vers_parts[1] %}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:# {% set cuda_vers_int = cuda_vers_major_minor|int %}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:# {% set cuda_path = environ.get('CUDA_PATH', '/usr/local/cuda') %}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:  string: cuda{{ cuda_version|replace(".", "") }}py{{ CONDA_PY }}h{{ PKG_HASH }}_{{ PKG_BUILDNUM }}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    - cp deploy/pkg/pyproject/.pyproject_cuda.toml pyproject.toml
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    # Enforce OpenMP and CUDA support.
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    - PY_CUDA=1
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    # Use Nvidia channel libraries.
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    - cuda-nvcc
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    - cuda-version {{ cuda_version }}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    - cuda-version {{ cuda_version }}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:    - {{ pin_compatible('cuda-version', min_pin='x', max_pin='x.x') }}
deploy/pkg/conda_recipe_cuda_xp/meta.yaml:# Skip tests as no GPU host is available.
deploy/pkg/conda_recipe_cuda_xp/conda_build_config.yaml:cuda_version:
deploy/pkg/conda_recipe_cuda/meta.yaml:{% set name = 'Triumvirate-CUDA' %}
deploy/pkg/conda_recipe_cuda/meta.yaml:# {% set cuda_vers = environ.get('CUDA_VERSION', '12.0') %}
deploy/pkg/conda_recipe_cuda/meta.yaml:# {% set cuda_vers_parts = cuda_vers.split('.') %}
deploy/pkg/conda_recipe_cuda/meta.yaml:# {% set cuda_vers_major_minor = cuda_vers_parts[0] ~ cuda_vers_parts[1] %}
deploy/pkg/conda_recipe_cuda/meta.yaml:# {% set cuda_vers_int = cuda_vers_major_minor|int %}
deploy/pkg/conda_recipe_cuda/meta.yaml:# {% set cuda_path = environ.get('CUDA_PATH', '/usr/local/cuda') %}
deploy/pkg/conda_recipe_cuda/meta.yaml:  string: cuda{{ cuda_version|replace(".", "") }}py{{ CONDA_PY }}h{{ PKG_HASH }}_{{ PKG_BUILDNUM }}
deploy/pkg/conda_recipe_cuda/meta.yaml:    - cp deploy/pkg/pyproject/.pyproject_cuda.toml pyproject.toml
deploy/pkg/conda_recipe_cuda/meta.yaml:    # Enforce OpenMP and CUDA support.
deploy/pkg/conda_recipe_cuda/meta.yaml:    - PY_CUDA=1
deploy/pkg/conda_recipe_cuda/meta.yaml:    # Use Nvidia channel libraries.
deploy/pkg/conda_recipe_cuda/meta.yaml:    - cuda-nvcc
deploy/pkg/conda_recipe_cuda/meta.yaml:    - cuda-version {{ cuda_version }}
deploy/pkg/conda_recipe_cuda/meta.yaml:    - cuda-version {{ cuda_version }}
deploy/pkg/conda_recipe_cuda/meta.yaml:    - {{ pin_compatible('cuda-version', min_pin='x', max_pin='x.x') }}
deploy/pkg/conda_recipe_cuda/meta.yaml:# Skip tests as no GPU host is available.
deploy/pkg/conda_recipe_cuda/conda_build_config.yaml:cuda_version:
deploy/pkg/pyproject/.pyproject_hip.toml:name = 'Triumvirate-CUDA'
deploy/pkg/pyproject/.pyproject_hip.toml:    "Environment :: GPU",
deploy/pkg/pyproject/.pyproject_hip.toml:environment = { PY_OMP='1', PY_CUDA='1', PY_BUILD_PARALLEL='-j' }
deploy/pkg/pyproject/.pyproject_hip.toml:# Install CUDA Toolkit inside Docker container using package manager,
deploy/pkg/pyproject/.pyproject_hip.toml:# matching repository with image OS, and optionally matching CUDA version
deploy/pkg/pyproject/.pyproject_hip.toml:    "yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo",
deploy/pkg/pyproject/.pyproject_hip.toml:    "yum install -y cuda-toolkit",
deploy/pkg/pyproject/.pyproject_hip.toml:    # "yum install -y cuda-toolkit-12-6",
deploy/pkg/pyproject/.pyproject_hip.toml:environment = { PY_CXX="/usr/local/cuda/bin/nvcc", PY_CXXFLAGS="-I/usr/local/cuda/include", PY_LDFLAGS="-L/usr/local/cuda/lib64", PY_OMP='1', PY_CUDA='1', PY_BUILD_PARALLEL='-j' }
deploy/pkg/pyproject/.pyproject_hipcuda.toml:name = 'Triumvirate-CUDA'
deploy/pkg/pyproject/.pyproject_hipcuda.toml:    "Environment :: GPU",
deploy/pkg/pyproject/.pyproject_hipcuda.toml:    "Environment :: GPU :: NVIDIA CUDA",
deploy/pkg/pyproject/.pyproject_hipcuda.toml:environment = { PY_OMP='1', PY_CUDA='1', PY_BUILD_PARALLEL='-j' }
deploy/pkg/pyproject/.pyproject_hipcuda.toml:# TODO: Modify the following for hybrid CUDA/HIP.
deploy/pkg/pyproject/.pyproject_hipcuda.toml:# Install CUDA Toolkit inside Docker container using package manager,
deploy/pkg/pyproject/.pyproject_hipcuda.toml:# matching repository with image OS, and optionally matching CUDA version
deploy/pkg/pyproject/.pyproject_hipcuda.toml:    "yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo",
deploy/pkg/pyproject/.pyproject_hipcuda.toml:    "yum install -y cuda-toolkit",
deploy/pkg/pyproject/.pyproject_hipcuda.toml:    # "yum install -y cuda-toolkit-12-6",
deploy/pkg/pyproject/.pyproject_hipcuda.toml:environment = { PY_CXX="/usr/local/cuda/bin/nvcc", PY_CXXFLAGS="-I/usr/local/cuda/include", PY_LDFLAGS="-L/usr/local/cuda/lib64", PY_OMP='1', PY_CUDA='1', PY_BUILD_PARALLEL='-j' }
deploy/pkg/pyproject/.pyproject_cuda.toml:name = 'Triumvirate-CUDA'
deploy/pkg/pyproject/.pyproject_cuda.toml:    "Environment :: GPU :: NVIDIA CUDA",
deploy/pkg/pyproject/.pyproject_cuda.toml:environment = { PY_OMP='1', PY_CUDA='1', PY_BUILD_PARALLEL='-j' }
deploy/pkg/pyproject/.pyproject_cuda.toml:# Install CUDA Toolkit inside Docker container using package manager,
deploy/pkg/pyproject/.pyproject_cuda.toml:# matching repository with image OS, and optionally matching CUDA version
deploy/pkg/pyproject/.pyproject_cuda.toml:    "yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo",
deploy/pkg/pyproject/.pyproject_cuda.toml:    "yum install -y cuda-toolkit",
deploy/pkg/pyproject/.pyproject_cuda.toml:    # "yum install -y cuda-toolkit-12-6",
deploy/pkg/pyproject/.pyproject_cuda.toml:environment = { PY_CXX="/usr/local/cuda/bin/nvcc", PY_CXXFLAGS="-I/usr/local/cuda/include", PY_LDFLAGS="-L/usr/local/cuda/lib64", PY_OMP='1', PY_CUDA='1', PY_BUILD_PARALLEL='-j' }
src/triumvirate/include/fftlog.hpp:#if defined(TRV_USE_CUDA)
src/triumvirate/include/fftlog.hpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/include/fftlog.hpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/include/fftlog.hpp:#endif  // TRV_USE_CUDA
src/triumvirate/include/parameters.hpp:#if defined(TRV_USE_CUDA)
src/triumvirate/include/parameters.hpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/include/parameters.hpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/include/parameters.hpp:#endif  // TRV_USE_CUDA
src/triumvirate/include/monitor.hpp:#if defined(TRV_USE_CUDA)
src/triumvirate/include/monitor.hpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/include/monitor.hpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/include/monitor.hpp:#endif  // TRV_USE_CUDA
src/triumvirate/include/threept.hpp:#if defined(TRV_USE_CUDA)
src/triumvirate/include/threept.hpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/include/threept.hpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/include/threept.hpp:#endif  // TRV_USE_CUDA
src/triumvirate/include/field.hpp:#if defined(TRV_USE_CUDA)
src/triumvirate/include/field.hpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/include/field.hpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/include/field.hpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/monitor.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/monitor.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/monitor.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/monitor.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if !defined(TRV_USE_CUDA) && !defined(TRV_USE_HIP)
src/triumvirate/src/field.cpp:#endif  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#if !defined(TRV_USE_CUDA) && !defined(TRV_USE_HIP)
src/triumvirate/src/field.cpp:#endif  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#if !defined(TRV_USE_CUDA) && !defined(TRV_USE_HIP)
src/triumvirate/src/field.cpp:#endif  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#if !defined(TRV_USE_CUDA) && !defined(TRV_USE_HIP)
src/triumvirate/src/field.cpp:#endif  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/field.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/field.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/field.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/field.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/twopt.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/twopt.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/twopt.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/twopt.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/twopt.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/twopt.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/twopt.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/twopt.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/fftlog.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/fftlog.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/fftlog.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/fftlog.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/fftlog.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/fftlog.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/fftlog.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/fftlog.cpp:#endif  // TRV_USE_CUDA
src/triumvirate/src/threept.cpp:// #if defined(TRV_USE_CUDA)
src/triumvirate/src/threept.cpp:// #elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/threept.cpp:// #else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/threept.cpp:// #endif  // TRV_USE_CUDA
src/triumvirate/src/parameters.cpp:#if defined(TRV_USE_CUDA)
src/triumvirate/src/parameters.cpp:#elif defined(TRV_USE_HIP) // !TRV_USE_CUDA && TRV_USE_HIP
src/triumvirate/src/parameters.cpp:#else  // !TRV_USE_CUDA && !TRV_USE_HIP
src/triumvirate/src/parameters.cpp:#endif  // TRV_USE_CUDA

```
