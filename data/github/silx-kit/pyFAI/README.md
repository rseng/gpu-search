# https://github.com/silx-kit/pyFAI

```console
plugins/Lima/demo_pyFAI.py:import pyopencl
plugins/Lima/pyFAI_lima.py:import pyopencl
plugins/Lima/limaFAI.py:            worker.method = "lut_ocl_gpu"
plugins/Lima/DistortionCorrection.py:    import pyopencl
plugins/Lima/DistortionCorrection.py:    pyopencl = None
plugins/Lima/DistortionCorrection.py:    logger.warning("Unable to import pyopencl, will use OpenMP (if available)")
plugins/Lima/DistortionCorrection.py:        if pyopencl and self.ocl_integrator:
plugins/Lima/DistortionCorrection.py:                if pyopencl:
plugins/Tango/DistortionDS.py:    import pyopencl
plugins/Tango/DistortionDS.py:    pyopencl = None
plugins/Tango/DistortionDS.py:    logger.warning("Unable to import pyopencl, will use OpenMP (if available)")
README.rst:and even more using OpenCL and GPU).
README.rst:`pyopencl <http://mathema.tician.de/software/pyopencl>`_
README.rst:* ``pyopencl``	 - http://mathema.tician.de/software/pyopencl/
README.rst:* ``python3-pyopencl``
README.rst:OpenCL is hence greately adviced on Apple systems.
package/debian12/control:Recommends: python3-pyopencl,
package/debian9/control:         python3-pyopencl,
package/debian9/control:Recommends: python3-pyopencl,
package/debian11/control:Recommends: python3-pyopencl,
package/debian10/control:         python3-pyopencl,
package/debian10/control:Recommends: python3-pyopencl,
ci/github/workflows/release.yml:          PYFAI_OPENCL: "False"  # skip GPU tests
ci/github/workflows/release.yml:          PYFAI_OPENCL: "False"  # skip GPU tests
ci/github/workflows/python-package.yml:        if [ -f ci/install_pyopencl.sh ]; then bash ci/install_pyopencl.sh ; fi
ci/github/workflows/python-package.yml:        #if [ -f ci/intel_opencl_icd.sh ]; then source ci/intel_opencl_icd.sh ; fi
ci/github/workflows/python-package.yml:        export OCL_ICD_VENDORS=$(pwd)/intel_opencl_icd/vendors
ci/github/workflows/python-package.yml:        python3 -c "import pyopencl; print(pyopencl.get_platforms())"
ci/github/workflows/python-package.yml:        python3 -c "import silx.opencl; print(silx.opencl.ocl)"
ci/requirements_gh.txt:pyopencl
ci/install_pyopencl.sh:# Compile & install pyopencl
ci/install_pyopencl.sh:if [ -f ci/intel_opencl_icd.sh ];
ci/install_pyopencl.sh:    source ci/intel_opencl_icd.sh
ci/install_pyopencl.sh:    pip install wheel pybind11 mako pyopencl
ci/install_pyopencl.sh:    python3 -c "import pyopencl; print(pyopencl.get_platforms())"
ci/install_pyopencl.sh:    python3 -c "import silx.opencl; print(silx.opencl.ocl)"
ci/info_platform.py:    import pyopencl
ci/info_platform.py:    print("Unable to import pyopencl: %s" % error)
ci/info_platform.py:    print("PyOpenCL platform:")
ci/info_platform.py:    for p in pyopencl.get_platforms():
ci/before_install-linux.sh:source ./ci/intel_opencl_icd.sh;
ci/requirements_travis.txt:#pyopencl
ci/intel_opencl_icd.sh:# Download the intel OpenCL ICD and setup the environment for using it.
ci/intel_opencl_icd.sh:URL="http://www.silx.org/pub/OpenCL/"
ci/intel_opencl_icd.sh:FILENAME="intel_opencl_icd-6.4.0.38.tar.gz"
ci/intel_opencl_icd.sh:echo $(pwd)/intel_opencl_icd/icd/libintelocl.so > intel_opencl_icd/vendors/intel64.icd
ci/intel_opencl_icd.sh:export OCL_ICD_VENDORS=$(pwd)/intel_opencl_icd/vendors
ci/intel_opencl_icd.sh:export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)/intel_opencl_icd/lib
ci/intel_opencl_icd.sh:export C_INCLUDE_PATH=${C_INCLUDE_PATH}:$(pwd)/intel_opencl_icd/include
ci/intel_opencl_icd.sh:#ldd $(pwd)/intel_opencl_icd/bin/clinfo
ci/intel_opencl_icd.sh:#echo libOpenCL:
ci/intel_opencl_icd.sh:#ldd $(pwd)/intel_opencl_icd/lib/libOpenCL.so.1.0.0
ci/intel_opencl_icd.sh:#for i in $(pwd)/intel_opencl_icd/icd/*.so
ci/intel_opencl_icd.sh:$(pwd)/intel_opencl_icd/bin/clinfo
ci/action.yml:        if [ -f ci/install_pyopencl.sh ]; then bash ci/install_pyopencl.sh ; fi
ci/action.yml:        #if [ -f ci/intel_opencl_icd.sh ]; then source ci/intel_opencl_icd.sh ; fi
ci/action.yml:        export OCL_ICD_VENDORS=$(pwd)/intel_opencl_icd/vendors
ci/action.yml:        python3 -c "import pyopencl; print(pyopencl.get_platforms())"
ci/action.yml:        python3 -c "import silx.opencl; print(silx.opencl.ocl)"
doc/source/changelog.rst:    + silent some noisy test (OpenCL on windows)
doc/source/changelog.rst:    + Require silx 1.1 (for OpenCL), scipy and matplotlib
doc/source/changelog.rst:    + GPU implementation tutorial
doc/source/changelog.rst:- Support extra dtype in OpenCL (contribution from Desy)
doc/source/changelog.rst:- Fix bug in OpenCL distortion correction (collaboration with Soleil)
doc/source/changelog.rst:    - Implementation in Python, Cython and OpenCL with poissonian and azimuthal error-model
doc/source/changelog.rst:* Sigma clipping and sparsification of single crystal data (OpenCL only)
doc/source/changelog.rst:* Improved distortion correction (also on GPU, ...)
doc/source/changelog.rst:* Drop deprecated OpenCL integrator
doc/source/changelog.rst:  implemented in Python, Cython and OpenCL.
doc/source/changelog.rst:* Sigma-clipping implemented in OpenCL
doc/source/changelog.rst:  Python, Cython and OpenCL. Now, propagates the variance properly !
doc/source/changelog.rst:* Better POCL integration (debugged on cuda, x87, Power9, ...)
doc/source/changelog.rst:* Rely on *silx* mechanics for the build, test, download, GUI, opencl ...
doc/source/changelog.rst:* Fix performance regression with pyopencl >2015.2 (Thanks Andreas)
doc/source/changelog.rst:* Common pre-processing factorization on Python, Cython and OpenCL
doc/source/changelog.rst:* GPU accelerate version of ai.separate (Bragg & amorphous) **ID13**
doc/source/changelog.rst:* OpenCL Bitonic sort (to be integrated into Bragg/Amorphous separation)
doc/source/changelog.rst:* Update tests and OpenCL -> works with Beignet and pocl open source drivers
doc/source/changelog.rst:* include spec of Maxwell GPU
doc/source/changelog.rst:* fix issues with intel OpenCL icd v4.4
doc/source/changelog.rst:* unified distortion class: merge OpenCL & OpenMP implementation #108
doc/source/changelog.rst:* LUT implementation in 1D & 2D (fully tested) both with OpenMP and with OpenCL
doc/source/changelog.rst:* Switch from C++/Cython OpenCL framework to PyOpenCL
doc/source/changelog.rst:* Port opencl code to both Windows 32/64 bits and MacOSX
doc/source/changelog.rst:* Use fast-CRC checksum on x86 using SSE4 (when available) to track array change on GPU buffers
doc/source/changelog.rst:* Enhanced tests, especially for Saxs and OpenCL
doc/source/changelog.rst:Implementation of look-up table based integration and OpenCL version of it
doc/source/changelog.rst:* OpenCL flavor works well on GPU in double precision with device selection
doc/source/changelog.rst:* Include OpenCL version of azimuthal integration (based on histograms)
doc/source/usage/tutorial/Parallelization/index.rst:be compatible with GPU processing. One GPU can host up to 15 parallel instances.
doc/source/usage/tutorial/Parallelization/index.rst:3. Full GPU processing: this requires the hardware (starts ~10k€) and
doc/source/usage/tutorial/Parallelization/index.rst:the azimuthal integration to be all performed on the GPU. The advantage
doc/source/usage/tutorial/Parallelization/index.rst:little data to the GPU. Probably one of the best solution with its simple design
doc/source/usage/tutorial/Parallelization/index.rst:(but complicated GPU code under the hood) which became much simpler recently with
doc/source/usage/tutorial/Parallelization/index.rst:to GPU but requires even more specialized hardware and staff: both decompression
doc/source/usage/tutorial/Parallelization/index.rst:   GPU-decompression
doc/source/usage/tutorial/Parallelization/index.rst:   MultiGPU
doc/source/operations/index.rst:* pyopencl (for GPU computing)
doc/source/operations/index.rst:* Apple: clang modified version for mac computer without support for OpenMP, please use OpenCL for parallelization.
doc/source/operations/index.rst:* PYFAI_OPENCL: set to "0" to disable the use of OpenCL
doc/source/operations/macosx.rst:The absence of OpenMP is mitigated on Apple computer by the support of OpenCL which provied parallel intgeration.
doc/source/coverage.rst:   "gui/dialog/OpenClDeviceDialog.py", "116", "14", "12.1 %"
doc/source/coverage.rst:   "gui/widgets/OpenClDeviceLabel.py", "52", "22", "42.3 %"
doc/source/coverage.rst:   "opencl/OCLFullSplit.py", "199", "24", "12.1 %"
doc/source/coverage.rst:   "opencl/__init__.py", "42", "35", "83.3 %"
doc/source/coverage.rst:   "opencl/azim_csr.py", "565", "407", "72.0 %"
doc/source/coverage.rst:   "opencl/azim_hist.py", "474", "340", "71.7 %"
doc/source/coverage.rst:   "opencl/azim_lut.py", "330", "247", "74.8 %"
doc/source/coverage.rst:   "opencl/ocl_hist_pixelsplit.py", "223", "27", "12.1 %"
doc/source/coverage.rst:   "opencl/peak_finder.py", "474", "368", "77.6 %"
doc/source/coverage.rst:   "opencl/preproc.py", "225", "173", "76.9 %"
doc/source/coverage.rst:   "opencl/sort.py", "282", "223", "79.1 %"
doc/source/coverage.rst:   "opencl/test/__init__.py", "25", "25", "100.0 %"
doc/source/api/gui/widgets.rst:pyFAI.gui.widgets.OpenClDeviceLabel module
doc/source/api/gui/widgets.rst:.. automodule:: pyFAI.gui.widgets.OpenClDeviceLabel
doc/source/api/gui/dialog.rst:pyFAI.gui.dialog.OpenClDeviceDialog module
doc/source/api/gui/dialog.rst:.. automodule:: pyFAI.gui.dialog.OpenClDeviceDialog
doc/source/api/opencl/index.rst:pyFAI.opencl package
doc/source/api/opencl/index.rst:pyFAI.opencl.azim_csr module
doc/source/api/opencl/index.rst:.. automodule:: pyFAI.opencl.azim_csr
doc/source/api/opencl/index.rst:pyFAI.opencl.azim_hist module
doc/source/api/opencl/index.rst:.. automodule:: pyFAI.opencl.azim_hist
doc/source/api/opencl/index.rst:pyFAI.opencl.azim_lut module
doc/source/api/opencl/index.rst:.. automodule:: pyFAI.opencl.azim_lut
doc/source/api/opencl/index.rst:pyFAI.opencl.preproc module
doc/source/api/opencl/index.rst:.. automodule:: pyFAI.opencl.preproc
doc/source/api/opencl/index.rst:pyFAI.opencl.sort module
doc/source/api/opencl/index.rst:.. automodule:: pyFAI.opencl.sort
doc/source/api/opencl/index.rst:.. automodule:: pyFAI.opencl
doc/source/api/pyFAI.rst:    opencl/index
doc/source/project.rst:OpenCL... but only a C-compiler is needed to build it.
doc/source/project.rst:* 8000 lines of OpenCL kernels
doc/source/project.rst:The OpenCL code has been tested using:
doc/source/project.rst:* Nvidia OpenCL v1.1 and v1.2 on Linux, Windows (GPU device)
doc/source/project.rst:* Intel OpenCL v1.2 on Linux and Windows (CPU and ACC (Phi) devices)
doc/source/project.rst:* AMD OpenCL v1.2 on Linux and Windows (CPU and GPU device)
doc/source/project.rst:* Apple OpenCL v1.2 on MacOSX  (CPU and GPU)
doc/source/project.rst:* Beignet OpenCL v1.2 on Linux (GPU device)
doc/source/project.rst:* Pocl OpenCL v1.2 on Linux (CPU device)
doc/source/project.rst:* pyopencl (optional)
doc/source/project.rst:**Note:**: The test coverage tool does not count lines of Cython, nor those of OpenCL.
doc/source/project.rst:* LinkSCEEM project: porting to OpenCL
doc/source/pyFAI.rst:[SciPy]_ and [Matplotlib]_ plus the [OpenCL]_ binding [PyOpenCL]_ for performances.
doc/source/pyFAI.rst:especially on GPU where thousands of threads are executed simultaneously.
doc/source/pyFAI.rst:This algorithm was implemented both in [Cython]_-OpenMP and OpenCL.
doc/source/pyFAI.rst:Secondly, the CSR  implementation in OpenCL is using an algorithm based on multiple parallel
doc/source/pyFAI.rst:This makes it very well suited to run on GPUs and accelerators
doc/source/pyFAI.rst:When using OpenCL for the GPU we used a compensated (or Kahan_summation_), to reduce
doc/source/pyFAI.rst:Double precision operations are currently limited to high price and performance computing dedicated GPUs.
doc/source/pyFAI.rst:the higher number of single precision units and that the GPU is usually limited by the memory bandwidth anyway.
doc/source/pyFAI.rst:on a GPUs as far as we are aware of, and the stated twenty-fold speed up
doc/source/pyFAI.rst:Porting pyFAI to GPU would have not been possible without
doc/source/performance.rst:OpenCL implementations accumulates using error-compensated_arithmetics_ with
doc/source/performance.rst:In this plot, one OpenCL device has been added (plotted with dashed lines), it is a high-end GPU.
doc/source/performance.rst:GPU provides the best performances when it comes to azimuthal integration, it is usually the upper most curve,
doc/source/performance.rst:with speed up to 1000 or 2000 1Mpixel frames processed per second (on high-end GPU).
doc/source/biblio.rst:.. [EPDIC13] PyFAI: a Python library for high performance azimuthal integration on GPU
doc/source/biblio.rst:.. [OpenCL] Khronos OpenCL Working Group 2010 The OpenCL Specification, version 1.1 URL http://www.khronos.org/registry/cl/specs/opencl-1.1.pdf
doc/source/biblio.rst:.. [PyOpenCL]  PyCUDA and PyOpenCL: A scripting-based approach to GPU run-time code generation
doc/source/design/ai.rst:Those transformation could be GPU-ized in the future.
doc/source/design/ai.rst:The distortion of the detector is handled here and could be GPU-ized in the future.
doc/source/design/ai.rst:    Implementations exists in Cython+OpenMP, OpenCL and even in Python using scipy.sparse.
doc/source/design/index.rst:0. Common basement: Python, Numpy, Cython, PyOpenCL and silx
doc/source/design/index.rst:Level 1. is often written in Cython, or OpenCL and requires low-level expertise.
doc/source/ecosystem.rst:[SciPy]_, [Matplotlib]_, [PyOpenCL]_ but also on some ESRF-developped code:
doc/source/ecosystem.rst:`The silx toolkit <http://www.silx.org>`_  provides the basements for all GUI-application of pyFAI and also the OpenCL compute framework.
doc/source/ecosystem.rst:A EDNA data analysis server is using pyFAI as an integration engine (on the GPU)
doc/source/publications.rst:  Initial publication where the usage of GPU is envisaged to overcome
doc/source/publications.rst:* *PyFAI: a Python library for high performance azimuthal integration on GPU*
doc/source/publications.rst:* *PyFAI: a Python library for high performance azimuthal integration on GPU*;
doc/source/man/pyFAI-integrate.rst:to select a GPU (or an openCL platform) to perform the calculation on.
doc/source/man/diff_map.rst:**-g**, **--gpu**
doc/source/man/diff_map.rst:   process using OpenCL on GPU
doc/source/man/diff_tomo.rst:**-g**, **--gpu**
doc/source/man/diff_tomo.rst:   process using OpenCL on GPU
doc/source/man/pyFAI-benchmark.rst:OpenCL devices can be probed with options "-c", "-g" and "-a".
doc/source/man/pyFAI-benchmark.rst:   perform benchmark using OpenCL on the CPU
doc/source/man/pyFAI-benchmark.rst:**-g**, **--gpu**
doc/source/man/pyFAI-benchmark.rst:   perform benchmark using OpenCL on the GPU
doc/source/man/pyFAI-benchmark.rst:   perform benchmark using OpenCL on the Accelerator (like XeonPhi/MIC)
doc/source/man/pyFAI-benchmark.rst:   opencl, all
doc/source/man/pyFAI-diffmap.rst:**-g**, **--gpu**
doc/source/man/pyFAI-diffmap.rst:   process using OpenCL on GPU
doc/source/man/sparsify-Bragg.rst:This program requires OpenCL. The device needs be properly selected.
doc/source/man/sparsify-Bragg.rst:Opencl setup options:
doc/source/man/sparsify-Bragg.rst:   Enforce the workgroup size for OpenCL kernel. Impacts only on the
doc/source/man/sparsify-Bragg.rst:   device type like \`cpu\` or \`gpu\` or \`acc`. Can help to select the
doc/source/man/peakfinder.rst:This program requires OpenCL. The device needs be properly selected.
doc/source/man/peakfinder.rst:Opencl setup options:
doc/source/man/peakfinder.rst:   Enforce the workgroup size for OpenCL kernel. Impacts only on the
doc/source/man/peakfinder.rst:   device type like \`cpu\` or \`gpu\` or \`acc`. Can help to select the
run_tests.py:PYFAI_OPENCL=False to disable OpenCL tests.
copyright:Files: pyFAI/resources/openCL/bitonic.cl
copyright:       openCL/bsort.cl
pyproject.toml:opencl = [ "pyopencl" ]
pyproject.toml:all = ["PyQt5", "pyopencl", "hdf5plugin"]
MANIFEST.in:recursive-include pyFAI/resources/openCL *.cl *.h
.travis.yml:  - "if [ -f ci/install_pyopencl.sh ]; then bash ci/install_pyopencl.sh ; fi"
src/pyFAI/gui/helper/ProcessingWidget.py:from silx.gui.widgets.WaitingPushButton import WaitingPushButton
src/pyFAI/gui/helper/ProcessingWidget.py:    button = WaitingPushButton(parent)
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:    import pyopencl
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:    pyopencl = None
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:class OpenClDeviceDialog(qt.QDialog):
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:    """Dialog to select an OpenCl device. It could be both select an available
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:    This dialog do not expect PyOpenCL to installed.
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        super(OpenClDeviceDialog, self).__init__(parent)
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        filename = get_ui_file("opencl-device-dialog.ui")
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        self._group.addButton(self._anyGpuButton)
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        if pyopencl is None:
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:            self._availableButton.setToolTip("PyOpenCL has to be installed to display available devices.")
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        if pyopencl is None:
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        for platformId, platform in enumerate(pyopencl.get_platforms()):
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:            for deviceId, device in enumerate(platform.get_devices(pyopencl.device_type.ALL)):
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:                typeName = pyopencl.device_type.to_string(device.type)
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        """Select an OpenCL device displayed on this dialog.
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        elif device == "gpu":
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:            self._anyGpuButton.setChecked(True)
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        """Returns the selected OpenCL device.
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:        if self._anyGpuButton.isChecked():
src/pyFAI/gui/dialog/OpenClDeviceDialog.py:            return "gpu"
src/pyFAI/gui/dialog/meson.build:    'OpenClDeviceDialog.py',
src/pyFAI/gui/dialog/IntegrationMethodDialog.py:    """Label displaying a specific OpenCL device.
src/pyFAI/gui/dialog/IntegrationMethodDialog.py:        "opencl": "OpenCL",
src/pyFAI/gui/dialog/IntegrationMethodDialog.py:        "opencl": "Use an OpenCL implementation based on hardware accelerators. Fastest but hardware/driver dependant",
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:    import pyopencl
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:    pyopencl = None
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:class OpenClDeviceLabel(qt.QLabel):
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:    """Label displaying a specific OpenCL device.
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        super(OpenClDeviceLabel, self).__init__(parent)
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:    def __getOpenClDevice(self, platformId, deviceId):
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        if pyopencl is None:
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        if not (0 <= platformId < len(pyopencl.get_platforms())):
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        platform = pyopencl.get_platforms()[platformId]
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:            label = "No OpenCL device selected"
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        elif self.__device == "gpu":
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:            label = "Any available GPU"
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:            device = self.__getOpenClDevice(platformId, deviceId)
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        """Select an OpenCL device displayed on this dialog.
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        """Returns the selected OpenCL device.
src/pyFAI/gui/widgets/OpenClDeviceLabel.py:        A device can be identified as a string like 'any', 'cpu' or 'gpu' or a
src/pyFAI/gui/widgets/WorkerConfigurator.py:from ..dialog.OpenClDeviceDialog import OpenClDeviceDialog
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.__openclDevice = None
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.opencl_config_button.clicked.connect(self.selectOpenclDevice)
src/pyFAI/gui/widgets/WorkerConfigurator.py:            if method.impl == "opencl":
src/pyFAI/gui/widgets/WorkerConfigurator.py:                config["opencl_device"] = self.__openclDevice
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.__setOpenclDevice(method.target)
src/pyFAI/gui/widgets/WorkerConfigurator.py:    def selectOpenclDevice(self):
src/pyFAI/gui/widgets/WorkerConfigurator.py:        dialog = OpenClDeviceDialog(self)
src/pyFAI/gui/widgets/WorkerConfigurator.py:        dialog.selectDevice(self.__openclDevice)
src/pyFAI/gui/widgets/WorkerConfigurator.py:            self.__setOpenclDevice(device)
src/pyFAI/gui/widgets/WorkerConfigurator.py:    def __setOpenclDevice(self, device):
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.__openclDevice = device
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.opencl_label.setDevice(device)
src/pyFAI/gui/widgets/WorkerConfigurator.py:        openclEnabled = (method.impl if method is not None else "") == "opencl"
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.opencl_title.setEnabled(openclEnabled)
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.opencl_label.setEnabled(openclEnabled)
src/pyFAI/gui/widgets/WorkerConfigurator.py:        self.opencl_config_button.setEnabled(openclEnabled)
src/pyFAI/gui/widgets/MethodLabel.py:        "opencl": "OpenCL",
src/pyFAI/gui/widgets/meson.build:    'OpenClDeviceLabel.py',
src/pyFAI/integrator/load_engines.py:    from ..opencl import ocl
src/pyFAI/integrator/load_engines.py:        from ..opencl import azim_hist as ocl_azim  # IGNORE:F0401
src/pyFAI/integrator/load_engines.py:        logger.error("Unable to import pyFAI.opencl.azim_hist: %s", error)
src/pyFAI/integrator/load_engines.py:            IntegrationMethod(1, "no", "histogram", "OpenCL",
src/pyFAI/integrator/load_engines.py:            IntegrationMethod(2, "no", "histogram", "OpenCL",
src/pyFAI/integrator/load_engines.py:        from ..opencl import azim_csr as ocl_azim_csr  # IGNORE:F0401
src/pyFAI/integrator/load_engines.py:        logger.error("Unable to import pyFAI.opencl.azim_csr: %s", error)
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(1, "bbox", "CSR", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(2, "bbox", "CSR", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(1, "no", "CSR", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(2, "no", "CSR", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(1, "full", "CSR", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(2, "full", "CSR", "OpenCL",
src/pyFAI/integrator/load_engines.py:        from ..opencl import azim_lut as ocl_azim_lut  # IGNORE:F0401
src/pyFAI/integrator/load_engines.py:        logger.error("Unable to import pyFAI.opencl.azim_lut: %s", error)
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(1, "bbox", "LUT", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(2, "bbox", "LUT", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(1, "no", "LUT", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(2, "no", "LUT", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(1, "full", "LUT", "OpenCL",
src/pyFAI/integrator/load_engines.py:                IntegrationMethod(2, "full", "LUT", "OpenCL",
src/pyFAI/integrator/load_engines.py:        from ..opencl import sort as ocl_sort
src/pyFAI/integrator/load_engines.py:        logger.error("Unable to import pyFAI.opencl.sort: %s", error)
src/pyFAI/integrator/azimuthal.py:            # This whole block uses CSR, Now we should treat all the various implementation: Cython, OpenCL and finally Python.
src/pyFAI/integrator/azimuthal.py:            else:  # method.impl_lower in ("opencl", "python"):
src/pyFAI/integrator/azimuthal.py:                        if method.impl_lower == "opencl":
src/pyFAI/integrator/azimuthal.py:                if method.impl_lower == "opencl":
src/pyFAI/integrator/azimuthal.py:        elif method.method[1:4] == ("no", "histogram", "opencl"):
src/pyFAI/integrator/azimuthal.py:            # This whole block uses CSR, Now we should treat all the various implementation: Cython, OpenCL and finally Python.
src/pyFAI/integrator/azimuthal.py:                # method.impl_lower in ("opencl", "python"):
src/pyFAI/integrator/azimuthal.py:                        if (method.impl_lower == "opencl"):
src/pyFAI/integrator/azimuthal.py:            if intpl is None:  # fallback if OpenCL failed or default cython
src/pyFAI/integrator/azimuthal.py:                if method.impl_lower == "opencl":
src/pyFAI/integrator/azimuthal.py:                    logger.debug("integrate2d uses OpenCL histogram implementation")
src/pyFAI/integrator/azimuthal.py:        if (method.impl_lower == "opencl") and npt_azim and (npt_azim > 1):
src/pyFAI/integrator/azimuthal.py:        if (method.impl_lower == "opencl"):
src/pyFAI/integrator/azimuthal.py:                    logger.info("reset opencl sorter")
src/pyFAI/integrator/azimuthal.py:        if (method.impl_lower == "opencl") and npt_azim and (npt_azim > 1):
src/pyFAI/integrator/azimuthal.py:        if (method.impl_lower == "opencl"):
src/pyFAI/integrator/azimuthal.py:                    logger.info("reset opencl sorter")
src/pyFAI/integrator/azimuthal.py:                # This whole block uses CSR, Now we should treat all the various implementation: Cython, OpenCL and finally Python.
src/pyFAI/integrator/azimuthal.py:                    if method.impl_lower == "opencl":
src/pyFAI/integrator/common.py:        :param block_size: size of the block for OpenCL integration (unused?)
src/pyFAI/integrator/common.py:        :param profile: set to True to enable profiling in OpenCL
src/pyFAI/integrator/common.py:                    if method.impl_lower == "opencl":
src/pyFAI/integrator/common.py:                    if method.impl_lower == "opencl":
src/pyFAI/integrator/common.py:                        # TODO: manage OpenCL targets
src/pyFAI/integrator/common.py:        if method.method[1:3] == ("no", "histogram") and method.impl_lower != "opencl":
src/pyFAI/integrator/common.py:                    if method.impl_lower == "opencl":
src/pyFAI/integrator/common.py:                    if method.impl_lower == "opencl":
src/pyFAI/integrator/common.py:        if method.method[1:3] == ("no", "histogram") and method.impl_lower != "opencl":
src/pyFAI/directories.py: * openCL directory with OpenCL kernels
src/pyFAI/app/benchmark.py:                        action="store_true", dest="opencl_cpu", default=False,
src/pyFAI/app/benchmark.py:                        help="perform benchmark using OpenCL on the CPU")
src/pyFAI/app/benchmark.py:    parser.add_argument("-g", "--gpu",
src/pyFAI/app/benchmark.py:                        action="store_true", dest="opencl_gpu", default=False,
src/pyFAI/app/benchmark.py:                        help="perform benchmark using OpenCL on the GPU")
src/pyFAI/app/benchmark.py:                        action="store_true", dest="opencl_acc", default=False,
src/pyFAI/app/benchmark.py:                        help="perform benchmark using OpenCL on the Accelerator (like XeonPhi/MIC)")
src/pyFAI/app/benchmark.py:                        dest="implementation", default=["cython", "opencl"], type=str, nargs="+",
src/pyFAI/app/benchmark.py:                        help="Benchmark using specific algorithm implementations: python, cython, opencl, all")
src/pyFAI/app/benchmark.py:    if options.opencl_cpu:
src/pyFAI/app/benchmark.py:    if options.opencl_gpu:
src/pyFAI/app/benchmark.py:        devices.append("gpu")
src/pyFAI/app/benchmark.py:    if options.opencl_acc:
src/pyFAI/app/integrate.py:to select a GPU (or an openCL platform) to perform the calculation on."""
src/pyFAI/app/peakfinder.py:This program requires OpenCL. The device needs be properly selected.
src/pyFAI/app/peakfinder.py:from ..opencl import ocl
src/pyFAI/app/peakfinder.py:    logger.error("Peakfinder requires a valid OpenCL stack to be installed")
src/pyFAI/app/peakfinder.py:    from ..opencl.peak_finder import OCL_PeakFinder
src/pyFAI/app/peakfinder.py:    group = parser.add_argument_group("Opencl setup options")
src/pyFAI/app/peakfinder.py:                       help="Enforce the workgroup size for OpenCL kernel. Impacts only on the execution speed, not on the result.")
src/pyFAI/app/peakfinder.py:                       help="device type like `cpu` or `gpu` or `acc`. Can help to select the proper device.")
src/pyFAI/app/peakfinder.py:            raise RuntimeError("sparsify-Brgg requires _really_ a valide OpenCL environment. Please install pyopencl !")
src/pyFAI/app/peakfinder.py:    logger.debug("Initialize the OpenCL device")
src/pyFAI/app/peakfinder.py:        pb.update(0, message="Initialize the OpenCL device")
src/pyFAI/app/diff_tomo.py:        parser.add_argument("-g", "--gpu", dest="gpu", action="store_true",
src/pyFAI/app/diff_tomo.py:                            help="process using OpenCL on GPU ", default=False)
src/pyFAI/app/diff_tomo.py:        self.use_gpu = options.gpu
src/pyFAI/app/sparsify.py:This program requires OpenCL. The device needs be properly selected.
src/pyFAI/app/sparsify.py:from ..opencl import ocl
src/pyFAI/app/sparsify.py:    logger.error("Sparsify requires a valid OpenCL stack to be installed")
src/pyFAI/app/sparsify.py:    from ..opencl.peak_finder import OCL_PeakFinder
src/pyFAI/app/sparsify.py:    group = parser.add_argument_group("Opencl setup options")
src/pyFAI/app/sparsify.py:                       help="Enforce the workgroup size for OpenCL kernel. Impacts only on the execution speed, not on the result.")
src/pyFAI/app/sparsify.py:                       help="device type like `cpu` or `gpu` or `acc`. Can help to select the proper device.")
src/pyFAI/app/sparsify.py:            raise RuntimeError("sparsify-Brgg requires _really_ a valide OpenCL environment. Please install pyopencl !")
src/pyFAI/app/sparsify.py:    logger.debug("Initialize the OpenCL device")
src/pyFAI/app/sparsify.py:        pb.update(0, message="Initialize the OpenCL device")
src/pyFAI/test/test_error_model.py:            for impl in ("python", "cython", "opencl"):
src/pyFAI/test/test_bug_regression.py:            if not UtilsTest.WITH_OPENCL_TEST:
src/pyFAI/test/test_bug_regression.py:                if "opencl" in elements:
src/pyFAI/test/test_bug_regression.py:                    logger.warning("Skip %s. OpenCL tests disabled", path)
src/pyFAI/test/test_bug_regression.py:                            "pyopencl is not installed" in err.__str__() or
src/pyFAI/test/test_bug_regression.py:        d.use_gpu # used to raise AttributeError
src/pyFAI/test/test_bug_regression.py:        d.use_gpu = True # used to raise AttributeError
src/pyFAI/test/test_bug_regression.py:                       # ("no", "csr", "opencl"),  # Known broken
src/pyFAI/test/test_integrate_config.py:    def test_opencl(self):
src/pyFAI/test/test_integrate_config.py:        config = {"do_OpenCL": True}
src/pyFAI/test/test_integrate_config.py:        self.assertNotIn("do_OpenCL", config)
src/pyFAI/test/test_integrate_config.py:        self.assertEqual(config["method"], ('*', 'csr', 'opencl'))
src/pyFAI/test/test_integrate_config.py:    def test_opencl_device(self):
src/pyFAI/test/test_integrate_config.py:        self.assertNotIn("do_OpenCL", config)
src/pyFAI/test/test_integrate_config.py:        self.assertEqual(config["method"], ('*', 'csr', 'opencl'))
src/pyFAI/test/test_integrate_config.py:        self.assertEqual(config["opencl_device"], (1, 1))
src/pyFAI/test/test_integrate_config.py:    def test_opencl_cpu_device(self):
src/pyFAI/test/test_integrate_config.py:        self.assertNotIn("do_OpenCL", config)
src/pyFAI/test/test_integrate_config.py:        self.assertEqual(config["method"], ('*', 'lut', 'opencl'))
src/pyFAI/test/test_integrate_config.py:        self.assertEqual(config["opencl_device"], "cpu")
src/pyFAI/test/test_integrate_config.py:        when parsing a config json file in diffmap + enforce the usage of the GPU, the splitting gets changed
src/pyFAI/test/test_integrate_config.py:        from ..opencl import ocl
src/pyFAI/test/test_integrate_config.py:        expected_without_gpu = ["no", "lut", "cython"]
src/pyFAI/test/test_integrate_config.py:        expected_with_gpu = ["no", "lut", "opencl"]
src/pyFAI/test/test_integrate_config.py:        config = {"ai": {"method": expected_without_gpu}}
src/pyFAI/test/test_integrate_config.py:        #without GPU option -g
src/pyFAI/test/test_integrate_config.py:        self.assertEqual(parsed_config["ai"]["method"], expected_without_gpu, "method matches without -g option")
src/pyFAI/test/test_integrate_config.py:        #with GPU option -g
src/pyFAI/test/test_integrate_config.py:        expected = expected_with_gpu if ocl else expected_without_gpu
src/pyFAI/test/test_mask.py:    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/test/test_mask.py:        meth = ("bbox", "lut", "opencl")
src/pyFAI/test/test_mask.py:    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/test/test_mask.py:        meth = ("bbox", "csr", "opencl")
src/pyFAI/test/test_mask.py:    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/test/test_mask.py:    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/test/test_integrate.py:        if UtilsTest.opencl:
src/pyFAI/test/test_integrate.py:                    continue  # Skip OpenCL
src/pyFAI/test/test_csr.py:from .. import opencl
src/pyFAI/test/test_csr.py:if opencl.ocl:
src/pyFAI/test/test_csr.py:    from ..opencl import azim_csr as ocl_azim_csr
src/pyFAI/test/test_csr.py:    @unittest.skipIf((utilstest.UtilsTest.opencl is None) or (opencl.ocl is None), "Test on OpenCL disabled")
src/pyFAI/test/test_csr.py:    def test_opencl_csr(self):
src/pyFAI/test/test_csr.py:                if not opencl.ocl:
src/pyFAI/test/test_csr.py:                    except (opencl.pyopencl.MemoryError, MemoryError):
src/pyFAI/test/test_method_registry.py:            ("lut_ocl", Method(dim=None, split='*', algo='lut', impl='opencl', target=None)),
src/pyFAI/test/test_method_registry.py:            ("csr_ocl", Method(dim=None, split='*', algo='csr', impl='opencl', target=None)),
src/pyFAI/test/test_method_registry.py:            ("csr_ocl_1,5", Method(dim=None, split='*', algo='csr', impl='opencl', target=(1, 5))),
src/pyFAI/test/test_method_registry.py:            ("ocl_2,3", Method(dim=None, split='*', algo='*', impl='opencl', target=(2, 3))),
src/pyFAI/test/test_split_pixel.py:            if v.dimension == 1 and  v.target is None:  # exclude OpenCL engines
src/pyFAI/test/utilstest.py:        self.WITH_OPENCL_TEST = True
src/pyFAI/test/utilstest.py:        """OpenCL tests are included"""
src/pyFAI/test/utilstest.py:        return f"TestOptions: WITH_QT_TEST={self.WITH_QT_TEST} WITH_OPENCL_TEST={self.WITH_OPENCL_TEST} "\
src/pyFAI/test/utilstest.py:    def opencl(self):
src/pyFAI/test/utilstest.py:        return self.WITH_OPENCL_TEST
src/pyFAI/test/utilstest.py:        if parsed_options is not None and not parsed_options.opencl:
src/pyFAI/test/utilstest.py:            self.WITH_OPENCL_TEST = False
src/pyFAI/test/utilstest.py:            # That's an easy way to skip OpenCL tests
src/pyFAI/test/utilstest.py:            # It disable the use of OpenCL on the full silx project
src/pyFAI/test/utilstest.py:            os.environ['PYFAI_OPENCL'] = "False"
src/pyFAI/test/utilstest.py:        elif os.environ.get('PYFAI_OPENCL', 'True') == 'False':
src/pyFAI/test/utilstest.py:            self.WITH_OPENCL_TEST = False
src/pyFAI/test/utilstest.py:            # That's an easy way to skip OpenCL tests
src/pyFAI/test/utilstest.py:            # It disable the use of OpenCL on the full silx project
src/pyFAI/test/utilstest.py:            os.environ['PYFAI_OPENCL'] = "False"
src/pyFAI/test/utilstest.py:        parser.add_argument("-o", "--no-opencl", dest="opencl", default=True,
src/pyFAI/test/utilstest.py:                            help="Disable the test of the OpenCL part")
src/pyFAI/test/test_distortion.py:        from ..opencl import ocl, pyopencl
src/pyFAI/test/test_distortion.py:            platform_id = pyopencl.get_platforms().index(oplat)
src/pyFAI/test/test_distortion.py:            self.assertEqual(len(w[0]), d.mask.sum(), "masked pixels are all missing, opencl")
src/pyFAI/test/test_flat.py:from ..opencl import ocl
src/pyFAI/test/test_flat.py:                "OpenCL atomic are not that good !"
src/pyFAI/test/test_flat.py:        if ocl and UtilsTest.opencl:
src/pyFAI/test/test_flat.py:            for device in ["cpu", "gpu", "acc"]:
src/pyFAI/test/test_flat.py:            except (MemoryError, pyFAI.opencl.pyopencl.MemoryError):
src/pyFAI/test/test_flat.py:                logger.warning("Got MemoryError from OpenCL device")
src/pyFAI/test/test_azimuthal_integrator.py:    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertLess(rwp, 1, "Rwp medfilt1d Cython/OpenCL: %.3f" % rwp)
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertLess(rwp, 3, "Rwp trimmed-mean Cython/OpenCL: %.3f" % rwp)
src/pyFAI/test/test_azimuthal_integrator.py:                                       method=("full", "CSR", "opencl"))
src/pyFAI/test/test_azimuthal_integrator.py:        if UtilsTest.opencl and os.name != 'nt':
src/pyFAI/test/test_azimuthal_integrator.py:        method = ("no", "histogram", "opencl")
src/pyFAI/test/test_azimuthal_integrator.py:            reason = "Skipping TestIntergrationNextGeneration.test_histo as OpenCL method not available"
src/pyFAI/test/test_azimuthal_integrator.py:        opencl = ai._integrate1d_ng(data, 100, method=method, error_model="poisson")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertEqual(opencl.compute_engine, "pyFAI.opencl.azim_hist.OCL_Histogram1d")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertEqual(str(opencl.unit), "q_nm^-1")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertTrue(numpy.allclose(opencl.radial, python.radial), "opencl position are the same")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertTrue(numpy.allclose(opencl.intensity, python.intensity), "opencl intensities are the same")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertTrue(numpy.allclose(opencl.sigma, python.sigma), "opencl errors are the same")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertTrue(numpy.allclose(opencl.sum_signal.sum(axis=-1), python.sum_signal), "opencl sum_signal are the same")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertTrue(numpy.allclose(opencl.sum_variance.sum(axis=-1), python.sum_variance),
src/pyFAI/test/test_azimuthal_integrator.py:                        f"opencl sum_variance are the same {abs(opencl.sum_variance.sum(axis=-1) - python.sum_variance).max()}")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertTrue(numpy.allclose(opencl.sum_normalization.sum(axis=-1), python.sum_normalization), "opencl sum_normalization are the same")
src/pyFAI/test/test_azimuthal_integrator.py:        self.assertTrue(numpy.allclose(opencl.count, python.count), "opencl count are the same")
src/pyFAI/test/test_azimuthal_integrator.py:                         # 'opencl' #TODO
src/pyFAI/test/test_azimuthal_integrator.py:            if method.impl == "OpenCL":
src/pyFAI/test/test_azimuthal_integrator.py:            if method.impl == "OpenCL":
src/pyFAI/test/test_all.py:from ..opencl import test as test_opencl
src/pyFAI/test/test_all.py:    testsuite.addTest(test_opencl.suite())
src/pyFAI/test/test_preproc.py:    @unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/test/test_preproc.py:    def test_opencl(self):
src/pyFAI/test/test_preproc.py:        from ..opencl import ocl
src/pyFAI/test/test_preproc.py:            self.skipTest("OpenCL not available")
src/pyFAI/test/test_preproc.py:        from ..opencl import preproc as ocl_preproc
src/pyFAI/test/test_worker.py:                 'method': [1, 'full', 'csr', 'opencl', 'gpu'],
src/pyFAI/worker.py:        :param device: Used to influance OpenCL behavour: can be "cpu", "GPU", "Acc" or even an OpenCL context
src/pyFAI/worker.py:        :param device: Used to influance OpenCL behavour: can be "cpu", "GPU", "Acc" or even an OpenCL context
src/pyFAI/io/integration_config.py:    * do_openCL is now deprecated, replaced with the method (which can still be a string)
src/pyFAI/io/integration_config.py:    use_opencl = config.pop("do_OpenCL", False)
src/pyFAI/io/integration_config.py:    if use_opencl is not None and method is not None:
src/pyFAI/io/integration_config.py:        if use_opencl:
src/pyFAI/io/integration_config.py:            _logger.warning("Both 'method' and 'do_OpenCL' are defined. 'do_OpenCL' is ignored.")
src/pyFAI/io/integration_config.py:        if use_opencl:
src/pyFAI/io/integration_config.py:    when 5-tuple, there is a reference to the opencl-target as well.
src/pyFAI/io/integration_config.py:    The prefered version is to have method and opencl_device separated for ease of parsing.
src/pyFAI/io/integration_config.py:    config["opencl_device"] = method.target
src/pyFAI/io/integration_config.py:        target = self._config.pop("opencl_device", None)
src/pyFAI/distortion.py:from .opencl import ocl
src/pyFAI/distortion.py:    from .opencl import azim_lut as ocl_azim_lut
src/pyFAI/distortion.py:    from .opencl import azim_csr as ocl_azim_csr
src/pyFAI/distortion.py:        :param device: Name of the device: None for OpenMP, "cpu" or "gpu" or the id of the OpenCL device a 2-tuple of integer
src/pyFAI/distortion.py:        :param workgroup: workgroup size for CSR on OpenCL
src/pyFAI/distortion.py:        :param device: can be None, "cpu" or "gpu" or the id as a 2-tuple of integer
src/pyFAI/opencl/azim_hist.py:from . import ocl, pyopencl
src/pyFAI/opencl/azim_hist.py:if pyopencl is not None:
src/pyFAI/opencl/azim_hist.py:    mf = pyopencl.mem_flags
src/pyFAI/opencl/azim_hist.py:    raise ImportError("pyopencl is not installed")
src/pyFAI/opencl/azim_hist.py:from . import concatenate_cl_kernel, get_x87_volatile_option, processing, OpenclProcessing
src/pyFAI/opencl/azim_hist.py:class OCL_Histogram1d(OpenclProcessing):
src/pyFAI/opencl/azim_hist.py:    """Class in charge of performing histogram calculation in OpenCL using
src/pyFAI/opencl/azim_hist.py:    kernel_files = ["silx:opencl/doubleword.cl",
src/pyFAI/opencl/azim_hist.py:                    "pyfai:openCL/preprocess.cl",
src/pyFAI/opencl/azim_hist.py:                    "pyfai:openCL/ocl_histo.cl"
src/pyFAI/opencl/azim_hist.py:        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
src/pyFAI/opencl/azim_hist.py:        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
src/pyFAI/opencl/azim_hist.py:                           " but it can be present and not declared as Nvidia does",
src/pyFAI/opencl/azim_hist.py:        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
src/pyFAI/opencl/azim_hist.py:        """Call the OpenCL compiler
src/pyFAI/opencl/azim_hist.py:        # concatenate all needed source files into a single openCL module
src/pyFAI/opencl/azim_hist.py:            OpenclProcessing.compile_kernels(self, kernels, compile_options)
src/pyFAI/opencl/azim_hist.py:                OpenclProcessing.compile_kernels(self, ["pyfai:openCL/deactivate_atomic64.cl"] + kernels, compile_options)
src/pyFAI/opencl/azim_hist.py:                logger.warning("Your OpenCL compiler wrongly claims it support 64-bit atomics. Degrading to 32 bits atomics!")
src/pyFAI/opencl/azim_hist.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/azim_hist.py:        if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/azim_hist.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], data.data)
src/pyFAI/opencl/azim_hist.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], data.data)
src/pyFAI/opencl/azim_hist.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], numpy.ascontiguousarray(data, dest_type))
src/pyFAI/opencl/azim_hist.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], numpy.ascontiguousarray(data))
src/pyFAI/opencl/azim_hist.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/azim_hist.py:        :param histo_signal: destination array or pyopencl array for sum of signals
src/pyFAI/opencl/azim_hist.py:        :param histo_normalization: destination array or pyopencl array for sum of normalization
src/pyFAI/opencl/azim_hist.py:        :param histo_normalization_sq: destination array or pyopencl array for sum of normalization squared
src/pyFAI/opencl/azim_hist.py:        :param histo_count: destination array or pyopencl array for counting pixels
src/pyFAI/opencl/azim_hist.py:        :param intensity: destination PyOpenCL array for integrated intensity
src/pyFAI/opencl/azim_hist.py:        :param std: destination PyOpenCL array for standard deviation
src/pyFAI/opencl/azim_hist.py:        :param sem: destination PyOpenCL array for standard error of the mean
src/pyFAI/opencl/azim_hist.py:                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output4"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_signal, self.cl_mem["histo_sig"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_variance, self.cl_mem["histo_var"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_normalization, self.cl_mem["histo_nrm"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_normalization_sq, self.cl_mem["histo_nrm2"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_count, self.cl_mem["histo_cnt"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, intensity, self.cl_mem["intensity"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
src/pyFAI/opencl/azim_hist.py:    """Class in charge of performing histogram calculation in OpenCL using
src/pyFAI/opencl/azim_hist.py:    kernel_files = ["silx:opencl/doubleword.cl",
src/pyFAI/opencl/azim_hist.py:                    "pyfai:openCL/preprocess.cl",
src/pyFAI/opencl/azim_hist.py:                    "pyfai:openCL/ocl_histo.cl"
src/pyFAI/opencl/azim_hist.py:        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
src/pyFAI/opencl/azim_hist.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/azim_hist.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/azim_hist.py:        :param histo_signal: destination array or pyopencl array for sum of signals
src/pyFAI/opencl/azim_hist.py:        :param histo_normalization: destination array or pyopencl array for sum of normalization
src/pyFAI/opencl/azim_hist.py:        :param histo_normalization_sq: destination PyOpenCL array or pyopencl array for sum of normalization squared
src/pyFAI/opencl/azim_hist.py:        :param histo_count: destination PyOpenCL array or pyopencl array for counting pixels
src/pyFAI/opencl/azim_hist.py:        :param intensity: destination PyOpenCL array for integrated intensity
src/pyFAI/opencl/azim_hist.py:        :param std: destination PyOpenCL array for standard deviation
src/pyFAI/opencl/azim_hist.py:        :param sem: destination PyOpenCL array for standard error of the mean
src/pyFAI/opencl/azim_hist.py:                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_signal, self.cl_mem["histo_sig"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_variance, self.cl_mem["histo_var"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_normalization, self.cl_mem["histo_nrm"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_normalization_sq, self.cl_mem["histo_nrm2"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, histo_count, self.cl_mem["histo_cnt"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, intensity, self.cl_mem["intensity"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
src/pyFAI/opencl/azim_hist.py:            ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
src/pyFAI/opencl/test/test_ocl_azim_csr.py:#    Project: Simple histogram in Python + OpenCL
src/pyFAI/opencl/test/test_ocl_azim_csr.py:from .. import ocl, get_opencl_code
src/pyFAI/opencl/test/test_ocl_azim_csr.py:    import pyopencl.array
src/pyFAI/opencl/test/test_ocl_azim_csr.py:from silx.opencl.common import _measure_workgroup_size
src/pyFAI/opencl/test/test_ocl_azim_csr.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_ocl_azim_csr.py:@unittest.skipUnless(ocl, "PyOpenCl is missing")
src/pyFAI/opencl/test/test_ocl_azim_csr.py:                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
src/pyFAI/opencl/test/test_ocl_azim_csr.py:                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
src/pyFAI/opencl/test/test_ocl_azim_csr.py:                cls.queue = pyopencl.CommandQueue(cls.ctx)
src/pyFAI/opencl/test/test_ocl_azim_csr.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_ocl_azim_csr.py:            logger.warning("This test is known to be complicated for AMD-GPU, relax the constrains for them")
src/pyFAI/opencl/test/test_ocl_azim_csr.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_ocl_azim_csr.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_ocl_sort.py:"""Test for OpenCL sorting on GPU"""
src/pyFAI/opencl/test/test_ocl_sort.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_ocl_sort.py:@unittest.skipIf(ocl is None, "OpenCL is not available")
src/pyFAI/opencl/test/test_ocl_sort.py:        import pyopencl
src/pyFAI/opencl/test/test_ocl_sort.py:            "cpu" in pyopencl.device_type.to_string(device.type).lower()):
src/pyFAI/opencl/test/test_ocl_sort.py:            "cpu" in pyopencl.device_type.to_string(device.type).lower()):
src/pyFAI/opencl/test/test_ocl_azim_lut.py:#    Project: Simple histogram in Python + OpenCL
src/pyFAI/opencl/test/test_ocl_azim_lut.py:from .. import ocl, get_opencl_code
src/pyFAI/opencl/test/test_ocl_azim_lut.py:    import pyopencl.array
src/pyFAI/opencl/test/test_ocl_azim_lut.py:from silx.opencl.common import _measure_workgroup_size
src/pyFAI/opencl/test/test_ocl_azim_lut.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_ocl_azim_lut.py:@unittest.skipUnless(ocl, "PyOpenCl is missing")
src/pyFAI/opencl/test/test_ocl_azim_lut.py:                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
src/pyFAI/opencl/test/test_ocl_azim_lut.py:                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
src/pyFAI/opencl/test/test_ocl_azim_lut.py:                cls.queue = pyopencl.CommandQueue(cls.ctx)
src/pyFAI/opencl/test/test_ocl_azim_lut.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_ocl_azim_lut.py:            logger.warning("This test is known to be complicated for AMD-GPU, relax the constrains for them")
src/pyFAI/opencl/test/test_ocl_histo.py:#    Project: Simple histogram in Python + OpenCL
src/pyFAI/opencl/test/test_ocl_histo.py:from .. import ocl, get_opencl_code
src/pyFAI/opencl/test/test_ocl_histo.py:    import pyopencl.array
src/pyFAI/opencl/test/test_ocl_histo.py:from silx.opencl.common import _measure_workgroup_size
src/pyFAI/opencl/test/test_ocl_histo.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_ocl_histo.py:@unittest.skipUnless(ocl, "PyOpenCl is missing")
src/pyFAI/opencl/test/test_ocl_histo.py:                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
src/pyFAI/opencl/test/test_ocl_histo.py:                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
src/pyFAI/opencl/test/test_ocl_histo.py:                cls.queue = pyopencl.CommandQueue(cls.ctx)
src/pyFAI/opencl/test/test_ocl_histo.py:                logger.warning("Decreasing precision on amdgpu")
src/pyFAI/opencl/test/test_ocl_histo.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_ocl_histo.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_addition.py:#    Project: Basic OpenCL test
src/pyFAI/opencl/test/test_addition.py:from .. import ocl, get_opencl_code
src/pyFAI/opencl/test/test_addition.py:    import pyopencl.array
src/pyFAI/opencl/test/test_addition.py:from silx.opencl.common import _measure_workgroup_size
src/pyFAI/opencl/test/test_addition.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_addition.py:@unittest.skipUnless(ocl, "PyOpenCl is missing")
src/pyFAI/opencl/test/test_addition.py:                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
src/pyFAI/opencl/test/test_addition.py:                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
src/pyFAI/opencl/test/test_addition.py:                cls.queue = pyopencl.CommandQueue(cls.ctx)
src/pyFAI/opencl/test/test_addition.py:                and cls.ctx.devices[0].type == pyopencl.device_type.GPU):
src/pyFAI/opencl/test/test_addition.py:                raise unittest.SkipTest("Skip test on Power9 GPU with PoCL driver")
src/pyFAI/opencl/test/test_addition.py:        self.d_array_img = pyopencl.array.to_device(self.queue, self.data)
src/pyFAI/opencl/test/test_addition.py:        self.d_array_5 = pyopencl.array.zeros_like(self.d_array_img) - 5
src/pyFAI/opencl/test/test_addition.py:        self.program = pyopencl.Program(self.ctx, get_opencl_code("addition")).build()
src/pyFAI/opencl/test/test_addition.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_addition.py:            d_array_result = pyopencl.array.empty_like(self.d_array_img)
src/pyFAI/opencl/test/test_addition.py:                max_valid_wg = self.program.addition.get_work_group_info(pyopencl.kernel_work_group_info.WORK_GROUP_SIZE, self.ctx.devices[0])
src/pyFAI/opencl/test/test_addition.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_collective.py:#    Project: Basic OpenCL test
src/pyFAI/opencl/test/test_collective.py:    import pyopencl.array
src/pyFAI/opencl/test/test_collective.py:from silx.opencl.common import _measure_workgroup_size
src/pyFAI/opencl/test/test_collective.py:from silx.opencl.utils import get_opencl_code
src/pyFAI/opencl/test/test_collective.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_collective.py:@unittest.skipUnless(ocl, "PyOpenCl is missing")
src/pyFAI/opencl/test/test_collective.py:                properties = pyopencl.command_queue_properties.PROFILING_ENABLE
src/pyFAI/opencl/test/test_collective.py:                cls.queue = pyopencl.CommandQueue(cls.ctx, properties=properties)
src/pyFAI/opencl/test/test_collective.py:                cls.queue = pyopencl.CommandQueue(cls.ctx)
src/pyFAI/opencl/test/test_collective.py:                and cls.ctx.devices[0].type == pyopencl.device_type.GPU):
src/pyFAI/opencl/test/test_collective.py:                raise unittest.SkipTest("Skip test on Power9 GPU with PoCL driver")
src/pyFAI/opencl/test/test_collective.py:        self.data_d = pyopencl.array.to_device(self.queue, self.data)
src/pyFAI/opencl/test/test_collective.py:        self.sum_d = pyopencl.array.zeros_like(self.data_d)
src/pyFAI/opencl/test/test_collective.py:        self.program = pyopencl.Program(self.ctx, get_opencl_code("pyfai:openCL/collective/reduction.cl")).build()
src/pyFAI/opencl/test/test_collective.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_collective.py:        # rec_workgroup = self.program.test_sum_int_reduction.get_work_group_info(pyopencl.kernel_work_group_info.WORK_GROUP_SIZE, self.ctx.devices[0])
src/pyFAI/opencl/test/test_collective.py:                                                          pyopencl.LocalMemory(4*wg))
src/pyFAI/opencl/test/test_collective.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_collective.py:                                                          pyopencl.LocalMemory(4*wg))
src/pyFAI/opencl/test/test_peak_finder.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_peak_finder.py:@unittest.skipUnless(ocl, "PyOpenCl is missing")
src/pyFAI/opencl/test/test_peak_finder.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_peak_finder.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_peak_finder.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_peak_finder.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/test_peak_finder.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/meson.build:    'test_openCL.py',
src/pyFAI/opencl/test/meson.build:  subdir: 'pyFAI/opencl/test'  # Folder relative to site-packages to install to
src/pyFAI/opencl/test/test_openCL.py:"test suite for OpenCL code"
src/pyFAI/opencl/test/test_openCL.py:    from .. import pyopencl, read_cl_file
src/pyFAI/opencl/test/test_openCL.py:    import pyopencl.array
src/pyFAI/opencl/test/test_openCL.py:    from pyopencl.elementwise import ElementwiseKernel
src/pyFAI/opencl/test/test_openCL.py:@unittest.skipIf(test_options.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_openCL.py:@unittest.skipIf(ocl is None, "OpenCL is not available")
src/pyFAI/opencl/test/test_openCL.py:        cls.tmp_dir = os.path.join(test_options.tempdir, "opencl")
src/pyFAI/opencl/test/test_openCL.py:                logger.info(f"OpenCL {method} has R={r}  (vs cython) for dataset {ds}")
src/pyFAI/opencl/test/test_openCL.py:                self.assertLess(r, 3, "Rwp=%.3f for OpenCL histogram processing of %s" % (r, ds))
src/pyFAI/opencl/test/test_openCL.py:    def test_OpenCL_sparse(self):
src/pyFAI/opencl/test/test_openCL.py:                logger.info(f"OpenCL {method} has R={r}  (vs cython) for dataset {ds}")
src/pyFAI/opencl/test/test_openCL.py:                self.assertLess(r, 3, "Rwp=%.3f for OpenCL histogram processing of %s" % (r, ds))
src/pyFAI/opencl/test/test_openCL.py:    def test_OpenCL_sigma_clip(self):
src/pyFAI/opencl/test/test_openCL.py:        logger.info("Testing OpenCL sigma-clipping")
src/pyFAI/opencl/test/test_openCL.py:                except (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError) as error:
src/pyFAI/opencl/test/test_openCL.py:                    logger.info("OpenCL sigma clipping has R= %.3f for dataset %s", r, ds)
src/pyFAI/opencl/test/test_openCL.py:                    self.assertLess(r, 3, "Rwp=%.3f for OpenCL CSR processing of %s" % (r, ds))
src/pyFAI/opencl/test/test_openCL.py:@unittest.skipIf(test_options.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_openCL.py:@unittest.skipIf(ocl is None, "OpenCL is not available")
src/pyFAI/opencl/test/test_openCL.py:        cls.ctx = ocl.create_context(devicetype="GPU")
src/pyFAI/opencl/test/test_openCL.py:            devtype = pyopencl.device_type.to_string(device.type).upper()
src/pyFAI/opencl/test/test_openCL.py:            logger.info("For Apple's OpenCL on CPU: enforce max_work_goup_size=1")
src/pyFAI/opencl/test/test_openCL.py:        cls.queue = pyopencl.CommandQueue(cls.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
src/pyFAI/opencl/test/test_openCL.py:        cls.local_mem = pyopencl.LocalMemory(cls.ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
src/pyFAI/opencl/test/test_openCL.py:        src = read_cl_file("pyfai:openCL/bitonic.cl")
src/pyFAI/opencl/test/test_openCL.py:        cls.prg = pyopencl.Program(cls.ctx, src).build()
src/pyFAI/opencl/test/test_openCL.py:            "cpu" in pyopencl.device_type.to_string(device.type).lower()):
src/pyFAI/opencl/test/test_openCL.py:            "cpu" in pyopencl.device_type.to_string(device.type).lower()):
src/pyFAI/opencl/test/test_openCL.py:        d_data = pyopencl.array.to_device(self.queue, self.h_data)
src/pyFAI/opencl/test/test_openCL.py:        d_data = pyopencl.array.to_device(self.queue, self.h_data)
src/pyFAI/opencl/test/test_openCL.py:        d_data = pyopencl.array.to_device(self.queue, self.h_data)
src/pyFAI/opencl/test/test_openCL.py:        d2_data = pyopencl.array.to_device(self.queue, self.h2_data)
src/pyFAI/opencl/test/test_openCL.py:        d2_data = pyopencl.array.to_device(self.queue, self.h2_data)
src/pyFAI/opencl/test/test_openCL.py:@unittest.skipIf(test_options.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_openCL.py:@unittest.skipIf(ocl is None, "OpenCL is not available")
src/pyFAI/opencl/test/test_openCL.py:    Test the kernels for compensated math in OpenCL
src/pyFAI/opencl/test/test_openCL.py:        cls.queue = pyopencl.CommandQueue(cls.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
src/pyFAI/opencl/test/test_openCL.py:            and cls.ctx.devices[0].type == pyopencl.device_type.GPU):
src/pyFAI/opencl/test/test_openCL.py:            raise unittest.SkipTest("Skip test on Power9 GPU with PoCL driver")
src/pyFAI/opencl/test/test_openCL.py:        # this is running 32 bits OpenCL with POCL
src/pyFAI/opencl/test/test_openCL.py:        prg = pyopencl.Program(self.ctx, read_cl_file("pyfai:openCL/kahan.cl") + src).build(self.args)
src/pyFAI/opencl/test/test_openCL.py:        ones_d = pyopencl.array.to_device(self.queue, data)
src/pyFAI/opencl/test/test_openCL.py:        res_d = pyopencl.array.zeros(self.queue, 2, numpy.float32)
src/pyFAI/opencl/test/test_openCL.py:        prg = pyopencl.Program(self.ctx, read_cl_file("pyfai:openCL/kahan.cl") + src).build(self.args)
src/pyFAI/opencl/test/test_openCL.py:        ones_d = pyopencl.array.to_device(self.queue, data)
src/pyFAI/opencl/test/test_openCL.py:        res_d = pyopencl.array.zeros(self.queue, 2, numpy.float32)
src/pyFAI/opencl/test/test_openCL.py:    Test the kernels for compensated math in OpenCL
src/pyFAI/opencl/test/test_openCL.py:        if not test_options.WITH_OPENCL_TEST:
src/pyFAI/opencl/test/test_openCL.py:            raise unittest.SkipTest("User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_openCL.py:            raise unittest.SkipTest("OpenCL module (pyopencl) is not present or no device available")
src/pyFAI/opencl/test/test_openCL.py:        cls.ctx = ocl.create_context(devicetype="GPU")
src/pyFAI/opencl/test/test_openCL.py:        cls.queue = pyopencl.CommandQueue(cls.ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
src/pyFAI/opencl/test/test_openCL.py:        # this is running 32 bits OpenCL woth POCL
src/pyFAI/opencl/test/test_openCL.py:        cls.doubleword = read_cl_file("silx:opencl/doubleword.cl")
src/pyFAI/opencl/test/test_openCL.py:        a_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        b_g = pyopencl.array.to_device(self.queue, self.bl)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(a_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(a_g)
src/pyFAI/opencl/test/test_openCL.py:        a_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        b_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(a_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(a_g)
src/pyFAI/opencl/test/test_openCL.py:        a_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        b_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(a_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(a_g)
src/pyFAI/opencl/test/test_openCL.py:        ah_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        al_g = pyopencl.array.to_device(self.queue, self.al)
src/pyFAI/opencl/test/test_openCL.py:        b_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(b_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(b_g)
src/pyFAI/opencl/test/test_openCL.py:        ah_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        al_g = pyopencl.array.to_device(self.queue, self.al)
src/pyFAI/opencl/test/test_openCL.py:        bh_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        bl_g = pyopencl.array.to_device(self.queue, self.bl)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(bh_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(bh_g)
src/pyFAI/opencl/test/test_openCL.py:        ah_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        al_g = pyopencl.array.to_device(self.queue, self.al)
src/pyFAI/opencl/test/test_openCL.py:        b_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(b_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(b_g)
src/pyFAI/opencl/test/test_openCL.py:        ah_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        al_g = pyopencl.array.to_device(self.queue, self.al)
src/pyFAI/opencl/test/test_openCL.py:        bh_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        bl_g = pyopencl.array.to_device(self.queue, self.bl)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(bh_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(bh_g)
src/pyFAI/opencl/test/test_openCL.py:        ah_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        al_g = pyopencl.array.to_device(self.queue, self.al)
src/pyFAI/opencl/test/test_openCL.py:        b_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(b_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(b_g)
src/pyFAI/opencl/test/test_openCL.py:        ah_g = pyopencl.array.to_device(self.queue, self.ah)
src/pyFAI/opencl/test/test_openCL.py:        al_g = pyopencl.array.to_device(self.queue, self.al)
src/pyFAI/opencl/test/test_openCL.py:        bh_g = pyopencl.array.to_device(self.queue, self.bh)
src/pyFAI/opencl/test/test_openCL.py:        bl_g = pyopencl.array.to_device(self.queue, self.bl)
src/pyFAI/opencl/test/test_openCL.py:        res_l = pyopencl.array.empty_like(bh_g)
src/pyFAI/opencl/test/test_openCL.py:        res_h = pyopencl.array.empty_like(bh_g)
src/pyFAI/opencl/test/test_preproc.py:from .. import ocl, get_opencl_code
src/pyFAI/opencl/test/test_preproc.py:    import pyopencl.array
src/pyFAI/opencl/test/test_preproc.py:@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
src/pyFAI/opencl/test/test_preproc.py:@unittest.skipUnless(ocl, "PyOpenCl is missing")
src/pyFAI/opencl/test/test_preproc.py:    @unittest.skipUnless(ocl, "pyopencl is missing")
src/pyFAI/opencl/test/__init__.py:    if UtilsTest.opencl:
src/pyFAI/opencl/test/__init__.py:        from . import test_openCL
src/pyFAI/opencl/test/__init__.py:        testSuite.addTests(test_openCL.suite())
src/pyFAI/opencl/ocl_hist_pixelsplit.py:    from . import pyopencl, allocate_cl_buffers, release_cl_buffers
src/pyFAI/opencl/ocl_hist_pixelsplit.py:    mf = pyopencl.mem_flags
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        :param devicetype: can be "cpu","gpu","acc" or "all"
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        self.pos0_range = numpy.zeros(1, pyopencl.array.vec.float2)
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        self.pos1_range = numpy.zeros(1, pyopencl.array.vec.float2)
src/pyFAI/opencl/ocl_hist_pixelsplit.py:            # self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
src/pyFAI/opencl/ocl_hist_pixelsplit.py:            self._ctx = pyopencl.create_some_context()
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                self._queue = pyopencl.CommandQueue(self._ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                self._queue = pyopencl.CommandQueue(self._ctx)
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        except pyopencl.MemoryError as error:
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["pos"], self.pos)
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        pyopencl.enqueue_copy(self._queue, result, self._cl_mem["minmax"])
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        Allocate OpenCL buffers required for a specific configuration
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        Note that an OpenCL context also requires some memory, as well
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        as Event and other OpenCL functionalities which cannot and are
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        for a 9300m is ~15Mb In addition, a GPU will always have at
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        least 3-5Mb of memory in use.  Unfortunately, OpenCL does NOT
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        Call the OpenCL compiler
src/pyFAI/opencl/ocl_hist_pixelsplit.py:            self._program = pyopencl.Program(self._ctx, kernel_src).build(options=compile_options)
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        except pyopencl.MemoryError as error:
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image_u16"], numpy.ascontiguousarray(data))
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image"], numpy.ascontiguousarray(data))
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                copy_image = pyopencl.enqueue_copy(self._queue, self._cl_mem["image"], numpy.ascontiguousarray(data, dtype=numpy.float32))
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["dark"], numpy.ascontiguousarray(dark, dtype=numpy.float32))
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["flat"], numpy.ascontiguousarray(flat, dtype=numpy.float32))
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["solidangle"], numpy.ascontiguousarray(solidAngle, dtype=numpy.float32))
src/pyFAI/opencl/ocl_hist_pixelsplit.py:                    ev = pyopencl.enqueue_copy(self._queue, self._cl_mem["polarization"], numpy.ascontiguousarray(polarization, dtype=numpy.float32))
src/pyFAI/opencl/ocl_hist_pixelsplit.py:            ev = pyopencl.enqueue_copy(self._queue, outData, self._cl_mem["outData"])
src/pyFAI/opencl/ocl_hist_pixelsplit.py:            ev = pyopencl.enqueue_copy(self._queue, outCount, self._cl_mem["outCount"])
src/pyFAI/opencl/ocl_hist_pixelsplit.py:            ev = pyopencl.enqueue_copy(self._queue, outMerge, self._cl_mem["outMerge"])
src/pyFAI/opencl/ocl_hist_pixelsplit.py:        If we are in profiling mode, prints out all timing for every single OpenCL call
src/pyFAI/opencl/azim_csr.py:from . import pyopencl, dtype_converter
src/pyFAI/opencl/azim_csr.py:if pyopencl:
src/pyFAI/opencl/azim_csr.py:    mf = pyopencl.mem_flags
src/pyFAI/opencl/azim_csr.py:    raise ImportError("pyopencl is not installed")
src/pyFAI/opencl/azim_csr.py:from . import processing, OpenclProcessing
src/pyFAI/opencl/azim_csr.py:class OCL_CSR_Integrator(OpenclProcessing):
src/pyFAI/opencl/azim_csr.py:    """Class in charge of doing a sparse-matrix multiplication in OpenCL
src/pyFAI/opencl/azim_csr.py:    kernel_files = ["silx:opencl/doubleword.cl",
src/pyFAI/opencl/azim_csr.py:                    "pyfai:openCL/preprocess.cl",
src/pyFAI/opencl/azim_csr.py:                    "pyfai:openCL/memset.cl",
src/pyFAI/opencl/azim_csr.py:                    "pyfai:openCL/ocl_azim_CSR.cl"
src/pyFAI/opencl/azim_csr.py:        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
src/pyFAI/opencl/azim_csr.py:        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
src/pyFAI/opencl/azim_csr.py:        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
src/pyFAI/opencl/azim_csr.py:            self.workgroup_size["csr_integrate4_single"] = (1, 1)  # Very bad performances on AMD GPU for diverging threads!
src/pyFAI/opencl/azim_csr.py:        :param block_size: Input workgroup size (block is the cuda name)
src/pyFAI/opencl/azim_csr.py:            devtype = pyopencl.device_type.to_string(device.type).upper()
src/pyFAI/opencl/azim_csr.py:            if "nvidia" in  platform:
src/pyFAI/opencl/azim_csr.py:        Call the OpenCL compiler
src/pyFAI/opencl/azim_csr.py:        # concatenate all needed source files into a single openCL module
src/pyFAI/opencl/azim_csr.py:        OpenclProcessing.compile_kernels(self, kernels, compile_options)
src/pyFAI/opencl/azim_csr.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/azim_csr.py:                                                            ("shared", pyopencl.LocalMemory(16))))
src/pyFAI/opencl/azim_csr.py:                                                            ("shared", pyopencl.LocalMemory(32))
src/pyFAI/opencl/azim_csr.py:                                                              ("shared", pyopencl.LocalMemory(32))
src/pyFAI/opencl/azim_csr.py:        :param convert: if True (default) convert dtype on GPU, if false, leave as it is.
src/pyFAI/opencl/azim_csr.py:        if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/azim_csr.py:                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, data.data)
src/pyFAI/opencl/azim_csr.py:                copy_image = pyopencl.enqueue_copy(self.queue, tmp_buffer, data.data)
src/pyFAI/opencl/azim_csr.py:                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, converted_data.data)
src/pyFAI/opencl/azim_csr.py:                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, numpy.ascontiguousarray(data, dest_type))
src/pyFAI/opencl/azim_csr.py:                copy_image = pyopencl.enqueue_copy(self.queue, tmp_buffer, numpy.ascontiguousarray(data))
src/pyFAI/opencl/azim_csr.py:                copy_image = pyopencl.enqueue_copy(self.queue, dest_buffer, converted_data)
src/pyFAI/opencl/azim_csr.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/azim_csr.py:        :param out_merged: destination array or pyopencl array for averaged data
src/pyFAI/opencl/azim_csr.py:        :param out_sum_data: destination array or pyopencl array for sum of all data
src/pyFAI/opencl/azim_csr.py:        :param out_sum_count: destination array or pyopencl array for sum of the number of pixels
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
src/pyFAI/opencl/azim_csr.py:                kw_int["shared"] = pyopencl.LocalMemory(16 * wg_min)  # sizeof(float4) == 16
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged"])
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, sum_data, self.cl_mem["sum_data"])
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, sum_count, self.cl_mem["sum_count"])
src/pyFAI/opencl/azim_csr.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/azim_csr.py:        :param out_avgint: destination array or pyopencl array for average intensity
src/pyFAI/opencl/azim_csr.py:        :param out_sem: destination array or pyopencl array for standard deviation (of mean)
src/pyFAI/opencl/azim_csr.py:        :param out_std: destination array or pyopencl array for standard deviation (of pixels)
src/pyFAI/opencl/azim_csr.py:        :param out_merged: destination array or pyopencl array for averaged data (float8!)
src/pyFAI/opencl/azim_csr.py:                kw_int["shared"] = pyopencl.LocalMemory(32 * wg_min)  # sizeof(float8) == 32
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, avgint, self.cl_mem["averint"])
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
src/pyFAI/opencl/azim_csr.py:                    ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
src/pyFAI/opencl/azim_csr.py:                    ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
src/pyFAI/opencl/azim_csr.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/azim_csr.py:        :param out_avgint: destination array or pyopencl array for sum of all data
src/pyFAI/opencl/azim_csr.py:        :param out_sem: destination array or pyopencl array for uncertainty on mean value
src/pyFAI/opencl/azim_csr.py:        :param out_std: destination array or pyopencl array for uncertainty on pixel value
src/pyFAI/opencl/azim_csr.py:        :param out_merged: destination array or pyopencl array for averaged data (float8!)
src/pyFAI/opencl/azim_csr.py:            kw_int["shared"] = pyopencl.LocalMemory(32 * wg_min)
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, avgint, self.cl_mem["averint"])
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
src/pyFAI/opencl/azim_csr.py:                ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
src/pyFAI/opencl/azim_csr.py:            ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
src/pyFAI/opencl/OCLFullSplit.py:from . import ocl, pyopencl
src/pyFAI/opencl/OCLFullSplit.py:if pyopencl:
src/pyFAI/opencl/OCLFullSplit.py:    mf = pyopencl.mem_flags
src/pyFAI/opencl/OCLFullSplit.py:    raise ImportError("pyopencl is not installed")
src/pyFAI/opencl/OCLFullSplit.py:            self.pos0_range[0] = min(pos0_range)  # do it on GPU?
src/pyFAI/opencl/OCLFullSplit.py:            self.pos1_range[0] = min(pos1_range)  # do it on GPU?
src/pyFAI/opencl/OCLFullSplit.py:            logger.warning("This is a workaround for Apple's OpenCL on CPU: enforce BLOCK_SIZE=1")
src/pyFAI/opencl/OCLFullSplit.py:            self._ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[platformid].get_devices()[deviceid]])
src/pyFAI/opencl/OCLFullSplit.py:                self._queue = pyopencl.CommandQueue(self._ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
src/pyFAI/opencl/OCLFullSplit.py:                self._queue = pyopencl.CommandQueue(self._ctx)
src/pyFAI/opencl/OCLFullSplit.py:        except pyopencl.MemoryError as error:
src/pyFAI/opencl/OCLFullSplit.py:        Call the OpenCL compiler
src/pyFAI/opencl/OCLFullSplit.py:                kernel_file = get_cl_file("pyfai:openCL/" + kernel_name)
src/pyFAI/opencl/OCLFullSplit.py:            self._program = pyopencl.Program(self._ctx, kernel_src).build(options=compile_options)
src/pyFAI/opencl/OCLFullSplit.py:        except pyopencl.MemoryError as error:
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["pos"] = pyopencl.Buffer(self._ctx, mf.READ_ONLY, size_of_float * self.pos_size)
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["preresult"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * 4 * self.workgroup_size)
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["minmax"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * 4)
src/pyFAI/opencl/OCLFullSplit.py:        except pyopencl.MemoryError as error:
src/pyFAI/opencl/OCLFullSplit.py:            copy_pos = pyopencl.enqueue_copy(self._queue, self._cl_mem["pos"], self.pos)
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["outMax"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * self.bins)
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["lutsize"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * 1)
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["idx_ptr"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * (self.bins + 1))
src/pyFAI/opencl/OCLFullSplit.py:        except pyopencl.MemoryError as error:
src/pyFAI/opencl/OCLFullSplit.py:            get_lutsize = pyopencl.enqueue_copy(self._queue, self.lutsize, self._cl_mem["lutsize"])
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["indices"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_int * self.lutsize[0])
src/pyFAI/opencl/OCLFullSplit.py:            self._cl_mem["data"] = pyopencl.Buffer(self._ctx, mf.READ_WRITE, size_of_float * self.lutsize[0])
src/pyFAI/opencl/OCLFullSplit.py:        except pyopencl.MemoryError as error:
src/pyFAI/opencl/OCLFullSplit.py:                except pyopencl.LogicError:
src/pyFAI/opencl/peak_finder.py:from .azim_csr import OCL_CSR_Integrator, BufferDescription, EventDescription, mf, calc_checksum, pyopencl, OpenclProcessing
src/pyFAI/opencl/peak_finder.py:    kernel_files = ["silx:opencl/doubleword.cl",
src/pyFAI/opencl/peak_finder.py:                    "pyfai:openCL/preprocess.cl",
src/pyFAI/opencl/peak_finder.py:                    "pyfai:openCL/memset.cl",
src/pyFAI/opencl/peak_finder.py:                    "pyfai:openCL/ocl_azim_CSR.cl",
src/pyFAI/opencl/peak_finder.py:                    "pyfai:openCL/sparsify.cl",
src/pyFAI/opencl/peak_finder.py:                    "pyfai:openCL/peakfinder.cl",
src/pyFAI/opencl/peak_finder.py:        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
src/pyFAI/opencl/peak_finder.py:        :param block_size: Input workgroup size (block is the cuda name)
src/pyFAI/opencl/peak_finder.py:        Note: this function does not lock the OpenCL context!
src/pyFAI/opencl/peak_finder.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/peak_finder.py:        kw_int["shared"] = pyopencl.LocalMemory(32 * wg_min)
src/pyFAI/opencl/peak_finder.py:        *this function does not lock the OpenCL context!
src/pyFAI/opencl/peak_finder.py:        :param events: list of OpenCL events for timing
src/pyFAI/opencl/peak_finder.py:        kw_proj["shared"] = pyopencl.LocalMemory(wg * 4)  # stores int
src/pyFAI/opencl/peak_finder.py:        ev = pyopencl.enqueue_copy(self.queue, cnt, self.cl_mem["counter"])
src/pyFAI/opencl/peak_finder.py:        Note: this function does not lock the OpenCL context!
src/pyFAI/opencl/peak_finder.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/peak_finder.py:        kw_proj["local_highidx"] = pyopencl.LocalMemory(1 * buffer_size)
src/pyFAI/opencl/peak_finder.py:        kw_proj["local_peaks"] = pyopencl.LocalMemory(4 * buffer_size)
src/pyFAI/opencl/peak_finder.py:        kw_proj["local_buffer"] = pyopencl.LocalMemory(8 * (wg0 + 2 * hw) * (wg1 + 2 * hw))
src/pyFAI/opencl/peak_finder.py:        ev = pyopencl.enqueue_copy(self.queue, cnt, self.cl_mem["counter"])
src/pyFAI/opencl/peak_finder.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/peak_finder.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/peak_finder.py:            ev1 = pyopencl.enqueue_copy(self.queue, background_avg, self.cl_mem["averint"])
src/pyFAI/opencl/peak_finder.py:            ev2 = pyopencl.enqueue_copy(self.queue, background_std, self.cl_mem["std"])
src/pyFAI/opencl/peak_finder.py:                    ev_idx = pyopencl.enqueue_copy(self.queue, index, self.cl_mem["position"])
src/pyFAI/opencl/peak_finder.py:                    ev_pks = pyopencl.enqueue_copy(self.queue, peak4, self.cl_mem["descriptor"])
src/pyFAI/opencl/peak_finder.py:                ev1 = pyopencl.enqueue_copy(self.queue, indexes, self.cl_mem["position"])
src/pyFAI/opencl/peak_finder.py:                ev2 = pyopencl.enqueue_copy(self.queue, signal, self.cl_mem["descriptor"])
src/pyFAI/opencl/peak_finder.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/peak_finder.py:                idxp = pyopencl.enqueue_copy(self.queue, index, self.cl_mem["position"])
src/pyFAI/opencl/peak_finder.py:                idxd = pyopencl.enqueue_copy(self.queue, peaks, self.cl_mem["descriptor"])
src/pyFAI/opencl/peak_finder.py:class OCL_SimplePeakFinder(OpenclProcessing):
src/pyFAI/opencl/peak_finder.py:    kernel_files = ["pyfai:openCL/simple_peak_picker.cl"]
src/pyFAI/opencl/peak_finder.py:        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
src/pyFAI/opencl/peak_finder.py:        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
src/pyFAI/opencl/peak_finder.py:        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
src/pyFAI/opencl/peak_finder.py:        Call the OpenCL compiler
src/pyFAI/opencl/peak_finder.py:        # concatenate all needed source files into a single openCL module
src/pyFAI/opencl/peak_finder.py:        OpenclProcessing.compile_kernels(self, kernels, compile_options)
src/pyFAI/opencl/peak_finder.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/peak_finder.py:                                                                 ("local_high", pyopencl.LocalMemory(self.BLOCK_SIZE * 4)),
src/pyFAI/opencl/peak_finder.py:        if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/peak_finder.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], data.data)
src/pyFAI/opencl/peak_finder.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], data.data)
src/pyFAI/opencl/peak_finder.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], numpy.ascontiguousarray(data, dest_type))
src/pyFAI/opencl/peak_finder.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], numpy.ascontiguousarray(data))
src/pyFAI/opencl/peak_finder.py:        copy_count = pyopencl.enqueue_copy(self.queue, count, self.cl_mem["count"])
src/pyFAI/opencl/peak_finder.py:            copy_index = pyopencl.enqueue_copy(self.queue, indexes, self.cl_mem["output"])
src/pyFAI/opencl/peak_finder.py:            copy_value = pyopencl.enqueue_copy(self.queue, values, self.cl_mem["peak_intensity"])
src/pyFAI/opencl/preproc.py:OpenCL implementation of the preproc module
src/pyFAI/opencl/preproc.py:from . import pyopencl
src/pyFAI/opencl/preproc.py:if pyopencl is None:
src/pyFAI/opencl/preproc.py:    raise ImportError("pyopencl is not installed")
src/pyFAI/opencl/preproc.py:from . import mf, processing, OpenclProcessing, dtype_converter
src/pyFAI/opencl/preproc.py:class OCL_Preproc(OpenclProcessing):
src/pyFAI/opencl/preproc.py:    """OpenCL class for pre-processing ... mainly for demonstration"""
src/pyFAI/opencl/preproc.py:    kernel_files = ["pyfai:openCL/preprocess.cl"]
src/pyFAI/opencl/preproc.py:        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
src/pyFAI/opencl/preproc.py:        OpenclProcessing.__init__(self, ctx, devicetype, platformid, deviceid, block_size, profile)
src/pyFAI/opencl/preproc.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/preproc.py:        """Call the OpenCL compiler
src/pyFAI/opencl/preproc.py:        # concatenate all needed source files into a single openCL module
src/pyFAI/opencl/preproc.py:        OpenclProcessing.compile_kernels(self, kernel_files, compile_options)
src/pyFAI/opencl/preproc.py:        :param convert: if True (default) convert dtype on GPU, if false, leave as it is in buffer named `dest_raw`
src/pyFAI/opencl/preproc.py:                copy_image = pyopencl.enqueue_copy(self.queue, actual_dest, numpy.ascontiguousarray(data, dest_type))
src/pyFAI/opencl/preproc.py:                copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["temp"], numpy.ascontiguousarray(data))
src/pyFAI/opencl/preproc.py:            copy_image = pyopencl.enqueue_copy(self.queue, actual_dest, numpy.ascontiguousarray(data))
src/pyFAI/opencl/preproc.py:            copy_result = pyopencl.enqueue_copy(self.queue, dest, self.cl_mem["output"])
src/pyFAI/opencl/preproc.py:    r"""Common preprocessing step, implemented using OpenCL. May be inefficient
src/pyFAI/opencl/meson.build:  subdir: 'pyFAI/opencl'  # Folder relative to site-packages to install to
src/pyFAI/opencl/azim_lut.py:from . import pyopencl, dtype_converter
src/pyFAI/opencl/azim_lut.py:if pyopencl:
src/pyFAI/opencl/azim_lut.py:    mf = pyopencl.mem_flags
src/pyFAI/opencl/azim_lut.py:    raise ImportError("pyopencl is not installed")
src/pyFAI/opencl/azim_lut.py:from . import processing, OpenclProcessing
src/pyFAI/opencl/azim_lut.py:class OCL_LUT_Integrator(OpenclProcessing):
src/pyFAI/opencl/azim_lut.py:    """Class in charge of doing a sparse-matrix multiplication in OpenCL
src/pyFAI/opencl/azim_lut.py:    kernel_files = ["silx:opencl/doubleword.cl",
src/pyFAI/opencl/azim_lut.py:                    "pyfai:openCL/preprocess.cl",
src/pyFAI/opencl/azim_lut.py:                    "pyfai:openCL/memset.cl",
src/pyFAI/opencl/azim_lut.py:                    "pyfai:openCL/ocl_azim_LUT.cl"
src/pyFAI/opencl/azim_lut.py:        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
src/pyFAI/opencl/azim_lut.py:        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
src/pyFAI/opencl/azim_lut.py:            ev = pyopencl.enqueue_copy(self.queue, self.cl_mem["lut"], lut)
src/pyFAI/opencl/azim_lut.py:            ev = pyopencl.enqueue_copy(self.queue, self.cl_mem["lut"], lut.T.copy())
src/pyFAI/opencl/azim_lut.py:        Call the OpenCL compiler
src/pyFAI/opencl/azim_lut.py:        # concatenate all needed source files into a single openCL module
src/pyFAI/opencl/azim_lut.py:        OpenclProcessing.compile_kernels(self, kernels, compile_options)
src/pyFAI/opencl/azim_lut.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/azim_lut.py:        :param convert: if True (default) convert dtype on GPU, if false, leave as it is.
src/pyFAI/opencl/azim_lut.py:            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem[dest], numpy.ascontiguousarray(data, dest_type))
src/pyFAI/opencl/azim_lut.py:            copy_image = pyopencl.enqueue_copy(self.queue, self.cl_mem["image_raw"], numpy.ascontiguousarray(data))
src/pyFAI/opencl/azim_lut.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/azim_lut.py:        :param out_merged: destination array or pyopencl array for averaged data
src/pyFAI/opencl/azim_lut.py:        :param out_sum_data: destination array or pyopencl array for sum of all data
src/pyFAI/opencl/azim_lut.py:        :param out_sum_count: destination array or pyopencl array for sum of the number of pixels
src/pyFAI/opencl/azim_lut.py:                ev = pyopencl.enqueue_copy(self.queue, image, self.cl_mem["output"])
src/pyFAI/opencl/azim_lut.py:                ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged"])
src/pyFAI/opencl/azim_lut.py:                ev = pyopencl.enqueue_copy(self.queue, sum_data, self.cl_mem["sum_data"])
src/pyFAI/opencl/azim_lut.py:                ev = pyopencl.enqueue_copy(self.queue, sum_count, self.cl_mem["sum_count"])
src/pyFAI/opencl/azim_lut.py:        :param safe: if True (default) compares arrays on GPU according to their checksum, unless, use the buffer location is used
src/pyFAI/opencl/azim_lut.py:        :param out_avgint: destination array or pyopencl array for average intensity
src/pyFAI/opencl/azim_lut.py:        :param out_sem: destination array or pyopencl array for standard deviation (of mean)
src/pyFAI/opencl/azim_lut.py:        :param out_std: destination array or pyopencl array for standard deviation (of pixels)
src/pyFAI/opencl/azim_lut.py:        :param out_merged: destination array or pyopencl array for averaged data (float8!)
src/pyFAI/opencl/azim_lut.py:            ev = pyopencl.enqueue_copy(self.queue, avgint, self.cl_mem["averint"])
src/pyFAI/opencl/azim_lut.py:                ev = pyopencl.enqueue_copy(self.queue, std, self.cl_mem["std"])
src/pyFAI/opencl/azim_lut.py:                ev = pyopencl.enqueue_copy(self.queue, sem, self.cl_mem["sem"])
src/pyFAI/opencl/azim_lut.py:                    ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
src/pyFAI/opencl/azim_lut.py:                    ev = pyopencl.enqueue_copy(self.queue, merged, self.cl_mem["merged8"])
src/pyFAI/opencl/__init__.py:"""Contains all OpenCL implementation."""
src/pyFAI/opencl/__init__.py:if not pyFAI.use_opencl:
src/pyFAI/opencl/__init__.py:    pyopencl = None
src/pyFAI/opencl/__init__.py:elif os.environ.get("PYFAI_OPENCL") in ["0", "False"]:
src/pyFAI/opencl/__init__.py:    logger.info("Use of OpenCL has been disables from environment variable: PYFAI_OPENCL=0")
src/pyFAI/opencl/__init__.py:    pyopencl = None
src/pyFAI/opencl/__init__.py:    OpenclProcessing = None
src/pyFAI/opencl/__init__.py:    from silx.opencl import common
src/pyFAI/opencl/__init__.py:    from silx.opencl.common import pyopencl, mf, release_cl_buffers, allocate_cl_buffers, \
src/pyFAI/opencl/__init__.py:    from silx.opencl import utils
src/pyFAI/opencl/__init__.py:    from silx.opencl.utils import get_opencl_code, concatenate_cl_kernel, read_cl_file
src/pyFAI/opencl/__init__.py:    from silx.opencl import processing
src/pyFAI/opencl/__init__.py:    OpenclProcessing = processing.OpenclProcessing
src/pyFAI/opencl/__init__.py:        # this is running 32 bits OpenCL with POCL
src/pyFAI/opencl/sort.py:Module for 2D sort based on OpenCL for median filtering and Bragg/amorphous
src/pyFAI/opencl/sort.py:separation on GPU.
src/pyFAI/opencl/sort.py:    import pyopencl.array
src/pyFAI/opencl/sort.py:    from . import processing, OpenclProcessing
src/pyFAI/opencl/sort.py:    raise ImportError("pyopencl is not installed or no device is available")
src/pyFAI/opencl/sort.py:class Separator(OpenclProcessing):
src/pyFAI/opencl/sort.py:    Implementation of sort, median filter and trimmed-mean in  pyopencl
src/pyFAI/opencl/sort.py:    kernel_files = ["silx:opencl/doubleword.cl",
src/pyFAI/opencl/sort.py:                    "pyfai:openCL/bitonic.cl",
src/pyFAI/opencl/sort.py:                    "pyfai:openCL/separate.cl",
src/pyFAI/opencl/sort.py:                    "pyfai:openCL/sigma_clip.cl"]
src/pyFAI/opencl/sort.py:        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
src/pyFAI/opencl/sort.py:        lst = ["OpenCL implementation of sort/median_filter/trimmed_mean"]
src/pyFAI/opencl/sort.py:        Allocate OpenCL buffers required for a specific configuration
src/pyFAI/opencl/sort.py:        Note that an OpenCL context also requires some memory, as well
src/pyFAI/opencl/sort.py:        as Event and other OpenCL functionalities which cannot and are
src/pyFAI/opencl/sort.py:        for a 9300m is ~15Mb In addition, a GPU will always have at
src/pyFAI/opencl/sort.py:        least 3-5Mb of memory in use.  Unfortunately, OpenCL does NOT
src/pyFAI/opencl/sort.py:                    mem[name] = pyopencl.array.Array(self.queue, shape=shape, dtype=dtype)
src/pyFAI/opencl/sort.py:            except pyopencl.MemoryError as error:
src/pyFAI/opencl/sort.py:        """Tie arguments of OpenCL kernel-functions to the actual kernels
src/pyFAI/opencl/sort.py:        :param data: numpy or pyopencl array
src/pyFAI/opencl/sort.py:        :return: pyopencl array
src/pyFAI/opencl/sort.py:            if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:            if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:                evt = pyopencl.enqueue(data.queue, self.cl_mem["input_data"].data, data.data)
src/pyFAI/opencl/sort.py:            kargs["l_data"] = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
src/pyFAI/opencl/sort.py:        :param data: numpy or pyopencl array
src/pyFAI/opencl/sort.py:        :return: pyopencl array
src/pyFAI/opencl/sort.py:#             if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:        if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:            evt = pyopencl.enqueue(data.queue, self.cl_mem["input_data"].data, data.data)
src/pyFAI/opencl/sort.py:            kargs["l_data"] = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
src/pyFAI/opencl/sort.py:        :param data: numpy or pyopencl array
src/pyFAI/opencl/sort.py:        :return: pyopencl array
src/pyFAI/opencl/sort.py:        :param data: numpy or pyopencl array
src/pyFAI/opencl/sort.py:        :return: pyopencl array
src/pyFAI/opencl/sort.py:        :param data: numpy or pyopencl array
src/pyFAI/opencl/sort.py:        :return: pyopencl array
src/pyFAI/opencl/sort.py:        :param data: numpy or pyopencl array
src/pyFAI/opencl/sort.py:        :return: pyopencl array
src/pyFAI/opencl/sort.py:            if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
src/pyFAI/opencl/sort.py:                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
src/pyFAI/opencl/sort.py:            if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
src/pyFAI/opencl/sort.py:                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
src/pyFAI/opencl/sort.py:            if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
src/pyFAI/opencl/sort.py:                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
src/pyFAI/opencl/sort.py:            if isinstance(data, pyopencl.array.Array):
src/pyFAI/opencl/sort.py:                evt = pyopencl.enqueue_copy(data.queue, self.cl_mem["input_data"].data, data.data)
src/pyFAI/opencl/sort.py:                kargs["l_data"] = pyopencl.LocalMemory(wg * 20)  # 5 float per thread
src/pyFAI/geometry/core.py:        :param with_checksum: calculate also the checksum (used with OpenCL integration)
src/pyFAI/diffmap.py:from .opencl import ocl
src/pyFAI/diffmap.py:        parser.add_argument("-g", "--gpu", dest="gpu", action="store_true",
src/pyFAI/diffmap.py:                            help="process using OpenCL on GPU ", default=False)
src/pyFAI/diffmap.py:        if ocl and options.gpu:
src/pyFAI/diffmap.py:            ai["opencl_device"] = ocl.select_device(type="gpu")
src/pyFAI/diffmap.py:                method[2] = "opencl"
src/pyFAI/diffmap.py:                method[3] = "opencl"
src/pyFAI/diffmap.py:    def get_use_gpu(self):
src/pyFAI/diffmap.py:        return self.worker._method.impl_lower == "opencl"
src/pyFAI/diffmap.py:    def set_use_gpu(self, value):
src/pyFAI/diffmap.py:                method = self.worker._method.method.fixed("opencl")
src/pyFAI/diffmap.py:    use_gpu = property(get_use_gpu, set_use_gpu)
src/pyFAI/method_registry.py:        if self.impl == "opencl":
src/pyFAI/method_registry.py:            "ocl_gpu", "ocl_0,0"
src/pyFAI/method_registry.py:            impl = "opencl"
src/pyFAI/method_registry.py:                elif target_string == "gpu":
src/pyFAI/method_registry.py:                    target = "gpu"
src/pyFAI/method_registry.py:    AVAILABLE_IMPLS = ("python", "cython", "opencl")
src/pyFAI/method_registry.py:        if target in ["cpu", "gpu", None]:
src/pyFAI/method_registry.py:        :param impl: "python", "cython" or "opencl" to describe the implementation
src/pyFAI/method_registry.py:            if impl == "opencl":
src/pyFAI/method_registry.py:                if len(smth) >= 4 and impl == "opencl":
src/pyFAI/method_registry.py:        :param impl: "python", "cython" or "opencl" to describe the implementation
src/pyFAI/method_registry.py:        :param target: the OpenCL device as 2-tuple of indices
src/pyFAI/method_registry.py:        :param target_name: Full name of the OpenCL device
src/pyFAI/resources/gui/worker-configurator.ui:       <widget class="QLabel" name="opencl_title">
src/pyFAI/resources/gui/worker-configurator.ui:         <string>OpenCL device:</string>
src/pyFAI/resources/gui/worker-configurator.ui:       <widget class="QToolButton" name="opencl_config_button">
src/pyFAI/resources/gui/worker-configurator.ui:       <widget class="OpenClDeviceLabel" name="opencl_label">
src/pyFAI/resources/gui/worker-configurator.ui:   <class>OpenClDeviceLabel</class>
src/pyFAI/resources/gui/worker-configurator.ui:   <header>pyFAI.gui.widgets.OpenClDeviceLabel</header>
src/pyFAI/resources/gui/worker-configurator.ui:  <tabstop>opencl_config_button</tabstop>
src/pyFAI/resources/gui/calibration-geometry.ui:           <widget class="WaitingPushButton" name="_resetButton">
src/pyFAI/resources/gui/calibration-geometry.ui:           <widget class="WaitingPushButton" name="_fitButton">
src/pyFAI/resources/gui/calibration-geometry.ui:   <class>WaitingPushButton</class>
src/pyFAI/resources/gui/calibration-geometry.ui:   <header>silx.gui.widgets.WaitingPushButton</header>
src/pyFAI/resources/gui/meson.build:    'opencl-device-dialog.ui',
src/pyFAI/resources/gui/opencl-device-dialog.ui:   <string>Select an OpenCL device</string>
src/pyFAI/resources/gui/opencl-device-dialog.ui:      <string>OpenCL platform index in the targetted machine</string>
src/pyFAI/resources/gui/opencl-device-dialog.ui:      <string>OpenCL device index in the targetted machine</string>
src/pyFAI/resources/gui/opencl-device-dialog.ui:    <widget class="QRadioButton" name="_anyGpuButton">
src/pyFAI/resources/gui/opencl-device-dialog.ui:      <string>Use any GPU</string>
src/pyFAI/resources/gui/opencl-device-dialog.ui:  <tabstop>_anyGpuButton</tabstop>
src/pyFAI/resources/openCL/sparsify.cl: *            OpenCL Kernels
src/pyFAI/resources/openCL/ocl_azim_LUT.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/ocl_azim_LUT.cl: * \brief OpenCL kernels for 1D azimuthal integration
src/pyFAI/resources/openCL/ocl_azim_LUT.cl: *   ON_CPU: 0 for GPU, 1 for CPU and probably Xeon Phi
src/pyFAI/resources/openCL/ocl_azim_LUT.cl:                //On GPU best performances are obtained  when threads are reading adjacent memory
src/pyFAI/resources/openCL/ocl_azim_LUT.cl: * \brief OpenCL function for 1d azimuthal integration based on LUT matrix multiplication after normalization !
src/pyFAI/resources/openCL/ocl_azim_LUT.cl:                //On GPU best performances are obtained  when threads are reading adjacent memory
src/pyFAI/resources/openCL/peakfinder.cl: *            OpenCL Kernels
src/pyFAI/resources/openCL/collective/meson.build:  subdir: 'pyFAI/resources/openCL/collective'  # Folder relative to site-packages to install to
src/pyFAI/resources/openCL/kahan.cl: * OpenCL library for 32-bits floating point calculation using compensated arithmetics
src/pyFAI/resources/openCL/ocl_azim_CSR.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/ocl_azim_CSR.cl: * \brief OpenCL kernels for 1D azimuthal integration using CSR sparse matrix representation
src/pyFAI/resources/openCL/ocl_azim_CSR.cl: * \brief OpenCL workgroup function for sparse matrix-dense vector multiplication
src/pyFAI/resources/openCL/ocl_azim_CSR.cl: * \brief OpenCL function for 1d azimuthal integration based on CSR matrix multiplication
src/pyFAI/resources/openCL/ocl_azim_CSR.cl: * \brief CSRxVec4 OpenCL function for 1d azimuthal integration based on CSR matrix multiplication after normalization !
src/pyFAI/resources/openCL/ocl_azim_CSR.cl: * \brief CSRxVec4_single OpenCL function for 1d azimuthal integration based on CSR matrix multiplication after normalization !
src/pyFAI/resources/openCL/ocl_azim_CSR.cl: * \brief OpenCL function for sigma clipping CSR look up table. Sets count to NAN
src/pyFAI/resources/openCL/ocl_lut_pixelsplit2.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/simple_peak_picker.cl: *            OpenCL Kernels
src/pyFAI/resources/openCL/memset.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/memset.cl: * \brief OpenCL kernels for image array casting, array mem-setting and normalizing
src/pyFAI/resources/openCL/for_eclipse.h:#ifndef __OPENCL_VERSION__
src/pyFAI/resources/openCL/ocl_lut_pixelsplit.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/ocl_lut_pixelsplit3.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/meson.build:  subdir: 'pyFAI/resources/openCL'  # Folder relative to site-packages to install to
src/pyFAI/resources/openCL/ocl_histo.cl: *   Project: Azimuthal regrouping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/ocl_histo.cl:#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
src/pyFAI/resources/openCL/ocl_histo.cl:#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
src/pyFAI/resources/openCL/ocl_lut_pixelsplit_test.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/ocl_hist_pixelsplit.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/bitonic.cl:# Taken from his book "OpenCL in Action",
src/pyFAI/resources/openCL/bitonic.cl:// The _FILE extension correspond to the formula found in the "OpenCL in Action" supplementary files
src/pyFAI/resources/openCL/preprocess.cl: *   Project: Azimuthal regroupping OpenCL kernel for PyFAI.
src/pyFAI/resources/openCL/preprocess.cl: * \brief OpenCL kernels for image array casting, array mem-setting and normalizing
src/pyFAI/resources/meson.build:subdir('openCL')
src/pyFAI/meson.build:subdir('opencl')
src/pyFAI/benchmark/__init__.py:from ..opencl import pyopencl, ocl
src/pyFAI/benchmark/__init__.py:                    if isinstance(method, Method) and method.impl == "opencl":
src/pyFAI/benchmark/__init__.py:class BenchTestGpu(BenchTest):
src/pyFAI/benchmark/__init__.py:    """Test XRPD in OpenCL"""
src/pyFAI/benchmark/__init__.py:        self.ai.xrpd_OpenCL(self.data, self.N, devicetype=self.devicetype, useFp64=self.useFp64, platformid=self.platformid, deviceid=self.deviceid)
src/pyFAI/benchmark/__init__.py:        return self.ai.xrpd_OpenCL(self.data, self.N, safe=False)
src/pyFAI/benchmark/__init__.py:              ("bbox", "lut", "opencl"): "LUT",
src/pyFAI/benchmark/__init__.py:              ("bbox", "csr", "opencl"): "CSR",
src/pyFAI/benchmark/__init__.py:                        "gpu": self.get_gpu()}
src/pyFAI/benchmark/__init__.py:    def get_gpu(self, devicetype="gpu", useFp64=False, platformid=None, deviceid=None):
src/pyFAI/benchmark/__init__.py:            return "NoGPU"
src/pyFAI/benchmark/__init__.py:            return "NoGPU"
src/pyFAI/benchmark/__init__.py:    def bench_1d(self, method="splitBBox", check=False, opencl=None, function="integrate1d"):
src/pyFAI/benchmark/__init__.py:        :param opencl: dict containing platformid, deviceid and devicetype
src/pyFAI/benchmark/__init__.py:        if opencl:
src/pyFAI/benchmark/__init__.py:                print("No pyopencl")
src/pyFAI/benchmark/__init__.py:            if (opencl.get("platformid") is None) or (opencl.get("deviceid") is None):
src/pyFAI/benchmark/__init__.py:                platdev = ocl.select_device(opencl.get("devicetype"))
src/pyFAI/benchmark/__init__.py:                    print("No such OpenCL device: skipping benchmark")
src/pyFAI/benchmark/__init__.py:                platformid, deviceid = opencl["platformid"], opencl["deviceid"] = platdev
src/pyFAI/benchmark/__init__.py:                platformid, deviceid = opencl["platformid"], opencl["deviceid"]
src/pyFAI/benchmark/__init__.py:            devicetype = opencl["devicetype"] = ocl.platforms[platformid].devices[deviceid].type
src/pyFAI/benchmark/__init__.py:                                                      target=(opencl["platformid"], opencl["deviceid"]))[0]
src/pyFAI/benchmark/__init__.py:            memory_error = (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError)
src/pyFAI/benchmark/__init__.py:                        if opencl:
src/pyFAI/benchmark/__init__.py:    def bench_2d(self, method="splitBBox", check=False, opencl=None):
src/pyFAI/benchmark/__init__.py:        if opencl:
src/pyFAI/benchmark/__init__.py:                print("No pyopencl")
src/pyFAI/benchmark/__init__.py:            if (opencl.get("platformid") is None) or (opencl.get("deviceid") is None):
src/pyFAI/benchmark/__init__.py:                platdev = ocl.select_device(opencl.get("devicetype"))
src/pyFAI/benchmark/__init__.py:                    print("No such OpenCL device: skipping benchmark")
src/pyFAI/benchmark/__init__.py:                platformid, deviceid = opencl["platformid"], opencl["deviceid"] = platdev
src/pyFAI/benchmark/__init__.py:                platformid, deviceid = opencl["platformid"], opencl["deviceid"]
src/pyFAI/benchmark/__init__.py:            memory_error = (pyopencl.MemoryError, MemoryError, pyopencl.RuntimeError, RuntimeError)
src/pyFAI/benchmark/__init__.py:    def bench_gpu1d(self, devicetype="gpu", useFp64=True, platformid=None, deviceid=None):
src/pyFAI/benchmark/__init__.py:            print("No pyopencl or no such device: skipping benchmark")
src/pyFAI/benchmark/__init__.py:        label = "Forward_OpenCL_%s_%s_bits" % (devicetype, ("64" if useFp64 else"32"))
src/pyFAI/benchmark/__init__.py:                res = ai.xrpd_OpenCL(data, N, devicetype=devicetype, useFp64=useFp64, platformid=platformid, deviceid=deviceid)
src/pyFAI/benchmark/__init__.py:                print("Failed to find an OpenCL GPU (useFp64:%s) %s" % (useFp64, error))
src/pyFAI/benchmark/__init__.py:            test = BenchTestGpu(param, file_name, devicetype, useFp64, platformid, deviceid)
src/pyFAI/benchmark/__init__.py:            self.ax.set_title(f'CPU: {self.get_cpu()}\nGPU: {self.get_gpu()}')
src/pyFAI/benchmark/__init__.py:                  split_list=["bbox"], algo_list=["histogram", "CSR"], impl_list=["cython", "opencl"], function="all",
src/pyFAI/benchmark/__init__.py:    :devices: "all", "cpu", "gpu" or "acc" or a list of devices [(proc_id, dev_id)]
src/pyFAI/benchmark/__init__.py:                    if "gpu" in devices:
src/pyFAI/benchmark/__init__.py:                        ocl_devices += [(i.id, j.id) for j in i.devices if j.type == "GPU"]
src/pyFAI/benchmark/__init__.py:    # Separate non-opencl from opencl methods
src/pyFAI/benchmark/__init__.py:        if method.impl_lower == 'opencl':
src/pyFAI/benchmark/__init__.py:        # Benchmark No OpenCL devices
src/pyFAI/benchmark/__init__.py:                        opencl=None,
src/pyFAI/benchmark/__init__.py:        # Benchmark OpenCL devices
src/pyFAI/benchmark/__init__.py:                        opencl={"platformid": method.target[0], "deviceid": method.target[1]},
src/pyFAI/benchmark/__init__.py:        # Benchmark No OpenCL devices
src/pyFAI/benchmark/__init__.py:                    opencl=False,
src/pyFAI/benchmark/__init__.py:        # Benchmark OpenCL devices
src/pyFAI/benchmark/__init__.py:                    opencl={"platformid": method.target[0], "deviceid": method.target[1]},
src/pyFAI/utils/__init__.py:    """get the full path of a openCL resource file
src/pyFAI/utils/__init__.py:    :param str resource: Resource name. File name contained if the `opencl`
src/pyFAI/utils/__init__.py:    :return: the full path of the openCL source file
src/pyFAI/utils/__init__.py:                                        default_directory="opencl")
src/pyFAI/__init__.py:use_opencl = True
src/pyFAI/__init__.py:"""Global configuration which allow to disable OpenCL programatically.
src/pyFAI/__init__.py:It must be set before requesting any OpenCL modules.
src/pyFAI/__init__.py:    pyFAI.use_opencl = False
sandbox/_distortionCSR.c:static char __pyx_k_GPU[] = "GPU";
sandbox/_distortionCSR.c:static PyObject *__pyx_n_s_GPU;
sandbox/_distortionCSR.c: *             self.integrator = ocl_azim_csr_dis.OCL_CSR_Integrator(self.LUT, self.bins, "GPU", block_size=self.workgroup_size)
sandbox/_distortionCSR.c: *             self.integrator = ocl_azim_csr_dis.OCL_CSR_Integrator(self.LUT, self.bins, "GPU", block_size=self.workgroup_size)
sandbox/_distortionCSR.c: *             self.integrator = ocl_azim_csr_dis.OCL_CSR_Integrator(self.LUT, self.bins, "GPU", block_size=self.workgroup_size)             # <<<<<<<<<<<<<<
sandbox/_distortionCSR.c:    __Pyx_INCREF(__pyx_n_s_GPU);
sandbox/_distortionCSR.c:    PyTuple_SET_ITEM(__pyx_t_6, 2, __pyx_n_s_GPU);
sandbox/_distortionCSR.c:    __Pyx_GIVEREF(__pyx_n_s_GPU);
sandbox/_distortionCSR.c: *             self.integrator = ocl_azim_csr_dis.OCL_CSR_Integrator(self.LUT, self.bins, "GPU", block_size=self.workgroup_size)
sandbox/_distortionCSR.c: *             self.integrator = ocl_azim_csr_dis.OCL_CSR_Integrator(self.LUT, self.bins, "GPU", block_size=self.workgroup_size)
sandbox/_distortionCSR.c:  {&__pyx_n_s_GPU, __pyx_k_GPU, sizeof(__pyx_k_GPU), 0, 0, 1, 1},
sandbox/debug_ocl_sort.py:import pyFAI, pyFAI.opencl
sandbox/debug_ocl_sort.py:from pyFAI.opencl import pyopencl, ocl
sandbox/debug_ocl_sort.py:import pyopencl.array
sandbox/debug_ocl_sort.py:ctx = ocl.create_context("GPU")
sandbox/debug_ocl_sort.py:queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
sandbox/debug_ocl_sort.py:d_data = pyopencl.array.to_device(queue, h_data)
sandbox/debug_ocl_sort.py:local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
sandbox/debug_ocl_sort.py:prg = pyopencl.Program(ctx, src).build()
sandbox/profile_pixelsplitFull.py:import pyopencl as cl
sandbox/profile_pixelsplitFull.py:with open("../openCL/ocl_lut_pixelsplit.cl", "r") as kernelFile:
sandbox/profile_ocl_lut.py:import fabio, pyopencl
sandbox/profile_ocl_lut.py:gpu = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(lut, data.size, "GPU")
sandbox/profile_ocl_lut.py:print(gpu.device)
sandbox/profile_ocl_lut.py:pyopencl.enqueue_copy(gpu._queue, img, gpu._cl_mem["image"])#.wait()
sandbox/profile_ocl_lut.py:out_ocl = gpu.integrate(data)[0]
sandbox/profile_ocl_lut.py:out_ocl = gpu.integrate(data, dummy= -2, delta_dummy=1.5)[0]
sandbox/profile_ocl_lut.py:out_ocl = gpu.integrate(data, dark=dark)[0]
sandbox/profile_ocl_lut.py:out_ocl = gpu.integrate(data, flat=flat)[0]
sandbox/profile_ocl_lut.py:out_ocl = gpu.integrate(data, solidAngle=solidAngle)[0]
sandbox/profile_ocl_lut.py:out_ocl = gpu.integrate(data, polarization=polarization)[0]
sandbox/profile_ocl_lut.py:#pyopencl.enqueue_copy(gpu._queue, img, gpu._cl_mem["image"]).wait()
sandbox/profile_ocl_lut.py:#pyopencl.enqueue_copy(gpu._queue, outData, gpu._cl_mem["outData"])#.wait()
sandbox/profile_ocl_lut.py:#pyopencl.enqueue_copy(gpu._queue, outCount, gpu._cl_mem["outCount"])#.wait()
sandbox/profile_ocl_lut.py:#pyopencl.enqueue_copy(gpu._queue, outMerge, gpu._cl_mem["outMerge"])#.wait()
sandbox/profile_ocl_lut.py:out = gpu.integrate(data, dummy= -2, delta_dummy=1.5)
sandbox/profile_ocl_lut.py:out = gpu.integrate(data)
sandbox/profile_csr.py:import fabio, pyopencl
sandbox/profile_csr.py:ocl_lut = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(cyt_lut.lut, data.size, "GPU",profile=True)
sandbox/profile_csr.py:print("OpenCL Device", ocl_lut.device)
sandbox/profile_csr.py:print("lut cpu vs lut gpu",abs(out_cyt_lut - out_ocl_lut).max())
sandbox/profile_ocl_separate.py:pr.dump_stats(__file__ + ".opencl.log")
sandbox/profile_OCLFullSplit.py:import pyopencl as cl
sandbox/profile_splitPixelFullLUT.py:import pyopencl as cl
sandbox/debug_peakfinder_p9.py:import pyFAI.opencl.peak_finder
sandbox/debug_peakfinder_p9.py:pf = pyFAI.opencl.peak_finder.OCL_PeakFinder(csr, img.size, bin_centers=res[0], radius=r2, profile=True)
sandbox/bug_ocl_cpu.py:ai.xrpd_OpenCL(numpy.ones(shape), 500, devicetype="cpu", useFp64=False)
sandbox/profile_csr_padded_csr.py:import fabio, pyopencl
sandbox/profile_csr_padded_csr.py:ocl_lut = pyFAI.ocl_azim_lut.OCL_LUT_Integrator(cyt_lut.lut, data.size, "GPU",profile=True)
sandbox/profile_csr_padded_csr.py:print("OpenCL LUT on: ", ocl_lut.device)
sandbox/profile_csr_padded_csr.py:ocl_csr = ocl_azim_csr.OCL_CSR_Integrator(cyt_csr.lut, data.size, "GPU",profile=True, block_size=workgroup_size)
sandbox/profile_csr_padded_csr.py:ocl_csr_padded = ocl_azim_csr.OCL_CSR_Integrator(cyt_csr_padded.lut, data.size, "GPU",profile=True, block_size=workgroup_size)
sandbox/profile_csr_all_platforms.py:import fabio, pyopencl
sandbox/profile_pixelsplitFullLUT2.py:import pyopencl as cl
sandbox/profile_ocl_hist_pixelsplit.py:import pyopencl as cl
sandbox/profile_ocl_hist_pixelsplit.py:with open("../../openCL/ocl_hist_pixelsplit.cl", "r") as kernelFile:
sandbox/demo_bitonic.py:import numpy, pyopencl, pyopencl.array, time
sandbox/demo_bitonic.py:ctx = pyopencl.create_some_context()
sandbox/demo_bitonic.py:queue = pyopencl.CommandQueue(ctx, properties=pyopencl.command_queue_properties.PROFILING_ENABLE)
sandbox/demo_bitonic.py:d_data = pyopencl.array.to_device(queue, h_data)
sandbox/demo_bitonic.py:local_mem = pyopencl.LocalMemory(ws * 32)  # 2float4 = 2*4*4 bytes per workgroup size
sandbox/demo_bitonic.py:src = open("../openCL/bitonic.cl").read().strip()
sandbox/demo_bitonic.py:prg = pyopencl.Program(ctx, src).build()
sandbox/demo_bitonic.py:d_data = pyopencl.array.to_device(queue, h_data)
sandbox/demo_bitonic.py:d2_data = pyopencl.array.to_device(queue, h2_data)
sandbox/demo_bitonic.py:d2_data = pyopencl.array.to_device(queue, h2_data)
sandbox/test_opencl_reduction_test4.py:import pyopencl as cl
sandbox/test_opencl_reduction_test4.py:from pyopencl import array
sandbox/test_opencl_reduction_test4.py:with open("pyFAI/resources/openCL/reduction_test4.cl", "r") as kernelFile:
sandbox/test_integrate_ui.py:from pyFAI.opencl import ocl
sandbox/test_integrate_ui.py:        self.connect(self.do_OpenCL, SIGNAL("clicked()"), self.openCL_changed)
sandbox/test_integrate_ui.py:            if self.do_OpenCL.isChecked():
sandbox/test_integrate_ui.py:    def openCL_changed(self):
sandbox/test_integrate_ui.py:        logger.debug("do_OpenCL")
sandbox/test_integrate_ui.py:        do_ocl = bool(self.do_OpenCL.isChecked())
sandbox/test_integrate_ui.py:                self.do_OpenCL.setChecked(0)
sandbox/test_integrate_ui.py:            self.do_OpenCL.setChecked(0)
sandbox/profile_csr_2d.py:import fabio, pyopencl
sandbox/profile_ocl_lut_pixelsplit3.py:import pyopencl as cl
sandbox/profile_ocl_lut_pixelsplit3.py:with open("../openCL/ocl_lut_pixelsplit.cl", "r") as kernelFile:
sandbox/profile_pixelsplitFullLUT.py:import pyopencl as cl
sandbox/_distortionCSR.pyx:            self.integrator = ocl_azim_csr_dis.OCL_CSR_Integrator(self.LUT, self.bins, "GPU", block_size=self.workgroup_size)
sandbox/profile_csr_fullsplit.py:import fabio, pyopencl
sandbox/profile_csr_fullsplit.py:foo2 = ocl_azim_csr.OCL_CSR_Integrator(foo.lut, data.size, "GPU", block_size=32)
sandbox/bug_ocl_gpu.py:ai.xrpd_OpenCL(a, 1000, devicetype="gpu")
sandbox/profile_ocl_lut_pixelsplit2.py:import pyopencl as cl
sandbox/profile_ocl_lut_pixelsplit2.py:with open("../openCL/ocl_lut_pixelsplit.cl", "r") as kernelFile:
sandbox/profile_ocl_lut_pixelsplit.py:import pyopencl as cl
sandbox/profile_ocl_lut_pixelsplit.py:with open("../openCL/ocl_lut_pixelsplit.cl", "r") as kernelFile:
sandbox/profile_lut_pixelsplitFull.py:import pyopencl as cl
sandbox/profile_lut_pixelsplitFull.py:with open("../openCL/ocl_lut_pixelsplit.cl", "r") as kernelFile:

```
