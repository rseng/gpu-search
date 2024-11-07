# https://github.com/hpparvi/PyTransit

```console
CHANGELOG.md: - Changed several of the OpenCL models use the Taylor series expansion approach to calculate
CHANGELOG.md:most models both in CPU and GPU versions.
tests/test_ma_quadratic.py:    # TODO: Set up OpenCL in Travis
tests/test_ma_quadratic.py:    #def test_to_opencl(self):
tests/test_ma_quadratic.py:    #    tm2 = tm.to_opencl()
doc/source/api/models/quadraticcl.rst:OpenCL Quadratic model
doc/source/api/models/uniformcl.rst:OpenCL Uniform model
doc/source/index.rst:that offers optimised CPU and GPU implementations of exoplanet transit models with a unified interface (API). Transit model
doc/source/models.rst:  data transfer between the GPU and main memory would create a bottleneck.
doc/source/models.rst:- OpenCL implementations for GPU computing. These can be orders of magnitude faster than the CPU
doc/source/models.rst:  implementations if ran in a powerful GPU, especially when modelling long cadence data where
doc/source/models.rst:The CPU and GPU implementations aim to offer equal functionality, but, at the moment of writing,
doc/source/models.rst:OpenCL
doc/source/models.rst:The OpenCL versions of the models work identically to the Python version, except
doc/source/models.rst:that the OpenCL context and queue can be given as arguments in the initialiser, and the model evaluation method can be
doc/source/models.rst:told to not to copy the model from the GPU memory. If the context and queue are not given, the model creates a default
doc/source/models.rst:    import pyopencl as cl
requirements.txt:pyopencl
MANIFEST.in:include pytransit/models/opencl/*.cl
pytransit/models/ma_uniform_cl.py:"""OpenCL implementation of the transit over a uniform disk (Mandel & Agol, ApJ 580, L171-L175 2002).
pytransit/models/ma_uniform_cl.py:import pyopencl as cl
pytransit/models/ma_uniform_cl.py:from pyopencl import CompilerWarning
pytransit/models/ma_uniform_cl.py:        # Initialize the OpenCL context and queue
pytransit/models/ma_uniform_cl.py:        rd = Path(resource_filename('pytransit', 'models/opencl'))
pytransit/models/general.py:    def to_opencl(self):
pytransit/models/general.py:        """Creates an OpenCL clone (`QuadraticModelCL`) of the transit model.
pytransit/models/qpower2_cl.py:import pyopencl as cl
pytransit/models/qpower2_cl.py:from pyopencl import CompilerWarning
pytransit/models/qpower2_cl.py:        self.prg = cl.Program(self.ctx, open(join(dirname(__file__),'opencl','qpower2.cl'),'r').read()).build()
pytransit/models/qpower2_cl.py:        # Release and reinitialise the GPU buffers if the sizes of the time or
pytransit/models/qpower2_cl.py:        # Copy the limb darkening coefficient array to the GPU
pytransit/models/qpower2_cl.py:        # Copy the parameter vector to the GPU
pytransit/models/qpower2_cl.py:        # Release and reinitialise the GPU buffers if the sizes of the time or
pytransit/models/qpower2_cl.py:        # Copy the limb darkening coefficient array to the GPU
pytransit/models/qpower2_cl.py:        # Copy the parameter vector to the GPU
pytransit/models/roadrunner/rrmodel_cl.py:import pyopencl as cl
pytransit/models/roadrunner/rrmodel_cl.py:from pyopencl import CompilerWarning
pytransit/models/roadrunner/rrmodel_cl.py:        # Release and reinitialise the GPU buffers if the parameter vector size changes
pytransit/models/roadrunner/rrmodel_cl.py:        # Copy the limb darkening profiles and their integrals to the GPU
pytransit/models/roadrunner/rrmodel_cl.py:        # Copy the parameter vector to the GPU
pytransit/models/ma_quadratic.py:    def to_opencl(self):
pytransit/models/ma_quadratic.py:        """Creates an OpenCL clone (`QuadraticModelCL`) of the transit model.
pytransit/models/ma_quadratic_cl.py:import pyopencl as cl
pytransit/models/ma_quadratic_cl.py:from pyopencl import CompilerWarning
pytransit/models/ma_quadratic_cl.py:    OpenCL implementation of the transit light curve model with quadratic limb darkening by Mandel and Agol (2002).
pytransit/models/ma_quadratic_cl.py:    This class implements the quadratic transit model by Mandel & Agol (ApJ 580, L171-L175, 2002) in OpenCL. The class
pytransit/models/ma_quadratic_cl.py:            OpenCL context.
pytransit/models/ma_quadratic_cl.py:            OpenCL queue
pytransit/models/ma_quadratic_cl.py:        rd = Path(resource_filename('pytransit', 'models/opencl'))
pytransit/models/ma_quadratic_cl.py:        # Release and reinitialise the GPU buffers if the sizes of the time or
pytransit/models/ma_quadratic_cl.py:        # Copy the limb darkening coefficient array to the GPU
pytransit/models/ma_quadratic_cl.py:        # Copy the parameter vector to the GPU
pytransit/models/ma_quadratic_cl.py:        # Release and reinitialise the GPU buffers if the sizes of the time or
pytransit/models/ma_quadratic_cl.py:        # Copy the limb darkening coefficient array to the GPU
pytransit/models/ma_quadratic_cl.py:        # Copy the parameter vector to the GPU
pytransit/models/ma_quadratic_cl.py:        # Release and reinitialise the GPU buffers if the sizes of the time or
pytransit/models/ma_quadratic_cl.py:        # Copy the limb darkening coefficient array to the GPU
pytransit/models/ma_quadratic_cl.py:        # Copy the parameter vector to the GPU
pytransit/lpf/ocltdvlpf.py:import pyopencl as cl
pytransit/lpf/ocllpf.py:import pyopencl as cl
pytransit/lpf/loglikelihood/clloglikelihood.py:import pyopencl as cl
pytransit/lpf/loglikelihood/clloglikelihood.py:        # Define the OpenCL buffers
pytransit/lpf/loglikelihood/clloglikelihood.py:        # Release OpenCL buffers if they're initialised
pytransit/lpf/loglikelihood/clloglikelihood.py:        # Initialise OpenCL buffers
pytransit/__init__.py:models implemented in Python (with Numba acceleration) and OpenCL.
pytransit/__init__.py:# OpenCL models
.travis.yml:#  - if [ $TRAVIS_OS_NAME = linux ]; then sudo apt-get install libpocl-dev pocl-opencl-icd; fi

```
