# https://github.com/henrysky/astroNN

```console
docs/source/neuralnets/basic_usage.rst:NeuralNetBase consists of a pre-training checking (check input and labels shape, cpu/gpu check and create astroNN
docs/source/neuralnets/layers.rst:If you wnat fast MC inference on GPU and you are using keras models, you should just use FastMCInference_.
docs/source/neuralnets/layers.rst:The advantage of `FastMCInferenceMeanVar` layer is you can copy the data and calculate the mean and variance on GPU (if any)
docs/source/neuralnets/layers.rst:If you wnat fast MC inference on GPU and you are using keras models, you should just use FastMCInference_.
docs/source/neuralnets/layers.rst:The advantage of `FastMCRepeat` layer is you can copy the data and calculate the mean and variance on GPU (if any)
docs/source/neuralnets/layers.rst:`FastMCInference` is a layer designed for fast Monte Carlo Inference on GPU. One of the main challenge of MC integration
docs/source/neuralnets/layers.rst:on GPU is you want the data stay on GPU and you do MC integration on GPU entirely, moving data from drives to GPU is
docs/source/neuralnets/layers.rst:a very expensive operation. `FastMCInference` will create a new keras model such that it will replicate data on GPU, do
docs/source/neuralnets/layers.rst:Monte Carlo integration and calculate mean and variance on GPU, and get back the result.
docs/source/neuralnets/layers.rst:Benchmark (Nvidia GTX1060 6GB): 98,000 7514 pixles APOGEE Spectra, traditionally the 25 forward pass spent ~270 seconds,
docs/source/neuralnets/layers.rst:    # fast_mc_model is the new keras model capable to do fast monte carlo integration on GPU
docs/source/quick_start.rst:    gpu_mem_ratio = True
tests/test_utilities.py:def test_cpu_gpu_management():
CONTRIBUTING.rst:.. topic:: GPU/performance related issues
CONTRIBUTING.rst:    * Data reduction pipeline on GPU?
CONTRIBUTING.rst:    * Multiple GPU support!
ISSUE_TEMPLATE.md:- **CUDA & cuDNN version (if applicable)**:
ISSUE_TEMPLATE.md:- **GPU model and memor (if applicable)y**:
src/astroNN/models/base_bayesian_cnn.py:        self.mc_num = 100  # increased to 100 due to high performance VI on GPU implemented on 14 April 2018 (Henry)
src/astroNN/models/base_bayesian_cnn.py:        Test model, High performance version designed for fast variational inference on GPU
src/astroNN/models/apogee_models.py:        # new astroNN high performance dropout variational inference on GPU expects single output
src/astroNN/models/apogee_models.py:        # new astroNN high performance dropout variational inference on GPU expects single output
src/astroNN/models/apogee_models.py:        # new astroNN high performance dropout variational inference on GPU expects single output
src/astroNN/models/nn_base.py:from astroNN.config import _astroNN_MODEL_NAME, cpu_gpu_reader
src/astroNN/models/nn_base.py:        fallback_cpu = cpu_gpu_reader()
src/astroNN/nn/layers.py:    Turn a model for fast Monte Carlo (Dropout, Flipout, etc) Inference on GPU
src/astroNN/config.py:def cpu_gpu_reader():
src/astroNN/config.py:    NAME: cpu_gpu_reader
src/astroNN/config.py:    PURPOSE: to read cpu gpu setting in config
src/astroNN/config.py:        return cpu_gpu_reader()
src/astroNN/shared/nn_tools.py:    A function to force Keras backend to use CPU even Nvidia GPU is presented
src/astroNN/shared/nn_tools.py:    :param flag: `True` to fallback to CPU, `False` to un-manage CPU or GPU
src/astroNN/shared/nn_tools.py:                tf.config.set_visible_devices([], "GPU")
src/astroNN/shared/nn_tools.py:                "torch_device", "cuda"
src/astroNN/shared/nn_tools.py:                gpu_phy_devices = tf.config.list_physical_devices("GPU")
src/astroNN/shared/nn_tools.py:                tf.config.set_visible_devices(gpu_phy_devices, "GPU")

```
