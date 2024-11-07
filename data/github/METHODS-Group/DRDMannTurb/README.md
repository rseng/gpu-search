# https://github.com/METHODS-Group/DRDMannTurb

```console
conda_env_files/conda-lock.yml:- name: cuda-version
conda_env_files/conda-lock.yml:  url: https://conda.anaconda.org/conda-forge/noarch/cuda-version-11.8-h70ddcb2_3.conda
conda_env_files/conda-lock.yml:- name: cudatoolkit
conda_env_files/conda-lock.yml:  url: https://conda.anaconda.org/conda-forge/linux-64/cudatoolkit-11.8.0-h4ba93d1_13.conda
conda_env_files/conda-lock.yml:    cuda-version: '>=11.0,<12.0a0'
conda_env_files/conda-lock.yml:    cudatoolkit: 11.*
conda_env_files/conda-lock.yml:    cudatoolkit: '>=11.8,<12'
conda_env_files/conda-lock.yml:    cudatoolkit: '>=11.8,<12'
conda_env_files/conda-lock.yml:    cudatoolkit: '>=11.8,<12'
conda_env_files/conda-lock.yml:    nccl: '>=2.22.3.1,<3.0a0'
conda_env_files/conda-lock.yml:  url: https://conda.anaconda.org/conda-forge/linux-64/libtorch-2.4.0-cuda118_h8db9d67_301.conda
conda_env_files/conda-lock.yml:- name: nccl
conda_env_files/conda-lock.yml:    cuda-version: '>=11.0,<12.0a0'
conda_env_files/conda-lock.yml:  url: https://conda.anaconda.org/conda-forge/linux-64/nccl-2.22.3.1-hee583db_1.conda
conda_env_files/conda-lock.yml:    __cuda: ''
conda_env_files/conda-lock.yml:    cudatoolkit: '>=11.8,<12'
conda_env_files/conda-lock.yml:    nccl: '>=2.22.3.1,<3.0a0'
conda_env_files/conda-lock.yml:  url: https://conda.anaconda.org/conda-forge/linux-64/pytorch-2.4.0-cuda118_py39hbf661d7_301.conda
drdmannturb/spectra_fitting/calibration.py:            One of the strings ``"cpu", "cuda", "mps"`` indicating the torch device to use
drdmannturb/spectra_fitting/calibration.py:        """Initializes the device (CPU or GPU) on which computation is performed.
drdmannturb/spectra_fitting/calibration.py:            string following PyTorch conventions -- "cuda" or "cpu"
drdmannturb/spectra_fitting/calibration.py:        if device == "cuda" and torch.cuda.is_available():
drdmannturb/spectra_fitting/calibration.py:            torch.set_default_tensor_type("torch.cuda.FloatTensor")
drdmannturb/spectra_fitting/calibration.py:            parameters setter method. This automatically offloads any model parameters that were on the GPU, if any.
drdmannturb/spectra_fitting/calibration.py:                if NN_parameters.is_cuda
drdmannturb/spectra_fitting/calibration.py:            )  # TODO: this should also properly load on GPU, issue #28
drdmannturb/spectra_fitting/data_generator.py:                if torch.cuda.is_available()
drdmannturb/common.py:        This function depends on SciPy for evaluating the hypergeometric function, meaning a GPU tensor will be returned to the CPU for a single evaluation and then converted back to a GPU tensor. This incurs a substantial loss of performance.
drdmannturb/common.py:    where :math:`\alpha, \beta` are obtained from a linear regression on the hypergeometric function on the domain of interest. In particular, using this function requires that a linear regression has already been performed on the basis of the above function depending on the hypergeometric function, which is an operation performed once on the CPU. The rest of this subroutine is on the GPU and unlike the full hypergeometric approximation, will not incur any slow-down of the rest of the spectra fitting.
drdmannturb/fluctuation_generation/fluctuation_field_generator.py:            device = "cuda" if torch.cuda.is_available() else "cpu"
drdmannturb/fluctuation_generation/nn_covariance.py:                if beta_torch.is_cuda
docs/source/index.rst:`DRDMannTurb` (Deep Rapid Distortion Mann Turbulence) is a GPU-accelerated, data-driven Python framework for
docs/source/index.rst:wind engineers in applications requiring rapid simulation of realistic wind turbulence. It is based off of the Deep Rapid Distortion models presented in `Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_. The data-driven functionalities are GPU-accelerated via a `PyTorch <https://pytorch.org/docs/stable/index.html>`_  implementation. 
docs/source/datatypes.rst:        This is a CPU-bound function since the hypergeometric function has no readily use-able CUDA implementation, which considerably impacts performance. Consider using an approximation to this function.
docs/source/datatypes.rst:#. ``EddyLifetimeType.MANN_APPROX``, which performs a linear regression on a single evaluation of the Mann eddy lifetime function in log-log space. This results in a GPU function of the form :math:`\tau \approx \exp (\alpha \boldsymbol{k} + \beta)`. 
test/eddy_lifetime/test_symmetries.py:device = "cuda" if torch.cuda.is_available() else "cpu"
test/eddy_lifetime/test_symmetries.py:# v2: torch.set_default_device('cuda:0')
test/eddy_lifetime/test_symmetries.py:if torch.cuda.is_available():
test/eddy_lifetime/test_symmetries.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
test/fluctuation_field/test_field_properties.py:device = "cuda" if torch.cuda.is_available() else "cpu"
test/fluctuation_field/test_field_properties.py:# v2: torch.set_default_device('cuda:0')
test/fluctuation_field/test_field_properties.py:if torch.cuda.is_available():
test/fluctuation_field/test_field_properties.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
test/fluctuation_field/test_field_properties.py:# TODO: this requires the GPU for testing, but passes as of release 0.1.0
test/fluctuation_field/test_field_properties.py:@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available")
test/io/test_data_io.py:device = "cuda" if torch.cuda.is_available() else "cpu"
test/io/test_data_io.py:# v2: torch.set_default_device('cuda:0')
test/io/test_data_io.py:if torch.cuda.is_available():
test/io/test_data_io.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
test/io/test_model_params.py:device = "cuda" if torch.cuda.is_available() else "cpu"
test/io/test_model_params.py:if torch.cuda.is_available():
test/io/test_model_params.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
test/perf/gpu_use.py:"""Basic tests for assessing GPU use of the package. During the initial release, the GPU utilization during training was >=95% throughout training. Changes that drop GPU utilization should be considered regressions to package performance."""
test/perf/gpu_use.py:device = "cuda" if torch.cuda.is_available() else "cpu"
test/perf/gpu_use.py:# v2: torch.set_default_device('cuda:0')
test/perf/gpu_use.py:if torch.cuda.is_available():
test/perf/gpu_use.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
test/perf/gpu_use.py:def test_gpu_utilization_synth_fit():
test/perf/gpu_use.py:    if torch.cuda.is_available() and util.find_spec("pynvml") is not None:
test/perf/gpu_use.py:        assert torch.cuda.utilization() >= 95
test/perf/gpu_use.py:            "CUDA must be available in test runner with pynvml installed in the environment."
README.md:Note also that certain components of the test suite require CUDA; these are also
README.md:skipped if a CUDA device is not available.
paper/paper.md:`NumPy` and `PyTorch`. The implementation makes DRD models easily portable to GPU and other backends via `PyTorch`. 
examples/08_mann_box_generation_IEC.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/08_mann_box_generation_IEC.py:if torch.cuda.is_available():
examples/08_mann_box_generation_IEC.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/07_mann_linear_regression.py:to show how a GPU kernel of a linear approximation (in log-log space) of the Mann eddy lifetime can
examples/07_mann_linear_regression.py:be generated automatically to speed up tasks that require the GPU. As before, the Kaimal spectra is
examples/07_mann_linear_regression.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/07_mann_linear_regression.py:if torch.cuda.is_available():
examples/07_mann_linear_regression.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/09_drd_box.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/09_drd_box.py:# v2: torch.set_default_device('cuda:0')
examples/09_drd_box.py:if torch.cuda.is_available():
examples/09_drd_box.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/06_custom_data_interpolate_and_fit.py:# working directory and dataset path, and choose to use CUDA if it is available.
examples/06_custom_data_interpolate_and_fit.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/06_custom_data_interpolate_and_fit.py:if torch.cuda.is_available():
examples/06_custom_data_interpolate_and_fit.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/06_custom_data_interpolate_and_fit.py:        k1_data_pts.cpu().numpy() if torch.cuda.is_available() else k1_data_pts.numpy()
examples/04_eddy-lifetime_fit_gelu.py:# CUDA if it is available.
examples/04_eddy-lifetime_fit_gelu.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/04_eddy-lifetime_fit_gelu.py:if torch.cuda.is_available():
examples/04_eddy-lifetime_fit_gelu.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/05_custom_noisy_data_fit.py:# working directory and dataset path, and choose to use CUDA if it is available.
examples/05_custom_noisy_data_fit.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/05_custom_noisy_data_fit.py:if torch.cuda.is_available():
examples/05_custom_noisy_data_fit.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/01_basic_mann_parameters_fit.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/01_basic_mann_parameters_fit.py:if torch.cuda.is_available():
examples/01_basic_mann_parameters_fit.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/02_eddy-lifetime_fit.py:# CUDA if it is available.
examples/02_eddy-lifetime_fit.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/02_eddy-lifetime_fit.py:if torch.cuda.is_available():
examples/02_eddy-lifetime_fit.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")
examples/03_eddy-lifetime_fit_3term_loss.py:# CUDA if it is available.
examples/03_eddy-lifetime_fit_3term_loss.py:device = "cuda" if torch.cuda.is_available() else "cpu"
examples/03_eddy-lifetime_fit_3term_loss.py:if torch.cuda.is_available():
examples/03_eddy-lifetime_fit_3term_loss.py:    torch.set_default_tensor_type("torch.cuda.FloatTensor")

```
