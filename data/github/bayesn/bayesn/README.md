# https://github.com/bayesn/bayesn

```console
docs/running.rst:GPUs. However, if you want to fit/analyse a small sample within a Jupyter notebook/custom Python script you can also do
docs/running.rst:- ``chain_method``: The method to use for running multiple chains in numpyro. If 'sequential', chains will be run one-after-the-other until all are complete. If 'parallel', the chains will be run in parallel over multiple devices - with 4 chains and a node with 4 GPUs, the chains will be run simultaneously in parallel. If 'vectorized', chains will be run in parallel on a single device which may or may not be quicker than running them sequentially depending on the device you are using, and may result in memory issues unless you are using a large GPU.
docs/intro.rst:GPU acceleration
docs/intro.rst:This code is built on numpyro and jax, and has been designed with GPU acceleration in mind;  running on a GPU should
docs/intro.rst:it is important to note that GPUs will show the most benefit running large scale jobs. If you want to fit samples of
docs/intro.rst:Depending on the GPUs you have access to and the cadence of your light curves, you should be able to fit thousands of
docs/intro.rst:SNe in parallel. In testing using 4 A100 GPUs, fits have successfully run upwards of 15,000 *griz*
docs/intro.rst:relating to the residual intrinsic SN colour distribution). This is a considerable speed increase enabled using GPUs.
docs/index.rst:for GPU acceleration.
docs/installation.rst:taken when installing the dependencies to allow BayeSN to run on GPUs - please see below.
docs/installation.rst:If you want to use GPUs, you must take care to install the correct version of jax following instructions below.
docs/installation.rst:Requirements for GPU
docs/installation.rst:* cudatoolkit > 11.8
docs/installation.rst:* jax version which matches cudatoolkit/cudnn version, instructions below
docs/installation.rst:To use GPUs, you need to install a version of jax specific for GPUs - the default pip install is CPU only. In addition,
docs/installation.rst:the jax version will need to match the version of cudatoolkit and cudnn you have installed. Full installation
docs/installation.rst:instructions for jax GPU can be found here: https://github.com/google/jax#installation.
docs/installation.rst:**However, take care if you want to run using GPUs**. In this case, you must install a version of jax compatible with
docs/installation.rst:GPUs **before** pip installing BayeSN, following the instructions above. This is because installing via pip will also
README.md:GPU acceleration, as discussed in Grayling+2024 (https://arxiv.org/abs/2401.08755, accepted by MNRAS).
README.md:### GPU Acceleration
README.md:This code has been designed with GPU acceleration in mind, and running on a GPU should yield a considerable (~100 times)
README.md:increase in performance. However, it is important to note that GPUs will show the most benefit running large scale jobs.
README.md:handful. With only 1 object, you are likely better off running on CPU than GPU.
bayesn/bayesn_model.py:        size. This is required because to benefit from the GPU, we need to have a fixed array structure allowing us to

```
