# https://github.com/scikit-hep/pyhf

```console
README.rst:|PyPI version| |Conda-forge version| |Supported Python versions| |Docker Hub pyhf| |Docker Hub pyhf CUDA|
README.rst:and GPU acceleration.
README.rst:.. |Docker Hub pyhf CUDA| image:: https://img.shields.io/badge/pyhf-CUDA-blue?logo=Docker
README.rst:   :target: https://hub.docker.com/r/pyhf/cuda/tags
docs/JOSS/paper.md:These alternative backends support hardware acceleration on GPUs, and in the case of JAX JIT compilation, as well as auto-differentiation allowing for calculating the full gradient of the likelihood function &mdash; all contributing to speeding up fits.
docs/faq.rst:    This is may be the result of a conflict with the NVIDIA drivers that you
docs/faq.rst:        sudo apt-get purge nvidia*
docs/bib/general_citations.bib:    title = "{GPU coprocessors as a service for deep learning inference in high energy physics}",
docs/outreach.rst:    auto-differentiation and GPU acceleration.
docs/outreach.rst:        auto-differentiation and GPU acceleration.
docs/governance/ROADMAP.rst:-  GPU support and testing
docs/governance/ROADMAP.rst:      "`GPU/accelerator-based implementation of statistical and other
docs/release-notes/v0.6.2.rst:* CUDA enabled Docker images are now available for release ``v0.6.1`` and later
docs/release-notes/v0.6.2.rst:  on `Docker Hub <https://hub.docker.com/r/pyhf/cuda>`__ and the `GitHub
docs/release-notes/v0.6.2.rst:  Container Registry <https://github.com/pyhf/cuda-images/pkgs/container/cuda-images>`__.
docs/release-notes/v0.6.2.rst:  Visit `github.com/pyhf/cuda-images <https://github.com/pyhf/cuda-images>`_ for more
CITATION.cff:  to make use of features such as autodifferentiation and GPU acceleration.
docker/gpu/Dockerfile:FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 AS base
docker/gpu/Dockerfile:COPY ./docker/gpu/install_backend.sh /code/install_backend.sh
docker/gpu/install_backend.sh:function get_JAXLIB_GPU_WHEEL {
docker/gpu/install_backend.sh:  local CUDA_VERSION # alternatives: cuda90, cuda92, cuda100, cuda101
docker/gpu/install_backend.sh:  CUDA_VERSION="cuda"$(< /usr/local/cuda/version.txt awk '{print $NF}' | awk '{split($0, rel, "."); print rel[1]rel[2]}')
docker/gpu/install_backend.sh:  local JAXLIB_GPU_WHEEL="${BASE_URL}/${CUDA_VERSION}/jaxlib-${JAXLIB_VERSION}-${PYTHON_VERSION}-none-${PLATFORM}.whl"
docker/gpu/install_backend.sh:  echo "${JAXLIB_GPU_WHEEL}"
docker/gpu/install_backend.sh:    python3 -m pip install --no-cache-dir "$(get_JAXLIB_GPU_WHEEL)"

```
