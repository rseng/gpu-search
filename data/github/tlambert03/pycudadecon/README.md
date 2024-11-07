# https://github.com/tlambert03/pycudadecon

```console
docs/cli.md:If you installed `pycudadecon` using conda, the original binaries for OTF
docs/cli.md:generation (`radialft`) and deconvolution (`cudadecon`) will also be
docs/cli.md:installed in your conda environment. The CLI is not created by pycudadecon;
docs/cli.md:it is entirely defined by [cudadecon](https://github.com/scopetools/cudadecon).
docs/cli.md:## cudadecon
docs/cli.md:The `cudadecon` command runs deconvolution (with deskewing and rotation if
docs/cli.md:cudadecon /folder/of/images 488nm /path/to/488nm_otf.tif -z 0.3
docs/cli.md:cudadecon /folder/of/images 488nm /path/to/488nm_otf.tif -z 0.3 -D 31.5 -M 0 0 1
docs/cli.md:Run `cudadecon --help` at the command prompt for the full menu of options.
docs/cli.md:$ cudadecon -h
docs/cli.md:cudaDeconv.  Version: 0.7.0
docs/cli.md:                                    deconvolved (i.e. exist in the GPUdecon
docs/cli.md:  -Q [ --DevQuery ]                 Show info and indices of available GPUs
docs/cli.md:complex OTF file that can be used by `cudaDecon` (or a 3D OTF file if
docs/_config.yml:title: pycudadecon
docs/_config.yml:  url: https://github.com/tlambert03/pycudadecon
docs/affine.md:.. automodule:: pycudadecon
docs/affine.md:    :members: deskewGPU, rotateGPU, affineGPU
docs/_toc.yml:- url: https://github.com/tlambert03/pycudadecon
docs/installation.md:conda install -c conda-forge pycudadecon
docs/installation.md:## GPU requirements
docs/installation.md:This software requires a CUDA-compatible NVIDIA GPU. The underlying cudadecon
docs/installation.md:libraries have been compiled against different versions of the CUDA toolkit.
docs/installation.md:The required CUDA libraries are bundled in the conda distributions so you don't
docs/installation.md:need to install the CUDA toolkit separately.  If desired, you can pick which
docs/installation.md:version of CUDA you'd like based on your needs, but please note that different
docs/installation.md:versions of the CUDA toolkit have different GPU driver requirements:
docs/installation.md:To specify a specific cudatoolkit version, install as follows:
docs/installation.md:conda install -c condaforge pycudadecon cuda-version=<11 or 12>
docs/installation.md:* - CUDA toolkit
docs/installation.md:For the most recent information on GPU driver compatibility, please see the
docs/installation.md:[NVIDIA CUDA Toolkit Release
docs/installation.md:Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
docs/installation.md:github](https://github.com/tlambert03/pycudadecon/issues) and describe your
docs/deconvolution.md:The primary function for performing deconvolution is {func}`~pycudadecon.decon`.
docs/deconvolution.md:will handle setting up and breaking down the FFT plan on the GPU for all files
docs/deconvolution.md:{class}`~pycudadecon.RLContext` context manager or the
docs/deconvolution.md:{func}`~pycudadecon.make_otf` {func}`~pycudadecon.rl_decon` functions.
docs/deconvolution.md:The setup and breakdown for the GPU-deconvolution can also be performed
docs/deconvolution.md:1. call {func}`~pycudadecon.rl_init` with the shape of the raw data and path to
docs/deconvolution.md:2. perform deconvolution(s) with {func}`~pycudadecon.rl_decon`.
docs/deconvolution.md:3. cleanup with {func}`~pycudadecon.rl_cleanup`
docs/deconvolution.md:As a convenience, the {class}`~pycudadecon.RLContext` context manager will
docs/deconvolution.md:.. automodule:: pycudadecon
docs/otf.md:the {class}`~pycudadecon.make_otf` function, or use the
docs/otf.md:{class}`~pycudadecon.TemporaryOTF` context manager to create and delete a
docs/otf.md:.. automodule:: pycudadecon
docs/index.md:# pyCUDAdecon
docs/index.md:[cudaDecon](https://github.com/scopetools/cudaDecon), which is a CUDA/C++
docs/index.md:algorithm {cite}`biggs_97`. ``cudaDecon`` was originally
docs/index.md:makes use of a shared library interface that I wrote for cudaDecon while
docs/index.md:- CUDA accelerated deconvolution with a handful of artifact-reducing features.
docs/index.md:- CUDA-based camera-correction for
docs/index.md:conda install -c conda-forge pycudadecon
docs/index.md:see GPU requirements in [Installation](installation.md).
docs/index.md:out the {func}`pycudadecon.decon` function, which should be able to handle most
docs/index.md:from pycudadecon import decon
docs/index.md:{func}`~pycudadecon.make_otf`, and then use the {class}`~pycudadecon.RLContext`
docs/index.md:context manager to setup the GPU for use with the {func}`~pycudadecon.rl_decon`
docs/index.md:from pycudadecon import RLContext, rl_decon
docs/index.md:If you have a 3D PSF volume, the {class}`~pycudadecon.TemporaryOTF` context
docs/index.md:... and that bit of code is essentially what the {func}`~pycudadecon.decon`
CHANGELOG.md:## [v0.5.1](https://github.com/tlambert03/pycudadecon/tree/v0.5.1) (2024-08-15)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.5.0...v0.5.1)
CHANGELOG.md:- fix: fix finding library [\#61](https://github.com/tlambert03/pycudadecon/pull/61) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:## [v0.5.0](https://github.com/tlambert03/pycudadecon/tree/v0.5.0) (2024-08-15)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.4.1...v0.5.0)
CHANGELOG.md:- ci\(pre-commit.ci\): autoupdate [\#60](https://github.com/tlambert03/pycudadecon/pull/60) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- feat: support get\_version [\#59](https://github.com/tlambert03/pycudadecon/pull/59) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- chore: cleanup repo [\#58](https://github.com/tlambert03/pycudadecon/pull/58) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- feat: add arguments for 3D OTF generation [\#57](https://github.com/tlambert03/pycudadecon/pull/57) ([zichenzachwang](https://github.com/zichenzachwang))
CHANGELOG.md:- ci\(dependabot\): bump peaceiris/actions-gh-pages from 3 to 4 [\#56](https://github.com/tlambert03/pycudadecon/pull/56) ([dependabot[bot]](https://github.com/apps/dependabot))
CHANGELOG.md:- ci\(dependabot\): bump softprops/action-gh-release from 1 to 2 [\#53](https://github.com/tlambert03/pycudadecon/pull/53) ([dependabot[bot]](https://github.com/apps/dependabot))
CHANGELOG.md:- ci\(pre-commit.ci\): autoupdate [\#52](https://github.com/tlambert03/pycudadecon/pull/52) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- ci\(dependabot\): bump actions/setup-python from 4 to 5 [\#51](https://github.com/tlambert03/pycudadecon/pull/51) ([dependabot[bot]](https://github.com/apps/dependabot))
CHANGELOG.md:- ci\(pre-commit.ci\): autoupdate [\#50](https://github.com/tlambert03/pycudadecon/pull/50) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- ci\(pre-commit.ci\): autoupdate [\#49](https://github.com/tlambert03/pycudadecon/pull/49) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:## [v0.4.1](https://github.com/tlambert03/pycudadecon/tree/v0.4.1) (2023-09-13)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.4.0...v0.4.1)
CHANGELOG.md:- ci: update typing, pre-commit, and add all kwargs to decon func [\#48](https://github.com/tlambert03/pycudadecon/pull/48) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- fix: Fix \_cudadecon\_version annotation, added support for skewed decon argument [\#47](https://github.com/tlambert03/pycudadecon/pull/47) ([dmilkie](https://github.com/dmilkie))
CHANGELOG.md:- Fix typo [\#46](https://github.com/tlambert03/pycudadecon/pull/46) ([dmilkie](https://github.com/dmilkie))
CHANGELOG.md:- ci\(dependabot\): bump actions/checkout from 3 to 4 [\#45](https://github.com/tlambert03/pycudadecon/pull/45) ([dependabot[bot]](https://github.com/apps/dependabot))
CHANGELOG.md:- docs: Fix docs [\#42](https://github.com/tlambert03/pycudadecon/pull/42) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- ci\(dependabot\): bump actions/setup-python from 2 to 4 [\#41](https://github.com/tlambert03/pycudadecon/pull/41) ([dependabot[bot]](https://github.com/apps/dependabot))
CHANGELOG.md:- ci\(dependabot\): bump actions/checkout from 2 to 3 [\#40](https://github.com/tlambert03/pycudadecon/pull/40) ([dependabot[bot]](https://github.com/apps/dependabot))
CHANGELOG.md:## [v0.4.0](https://github.com/tlambert03/pycudadecon/tree/v0.4.0) (2022-11-07)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.3.0...v0.4.0)
CHANGELOG.md:- fix: Fix shared library interface for cudadecon ≥0.6.1 [\#38](https://github.com/tlambert03/pycudadecon/pull/38) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- Npe2 plugin [\#37](https://github.com/tlambert03/pycudadecon/pull/37) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- \[pre-commit.ci\] pre-commit autoupdate [\#31](https://github.com/tlambert03/pycudadecon/pull/31) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- \[pre-commit.ci\] pre-commit autoupdate [\#30](https://github.com/tlambert03/pycudadecon/pull/30) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- \[pre-commit.ci\] pre-commit autoupdate [\#29](https://github.com/tlambert03/pycudadecon/pull/29) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- \[pre-commit.ci\] pre-commit autoupdate [\#28](https://github.com/tlambert03/pycudadecon/pull/28) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- \[pre-commit.ci\] pre-commit autoupdate [\#27](https://github.com/tlambert03/pycudadecon/pull/27) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- jupyter books docs [\#26](https://github.com/tlambert03/pycudadecon/pull/26) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- \[pre-commit.ci\] pre-commit autoupdate [\#24](https://github.com/tlambert03/pycudadecon/pull/24) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
CHANGELOG.md:- update docs and readme [\#23](https://github.com/tlambert03/pycudadecon/pull/23) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:## [v0.3.0](https://github.com/tlambert03/pycudadecon/tree/v0.3.0) (2022-08-10)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.2.0...v0.3.0)
CHANGELOG.md:## [v0.2.0](https://github.com/tlambert03/pycudadecon/tree/v0.2.0) (2021-05-30)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.1.2...v0.2.0)
CHANGELOG.md:- remove setuptools [\#22](https://github.com/tlambert03/pycudadecon/pull/22) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- support py36 [\#21](https://github.com/tlambert03/pycudadecon/pull/21) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- numpy docstrings and signature refactor [\#20](https://github.com/tlambert03/pycudadecon/pull/20) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- add napari plugin [\#19](https://github.com/tlambert03/pycudadecon/pull/19) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- update setup [\#18](https://github.com/tlambert03/pycudadecon/pull/18) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:- General cleanup around library wrap [\#17](https://github.com/tlambert03/pycudadecon/pull/17) ([tlambert03](https://github.com/tlambert03))
CHANGELOG.md:## [v0.1.2](https://github.com/tlambert03/pycudadecon/tree/v0.1.2) (2019-03-12)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.1.1...v0.1.2)
CHANGELOG.md:## [v0.1.1](https://github.com/tlambert03/pycudadecon/tree/v0.1.1) (2019-03-11)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/v0.1.0...v0.1.1)
CHANGELOG.md:## [v0.1.0](https://github.com/tlambert03/pycudadecon/tree/v0.1.0) (2019-03-06)
CHANGELOG.md:[Full Changelog](https://github.com/tlambert03/pycudadecon/compare/4155fd1d86878be6a5de77b84eaebf27c4ccbd10...v0.1.0)
CHANGELOG.md:- Update README.md [\#2](https://github.com/tlambert03/pycudadecon/pull/2) ([tlambert03](https://github.com/tlambert03))
tests/test_otf.py:from pycudadecon import make_otf
tests/test_otf.py:from pycudadecon.util import imread, is_otf
tests/test_decon.py:from pycudadecon import RLContext, decon, rl_cleanup, rl_decon, rl_init
tests/test_decon.py:from pycudadecon.util import imread
tests/conftest.py:from pycudadecon.util import imread
tests/test_affine.py:from pycudadecon import affineGPU, deskewGPU, rotateGPU
tests/test_affine.py:    deskewed = deskewGPU(raw_image, dzdata=0.3, angle=31.5, pad_val=98)
tests/test_affine.py:        affineGPU(raw_image, np.eye(5))
tests/test_affine.py:        affineGPU(raw_image, np.eye(3))
tests/test_affine.py:        affineGPU(raw_image, np.eye(2))
tests/test_affine.py:    result = affineGPU(raw_image, np.eye(4))
tests/test_affine.py:    result = affineGPU(raw_image, T)
tests/test_affine.py:    result = affineGPU(raw_image, T, voxsize)
tests/test_affine.py:    """test that rotateGPU rotates the image on the Y axis by some angle"""
tests/test_affine.py:    result = rotateGPU(deskewed_image, 0.3, dxdata=0.1, angle=31.5)
README.md:# pycudadecon
README.md:[cudaDecon](https://github.com/scopetools/cudaDecon), which is a CUDA/C++
README.md:* CUDA accelerated deconvolution with a handful of artifact-reducing features.
README.md:* CUDA-based camera-correction for [sCMOS artifact correction](https://llspy.readthedocs.io/en/latest/camera.html)
README.md:The conda package includes the required pre-compiled libraries for Windows and Linux. See GPU driver requirements [below](#gpu-requirements)
README.md:conda install -c conda-forge pycudadecon
README.md:### 📖   &nbsp; [Documentation](http://www.talleylambert.com/pycudadecon)
README.md:### GPU requirements
README.md:This software requires a CUDA-compatible NVIDIA GPU.
README.md:versions of the CUDA toolkit. The required CUDA libraries are bundled in the
README.md:conda distributions so you don't need to install the CUDA toolkit separately.
README.md:If desired, you may specify cuda-version as follows:
README.md:conda install -c conda-forge pycudadecon cuda-version=<11 or 12>
README.md:[minimum required driver version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#cuda-11-and-later-defaults-to-minor-version-compatibility)
README.md:installed for the CUDA version you are using.
README.md:[`pycudadecon.decon()`](https://www.talleylambert.com/pycudadecon/deconvolution.html#pycudadecon.decon)
README.md:from pycudadecon import decon
README.md:[`pycudadecon.make_otf()`](https://www.talleylambert.com/pycudadecon/otf.html#pycudadecon.make_otf),
README.md:[`pycudadecon.RLContext`](https://www.talleylambert.com/pycudadecon/deconvolution.html#pycudadecon.RLContext)
README.md:context manager to setup the GPU for use with the
README.md:[`pycudadecon.rl_decon()`](https://www.talleylambert.com/pycudadecon/deconvolution.html#pycudadecon.rl_decon)
README.md:from pycudadecon import RLContext, rl_decon
README.md:If you have a 3D PSF volume, the [`pycudadecon.TemporaryOTF`](https://www.talleylambert.com/pycudadecon/otf.html#pycudadecon.TemporaryOTF) context manager facilitates temporary OTF generation...
README.md:... and that bit of code is essentially what the [`pycudadecon.decon()`](https://www.talleylambert.com/pycudadecon/deconvolution.html#pycudadecon.decon) function is doing, with a little bit of additional conveniences added in.
README.md:*Each of these functions has many options and accepts multiple keyword arguments. See the [documentation](https://www.talleylambert.com/pycudadecon/index.html) for further information on the respective functions.*
README.md:For examples and information on affine transforms, volume rotations, and deskewing (typical of light sheet volumes acquired with stage-scanning), see the [documentation on Affine Transformations](https://www.talleylambert.com/pycudadecon/affine.html)
pyproject.toml:name = "pycudadecon"
pyproject.toml:description = "Python wrapper for CUDA-accelerated 3D deconvolution"
pyproject.toml:keywords = ["deconvolution", "microscopy", "CUDA"]
pyproject.toml:  "Environment :: GPU :: NVIDIA CUDA",
pyproject.toml:Documentation = "https://tlambert03.github.io/pycudadecon/"
pyproject.toml:Source = "https://github.com/tlambert03/pycudadecon"
pyproject.toml:Tracker = "https://github.com/tlambert03/pycudadecon/issues"
pyproject.toml:pycudadecon = "pycudadecon:napari.yaml"
pyproject.toml:source = ["pycudadecon"]
.github_changelog_generator:project=pycudadecon
examples/nd2_deskew.py:"""Example of deskewing an ND2 file using pycudadecon."""
examples/nd2_deskew.py:from pycudadecon.affine import affineGPU
examples/nd2_deskew.py:            out = np.stack([affineGPU(chan, tmat).T for chan in frame])
.gitignore:pycudadecon/_version.py
src/pycudadecon/_libwrap.py:    lib = Library("libcudaDecon")
src/pycudadecon/_libwrap.py:        "Unable to find library 'lidbcudaDecon'\n"
src/pycudadecon/_libwrap.py:        "Please try `conda install -c conda-forge cudadecon`."
src/pycudadecon/_libwrap.py:    """Return the version of the cudadecon library. Example b'0.7.0'."""
src/pycudadecon/_libwrap.py:            fname = next(conda_meta.glob("cudadecon*.json"), None)
src/pycudadecon/_libwrap.py:    """Release GPU buffer and cleanup after deconvolution.
src/pycudadecon/_libwrap.py:    Call this before program quits to release global GPUBuffer d_interpOTF.
src/pycudadecon/_libwrap.py:    - Removes OTF from GPU buffer
src/pycudadecon/_libwrap.py:    - Releases GPU buffers
src/pycudadecon/_libwrap.py:def cuda_reset() -> None:
src/pycudadecon/_libwrap.py:    """Calls `cudaDeviceReset`.
src/pycudadecon/_libwrap.py:        "Please try `conda install -c conda-forge cudadecon`."
src/pycudadecon/camcor.py:    """Correct Flash residual pixel artifact on GPU."""
src/pycudadecon/camcor.py:    """Initialize camera correction on GPU.
src/pycudadecon/camcor.py:    """Perform residual pixel artifact correction on GPU."""
src/pycudadecon/otf.py:        :func:`pycudadecon.otf.make_otf` function
src/pycudadecon/affine.py:"""Affine transformations on 3D volumes using CUDA."""
src/pycudadecon/affine.py:def deskewGPU(
src/pycudadecon/affine.py:    """Deskew data acquired in stage-scanning mode on GPU.
src/pycudadecon/affine.py:def affineGPU(
src/pycudadecon/affine.py:    >>> rotated = affineGPU(im, T)
src/pycudadecon/affine.py:    (this is the underlying code for :func:`rotateGPU`)
src/pycudadecon/affine.py:    >>> rotated = affineGPU(im, T)
src/pycudadecon/affine.py:def rotateGPU(
src/pycudadecon/affine.py:    return affineGPU(im, T)
src/pycudadecon/deconvolution.py:    """Release GPU buffer and cleanup after deconvolution.
src/pycudadecon/deconvolution.py:    Call this before program quits to release global GPUBuffer d_interpOTF.
src/pycudadecon/deconvolution.py:    - Removes OTF from GPU buffer
src/pycudadecon/deconvolution.py:    - Releases GPU buffers
src/pycudadecon/deconvolution.py:    """Initialize GPU for deconvolution.
src/pycudadecon/deconvolution.py:    Must be used prior to :func:`pycudadecon.rl_decon`
src/pycudadecon/deconvolution.py:    Performs actual deconvolution. GPU must first be initialized with
src/pycudadecon/deconvolution.py:    :func:`pycudadecon.rl_init`
src/pycudadecon/deconvolution.py:        :class:`pycudadecon.RLContext` context, by default None
src/pycudadecon/deconvolution.py:    """Context manager to setup the GPU for RL decon.
src/pycudadecon/deconvolution.py:    Takes care of handing the OTF to the GPU, preparing a cuFFT plan,
src/pycudadecon/deconvolution.py:        provided as filenames created with :func:`pycudadecon.make_otf`)
src/pycudadecon/deconvolution.py:        # in which case we can prevent a lot of GPU IO
src/pycudadecon/napari.py:    from pycudadecon.deconvolution import decon
src/pycudadecon/util.py:"""Utility functions for pycudadecon."""
src/pycudadecon/__init__.py:"""Python wrapper for CUDA-accelerated 3D deconvolution."""
src/pycudadecon/__init__.py:    __version__ = version("pycudadecon")
src/pycudadecon/__init__.py:from .affine import affineGPU, deskewGPU, rotateGPU
src/pycudadecon/__init__.py:    "affineGPU",
src/pycudadecon/__init__.py:    "deskewGPU",
src/pycudadecon/__init__.py:    "rotateGPU",
src/pycudadecon/napari.yaml:name: pycudadecon
src/pycudadecon/napari.yaml:display_name: pyCUDAdecon
src/pycudadecon/napari.yaml:    - id: pycudadecon.deconvolve
src/pycudadecon/napari.yaml:      python_name: pycudadecon.napari:deconvolve
src/pycudadecon/napari.yaml:    - command: pycudadecon.deconvolve
src/pycudadecon/napari.yaml:      display_name: CUDA-Deconvolution

```
