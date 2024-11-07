# https://github.com/conda/conda

```console
docs/source/user-guide/tasks/manage-virtual.rst:  * ``__cuda``: Maximum version of CUDA supported by the display driver.
docs/source/user-guide/tasks/manage-virtual.rst:           virtual packages : __cuda=10.0
docs/source/user-guide/tasks/manage-virtual.rst:* ``CONDA_OVERRIDE_CUDA`` - CUDA version number or set to ``""`` for no CUDA
docs/source/user-guide/concepts/data-science.rst:  libraries (such as Intel’s MKL or NVIDIA’s CUDA) which speed up performance
docs/source/dev-guide/deep-dives/solvers.md:  like `numpy>=1.19`, `python=3.*` or `pytorch=1.8.*=*cuda*`, into instances of this class.
docs/source/dev-guide/deep-dives/solvers.md:  * `build`: the build string constraints (e.g. `*cuda*`); can be empty.
CHANGELOG.md:* Removed `conda.base.context.Context.cuda_version`. (#12948)
CHANGELOG.md:* Removed `conda.common.cuda` module. (#12948)
CHANGELOG.md:* Detect CUDA driver version in subprocess. (#11667)
CHANGELOG.md:* Fix support for CUDA version detection on WSL2. (#11626)
CHANGELOG.md:* Stop showing solver hints about CUDA when it is not a dependency (#10275)
CHANGELOG.md:* Improve cuda virtual package conflict messages to show the `__cuda` virtual package as part of the conflict (#8834)
CHANGELOG.md:* virtual packages (such as cuda) are represented by leading double underscores
CHANGELOG.md:* Implement support for "virtual" CUDA packages, to make conda consider the system-installed CUDA driver and act accordingly  (#8267)
tests/plugins/test_virtual_packages.py:from conda.plugins.virtual_packages import cuda
tests/plugins/test_virtual_packages.py:def test_cuda_detection(clear_cuda_version):
tests/plugins/test_virtual_packages.py:    # confirm that CUDA detection doesn't raise exception
tests/plugins/test_virtual_packages.py:    version = cuda.cuda_version()
tests/plugins/test_virtual_packages.py:def test_cuda_override(clear_cuda_version):
tests/plugins/test_virtual_packages.py:    with env_var("CONDA_OVERRIDE_CUDA", "4.5"):
tests/plugins/test_virtual_packages.py:        version = cuda.cached_cuda_version()
tests/plugins/test_virtual_packages.py:def test_cuda_override_none(clear_cuda_version):
tests/plugins/test_virtual_packages.py:    with env_var("CONDA_OVERRIDE_CUDA", ""):
tests/plugins/test_virtual_packages.py:        version = cuda.cuda_version()
tests/plugins/test_virtual_packages.py:    clear_cuda_version: None,
tests/plugins/test_virtual_packages.py:    monkeypatch.setenv("CONDA_OVERRIDE_CUDA", "")
tests/core/test_solve.py:    get_solver_cuda,
tests/core/test_solve.py:def test_virtual_package_solver(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cudatoolkit"),)
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "10.0"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:            # Check the cuda virtual package is included in the solver
tests/core/test_solve.py:            assert "__cuda" in ssc.specs_map.keys()
tests/core/test_solve.py:                if pkgs.name == "cudatoolkit":
tests/core/test_solve.py:                    # make sure this package depends on the __cuda virtual
tests/core/test_solve.py:                    assert "__cuda" in pkgs.depends[0]
tests/core/test_solve.py:def test_solve_msgs_exclude_vp(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "10.0"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:    assert "__cuda==10.0" not in str(exc.value).strip()
tests/core/test_solve.py:def test_cuda_1(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cudatoolkit"),)
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "9.2"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:            order = add_subdir_to_iter(("channel-1::cudatoolkit-9.0-0",))
tests/core/test_solve.py:def test_cuda_2(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cudatoolkit"),)
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "10.0"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:            order = add_subdir_to_iter(("channel-1::cudatoolkit-10.0-0",))
tests/core/test_solve.py:def test_cuda_fail_1(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cudatoolkit"),)
tests/core/test_solve.py:    # No cudatoolkit in index for CUDA 8.0
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "8.0"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:  - nothing provides __cuda >=9.0 needed by cudatoolkit-9.0-0"""
tests/core/test_solve.py:  - nothing provides __cuda >=10.0 needed by cudatoolkit-10.0-0"""
tests/core/test_solve.py:  - feature:/{context._native_subdir()}::__cuda==8.0=0
tests/core/test_solve.py:  - cudatoolkit -> __cuda[version='>=10.0|>=9.0']
tests/core/test_solve.py:def test_cuda_fail_2(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cudatoolkit"),)
tests/core/test_solve.py:    # No CUDA on system
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", ""):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:  - nothing provides __cuda >=9.0 needed by cudatoolkit-9.0-0"""
tests/core/test_solve.py:  - nothing provides __cuda >=10.0 needed by cudatoolkit-10.0-0"""
tests/core/test_solve.py:  - cudatoolkit -> __cuda[version='>=10.0|>=9.0']
tests/core/test_solve.py:def test_cuda_constrain_absent(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cuda-constrain"),)
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", ""):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:            order = add_subdir_to_iter(("channel-1::cuda-constrain-11.0-0",))
tests/core/test_solve.py:def test_cuda_constrain_sat(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cuda-constrain"),)
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "10.0"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:            order = add_subdir_to_iter(("channel-1::cuda-constrain-10.0-0",))
tests/core/test_solve.py:def test_cuda_constrain_unsat(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cuda-constrain"),)
tests/core/test_solve.py:    # No cudatoolkit in index for CUDA 8.0
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "8.0"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:  - feature:|@/{context.subdir}::__cuda==8.0=0
tests/core/test_solve.py:  - __cuda[version='>=10.0'] -> feature:/linux-64::__cuda==8.0=0
tests/core/test_solve.py:def test_cuda_glibc_sat(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cuda-glibc"),)
tests/core/test_solve.py:        env_var("CONDA_OVERRIDE_CUDA", "10.0"),
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:            order = add_subdir_to_iter(("channel-1::cuda-glibc-10.0-0",))
tests/core/test_solve.py:def test_cuda_glibc_unsat_depend(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cuda-glibc"),)
tests/core/test_solve.py:    with env_var("CONDA_OVERRIDE_CUDA", "8.0"), env_var("CONDA_OVERRIDE_GLIBC", "2.23"):
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:  - feature:|@/{context.subdir}::__cuda==8.0=0
tests/core/test_solve.py:  - __cuda[version='>=10.0'] -> feature:/linux-64::__cuda==8.0=0
tests/core/test_solve.py:def test_cuda_glibc_unsat_constrain(tmpdir, clear_cuda_version):
tests/core/test_solve.py:    specs = (MatchSpec("cuda-glibc"),)
tests/core/test_solve.py:        env_var("CONDA_OVERRIDE_CUDA", "10.0"),
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_solve.py:        with get_solver_cuda(tmpdir, specs) as solver:
tests/core/test_index.py:def test_supplement_index_with_system_cuda(clear_cuda_version):
tests/core/test_index.py:    with env_vars({"CONDA_OVERRIDE_CUDA": "3.2"}):
tests/core/test_index.py:    cuda_pkg = next(iter(_ for _ in index if _.name == "__cuda"))
tests/core/test_index.py:    assert cuda_pkg.version == "3.2"
tests/core/test_index.py:    assert cuda_pkg.package_type == PackageType.VIRTUAL_SYSTEM
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:    "include/unicode/icudataver.h",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.58.2.dylib",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.58.dylib",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.a",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.dylib",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "include/unicode/icudataver.h",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.58.2.dylib",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.58.dylib",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.a",
tests/data/env_metadata/py36-osx-whl/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.dylib",
tests/data/env_metadata/py36-osx-whl/conda-meta/hdf5-1.10.2-hfa1e0ec_1.json:    "include/H5Gpublic.h",
tests/data/env_metadata/py36-osx-whl/conda-meta/hdf5-1.10.2-hfa1e0ec_1.json:        "_path": "include/H5Gpublic.h",
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Scrapy-1.5.1.dist-info/RECORD:scrapy/contracts/__init__.py,sha256=zseWeIFdrgPuKBAW3Zm6uwDRb7gXi7wGYYGeTowVB08,5416
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/pytz-2018.5.dist-info/RECORD:pytz/zoneinfo/MET,sha256=H_8zGkQU6YCX0zvsGpu_KhVdmRtXrNG7TBH4VZ-05RQ,2102
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/pytz-2018.5.dist-info/RECORD:pytz/zoneinfo/America/Argentina/Rio_Gallegos,sha256=92IGeyXMfmFBsGpurnd2TKzMy_6HFvE3DDPhaewuFyM,1109
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/zope.interface-4.5.0.dist-info/RECORD:zope/interface/tests/test_document.py,sha256=j6D7CTGRLy8IHYtMvqQBcFtXsoLaZfIpAn9Rl9cNGPU,16637
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Django-2.1.dist-info/RECORD:django/conf/locale/ka/LC_MESSAGES/django.mo,sha256=QaAqOu78WJU2RBnimMvT12_PkmNccLd4uqAGGrvRHnc,24781
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Django-2.1.dist-info/RECORD:django/contrib/admin/locale/ca/LC_MESSAGES/djangojs.po,sha256=Mfn1tgpuhe05MBQShhMoJDZ7L5Nn2p1jFy_jYIt0H0g,5098
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/PyJWT-1.6.4.dist-info/RECORD:jwt/__init__.py,sha256=IqQ3GoUX91hf1vLOBv0ztUbdlPgNWxGPua1aEsSGv-U,810
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/urllib3-1.23.dist-info/RECORD:urllib3/poolmanager.py,sha256=FHBjb7odbP2LyQRzeitgpuh1AQAPyegzmrm2b3gSZlY,16821
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/python/_inotify.py,sha256=aYbn1L_5MM62Gl60GzlkiwOHTc6mB0lUCr1tP-FsgPU,3455
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/procmon.py,sha256=e7Rr8ANff3jDDirWYHQxTGm_F5sH24eKQBRZRulrkjg,12675
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/procmontap.py,sha256=3S6LdXV2qIzUrWyNvHlk8ZMpc2Pri79PXjBIB0p6oEw,2298
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/test/test_procmon.py,sha256=juFE0_OW7YLTIr9Ohe35EK84T7NVB0NXGrz4UsuqWNg,23346
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/test/test_procmontap.py,sha256=f237F_JUHlqJQ0IDXoVBB9C_Qu2FZHJGTGOduSXWyNM,2520
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/test/__pycache__/test_procmontap.cpython-36.pyc,,
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/test/__pycache__/test_procmon.cpython-36.pyc,,
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/__pycache__/procmontap.cpython-36.pyc,,
tests/data/env_metadata/py36-osx-whl/lib/python3.6/site-packages/Twisted-18.7.0.dist-info/RECORD:twisted/runner/__pycache__/procmon.cpython-36.pyc,,
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:    "include/unicode/icudataver.h",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.58.2.dylib",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.58.dylib",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.a",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:    "lib/libicudata.dylib",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "include/unicode/icudataver.h",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.58.2.dylib",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.58.dylib",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.a",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/icu-58.2-h4b95b61_1.json:        "_path": "lib/libicudata.dylib",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/hdf5-1.10.2-hfa1e0ec_1.json:    "include/H5Gpublic.h",
tests/data/env_metadata/py27-osx-no-binary/conda-meta/hdf5-1.10.2-hfa1e0ec_1.json:        "_path": "include/H5Gpublic.h",
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/procmontap.py
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/procmon.py
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/test/test_procmontap.py
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/test/test_procmon.py
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/procmontap.pyc
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/procmon.pyc
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/test/test_procmontap.pyc
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/installed-files.txt:../twisted/runner/test/test_procmon.pyc
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/SOURCES.txt:src/twisted/runner/procmon.py
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/SOURCES.txt:src/twisted/runner/procmontap.py
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/SOURCES.txt:src/twisted/runner/test/test_procmon.py
tests/data/env_metadata/py27-osx-no-binary/lib/python2.7/site-packages/Twisted-18.7.0-py2.7.egg-info/SOURCES.txt:src/twisted/runner/test/test_procmontap.py
tests/data/repodata/r_linux-64.json:    "_r-xgboost-mutex-1.0-gpu_0.tar.bz2": {
tests/data/repodata/r_linux-64.json:      "build": "gpu_0",
tests/data/repodata/r_linux-64.json:        "_r-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/r_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/r_linux-64.json:        "_r-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/r_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/r_linux-64.json:        "_r-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/r_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/r_linux-64.json:        "_r-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/r_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/r_linux-64.json:    "r-xgboost-gpu-0.72-mro343h872fb70_0.tar.bz2": {
tests/data/repodata/r_linux-64.json:      "name": "r-xgboost-gpu",
tests/data/repodata/r_linux-64.json:    "r-xgboost-gpu-0.72-mro343h8e6da59_0.tar.bz2": {
tests/data/repodata/r_linux-64.json:      "name": "r-xgboost-gpu",
tests/data/repodata/r_linux-64.json:    "r-xgboost-gpu-0.72-r343h23e6c04_0.tar.bz2": {
tests/data/repodata/r_linux-64.json:      "name": "r-xgboost-gpu",
tests/data/repodata/r_linux-64.json:    "r-xgboost-gpu-0.72-r343h7ba8e84_0.tar.bz2": {
tests/data/repodata/r_linux-64.json:      "name": "r-xgboost-gpu",
tests/data/repodata/main_linux-64.json:    "_mutex_mxnet-0.0.10-gpu_openblas.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_openblas",
tests/data/repodata/main_linux-64.json:    "_mutex_mxnet-0.0.20-gpu_mkl.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_mkl",
tests/data/repodata/main_linux-64.json:    "_py-xgboost-mutex-1.0-gpu_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_0",
tests/data/repodata/main_linux-64.json:    "_tflow_180_select-1.0-gpu.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu",
tests/data/repodata/main_linux-64.json:    "_tflow_190_select-0.0.1-gpu.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27h03f526a_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27h37b1cb2_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27h4dc7405_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27h749159d_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27h960b796_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27hdfd716b_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27he096b04_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py27heda4471_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35h03f526a_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35h37b1cb2_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35h4dc7405_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35h749159d_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35h960b796_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35hdfd716b_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35he096b04_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py35heda4471_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36h03f526a_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36h37b1cb2_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36h4dc7405_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36h749159d_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36h960b796_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36hdfd716b_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36he096b04_2.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "caffe-gpu-1.0-py36heda4471_3.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/main_linux-64.json:    "cudatoolkit-9.0-h13b8566_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "cudatoolkit",
tests/data/repodata/main_linux-64.json:    "cudnn-7.0.5-cuda8.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "cuda8.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*"
tests/data/repodata/main_linux-64.json:    "cudnn-7.1.2-cuda9.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "cuda9.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "cudnn-7.1.3-cuda8.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "cuda8.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*"
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.*"
tests/data/repodata/main_linux-64.json:      "license": "proprietary - Nvidia",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.*"
tests/data/repodata/main_linux-64.json:      "license": "proprietary - Nvidia",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.0.8-py27hde4dcf2_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.0.8-py35ha2fb4ba_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.0.8-py36h0585f72_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.2-py27_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.2-py35_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.2-py36_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.3-py27_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.3-py35_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.3-py36_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.4-py27_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.4-py35_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.4-py36_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.5-py27_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.5-py35_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.5-py36_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.6-py27_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.6-py35_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.1.6-py36_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.2.0-0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "keras-gpu-2.2.2-0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu"
tests/data/repodata/main_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/main_linux-64.json:    "libgpuarray-0.7.5-h14c3975_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/main_linux-64.json:    "libgpuarray-0.7.6-h14c3975_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/main_linux-64.json:    "libmxnet-1.2.1-gpu_mkl_h3d71631_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_mkl_h3d71631_1",
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "libmxnet-1.2.1-gpu_mkl_he87abd8_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_mkl_he87abd8_1",
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "libmxnet-1.2.1-gpu_openblas_h1d4bbbf_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_openblas_h1d4bbbf_1",
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.10 gpu_openblas",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "libmxnet-1.2.1-gpu_openblas_hf1ee61d_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_openblas_hf1ee61d_1",
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.10 gpu_openblas",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "mxnet-gpu-1.2.1-hd441e4d_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.10 gpu_openblas",
tests/data/repodata/main_linux-64.json:      "name": "mxnet-gpu",
tests/data/repodata/main_linux-64.json:    "mxnet-gpu-1.2.1-hf82a2c8_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_linux-64.json:      "name": "mxnet-gpu",
tests/data/repodata/main_linux-64.json:    "mxnet-gpu_mkl-1.2.1-hf82a2c8_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_linux-64.json:      "name": "mxnet-gpu_mkl",
tests/data/repodata/main_linux-64.json:    "mxnet-gpu_openblas-1.2.1-hd441e4d_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "_mutex_mxnet 0.0.10 gpu_openblas",
tests/data/repodata/main_linux-64.json:      "name": "mxnet-gpu_openblas",
tests/data/repodata/main_linux-64.json:    "nccl-1.3.5-cuda9.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "cuda9.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "nccl",
tests/data/repodata/main_linux-64.json:        "_py-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "_py-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "_py-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "_py-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "_py-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "_py-xgboost-mutex 1.0 gpu_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "py-xgboost-gpu-0.72-py27h895cc61_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "py-xgboost-gpu",
tests/data/repodata/main_linux-64.json:    "py-xgboost-gpu-0.72-py27hbd78df6_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "py-xgboost-gpu",
tests/data/repodata/main_linux-64.json:    "py-xgboost-gpu-0.72-py35h895cc61_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "py-xgboost-gpu",
tests/data/repodata/main_linux-64.json:    "py-xgboost-gpu-0.72-py35hbd78df6_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "py-xgboost-gpu",
tests/data/repodata/main_linux-64.json:    "py-xgboost-gpu-0.72-py36h895cc61_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "py-xgboost-gpu",
tests/data/repodata/main_linux-64.json:    "py-xgboost-gpu-0.72-py36hbd78df6_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "name": "py-xgboost-gpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.5-py27h14c3975_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.5",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.5-py35h14c3975_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.5",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.5-py36h14c3975_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.5",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.6-py27h035aef0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.6-py27h3010b51_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.6-py35h3010b51_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.6-py36h035aef0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.6-py36h3010b51_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pygpu-0.7.6-py37h035aef0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_linux-64.json:      "name": "pygpu",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py27cuda7.5cudnn5.1_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py27cuda7.5cudnn5.1_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py27cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py27cuda7.5cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py27cuda8.0cudnn5.1_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py27cuda8.0cudnn5.1_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py27cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py27cuda8.0cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py35cuda7.5cudnn5.1_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py35cuda7.5cudnn5.1_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py35cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py35cuda7.5cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py35cuda8.0cudnn5.1_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py35cuda8.0cudnn5.1_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py35cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py35cuda8.0cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py36cuda7.5cudnn5.1_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py36cuda7.5cudnn5.1_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py36cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py36cuda7.5cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py36cuda8.0cudnn5.1_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py36cuda8.0cudnn5.1_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.2.0-py36cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py36cuda8.0cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py27cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py27cuda7.5cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py27cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py27cuda8.0cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py27cuda8.0cudnn7.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py27cuda8.0cudnn7.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py35cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py35cuda7.5cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py35cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py35cuda8.0cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py35cuda8.0cudnn7.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py35cuda8.0cudnn7.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py36cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py36cuda7.5cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py36cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py36cuda8.0cudnn6.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:    "pytorch-0.3.0-py36cuda8.0cudnn7.0_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "py36cuda8.0cudnn7.0_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "nccl",
tests/data/repodata/main_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/main_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/main_linux-64.json:        "_tflow_180_select ==1.0 gpu",
tests/data/repodata/main_linux-64.json:        "_tflow_180_select ==1.0 gpu",
tests/data/repodata/main_linux-64.json:        "_tflow_180_select ==1.0 gpu",
tests/data/repodata/main_linux-64.json:        "_tflow_180_select ==1.0 gpu",
tests/data/repodata/main_linux-64.json:        "_tflow_180_select ==1.0 gpu",
tests/data/repodata/main_linux-64.json:        "_tflow_180_select ==1.0 gpu",
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py27h233f449_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py27h233f449_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py27h9f529ab_0"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py27h395d940_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py27h395d940_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py27h9f529ab_1"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py27hd3a791e_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py27hd3a791e_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py27h6ecc378_0"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py35h42d5ad8_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py35h42d5ad8_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py35h6ecc378_0"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py35h60c0932_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py35h60c0932_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py35h9f529ab_0"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py35hb39db67_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py35hb39db67_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py35h9f529ab_1"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py36h02c5d5e_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py36h02c5d5e_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py36h6ecc378_0"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py36h220e158_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py36h220e158_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py36h9f529ab_0"
tests/data/repodata/main_linux-64.json:    "tensorflow-1.9.0-gpu_py36h313df88_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py36h313df88_1",
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:        "tensorflow-base ==1.9.0 gpu_py36h9f529ab_1"
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py27h6ecc378_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py27h6ecc378_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py27h9f529ab_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py27h9f529ab_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py27h9f529ab_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py27h9f529ab_1",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py35h6ecc378_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py35h6ecc378_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py35h9f529ab_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py35h9f529ab_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py35h9f529ab_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py35h9f529ab_1",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py36h6ecc378_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py36h6ecc378_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py36h9f529ab_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py36h9f529ab_0",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-base-1.9.0-gpu_py36h9f529ab_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:      "build": "gpu_py36h9f529ab_1",
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-1.4.1-0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu-base 1.4.1",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-1.5.0-0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu-base 1.5.0",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-1.6.0-0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu-base 1.6.0"
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-1.7.0-0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "tensorflow-gpu-base 1.7.0"
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-1.8.0-h7b35bdc_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "_tflow_180_select ==1.0 gpu",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-1.9.0-hf154084_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.4.1-py27h01caf0a_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.4.1-py27ha7e2fe3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.4.1-py35h01caf0a_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.4.1-py35ha7e2fe3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.4.1-py36h01caf0a_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.4.1-py36ha7e2fe3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.5.0-py27h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.5.0-py27had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.5.0-py35h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.5.0-py35had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.5.0-py36h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.5.0-py36had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py27h5b7bae4_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py27h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py27had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py27hcdda91b_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py35h5b7bae4_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py35h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py35had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py35hcdda91b_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py36h5b7bae4_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py36h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py36had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.6.0-py36hcdda91b_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py27h5b7bae4_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py27h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py27had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py27hcdda91b_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py35h5b7bae4_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py35h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py35had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py35hcdda91b_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py36h5b7bae4_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py36h8a131e3_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py36had95abb_0.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:    "tensorflow-gpu-base-1.7.0-py36hcdda91b_1.tar.bz2": {
tests/data/repodata/main_linux-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_linux-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_win-64.json:    "_mutex_mxnet-0.0.20-gpu_mkl.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu_mkl",
tests/data/repodata/main_win-64.json:    "_tflow_190_select-0.0.1-gpu.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu",
tests/data/repodata/main_win-64.json:    "cudatoolkit-8.0-4.tar.bz2": {
tests/data/repodata/main_win-64.json:      "name": "cudatoolkit",
tests/data/repodata/main_win-64.json:    "cudatoolkit-9.0-1.tar.bz2": {
tests/data/repodata/main_win-64.json:      "name": "cudatoolkit",
tests/data/repodata/main_win-64.json:    "cudnn-7.1.4-cuda8.0_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "cuda8.0_0",
tests/data/repodata/main_win-64.json:        "cudatoolkit 8.0.*"
tests/data/repodata/main_win-64.json:    "cudnn-7.1.4-cuda9.0_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "cuda9.0_0",
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*"
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.0.8-py35hfd8c95c_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.0.8-py36hb5f7954_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.2-py35_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.2-py36_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.3-py35_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.3-py36_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.4-py35_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.4-py36_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.5-py35_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.5-py36_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.6-py35_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.1.6-py36_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.2.0-0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "keras-gpu-2.2.2-0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu"
tests/data/repodata/main_win-64.json:      "name": "keras-gpu",
tests/data/repodata/main_win-64.json:    "libgpuarray-0.7.5-h0c8e037_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "name": "libgpuarray",
tests/data/repodata/main_win-64.json:    "libgpuarray-0.7.5-hfa6e2cd_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "name": "libgpuarray",
tests/data/repodata/main_win-64.json:    "libgpuarray-0.7.6-h0c8e037_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "name": "libgpuarray",
tests/data/repodata/main_win-64.json:    "libgpuarray-0.7.6-hfa6e2cd_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "name": "libgpuarray",
tests/data/repodata/main_win-64.json:    "libmxnet-1.2.1-gpu_mkl_hc8d6281_1.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu_mkl_hc8d6281_1",
tests/data/repodata/main_win-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_win-64.json:        "cudatoolkit 8.0.*",
tests/data/repodata/main_win-64.json:    "libmxnet-1.2.1-gpu_mkl_hdf6cc24_1.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu_mkl_hdf6cc24_1",
tests/data/repodata/main_win-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:    "mxnet-gpu-1.2.1-hf82a2c8_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_win-64.json:      "name": "mxnet-gpu",
tests/data/repodata/main_win-64.json:    "mxnet-gpu_mkl-1.2.1-hf82a2c8_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "_mutex_mxnet 0.0.20 gpu_mkl",
tests/data/repodata/main_win-64.json:      "name": "mxnet-gpu_mkl",
tests/data/repodata/main_win-64.json:    "pygpu-0.7.5-py27h0c8e037_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "libgpuarray ==0.7.5",
tests/data/repodata/main_win-64.json:      "name": "pygpu",
tests/data/repodata/main_win-64.json:    "pygpu-0.7.5-py35hfa6e2cd_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "libgpuarray ==0.7.5",
tests/data/repodata/main_win-64.json:      "name": "pygpu",
tests/data/repodata/main_win-64.json:    "pygpu-0.7.5-py36hfa6e2cd_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "libgpuarray ==0.7.5",
tests/data/repodata/main_win-64.json:      "name": "pygpu",
tests/data/repodata/main_win-64.json:    "pygpu-0.7.6-py27hc997a72_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_win-64.json:      "name": "pygpu",
tests/data/repodata/main_win-64.json:    "pygpu-0.7.6-py35h452e1ab_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_win-64.json:      "name": "pygpu",
tests/data/repodata/main_win-64.json:    "pygpu-0.7.6-py36h452e1ab_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_win-64.json:      "name": "pygpu",
tests/data/repodata/main_win-64.json:    "pygpu-0.7.6-py37h452e1ab_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/main_win-64.json:      "name": "pygpu",
tests/data/repodata/main_win-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/main_win-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/main_win-64.json:    "tensorflow-1.9.0-gpu_py35h0075c17_1.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu_py35h0075c17_1",
tests/data/repodata/main_win-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_win-64.json:        "tensorflow-base ==1.9.0 gpu_py35h6e53903_0"
tests/data/repodata/main_win-64.json:    "tensorflow-1.9.0-gpu_py36hfdee9c2_1.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu_py36hfdee9c2_1",
tests/data/repodata/main_win-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_win-64.json:        "tensorflow-base ==1.9.0 gpu_py36h6e53903_0"
tests/data/repodata/main_win-64.json:    "tensorflow-base-1.9.0-gpu_py35h6e53903_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu_py35h6e53903_0",
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:    "tensorflow-base-1.9.0-gpu_py36h6e53903_0.tar.bz2": {
tests/data/repodata/main_win-64.json:      "build": "gpu_py36h6e53903_0",
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:    "tensorflow-gpu-1.8.0-h21ff451_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "tensorflow-gpu-base 1.8.0"
tests/data/repodata/main_win-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_win-64.json:    "tensorflow-gpu-1.9.0-hf154084_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "_tflow_190_select ==0.0.1 gpu",
tests/data/repodata/main_win-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/main_win-64.json:    "tensorflow-gpu-base-1.8.0-py35h376609f_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_win-64.json:    "tensorflow-gpu-base-1.8.0-py36h376609f_0.tar.bz2": {
tests/data/repodata/main_win-64.json:        "cudatoolkit 9.0.*",
tests/data/repodata/main_win-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/main_win-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/main_win-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/main_win-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/main_win-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_win-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_win-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_win-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_win-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_win-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/main_win-64.json:        "pygpu >=0.7,<0.8.0a0",
tests/data/repodata/free_linux-64.json:    "caffe-gpu-1.0.0rc5-np112py27_3.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/free_linux-64.json:    "caffe-gpu-1.0.0rc5-np112py27_nomkl_3.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/free_linux-64.json:    "caffe-gpu-1.0.0rc5-np112py35_3.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/free_linux-64.json:    "caffe-gpu-1.0.0rc5-np112py35_nomkl_3.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/free_linux-64.json:    "caffe-gpu-1.0.0rc5-np112py36_3.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/free_linux-64.json:    "caffe-gpu-1.0.0rc5-np112py36_nomkl_3.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "caffe-gpu",
tests/data/repodata/free_linux-64.json:    "cudatoolkit-7.5-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "license": "proprietary - Nvidia",
tests/data/repodata/free_linux-64.json:      "name": "cudatoolkit",
tests/data/repodata/free_linux-64.json:    "cudatoolkit-7.5-2.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "license": "proprietary - Nvidia",
tests/data/repodata/free_linux-64.json:      "name": "cudatoolkit",
tests/data/repodata/free_linux-64.json:    "cudatoolkit-8.0-1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "license": "proprietary - Nvidia",
tests/data/repodata/free_linux-64.json:      "name": "cudatoolkit",
tests/data/repodata/free_linux-64.json:    "cudatoolkit-8.0-3.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "name": "cudatoolkit",
tests/data/repodata/free_linux-64.json:    "cudnn-5.1.10-cuda7.5_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda7.5_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*"
tests/data/repodata/free_linux-64.json:    "cudnn-5.1.10-cuda8.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda8.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*"
tests/data/repodata/free_linux-64.json:    "cudnn-6.0.21-cuda7.5_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda7.5_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*"
tests/data/repodata/free_linux-64.json:    "cudnn-6.0.21-cuda8.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda8.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*"
tests/data/repodata/free_linux-64.json:        "caffe-gpu",
tests/data/repodata/free_linux-64.json:    "keras-gpu-2.0.2-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/free_linux-64.json:    "keras-gpu-2.0.2-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/free_linux-64.json:    "keras-gpu-2.0.2-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/free_linux-64.json:    "keras-gpu-2.0.5-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/free_linux-64.json:    "keras-gpu-2.0.5-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/free_linux-64.json:    "keras-gpu-2.0.5-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:      "name": "keras-gpu",
tests/data/repodata/free_linux-64.json:    "libgpuarray-0.6.2-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/free_linux-64.json:    "libgpuarray-0.6.4-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/free_linux-64.json:    "libgpuarray-0.6.8-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/free_linux-64.json:    "libgpuarray-0.6.9-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/free_linux-64.json:    "libtorch-gpu-0.1.12-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "libtorch-gpu",
tests/data/repodata/free_linux-64.json:    "libtorch-gpu-0.1.12-nomkl_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "libtorch-gpu",
tests/data/repodata/free_linux-64.json:    "magma-2.2.0-cuda7.5_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda7.5_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*"
tests/data/repodata/free_linux-64.json:    "magma-2.2.0-cuda8.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda8.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*"
tests/data/repodata/free_linux-64.json:    "nccl-1.3.4-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5"
tests/data/repodata/free_linux-64.json:      "name": "nccl",
tests/data/repodata/free_linux-64.json:    "nccl-1.3.4-cuda7.5_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda7.5_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*"
tests/data/repodata/free_linux-64.json:      "name": "nccl",
tests/data/repodata/free_linux-64.json:    "nccl-1.3.4-cuda8.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "cuda8.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*"
tests/data/repodata/free_linux-64.json:      "name": "nccl",
tests/data/repodata/free_linux-64.json:    "numbapro_cudalib-0.1-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "name": "numbapro_cudalib",
tests/data/repodata/free_linux-64.json:    "numbapro_cudalib-0.2-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "name": "numbapro_cudalib",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:        "cudatoolkit",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.2-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.2",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.2-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.2",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.2-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.2",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.4-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.4",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.4-py27_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.4",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.4-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.4",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.4-py35_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.4",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.4-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.4",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.4-py36_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.4",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.8-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.8-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.8-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.9-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.9-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pygpu-0.6.9-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/free_linux-64.json:      "name": "pygpu",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py27cuda7.5cudnn5.1_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda7.5cudnn5.1_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py27cuda7.5cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda7.5cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py27cuda8.0cudnn5.1_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda8.0cudnn5.1_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py27cuda8.0cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda8.0cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py35cuda7.5cudnn5.1_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda7.5cudnn5.1_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py35cuda7.5cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda7.5cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py35cuda8.0cudnn5.1_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda8.0cudnn5.1_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py35cuda8.0cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda8.0cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py36cuda7.5cudnn5.1_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda7.5cudnn5.1_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py36cuda7.5cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda7.5cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py36cuda8.0cudnn5.1_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda8.0cudnn5.1_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-0.1.12-py36cuda8.0cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda8.0cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:    "pytorch-gpu-0.1.12-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:        "libtorch-gpu 0.1.12",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:      "name": "pytorch-gpu",
tests/data/repodata/free_linux-64.json:    "pytorch-gpu-0.1.12-py27_nomkl_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:        "libtorch-gpu 0.1.12",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:      "name": "pytorch-gpu",
tests/data/repodata/free_linux-64.json:    "pytorch-gpu-0.1.12-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:        "libtorch-gpu 0.1.12",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:      "name": "pytorch-gpu",
tests/data/repodata/free_linux-64.json:    "pytorch-gpu-0.1.12-py35_nomkl_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:        "libtorch-gpu 0.1.12",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:      "name": "pytorch-gpu",
tests/data/repodata/free_linux-64.json:    "pytorch-gpu-0.1.12-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:        "libtorch-gpu 0.1.12",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:      "name": "pytorch-gpu",
tests/data/repodata/free_linux-64.json:    "pytorch-gpu-0.1.12-py36_nomkl_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:        "libtorch-gpu 0.1.12",
tests/data/repodata/free_linux-64.json:        "nccl",
tests/data/repodata/free_linux-64.json:      "name": "pytorch-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.0.1-py27_4.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.0.1-py35_4.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.0.1-py36_4.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.1.0-np111py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.1.0-np111py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.1.0-np111py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.1.0-np112py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.1.0-np112py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.1.0-np112py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "cudatoolkit ==7.5",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py27cuda7.5cudnn5.1_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda7.5cudnn5.1_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py27cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda7.5cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py27cuda8.0cudnn5.1_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda8.0cudnn5.1_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py27cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda8.0cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py35cuda7.5cudnn5.1_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda7.5cudnn5.1_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py35cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda7.5cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py35cuda8.0cudnn5.1_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda8.0cudnn5.1_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py35cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda8.0cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py36cuda7.5cudnn5.1_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda7.5cudnn5.1_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py36cuda7.5cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda7.5cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 7.5*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py36cuda8.0cudnn5.1_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda8.0cudnn5.1_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.2.1-py36cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda8.0cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-1.3.0-0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "tensorflow-gpu-base ==1.3.0",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-base-1.3.0-py27cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda8.0cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-base-1.3.0-py27cuda8.0cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py27cuda8.0cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-base-1.3.0-py35cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda8.0cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-base-1.3.0-py35cuda8.0cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py35cuda8.0cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-base-1.3.0-py36cuda8.0cudnn6.0_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda8.0cudnn6.0_0",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/free_linux-64.json:    "tensorflow-gpu-base-1.3.0-py36cuda8.0cudnn6.0_1.tar.bz2": {
tests/data/repodata/free_linux-64.json:      "build": "py36cuda8.0cudnn6.0_1",
tests/data/repodata/free_linux-64.json:        "cudatoolkit 8.0*",
tests/data/repodata/free_linux-64.json:      "name": "tensorflow-gpu-base",
tests/data/repodata/free_linux-64.json:        "pygpu >=0.6.2",
tests/data/repodata/free_linux-64.json:        "pygpu >=0.6.2",
tests/data/repodata/free_linux-64.json:        "pygpu >=0.6.2",
tests/data/repodata/free_linux-64.json:    "torchvision-gpu-0.1.8-py27_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "pytorch-gpu",
tests/data/repodata/free_linux-64.json:      "name": "torchvision-gpu",
tests/data/repodata/free_linux-64.json:    "torchvision-gpu-0.1.8-py35_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "pytorch-gpu",
tests/data/repodata/free_linux-64.json:      "name": "torchvision-gpu",
tests/data/repodata/free_linux-64.json:    "torchvision-gpu-0.1.8-py36_0.tar.bz2": {
tests/data/repodata/free_linux-64.json:        "pytorch-gpu",
tests/data/repodata/free_linux-64.json:      "name": "torchvision-gpu",
tests/data/repodata/conda-forge_linux-64.json:    "cudatoolkit-dev-9.2-py37_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "cudatoolkit-dev",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.2-np111py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.2-np111py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.2-np111py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.2-np112py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.2-np112py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.2-np112py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np111py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np111py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np111py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np112py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np112py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np112py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np113py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np113py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.5-np113py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.8-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.6.9-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.0-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.1-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.2-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.3-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.4-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.5-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.6-0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.6-h470a237_2.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "libgpuarray-0.7.6-h470a237_3.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "libgpuarray",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np111py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np111py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np111py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np112py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np112py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np112py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np113py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np113py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.8-np113py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.8",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np111py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np111py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np111py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np112py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np112py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np112py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np113py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np113py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.6.9-np113py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.6.9",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np111py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np111py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np111py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np112py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np112py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np112py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np113py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np113py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.0-np113py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.0",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np111py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np111py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np111py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np112py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np112py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np112py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np113py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np113py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.1-np113py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.1",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np111py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np111py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np111py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np112py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np112py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np112py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np113py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np113py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.3-np113py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.3",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.4-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.4",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.4-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.4",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.4-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.4",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.5-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.5",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.5-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.5",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.5-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.5",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.6-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.6",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.6-py27h7eb728f_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.6-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.6",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.6-py35h7eb728f_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray 0.7.6",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pygpu-0.7.6-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:        "libgpuarray ==0.7.6",
tests/data/repodata/conda-forge_linux-64.json:      "name": "pygpu",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2016.2-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2016.2-py35_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2016.2.1-py27_2.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2016.2.1-py35_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2016.2.1-py35_2.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2016.2.1-py36_2.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.1-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.1-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.1-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.1.1-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.1.1-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.1.1-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2.2-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2.2-py27_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2.2-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2.2-py35_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2.2-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2017.2.2-py36_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py27_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py27_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py27_2.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py35_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py35_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py35_2.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py36_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py36_1.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:    "pyopencl-2018.1.1-py36_2.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr >=0.1.5",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr >=0.1.5",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:    "r-ggpubr-0.1.6-r3.3.2_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:    "r-ggpubr-0.1.6-r3.4.1_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:    "r-ggpubr-0.1.7-r341_0.tar.bz2": {
tests/data/repodata/conda-forge_linux-64.json:      "name": "r-ggpubr",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr >=0.1.3",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr >=0.1.3",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr >=0.1.6",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr >=0.1.6",
tests/data/repodata/conda-forge_linux-64.json:        "r-ggpubr >=0.1.6",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:        "pyopencl",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:      "license": "LGPL v2 (AMD, BTF, etc), BSD 3-clause (UFget), GPL v2 (UMFPACK, RBIO, SPQR, GPUQRENGINE), Apache 2.0 (Metis)",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.6.5,<0.7",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.7.0,<0.8",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.7.0,<0.8",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.7.0,<0.8",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.7.0,<0.8",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.7.0,<0.8",
tests/data/repodata/conda-forge_linux-64.json:        "pygpu >=0.7.0,<0.8",
tests/data/index.json:  "cuda-glibc-10.0-0.tar.bz2": {
tests/data/index.json:      "__cuda>=10.0,<11"
tests/data/index.json:    "name": "cuda-glibc",
tests/data/index.json:  "cuda-constrain-10.0-0.tar.bz2": {
tests/data/index.json:      "__cuda>=10.0,<11"
tests/data/index.json:    "name": "cuda-constrain",
tests/data/index.json:  "cuda-constrain-11.0-0.tar.bz2": {
tests/data/index.json:      "__cuda>=11.0,<12"
tests/data/index.json:    "name": "cuda-constrain",
tests/data/index.json:  "cudatoolkit-9.0-0.tar.bz2": {
tests/data/index.json:      "__cuda>=9.0"
tests/data/index.json:    "name": "cudatoolkit",
tests/data/index.json:  "cudatoolkit-10.0-0.tar.bz2": {
tests/data/index.json:      "__cuda>=10.0"
tests/data/index.json:    "name": "cudatoolkit",
tests/conftest.py:def clear_cuda_version():
tests/conftest.py:    from conda.plugins.virtual_packages import cuda
tests/conftest.py:    cuda.cached_cuda_version.cache_clear()
durations/macOS.json:    "tests/core/test_index.py::test_supplement_index_with_system_cuda": 0.1903388600427573,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_1[classic]": 1.5257558877114645,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_1[libmamba]": 1.5457873964748703,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_2[classic]": 1.6598667405796872,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_2[libmamba]": 1.4533618975032567,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_constrain_absent[classic]": 1.61484762142756,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_constrain_absent[libmamba]": 1.510215914140262,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_constrain_sat[classic]": 0.0005860093060872043,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_constrain_sat[libmamba]": 0.0006494726604763479,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_constrain_unsat[classic]": 0.0005623726189371414,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_constrain_unsat[libmamba]": 0.0005631172780003643,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_fail_1[classic]": 1.6472192532389363,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_fail_1[libmamba]": 1.5531458280080805,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_fail_2[classic]": 1.6187558125376929,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_fail_2[libmamba]": 1.5638728693228319,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_glibc_sat[classic]": 0.0005549337197723068,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_glibc_sat[libmamba]": 0.0008479746759504536,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_constrain[classic]": 0.0005993239015118239,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_constrain[libmamba]": 0.0005891881583373657,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_depend[classic]": 0.0005286680629767148,
durations/macOS.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_depend[libmamba]": 0.0005749209688477625,
durations/macOS.json:    "tests/plugins/test_virtual_packages.py::test_cuda_detection": 1.1612103212711362,
durations/macOS.json:    "tests/plugins/test_virtual_packages.py::test_cuda_override": 0.0033165499078763624,
durations/macOS.json:    "tests/plugins/test_virtual_packages.py::test_cuda_override_none": 0.0029804656460904323,
durations/Windows.json:    "tests/core/test_index.py::test_supplement_index_with_system_cuda": 0.038674228611534724,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_1[classic]": 1.2364858709559894,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_1[libmamba]": 1.2586236254655605,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_2[classic]": 1.2352318487375624,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_2[libmamba]": 1.2692405070615829,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_constrain_absent[classic]": 1.241520706852123,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_constrain_absent[libmamba]": 1.2844530105258798,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_constrain_sat[classic]": 0.0005478225914695078,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_constrain_sat[libmamba]": 0.0005629955127413791,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_constrain_unsat[classic]": 0.0005349952374369462,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_constrain_unsat[libmamba]": 0.0005385892990225941,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_fail_1[classic]": 1.2411117603223156,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_fail_1[libmamba]": 1.2696036555782275,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_fail_2[classic]": 1.2374550967216185,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_fail_2[libmamba]": 1.2587812568815118,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_glibc_sat[classic]": 0.0005257884793779338,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_glibc_sat[libmamba]": 0.0005910879892379431,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_constrain[classic]": 0.0006193785543953502,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_constrain[libmamba]": 0.0005141328944800954,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_depend[classic]": 0.0005658310743745597,
durations/Windows.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_depend[libmamba]": 0.0005757016520291123,
durations/Windows.json:    "tests/plugins/test_virtual_packages.py::test_cuda_detection": 1.5489721566155017,
durations/Windows.json:    "tests/plugins/test_virtual_packages.py::test_cuda_override": 0.0026644248092564994,
durations/Windows.json:    "tests/plugins/test_virtual_packages.py::test_cuda_override_none": 0.0025749061873453307,
durations/Linux.json:    "tests/core/test_index.py::test_supplement_index_with_system_cuda": 0.025253741331072308,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_1[classic]": 0.7510277486996609,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_1[libmamba]": 0.761836418367606,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_2[classic]": 0.7436205641459507,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_2[libmamba]": 0.7512268185948918,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_constrain_absent[classic]": 0.7440319500279626,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_constrain_absent[libmamba]": 0.7520533520033565,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_constrain_sat[classic]": 0.0003609711114746547,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_constrain_sat[libmamba]": 0.0003753411341399905,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_constrain_unsat[classic]": 0.00034324481253897326,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_constrain_unsat[libmamba]": 0.00035064156899843095,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_fail_1[classic]": 0.763774410384478,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_fail_1[libmamba]": 0.7921399711397477,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_fail_2[classic]": 0.7724145429676323,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_fail_2[libmamba]": 0.7737236747621121,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_glibc_sat[classic]": 0.7477371281182639,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_glibc_sat[libmamba]": 0.7652764922638053,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_constrain[classic]": 0.00034433334638648796,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_constrain[libmamba]": 0.000353582887836666,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_depend[classic]": 0.00036427583717970046,
durations/Linux.json:    "tests/core/test_solve.py::test_cuda_glibc_unsat_depend[libmamba]": 0.0003766923768422445,
durations/Linux.json:    "tests/plugins/test_virtual_packages.py::test_cuda_detection": 0.642471484847279,
durations/Linux.json:    "tests/plugins/test_virtual_packages.py::test_cuda_override": 0.0018673300713381274,
durations/Linux.json:    "tests/plugins/test_virtual_packages.py::test_cuda_override_none": 0.0017909424623828855,
conda/plugins/virtual_packages/cuda.py:"""Detect CUDA version."""
conda/plugins/virtual_packages/cuda.py:def cuda_version():
conda/plugins/virtual_packages/cuda.py:    Attempt to detect the version of CUDA present in the operating system.
conda/plugins/virtual_packages/cuda.py:    On Windows and Linux, the CUDA library is installed by the NVIDIA
conda/plugins/virtual_packages/cuda.py:    rather than with the CUDA SDK (which is optional for running CUDA apps).
conda/plugins/virtual_packages/cuda.py:    On macOS, the CUDA library is only installed with the CUDA SDK, and
conda/plugins/virtual_packages/cuda.py:    Returns: version string (e.g., '9.2') or None if CUDA is not found.
conda/plugins/virtual_packages/cuda.py:    if "CONDA_OVERRIDE_CUDA" in os.environ:
conda/plugins/virtual_packages/cuda.py:        return os.environ["CONDA_OVERRIDE_CUDA"].strip() or None
conda/plugins/virtual_packages/cuda.py:        # Spawn a subprocess to detect the CUDA version
conda/plugins/virtual_packages/cuda.py:            target=_cuda_driver_version_detector_target,
conda/plugins/virtual_packages/cuda.py:            name="CUDA driver version detector",
conda/plugins/virtual_packages/cuda.py:def cached_cuda_version():
conda/plugins/virtual_packages/cuda.py:    """A cached version of the cuda detection system."""
conda/plugins/virtual_packages/cuda.py:    return cuda_version()
conda/plugins/virtual_packages/cuda.py:    cuda_version = cached_cuda_version()
conda/plugins/virtual_packages/cuda.py:    if cuda_version is not None:
conda/plugins/virtual_packages/cuda.py:        yield CondaVirtualPackage("cuda", cuda_version, None)
conda/plugins/virtual_packages/cuda.py:def _cuda_driver_version_detector_target(queue):
conda/plugins/virtual_packages/cuda.py:    Attempt to detect the version of CUDA present in the operating system in a
conda/plugins/virtual_packages/cuda.py:    On Windows and Linux, the CUDA library is installed by the NVIDIA
conda/plugins/virtual_packages/cuda.py:    rather than with the CUDA SDK (which is optional for running CUDA apps).
conda/plugins/virtual_packages/cuda.py:    On macOS, the CUDA library is only installed with the CUDA SDK, and
conda/plugins/virtual_packages/cuda.py:    Returns: version string (e.g., '9.2') or None if CUDA is not found.
conda/plugins/virtual_packages/cuda.py:    # Platform-specific libcuda location
conda/plugins/virtual_packages/cuda.py:            "libcuda.1.dylib",  # check library path first
conda/plugins/virtual_packages/cuda.py:            "libcuda.dylib",
conda/plugins/virtual_packages/cuda.py:            "/usr/local/cuda/lib/libcuda.1.dylib",
conda/plugins/virtual_packages/cuda.py:            "/usr/local/cuda/lib/libcuda.dylib",
conda/plugins/virtual_packages/cuda.py:            "libcuda.so",  # check library path first
conda/plugins/virtual_packages/cuda.py:            "/usr/lib64/nvidia/libcuda.so",  # RHEL/Centos/Fedora
conda/plugins/virtual_packages/cuda.py:            "/usr/lib/x86_64-linux-gnu/libcuda.so",  # Ubuntu
conda/plugins/virtual_packages/cuda.py:            "/usr/lib/wsl/lib/libcuda.so",  # WSL
conda/plugins/virtual_packages/cuda.py:        lib_filenames = [f"nvcuda{bits}.dll", "nvcuda.dll"]
conda/plugins/virtual_packages/cuda.py:        queue.put(None)  # CUDA not available for other operating systems
conda/plugins/virtual_packages/cuda.py:            libcuda = dll.LoadLibrary(lib_filename)
conda/plugins/virtual_packages/cuda.py:    # Empty `CUDA_VISIBLE_DEVICES` can cause `cuInit()` returns `CUDA_ERROR_NO_DEVICE`
conda/plugins/virtual_packages/cuda.py:    # Invalid `CUDA_VISIBLE_DEVICES` can cause `cuInit()` returns `CUDA_ERROR_INVALID_DEVICE`
conda/plugins/virtual_packages/cuda.py:    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
conda/plugins/virtual_packages/cuda.py:    # Get CUDA version
conda/plugins/virtual_packages/cuda.py:        cuInit = libcuda.cuInit
conda/plugins/virtual_packages/cuda.py:        cuDriverGetVersion = libcuda.cuDriverGetVersion
conda/plugins/virtual_packages/__init__.py:from . import archspec, conda, cuda, freebsd, linux, osx, windows
conda/plugins/virtual_packages/__init__.py:plugins = [archspec, conda, cuda, freebsd, linux, osx, windows]
conda/testing/helpers.py:# Do not memoize this get_index to allow different CUDA versions to be detected
conda/testing/helpers.py:def get_index_cuda(subdir=context.subdir, add_pip=True, merge_noarch=False):
conda/testing/helpers.py:    elif channel_id == "cuda":
conda/testing/helpers.py:        get_index_cuda(context.subdir, add_pip, merge_noarch)
conda/testing/helpers.py:def get_solver_cuda(
conda/testing/helpers.py:        "cuda",

```
