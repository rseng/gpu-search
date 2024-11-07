# https://github.com/ideasrule/platon

```console
docs/questions_and_answers.rst:  If you don't have a CUDA-capable GPU, get one!  Even a cheap gaming GPU, like a $300 RTX 3060 12 GB, will speed up PLATON many-fold.
docs/index.rst:**Major changes include GPU support, free retrievals, surface emission, higher-resolution (R=20k) and more up-to-date opacities, leave-one-out cross-validation, pymultinest, and much better plotting tools.**
docs/install.rst:PLATON 6.2 uses the GPU by default, and falls back on the CPU if it can't find one.  If you have a Nvidia GPU, we highly recommend that you install CUDA and cupy, which will speed up PLATON many-fold::
docs/install.rst:PLATON will automatically detect the existence of cupy and use the GPU.  You can force it to use the CPU by setting FORCE_CPU = True in _cupy_numpy.py.
README.md:**Oct 30, 2024: PLATON v6.2 is out, with major updates!  This corresponds most closely to the version described in [Zhang et al 2024](https://arxiv.org/abs/2410.22398).  Major changes include GPU support, free retrievals, surface emission, higher-resolution (R=20k) and more up-to-date opacities, leave-one-out cross-validation, pymultinest, and much better plotting tools.**
platon/_cupy_numpy.py:        print("cupy not found. Disabling GPU acceleration")

```
