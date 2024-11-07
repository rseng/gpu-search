# https://github.com/biocore/unifrac

```console
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
unifrac/_methods.py:    UNIFRAC_USE_GPU
unifrac/_methods.py:        Enable or disable GPU offload. If not defined, autodetect.
unifrac/_methods.py:        The GPU to use. If not defined, the first GPU will be used.
README.md:An example of installing UniFrac, and using it with CPUs as well as GPUs, can be be found on [Google Colabs](https://colab.research.google.com/drive/1yL0MdF1zNAkPg1_yESI1iABUH4ZHNGwj?usp=sharing).
README.md:## GPU support
README.md:On Linux platforms, Unifrac will run on a GPU, if one is found. 
README.md:To disable GPU offload, and thus force CPU-only execution, one can set:
README.md:    export UNIFRAC_USE_GPU=N
README.md:    export UNIFRAC_GPU_INFO=Y
README.md:Finally, Unifrac will only use one GPU at a time. 
README.md:If more than one GPU is present, one can select the one to use by setting:
README.md:    export ACC_DEVICE_NUM=gpunum
README.md:Note that there is no GPU support for MacOS.
README.md:        UNIFRAC_USE_GPU
README.md:            Enable or disable GPU offload. If not defined, autodetect.
README.md:            The GPU to use. If not defined, the first GPU will be used.
README.md:        UNIFRAC_USE_GPU
README.md:            Enable or disable GPU offload. If not defined, autodetect.
README.md:            The GPU to use. If not defined, the first GPU will be used.
README.md:        GPU offload can be disabled with UNIFRAC_USE_GPU=N. By default, if a NVIDIA GPU is detected, it will be used.
README.md:        A specific GPU can be selected with ACC_DEVICE_NUM. If not defined, the first GPU will be used.

```
