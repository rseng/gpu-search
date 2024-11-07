# https://github.com/SISPO-developers/sispo

```console
sispo/sim/render.py:        When device="AUTO" it is attempted to use GPU first, otherwise
sispo/sim/render.py:        if device in ("AUTO", "GPU"):
sispo/sim/render.py:            if "CUDA" in device_types:
sispo/sim/render.py:                used_device = "GPU"
sispo/sim/render.py:        if self.device in ("CPU", "GPU"):
sispo/sim/render.py:            if self.device == "GPU":
sispo/sim/render.py:                device_type = "CUDA"
sispo/sim/render.py:        if self.device == "GPU":
sispo/reconstruction/openmvs.py:        use_cuda=False
sispo/reconstruction/openmvs.py:        Despite being used by default, CUDA is specifically disabled as default
sispo/reconstruction/openmvs.py:        args.extend(["--use-cuda", str(int(use_cuda))])
sispo/reconstruction/reconstruction.py:        use_cuda=False,
sispo/reconstruction/reconstruction.py:        self.use_cuda = use_cuda
sispo/reconstruction/reconstruction.py:            self.use_cuda
sispo/reconstruction/reconstruction.py:            self.use_cuda
data/input/oneshot_itokawa.json:        "device": "GPU",
data/input/oneshot_67p_opengl.json:        "device": "GPU",
data/input/definition.json:        "device": "GPU",
data/input/definition_opengl.json:        "device": "GPU",
data/input/oneshot_67p.json:        "device": "GPU",
data/input/oneshot_itokawa_opengl.json:        "device": "GPU",
doc/source/setup.rst:To make SISPO perform well, it is beneficial to install the `Nvidia CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ in case an Nvidia graphics card is available.
doc/source/setup.rst:  * If available: Activate CUDA in the cmake project and recompile
build/blender/install.sh:	-DWITH_CYCLES_CUDA_BINARIES=ON \

```
