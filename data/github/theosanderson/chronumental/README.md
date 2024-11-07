# https://github.com/theosanderson/chronumental

```console
README.md:What sets Chronumental apart from most other tools is that it scales to extremely large trees, which can contain millions of nodes. Chronumental uses JAX to represent the task of computing a time tree in a differentiable graph for efficient calculation on a CPU or GPU.
src/chronumental/__main__.py:GPU_REQUESTED = "--use_gpu" in sys.argv
src/chronumental/__main__.py:if not GPU_REQUESTED:
src/chronumental/__main__.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
src/chronumental/__main__.py:if GPU_REQUESTED and platform == "cpu":
src/chronumental/__main__.py:    print("GPU requested but was not available")
src/chronumental/__main__.py:    print("This probably reflects your CUDA/jaxlib installation")
src/chronumental/__main__.py:        '--use_gpu',
src/chronumental/__main__.py:        ("Will attempt to use the GPU. You will need a version of CUDA installed to suit Numpyro."

```
