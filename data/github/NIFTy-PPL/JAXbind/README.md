# https://github.com/NIFTy-PPL/JAXbind

```console
docs/source/internals.rst:This constraint could be easily lifted once the need arises to bridge GPU code to JAX.
docs/source/internals.rst:An alternative to consider when thinking about writing native GPU code and bridging it to JAX is Pallas.
README.md:Currently, `JAXbind` only has CPU but no GPU support.
README.md:With some expertise on Python bindings for GPU kernels adding GPU support should be fairly simple.
README.md:The interfacing with the JAX automatic differentiation engine is identical for CPU and GPU.
paper/paper.bib:  title     = {Reverse-Mode Automatic Differentiation and Optimization of GPU Kernels via Enzyme},
paper/paper.bib:  keywords  = {CUDA, LLVM, ROCm, HPC, AD, GPU, automatic differentiation},
paper/paper.tex:custom C++ or CUDA extensions\footnote{\url{https://pytorch.org/tutorials/advanced/cpp_extension.html}}.
paper/paper.tex:CPU, although the JAX built-in C++ interface also allows for custom GPU
paper/paper.tex:for custom functions that can be executed on the CPU or GPU. Custom
paper/paper.tex:memory. In the future, GPU support could be added, which should work
paper/paper.tex:any additional bindings to work on the GPU.
paper/paper.tex:differentiation and optimization of GPU kernels via enzyme.
paper/paper.md:Additionally, PyTorch allows a user to interface its C++ backend with custom C++ or CUDA extensions^[[https://pytorch.org/tutorials/advanced/cpp_extension.html](https://pytorch.org/tutorials/advanced/cpp_extension.html)].
paper/paper.md:JAXbind, in contrast, currently only supports functions executed on the CPU, although the JAX built-in C++ interface also allows for custom GPU kernels.
paper/paper.md:TensorFlow includes a C++ interface^[[https://www.tensorflow.org/guide/create_op](https://www.tensorflow.org/guide/create_op)] for custom functions that can be executed on the CPU or GPU.
paper/paper.md:In the future, GPU support could be added, which should work analogously to the CPU support in most respects.
paper/paper.md:The automatic differentiation in JAX is backend agnostic and would thus not require any additional bindings to work on the GPU.
jaxbind/jaxbind.py:    elif _platform == "gpu":
jaxbind/jaxbind.py:        raise ValueError("No GPU support")
jaxbind/jaxbind.py:    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")
jaxbind/jaxbind.py:for platform in ["cpu", "gpu"]:

```
