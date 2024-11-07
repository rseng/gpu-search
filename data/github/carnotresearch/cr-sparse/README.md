# https://github.com/carnotresearch/cr-sparse

```console
README.rst:Python code to get efficiently compiled on CPU, GPU and TPU architectures
docs/tutorials/jax.rst:numerical code is running on GPU hardware).
docs/intro.rst:and GPU architectures.
docs/intro.rst:devices like CPUs, GPUs and custom accelerators (like Google TPUs).
docs/benchmarks/omp.rst:* Average time taken in CPU/GPU configurations
docs/benchmarks/omp.rst:* CPU and GPU configurations Google Colab have been used
docs/benchmarks/omp.rst:      - GPU 
docs/benchmarks/omp.rst:      - GPU + JIT
docs/benchmarks/omp.rst:      - GPU / GPU + JIT
docs/benchmarks/omp.rst:      - CPU + JIT / GPU + JIT
docs/benchmarks/omp.rst:  in both CPU and GPU architectures
docs/benchmarks/omp.rst:* Current implementation seems to be slower on GPU vs CPU with JIT. 
docs/benchmarks/omp.rst:* GPU speed gain over CPU (with JIT on) is relatively meager. 
docs/benchmarks/omp.rst:  On TensorFlow, people regularly report 30x improvements between CPU to GPU 
docs/benchmarks/omp.rst:* GPUs may not be great at solving triangular systems. 
docs/benchmarks/comparison.rst:  32-bit computation would be more relevant for GPUs.
paper/paper.md:Python code to get efficiently compiled on CPU, GPU and TPU architectures
paper/paper.md:Python wavelets implementation which can work across CPUs, GPUs and TPUs.
paper/paper.md:in the form of C/C++ extensions making portability to GPUs harder. 
paper/paper.md:written entirely in C. There are several attempts to port it on GPU
paper/paper.md:[`PyLops`](https://github.com/PyLops/pylops) includes GPU support. 
paper/paper.md:`NumPy` and [`CuPy`](https://cupy.dev/) for GPU support. 
paper/paper.md:NVIDIA GeForce GTX 1060 6GB GPU, 
paper/paper.md:NVidia driver version 495.29.05,
paper/paper.md:CUDA version 11.5.
paper/paper.md:We see significant though variable gains achieved by `CR-Sparse` on GPU. 
paper/paper.md:GPUs tend to perform better when problem size increases as the matrix/vector 
paper/paper.md:Following table compares the runtime of linear operators in `CR-Sparse` on GPU vs 

```
