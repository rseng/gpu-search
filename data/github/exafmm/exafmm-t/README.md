# https://github.com/exafmm/exafmm-t

```console
TODO.md:- [ ] M2M, L2L, L2P on GPU (Elket)
TODO.md:- [ ] GPU kernels
paper.bib:	title = {A sparse octree gravitational {N}-body code that runs entirely on the {GPU} processor},
paper.bib:	keywords = {GPU, -body, Gravity, Hierarchical, Parallel, Tree-code},
paper.bib:	  title={Treecode and fast multipole method for N-body simulation with CUDA},
paper.bib:	  booktitle={GPU Computing Gems Emerald Edition},
paper.bib:@inproceedings{choiCPUGPUHybrid2014,
paper.bib:	series = {{GPGPU}-7},
paper.bib:	title = {A {CPU}: {GPU} {Hybrid} {Implementation} and {Model}-{Driven} {Scheduling} of the {Fast} {Multipole} {Method}},
paper.bib:	abstract = {This paper presents an optimized CPU--GPU hybrid implementation and a GPU performance model for the kernel-independent fast multipole method (FMM). We implement an optimized kernel-independent FMM for GPUs, and combine it with our previous CPU implementation to create a hybrid CPU+GPU FMM kernel. When compared to another highly optimized GPU implementation, our implementation achieves as much as a 1.9× speedup. We then extend our previous lower bound analyses of FMM for CPUs to include GPUs. This yields a model for predicting the execution times of the different phases of FMM. Using this information, we estimate the execution times of a set of static hybrid schedules on a given system, which allows us to automatically choose the schedule that yields the best performance. In the best case, we achieve a speedup of 1.5× compared to our GPU-only implementation, despite the large difference in computational powers of CPUs and GPUs. We comment on one consequence of having such performance models, which is to enable speculative predictions about FMM scalability on future systems.},
paper.bib:	booktitle = {Proceedings of {Workshop} on {General} {Purpose} {Processing} {Using} {GPUs}},
paper.bib:	keywords = {exascale, fast multipole method, GPU, hybrid, multicore, performance model},
include/vec.h:#ifndef __CUDACC__
include/vec.h:#pragma message("Overloading vector operators for CUDA")
m4/libtool.m4:    nvcc*) # Cuda Compiler Driver 2.2
m4/libtool.m4:	nvcc*)	# Cuda Compiler Driver 2.2
history.md:As a bit of history of this project, it started in 2008 with [`PyFMM`](https://github.com/barbagroup/pyfmm), a 2D serial prototype in Python; then followed `PetFMM` in 2009, a PETSc-based parallel code with heavy templating [@cruz2011petfmm]; the third effort was [`GemsFMM`](https://github.com/barbagroup/gemsfmm) in 2010, a serial code with CUDA kernels for execution on GPUs [@yokota2011gems].
history.md:Branch: gpu, vanilla-m2l  
history.md:GPU: P2P, M2L  
history.md:GPU: no  
history.md:GPU: separate code (Bonsai hack)  
history.md:GPU: offload all kernels  
history.md:GPU: offload all kernels  
history.md:GPU: offload all kernels  
configure:    nvcc*) # Cuda Compiler Driver 2.2
configure:	nvcc*)	# Cuda Compiler Driver 2.2
configure:    nvcc*) # Cuda Compiler Driver 2.2
configure:	nvcc*)	# Cuda Compiler Driver 2.2
paper.md:`Bonsai` [@bedorfSparseOctreeGravitational2012] is a gravitational treecode that runs entirely on GPU hardware.
paper.md:It was GPU-enabled using CUDA, parallel with MPI and exploited multithreading using OpenMP.
paper.md:Other support includes faculty start-up funds at Boston University and George Washington University, and NVIDIA via hardware donations. 

```
