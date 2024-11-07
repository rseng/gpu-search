# https://github.com/JuliaLang/julia

```console
Make.inc:USE_POLLY_ACC := 0     # Enable GPU code-generation
stdlib/TOML/benchmark/files/Registry.toml:004fe424-f3b5-51a0-a814-026e7c39908e = { name = "CUTENSOR_CUDA110_jll", path = "C/CUTENSOR_CUDA110_jll" }
stdlib/TOML/benchmark/files/Registry.toml:052768ef-5323-5732-b1bb-66c8b64840ba = { name = "CUDA", path = "C/CUDA" }
stdlib/TOML/benchmark/files/Registry.toml:071ae1c0-96b5-11e9-1965-c90190d839ea = { name = "DiffEqGPU", path = "D/DiffEqGPU" }
stdlib/TOML/benchmark/files/Registry.toml:08131aa3-fb12-5dee-8b74-c09406e224a2 = { name = "OpenCL", path = "O/OpenCL" }
stdlib/TOML/benchmark/files/Registry.toml:0c68f7d7-f131-5f86-a1c3-88cf8149b2d7 = { name = "GPUArrays", path = "G/GPUArrays" }
stdlib/TOML/benchmark/files/Registry.toml:0e66bb0f-9c4c-572f-9142-addab54c8167 = { name = "CUDNN_CUDA101_jll", path = "C/CUDNN_CUDA101_jll" }
stdlib/TOML/benchmark/files/Registry.toml:12f4821f-d7ee-5ba6-b76b-566925c5fcc5 = { name = "AMDGPUnative", path = "A/AMDGPUnative" }
stdlib/TOML/benchmark/files/Registry.toml:17e71f27-613d-5023-914f-287e042b5c33 = { name = "CUDNN_CUDA90_jll", path = "C/CUDNN_CUDA90_jll" }
stdlib/TOML/benchmark/files/Registry.toml:21141c5a-9bdb-4563-92ae-f87d6854732e = { name = "AMDGPU", path = "A/AMDGPU" }
stdlib/TOML/benchmark/files/Registry.toml:3383e110-d2c8-5588-847f-51a846eee08b = { name = "CUTENSOR_CUDA102_jll", path = "C/CUTENSOR_CUDA102_jll" }
stdlib/TOML/benchmark/files/Registry.toml:3895d2a7-ec45-59b8-82bb-cfc6a382f9b3 = { name = "CUDAapi", path = "C/CUDAapi" }
stdlib/TOML/benchmark/files/Registry.toml:4f82f1eb-248c-5f56-a42e-99106d144614 = { name = "CUDA_full_jll", path = "C/CUDA_full_jll" }
stdlib/TOML/benchmark/files/Registry.toml:5fa5d4a9-0408-52e0-9638-7667eddd2fce = { name = "CUDNN_CUDA110_jll", path = "C/CUDNN_CUDA110_jll" }
stdlib/TOML/benchmark/files/Registry.toml:61eb1bfa-7361-4325-ad38-22787b887f55 = { name = "GPUCompiler", path = "G/GPUCompiler" }
stdlib/TOML/benchmark/files/Registry.toml:68e73e28-2238-4d5a-bf97-e5d4aa3c4be2 = { name = "DaggerGPU", path = "D/DaggerGPU" }
stdlib/TOML/benchmark/files/Registry.toml:75b8dd0a-07b1-5bf5-8859-e3022e43e992 = { name = "CUDNN_CUDA92_jll", path = "C/CUDNN_CUDA92_jll" }
stdlib/TOML/benchmark/files/Registry.toml:7ef05209-3a99-504a-91f2-5551118e1dbe = { name = "CUDAatomics", path = "C/CUDAatomics" }
stdlib/TOML/benchmark/files/Registry.toml:8ba91e8f-b0e5-5ca0-b631-aadbb9431ebf = { name = "CUDNN_CUDA102_jll", path = "C/CUDNN_CUDA102_jll" }
stdlib/TOML/benchmark/files/Registry.toml:91981db5-b9f1-5001-b8b4-81f16e12aa66 = { name = "CUDNN_CUDA100_jll", path = "C/CUDNN_CUDA100_jll" }
stdlib/TOML/benchmark/files/Registry.toml:a7aa756b-2b7f-562a-9e9d-e94076c5c8ee = { name = "OpenCL_Headers_jll", path = "O/OpenCL_Headers_jll" }
stdlib/TOML/benchmark/files/Registry.toml:ba82f77b-6841-5d2e-bd9f-4daf811aec27 = { name = "GPUifyLoops", path = "G/GPUifyLoops" }
stdlib/TOML/benchmark/files/Registry.toml:be33ccc6-a3ff-5ff2-a52e-74243cff1e17 = { name = "CUDAnative", path = "C/CUDAnative" }
stdlib/TOML/benchmark/files/Registry.toml:c2b537fd-2c7d-5f1c-8e77-78945a4d1c3a = { name = "CUTENSOR_CUDA101_jll", path = "C/CUTENSOR_CUDA101_jll" }
stdlib/TOML/benchmark/files/Registry.toml:c5f51814-7f29-56b8-a69c-e4d8f6be1fde = { name = "CUDAdrv", path = "C/CUDAdrv" }
stdlib/TOML/benchmark/files/Registry.toml:e9e359dc-d701-5aa8-82ae-09bbf812ea83 = { name = "CUDA_jll", path = "C/CUDA_jll" }
stdlib/TOML/benchmark/files/Registry.toml:eed1d86a-c4b9-5957-a3fd-78c3dc15849c = { name = "Darknet_CUDA_jll", path = "D/Darknet_CUDA_jll" }
stdlib/InteractiveUtils/test/runtests.jl:    #ccall((:OpenClipboard, "user32"), stdcall, Cint, (Ptr{Cvoid},), hDesktop) == 0 && Base.windowserror("OpenClipboard")
stdlib/InteractiveUtils/test/runtests.jl:    #    @test_throws Base.SystemError("OpenClipboard", 0, Base.WindowsErrorInfo(0x00000005, nothing)) clipboard() # ACCESS_DENIED
stdlib/InteractiveUtils/test/runtests.jl:    ccall((:OpenClipboard, "user32"), stdcall, Cint, (Ptr{Cvoid},), C_NULL) == 0 && Base.windowserror("OpenClipboard")
stdlib/InteractiveUtils/src/clipboard.jl:            if cause !== :OpenClipboard
stdlib/InteractiveUtils/src/clipboard.jl:        ccall((:OpenClipboard, "user32"), stdcall, Cint, (Ptr{Cvoid},), C_NULL) == 0 && return Base.windowserror(:OpenClipboard)
stdlib/InteractiveUtils/src/clipboard.jl:            if cause !== :OpenClipboard
stdlib/InteractiveUtils/src/clipboard.jl:        ccall((:OpenClipboard, "user32"), stdcall, Cint, (Ptr{Cvoid},), C_NULL) == 0 && return Base.windowserror(:OpenClipboard)
test/llvmpasses/loopinfo.jl:# Example from a GPU kernel where we want to unroll the outer loop
test/compiler/AbstractInterpreter.jl:raise_on_gpu1(x) = error(x)
test/compiler/AbstractInterpreter.jl:@overlay OVERLAY_MT @noinline raise_on_gpu1(x) = #=do something with GPU=# error(x)
test/compiler/AbstractInterpreter.jl:raise_on_gpu2(x) = error(x)
test/compiler/AbstractInterpreter.jl:@consistent_overlay OVERLAY_MT @noinline raise_on_gpu2(x) = #=do something with GPU=# error(x)
test/compiler/AbstractInterpreter.jl:raise_on_gpu3(x) = error(x)
test/compiler/AbstractInterpreter.jl:@consistent_overlay OVERLAY_MT @noinline Base.@assume_effects :foldable raise_on_gpu3(x) = #=do something with GPU=# error_on_gpu(x)
test/compiler/AbstractInterpreter.jl:gpu_factorial1(x::Int) = myfactorial(x, raise_on_gpu1)
test/compiler/AbstractInterpreter.jl:gpu_factorial2(x::Int) = myfactorial(x, raise_on_gpu2)
test/compiler/AbstractInterpreter.jl:gpu_factorial3(x::Int) = myfactorial(x, raise_on_gpu3)
test/compiler/AbstractInterpreter.jl:@test Base.infer_effects(gpu_factorial1, (Int,); interp=MTOverlayInterp()) |> !Core.Compiler.is_nonoverlayed
test/compiler/AbstractInterpreter.jl:@test Base.infer_effects(gpu_factorial2, (Int,); interp=MTOverlayInterp()) |> Core.Compiler.is_consistent_overlay
test/compiler/AbstractInterpreter.jl:let effects = Base.infer_effects(gpu_factorial3, (Int,); interp=MTOverlayInterp())
test/compiler/AbstractInterpreter.jl:    # N.B. the overlaid `raise_on_gpu3` is not :foldable otherwise since `error_on_gpu` is (intetionally) undefined.
test/compiler/AbstractInterpreter.jl:    Val(gpu_factorial2(3))
test/compiler/AbstractInterpreter.jl:    Val(gpu_factorial3(3))
test/compiler/AbstractInterpreter.jl:# GPUCompiler needs accurate inference through kwfunc with the overlay of `Core.throw_inexacterror`
test/compiler/AbstractInterpreter.jl:# https://github.com/JuliaGPU/CUDA.jl/issues/2241
test/compiler/AbstractInterpreter.jl:@newinterp Cuda2241Interp
test/compiler/AbstractInterpreter.jl:@MethodTable CUDA_2241_MT
test/compiler/AbstractInterpreter.jl:CC.method_table(interp::Cuda2241Interp) = CC.OverlayMethodTable(CC.get_inference_world(interp), CUDA_2241_MT)
test/compiler/AbstractInterpreter.jl:# NOTE CUDA.jl overlays `throw_boundserror` in a way that causes effects, but these effects
test/compiler/AbstractInterpreter.jl:const cuda_kernel_state = Ref{Any}()
test/compiler/AbstractInterpreter.jl:@consistent_overlay CUDA_2241_MT @inline Base.throw_boundserror(A, I) =
test/compiler/AbstractInterpreter.jl:    (cuda_kernel_state[] = (A, I); error())
test/compiler/AbstractInterpreter.jl:@test fully_eliminated(outer2241, (Nothing,); interp=Cuda2241Interp(), retval=nothing)
test/binaryplatforms.jl:    p = Platform("x86_64", "linux"; cuda = v"11")
test/binaryplatforms.jl:    Base.BinaryPlatforms.set_compare_strategy!(p, "cuda", Base.BinaryPlatforms.compare_version_cap)
test/binaryplatforms.jl:    @test R("x86_64-linux-gnu-march+x86_64-cuda+10.1") == P("x86_64", "linux"; march="x86_64", cuda="10.1")
deps/libsuitesparse.mk:	  -DSUITESPARSE_USE_CUDA=OFF \
deps/llvm.mk:LLVM_TARGETS := host;NVPTX;AMDGPU;WebAssembly;BPF;AVR
deps/llvm.mk:LLVM_CMAKE += -DPOLLY_ENABLE_GPGPU_CODEGEN=ON
deps/nvtx.mk:NVTX_GIT_URL := https://github.com/NVIDIA/NVTX.git
deps/nvtx.mk:NVTX_TAR_URL = https://api.github.com/repos/NVIDIA/NVTX/tarball/$1
doc/src/manual/parallel-computing.md:4. **GPU computing**:
doc/src/manual/parallel-computing.md:    The Julia GPU compiler provides the ability to run Julia code natively on GPUs. There
doc/src/manual/parallel-computing.md:    is a rich ecosystem of Julia packages that target GPUs. The [JuliaGPU.org](https://juliagpu.org)
doc/src/manual/parallel-computing.md:    website provides a list of capabilities, supported GPUs, related packages and documentation.
doc/src/manual/calling-c-and-fortran-code.md:This can be especially useful when targeting unusual platforms such as GPGPUs.
doc/src/manual/calling-c-and-fortran-code.md:For example, for [CUDA](https://llvm.org/docs/NVPTXUsage.html), we need to be able to read the thread index:
doc/src/manual/distributed-computing.md:A mention must be made of Julia's GPU programming ecosystem, which includes:
doc/src/manual/distributed-computing.md:1. [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) wraps the various CUDA libraries and supports compiling Julia kernels for Nvidia GPUs.
doc/src/manual/distributed-computing.md:2. [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) wraps the oneAPI unified programming model, and supports executing Julia kernels on supported accelerators. Currently only Linux is supported.
doc/src/manual/distributed-computing.md:3. [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) wraps the AMD ROCm libraries and supports compiling Julia kernels for AMD GPUs. Currently only Linux is supported.
doc/src/manual/distributed-computing.md:4. High-level libraries like [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), [Tullio.jl](https://github.com/mcabbott/Tullio.jl) and [ArrayFire.jl](https://github.com/JuliaComputing/ArrayFire.jl).
doc/src/manual/distributed-computing.md:In the following example we will use both `DistributedArrays.jl` and `CUDA.jl` to distribute an array across multiple
doc/src/manual/distributed-computing.md:julia> using CUDA
doc/src/manual/distributed-computing.md:In the following example we will use both `DistributedArrays.jl` and `CUDA.jl` to distribute an array across multiple
doc/src/devdocs/llvm-passes.md:This pass lowers a few last intrinsics to their final form targeting functions in the `libjulia` library. Separating this from `LateGCLowering` enables other backends (GPU compilation) to supply their own custom lowerings for these intrinsics, enabling the Julia pipeline to be used on those backends as well.
doc/src/devdocs/llvm-passes.md:Julia does not have the concept of a program stack as a place to allocate mutable objects. However, allocating objects on the stack reduces GC pressure and is critical for GPU compilation. Thus, `AllocOpt` performs heap to stack conversion of objects that it can prove do not [escape](https://en.wikipedia.org/wiki/Escape_analysis) the current function. It also performs a number of other optimizations on allocations, such as removing allocations that are never used, optimizing typeof calls to freshly allocated objects, and removing stores to allocations that are immediately overwritten. The escape analysis implementation is located in `llvm-alloc-helpers.cpp`. Currently, this pass does not use information from `EscapeAnalysis.jl`, though that may change in the future.
base/genericmemory.jl:`addrspace` can currently only be set to `Core.CPU`. It is designed to permit extension by other systems such as GPUs, which might define values such as:
base/genericmemory.jl:module CUDA
base/genericmemory.jl:const Generic = bitcast(Core.AddrSpace{CUDA}, 0)
base/genericmemory.jl:const Global = bitcast(Core.AddrSpace{CUDA}, 1)
base/process.jl:struct SyncCloseFD
base/process.jl:rawhandle(io::SyncCloseFD) = rawhandle(io.fd)
base/process.jl:const SpawnIO  = Union{IO, RawFD, OS_HANDLE, SyncCloseFD} # internal copy of Redirectable, removing FileRedirect and adding SyncCloseFD
base/process.jl:        syncd = Task[io.t for io in stdio if io isa SyncCloseFD]
base/process.jl:            return (SyncCloseFD(child, t), true)
base/process.jl:close_stdio(stdio::SyncCloseFD) = close_stdio(stdio.fd)
base/boot.jl:# n.b. This function exists for CUDA to overload to configure error behavior (see #48097)
base/binaryplatforms.jl:    Platform("x86_64", "windows"; cuda = "10.1")
contrib/generate_precompile.jl:precompile(Tuple{typeof(Base.setindex!), GenericMemory{:not_atomic, Union{Base.Libc.RawFD, Base.SyncCloseFD, IO}, Core.AddrSpace{Core}(0x00)}, Base.TTY, Int})
julia.spdx.json:            "copyrightText": "AMD, Copyright (c), 1996-2015, Timothy A. Davis,\nBTF, Copyright (C) 2004-2013, University of Florida\nCAMD, Copyright (c) by Timothy A. Davis, Yanqing Chen, Patrick R. Amestoy, and Iain S. Duff.  All Rights Reserved.\nCCOLAMD: Copyright (C) 2005-2016, Univ. of Florida.  Authors: Timothy A. Davis, Sivasankaran Rajamanickam, and Stefan Larimore.  Closely based on COLAMD by Davis, Stefan Larimore, in collaboration with Esmond Ng, and John Gilbert.\nCHOLMOD/Check Module.  Copyright (C) 2005-2006, Timothy A. Davis\nCHOLMOD/Cholesky module, Copyright (C) 2005-2006, Timothy A. Davis.\nCHOLMOD/Core Module.  Copyright (C) 2005-2006, Univ. of Florida.  Author: Timothy A. Davis.\nCHOLMOD/Demo Module.  Copyright (C) 2005-2006, Timothy A. Davis.\nCHOLMOD/Include/* files.  Copyright (C) 2005-2006, either Univ. of Florida or T. Davis, depending on the file\nCHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis.\nCHOLMOD/MatrixOps Module.  Copyright (C) 2005-2006, Timothy A. Davis.\nCHOLMOD/Modify Module.  Copyright (C) 2005-2006, Timothy A. Davis and William W. Hager.\nCHOLMOD/Partition Module.  Copyright (C) 2005-2006, Univ. of Florida.  Author: Timothy A. Davis\nCHOLMOD/Supernodal Module.  Copyright (C) 2005-2006, Timothy A. Davis\nCHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis\nCHOLMOD/Valgrind Module.  Copyright (C) 2005-2006, Timothy A. Davis.\nCOLAMD, Copyright 1998-2016, Timothy A. Davis.\nCSparse, Copyright (c) 2006, Timothy A. Davis.\nCXSparse: Copyright (c) 2006, Timothy A. Davis.\nGPUQREngine, Copyright (c) 2013, Timothy A. Davis, Sencer Nuri Yeralan, and Sanjay Ranka.\nKLU, Copyright (C) 2004-2013, University of Florida by Timothy A. Davis and Ekanathan Palamadai.\nLDL, Copyright (c) 2005-2013 by Timothy A. Davis.\nThe MATLAB_Tools collection of packages is Copyright (c), Timothy A. Davis, All Rights Reserved, with the exception of the spqr_rank package, which is Copyright (c), Timothy A. Davis and Les Foster, All Rights Reserved\nMATLAB_Tools, SSMULT, Copyright (c) 2007-2011, Timothy A. Davis,\nMongoose Graph Partitioning Library  Copyright (C) 2017-2018, Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager\nRBio toolbox.  Copyright (C) 2006-2009, Timothy A. Davis\nSLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno, Timothy A. Davis, Texas A&M University. \nSPQR, Copyright 2008-2016 by Timothy A. Davis.\nSuiteSparse_GPURuntime Copyright (c) 2013-2016, Timothy A. Davis, Sencer Nuri Yeralan, and Sanjay Ranka.\nUMFPACK, Copyright 1995-2009 by Timothy A. Davis.",
src/processor_arm.cpp:    // NVIDIA
src/processor_arm.cpp:    nvidia_denver1,
src/processor_arm.cpp:    nvidia_denver2,
src/processor_arm.cpp:    nvidia_carmel,
src/processor_arm.cpp:constexpr auto nvidia_denver1 = generic; // TODO? (crc, crypto)
src/processor_arm.cpp:constexpr auto nvidia_denver2 = armv8a_crc_crypto;
src/processor_arm.cpp:constexpr auto nvidia_carmel = armv8_2a_crypto | get_feature_masks(fullfp16);
src/processor_arm.cpp:    {"denver1", CPU::nvidia_denver1, CPU::generic, UINT32_MAX, Feature::nvidia_denver1},
src/processor_arm.cpp:    {"denver2", CPU::nvidia_denver2, CPU::generic, UINT32_MAX, Feature::nvidia_denver2},
src/processor_arm.cpp:    {"carmel", CPU::nvidia_carmel, CPU::generic, 110000, Feature::nvidia_carmel},
src/processor_arm.cpp:constexpr auto nvidia_denver1 = armv8a; // TODO? (crc, crypto)
src/processor_arm.cpp:constexpr auto nvidia_denver2 = armv8a_crc_crypto;
src/processor_arm.cpp:    {"denver1", CPU::nvidia_denver1, CPU::arm_cortex_a53, UINT32_MAX, Feature::nvidia_denver1},
src/processor_arm.cpp:    {"denver2", CPU::nvidia_denver2, CPU::arm_cortex_a57, UINT32_MAX, Feature::nvidia_denver2},
src/processor_arm.cpp:    case 0x4e: // 'N': NVIDIA
src/processor_arm.cpp:        case 0x000: return CPU::nvidia_denver1;
src/processor_arm.cpp:        case 0x003: return CPU::nvidia_denver2;
src/processor_arm.cpp:        case 0x004: return CPU::nvidia_carmel;
src/processor_arm.cpp:        CPU::nvidia_denver2,
src/processor_arm.cpp:        CPU::nvidia_carmel,
src/Makefile:CG_LLVMLINK += -lPollyPPCG -lGPURuntime
src/Makefile:FLAGS += -I$(shell $(LLVM_CONFIG_HOST) --src-root)/tools/polly/tools # Required to find GPURuntime/GPUJIT.h
src/aotcompile.cpp:#include <polly/Support/LinkGPURuntime.h>
src/aotcompile.cpp:// also be used be extern consumers like GPUCompiler.jl to obtain a module containing
src/aotcompile.cpp:// The `policy` flag switches between the default mode `0` and the extern mode `1` used by GPUCompiler.

```
