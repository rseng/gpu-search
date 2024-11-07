# https://github.com/RemoteSensingTools/vSmartMOM.jl

```console
docs/Manifest.toml:[[deps.CUDA]]
docs/Manifest.toml:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
docs/Manifest.toml:[[deps.CUDAKernels]]
docs/Manifest.toml:deps = ["Adapt", "CUDA", "Cassette", "KernelAbstractions", "SpecialFunctions", "StaticArrays"]
docs/Manifest.toml:[[deps.GPUArrays]]
docs/Manifest.toml:[[deps.GPUCompiler]]
docs/Manifest.toml:deps = ["CUDA", "CUDAKernels", "DataInterpolations", "DelimitedFiles", "DiffResults", "Distributions", "DocStringExtensions", "FastGaussQuadrature", "ForwardDiff", "InstrumentOperator", "Interpolations", "JLD2", "JSON", "KernelAbstractions", "LinearAlgebra", "NCDatasets", "NNlib", "NetCDF", "Parameters", "Pkg", "Polynomials", "ProgressMeter", "SpecialFunctions", "StaticArrays", "StatsBase", "TimerOutputs", "YAML"]
docs/src/pages/Absorption/Overview.md:The module also supports auto-differentiation (AD) of the profile, with respect to pressure and temperature. Calculations can be computed either on CPU or GPU (CUDA). 
docs/src/pages/Absorption/Example.md:model_voigt_GPU = make_hitran_model(hitran_data, Voigt(), architecture=Architectures.GPU())
docs/src/pages/vSmartMOM/Overview.md:- Enables GPU-accelerated computations of the resulting hyperspectral reflectances/transmittances
docs/src/pages/vSmartMOM/InputParametersGuide.md:**architecture**: Hardware architecture to use for calculations. Should be one among [`CPU`, `GPU`].
docs/src/index.md:By taking advantage of modern software tools, such as GPU acceleration and HPC computing, the software suite significantly accelerates computationally intensive calculations and models, while keeping the interface easy to use for researchers and students.
test/gpu_tests/rt_kernels.jl:function rt_interaction_GPU!(R⁻⁺, T⁺⁺, R⁺⁻, T⁻⁻, r⁻⁺, t⁺⁺, r⁺⁻, t⁻⁻, aux1, aux2, aux3,I_static)
test/gpu_tests/gpu_cpu_tests.jl:using CUDA
test/gpu_tests/gpu_cpu_tests.jl:if has_cuda_gpu()
test/gpu_tests/gpu_cpu_tests.jl:    CUDA.allowscalar(false)
test/gpu_tests/gpu_cpu_tests.jl:added_layer_GPU     = CoreRT.make_added_layer_rand(FT, CuArray, dims, nSpec); 
test/gpu_tests/gpu_cpu_tests.jl:#composite_layer_GPU = vSmartMOM.make_composite_layer(FT, CuArray, dims, nSpec)
test/gpu_tests/gpu_cpu_tests.jl:println("GPU runs:")
test/gpu_tests/gpu_cpu_tests.jl:@btime CoreRT.doubling!(pol_type, SFI, expk_, ndoubl, added_layer_GPU, I_static_, vSmartMOM.Architectures.GPU())
test/gpu_tests/inelastic.jl:using CUDA
test/gpu_tests/inelastic.jl:function gpu_MM(A, B, C)
test/gpu_tests/matrix_inv_test.jl:using CUDA
test/gpu_tests/matrix_inv_test.jl:include("CUDA_getri.jl")
test/gpu_tests/matrix_inv_test.jl:# And move to GPU as CuArray
test/gpu_tests/gpu_batched_interaction2.jl:using CUDA
test/gpu_tests/gpu_batched_interaction2.jl:if has_cuda_gpu()
test/gpu_tests/gpu_batched_interaction2.jl:    CUDA.allowscalar(false)
test/gpu_tests/gpu_batched_interaction2.jl:# And move to GPU as CuArray
test/gpu_tests/gpu_batched_interaction2.jl:println("RT Interaction GPU time:")
test/gpu_tests/gpu_batched_interaction2.jl:@testset "GPU-CPU consistency" begin
test/gpu_tests/gpu_batched_interaction2.jl:    for (matGPU, matCPU) in ((R⁻⁺, R⁻⁺_),
test/gpu_tests/gpu_batched_interaction2.jl:        @show Array(matGPU) ≈ matCPU
test/gpu_tests/gpu_batched_interaction2.jl:        @test Array(matGPU) ≈ matCPU    
test/gpu_tests/elemental_test.jl:using CUDA
test/gpu_tests/elemental_test.jl:device = KernelAbstractions.CUDADevice()
test/gpu_tests/D_matrix_test.jl:    # And move to GPU as CuArray
test/gpu_tests/D_matrix_test.jl:    @testset "GPU-CPU consistency $i" begin
test/gpu_tests/D_matrix_test.jl:        for (matGPU, matCPU) in ((r⁻⁺, r⁻⁺_),
test/gpu_tests/D_matrix_test.jl:            @test Array(matGPU) ≈ matCPU    
test/gpu_tests/D_matrix_test.jl:            for (matGPU, matCPU) in ((r⁻⁺, r⁺⁻),
test/gpu_tests/D_matrix_test.jl:                @test Array(matGPU) ≈ Array(matCPU)    
test/gpu_tests/gpu_batched_interaction.jl:using CUDA
test/gpu_tests/gpu_batched_interaction.jl:if has_cuda_gpu()
test/gpu_tests/gpu_batched_interaction.jl:    CUDA.allowscalar(false)
test/gpu_tests/gpu_batched_interaction.jl:# And move to GPU as CuArray
test/gpu_tests/gpu_batched_interaction.jl:# CUDA has no strided batched getri, but we can at least avoid constructing costly views (copied this over from gertf)
test/gpu_tests/gpu_batched_interaction.jl:    info = CUDA.zeros(Cint, size(A, 3))
test/gpu_tests/gpu_batched_interaction.jl:# CUDA has no strided batched getri, but we can at least avoid constructing costly views (copied this over from gertf)
test/gpu_tests/gpu_batched_interaction.jl:    info = CUDA.zeros(Cint, size(A, 3))
test/gpu_tests/gpu_batched_interaction.jl:println("RT Interaction GPU time:")
test/gpu_tests/gpu_batched_interaction.jl:@testset "GPU-CPU consistency" begin
test/gpu_tests/gpu_batched_interaction.jl:    for (matGPU, matCPU) in ((R⁻⁺, R⁻⁺_),
test/gpu_tests/gpu_batched_interaction.jl:        @test Array(matGPU) ≈ matCPU    
test/gpu_tests/gpu_batched_interaction.jl:println("RT Doubling GPU time:")
test/gpu_tests/gpu_batched_interaction.jl:@testset "GPU-CPU consistency" begin
test/gpu_tests/gpu_batched_interaction.jl:    for (matGPU, matCPU) in ((R⁻⁺, R⁻⁺_),
test/gpu_tests/gpu_batched_interaction.jl:        @test Array(matGPU) ≈ matCPU    
test/gpu_tests/gpu_batched_interaction.jl:println("RT Elemental GPU time:")
test/gpu_tests/gpu_batched_interaction.jl:@testset "GPU-CPU consistency" begin
test/gpu_tests/gpu_batched_interaction.jl:    for (matGPU, matCPU) in ((R⁻⁺, R⁻⁺_),
test/gpu_tests/gpu_batched_interaction.jl:        @test Array(matGPU) ≈ matCPU    
test/rami/RamiGas.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/testLeaf.jl:using CUDA
test/rami/RamiGasI_12.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiGasI_8a.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiNoGasI.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiGasI_11.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiGasI_4.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiGasI.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiNoGas.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiGasI_2.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/rami/RamiGasI_3.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/benchmarks/6SV1_1_simple.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/benchmarks/6SV1_1.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/benchmarks/natraj_I.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/benchmarks/natraj.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/benchmarks/natraj_v2.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/test_parameters/ThreeBandsParameters.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/test_parameters/CO2WParameters2.yaml:  # Architecture (default_architecture, Architectures.GPU(), Architectures.CPU())
test/test_parameters/CO2WParameters2.yaml:  architecture:       Architectures.GPU() # default_architecture
test/test_parameters/S2BandParameters.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/test_parameters/O2ParametersVS.yaml:  # Architecture (default_architecture, Architectures.GPU(), Architectures.CPU())
test/test_parameters/2BandParameters.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/test_parameters/3BandParameters.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/test_parameters/PureRayleighParameters.yaml:  # Architecture (default_architecture, GPU(), CPU())
test/test_parameters/O2Parameters.yaml:  # Architecture (default_architecture, Architectures.GPU(), Architectures.CPU())
test/test_parameters/O2Parameters.yaml:  architecture:       Architectures.GPU() # default_architecture
test/test_parameters/O2Parameters2.yaml:  # Architecture (default_architecture, Architectures.GPU(), Architectures.CPU())
test/test_parameters/O2Parameters2.yaml:  architecture:       Architectures.GPU() # default_architecture
test/test_parameters/O2ACO2WSParameters.yaml:  # Architecture (default_architecture, Architectures.GPU(), Architectures.CPU())
test/test_parameters/O2ACO2WSParameters.yaml:  architecture:       Architectures.GPU() # default_architecture
test/test_parameters/3BandParameters_canopy.yaml:  # Architecture (default_architecture, GPU(), CPU())
README.md:By taking advantage of modern software tools, such as GPU acceleration and HPC computing, the software suite significantly accelerates computationally-intensive calculations and models, while keeping the interface easy-to-use for researchers and students.
README.md:  3. Enables GPU-accelerated computations of the resulting hyperspectral reflectances/transmittances.
README.md:This module enables absorption cross-section calculations of atmospheric gases at different pressures, temperatures, and broadeners (Doppler, Lorentzian, Voigt). It uses the <a href=https://hitran.org>HITRAN</a> energy transition database for calculations. While it enables lineshape calculations from scratch, it also allows users to create and save an interpolator object at specified wavelength, pressure, and temperature grids. It can perform these computations either on CPU or GPU. <br><img src='docs/src/assets/CrossSectionGIF.gif' class='center'></img><br> Key functions:
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:CUDA = "4, 5"
src/Absorption/compute_absorption_cross_section.jl:            synchronize_if_gpu()
src/Absorption/Absorption.jl:using KernelAbstractions        # For heterogeneous (GPU+CPU) programming
src/Absorption/Absorption.jl:using CUDA.CUDAKernels               # Access to CUDADevice
src/Absorption/Absorption.jl:using CUDA                      # For GPU programming
src/Absorption/Absorption.jl:using ..Architectures           # For GPU/CPU convenience
src/Absorption/Absorption.jl:using ..Architectures: CPU, GPU # Again for GPU/CPU convenience
src/Absorption/make_model_helpers.jl:    if architecture isa GPU && !(CEF isa HumlicekWeidemann32SDErrorFunction)
src/Absorption/make_model_helpers.jl:        @warn "Cross-section calculations on GPU may or may not work with this CEF (use HumlicekWeidemann32SDErrorFunction if you encounter issues)"
src/Absorption/make_model_helpers.jl:    if architecture isa Architectures.GPU && !(CEF isa HumlicekWeidemann32SDErrorFunction)
src/Absorption/make_model_helpers.jl:        @warn "Cross-section calculations on GPU may or may not work with this CEF (use HumlicekWeidemann32SDErrorFunction if you encounter issues)"
src/Scattering/Scattering.jl:using KernelAbstractions        # For heterogeneous (GPU+CPU) programming
src/Scattering/Scattering.jl:using CUDA                      # For GPU programming
src/vSmartMOM.jl:using CUDA
src/vSmartMOM.jl:export CPU, GPU, default_architecture, array_type
src/vSmartMOM.jl:# GPU/CPU Architecture (from Oceanigans)
src/vSmartMOM.jl:# Perform some GPU setup when the module is loaded
src/vSmartMOM.jl:    @hascuda begin
src/vSmartMOM.jl:        @info "CUDA-enabled GPU(s) detected"
src/vSmartMOM.jl:        for (gpu, dev) in enumerate(CUDA.devices())
src/vSmartMOM.jl:            @info "$dev: $(CUDA.name(dev))"
src/vSmartMOM.jl:	CUDA.allowscalar(false)
src/CoreRT/Surfaces/lambertian_surface.jl:    - `architecture` Compute architecture (GPU,CPU)
src/CoreRT/Surfaces/rpv_surface.jl:    - `architecture` Compute architecture (GPU,CPU)
src/CoreRT/Surfaces/rpv_surface.jl:    synchronize_if_gpu();
src/CoreRT/rt_run_bck.jl:                architecture::AbstractArchitecture)   # Whether to use CPU / GPU
src/CoreRT/CoreRT.jl:using CUDA                         # GPU CuArrays and functions
src/CoreRT/CoreRT.jl:using KernelAbstractions           # Abstracting code for CPU/GPU
src/CoreRT/CoreRT.jl:using CUDA.CUDAKernels
src/CoreRT/CoreRT.jl:# GPU
src/CoreRT/CoreRT.jl:include("gpu_batched.jl")                   # Batched operations
src/CoreRT/DefaultParameters.yaml:  # Architecture (default_architecture, GPU(), CPU())
src/CoreRT/types.jl:    "Architecture to use for calculations (CPU/GPU)"
src/CoreRT/CoreKernel/rt_kernel.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/rt_kernel.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/rt_kernel.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/rt_kernel.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/rt_kernel.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/rt_kernel.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/interaction.jl:    #CUBLAS.math_mode!(handle, CUDA.FAST_MATH)
src/CoreRT/CoreKernel/interaction.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interaction_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interaction_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interaction_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/elemental_inelastic.jl:        synchronize_if_gpu();   
src/CoreRT/CoreKernel/elemental_inelastic.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/rt_kernel_multisensor.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/rt_kernel_multisensor.jl:    else # This might not work yet on GPU!
src/CoreRT/CoreKernel/elemental_canopy.jl:        synchronize_if_gpu()
src/CoreRT/CoreKernel/elemental_canopy.jl:        synchronize_if_gpu()
src/CoreRT/CoreKernel/raman_kernel_test.jl:using CUDA.CUDAKernels
src/CoreRT/CoreKernel/raman_kernel_test.jl:using CUDA
src/CoreRT/CoreKernel/raman_kernel_test.jl:# Test GPU kernel version:
src/CoreRT/CoreKernel/raman_kernel_test.jl:if has_cuda()
src/CoreRT/CoreKernel/raman_kernel_test.jl:    device = CUDAKernels.CUDADevice()
src/CoreRT/CoreKernel/raman_kernel_test.jl:    cuda_iet⁺⁺ = Array(c_iet⁺⁺);
src/CoreRT/CoreKernel/raman_kernel_test.jl:    cuda_ier⁻⁺ = Array(c_ier⁻⁺);
src/CoreRT/CoreKernel/raman_kernel_test.jl:    cuda_iet⁺⁺ ≈ base_iet⁺⁺
src/CoreRT/CoreKernel/raman_kernel_test.jl:    cuda_ier⁻⁺ ≈ base_ier⁻⁺
src/CoreRT/CoreKernel/doubling_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/doubling_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/doubling_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/doubling_inelastic.jl:#        synchronize_if_gpu();
src/CoreRT/CoreKernel/doubling_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/doubling_inelastic.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interaction_hdrf.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interlayer_flux.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interlayer_flux.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:        synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental_inelastic_plus.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental.jl:            synchronize_if_gpu()
src/CoreRT/CoreKernel/elemental.jl:                synchronize_if_gpu()
src/CoreRT/CoreKernel/elemental.jl:        synchronize_if_gpu()
src/CoreRT/CoreKernel/elemental.jl:        synchronize_if_gpu()
src/CoreRT/CoreKernel/elemental.jl:    synchronize_if_gpu();
src/CoreRT/CoreKernel/elemental.jl:        synchronize_if_gpu();
src/CoreRT/CoreKernel/interaction_multisensor.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interaction_multisensor.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interaction_multisensor.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/interaction_multisensor.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/doubling.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/doubling.jl:    synchronize_if_gpu()
src/CoreRT/CoreKernel/doubling.jl:        synchronize_if_gpu();
src/CoreRT/CoreKernel/doubling.jl:    synchronize_if_gpu();
src/CoreRT/parameters_from_yaml.jl:              (["radiative_transfer", "architecture"], String, ["default_architecture", "Architectures.GPU()", "Architectures.CPU()", "GPU()", "CPU()"]),
src/CoreRT/parameters_from_yaml.jl:              (["radiative_transfer", "architecture"], String, ["default_architecture", "Architectures.GPU()", "Architectures.CPU()", "GPU()", "CPU()"]),
src/CoreRT/gpu_batched.jl:@inline synchronize() = CUDA.synchronize()
src/CoreRT/gpu_batched.jl:    #CUBLAS.math_mode!(CUBLAS.handle(), CUDA.FAST_MATH)
src/CoreRT/gpu_batched.jl:"Define batched matrix multiply for GPU and Duals"
src/CoreRT/gpu_batched.jl:"Define batched matrix multiply for GPU and Duals"
src/Architectures.jl:    @hascuda,
src/Architectures.jl:    AbstractArchitecture, CPU, GPU,
src/Architectures.jl:    synchronize_if_gpu
src/Architectures.jl:using CUDA
src/Architectures.jl:using CUDA.CUDAKernels
src/Architectures.jl:    GPU <: AbstractArchitecture
src/Architectures.jl:Run on a single NVIDIA CUDA GPU.
src/Architectures.jl:struct GPU <: AbstractArchitecture end
src/Architectures.jl:    @hascuda expr
src/Architectures.jl:A macro to compile and execute `expr` only if CUDA is installed and available. Generally used to
src/Architectures.jl:wrap expressions that can only be compiled if `CuArrays` and `CUDAnative` can be loaded.
src/Architectures.jl:macro hascuda(expr)
src/Architectures.jl:    return has_cuda() ? :($(esc(expr))) : :(nothing)
src/Architectures.jl:devi(::GPU) = CUDA.CUDABackend(; always_inline=true)
src/Architectures.jl:@hascuda architecture(::CuArray) = GPU()
src/Architectures.jl:@hascuda array_type(::GPU) = CuArray
src/Architectures.jl:default_architecture = has_cuda() ? GPU() : CPU()
src/Architectures.jl:synchronize_if_gpu() = has_cuda() ? CUDA.synchronize() : nothing

```
