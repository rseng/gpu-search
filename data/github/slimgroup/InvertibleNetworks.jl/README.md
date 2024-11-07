# https://github.com/slimgroup/InvertibleNetworks.jl

```console
test/test_layers/test_layer_conv1x1.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
test/test_layers/test_layer_conv1x1.jl:(device == gpu) && println("Testing on GPU"); 
test/test_networks/test_conditional_glow_network.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
test/test_networks/test_conditional_glow_network.jl:(device == gpu) && println("Testing on GPU"); 
test/test_networks/test_glow.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
test/test_networks/test_glow.jl:(device == gpu) && println("Testing on GPU"); 
README.md:- GPU support
README.md:## GPU support
README.md:GPU support is supported via Flux/CuArray. To use the GPU, move the input and the network layer to GPU via `|> gpu`
README.md:X = randn(Float32, nx, ny, k, batchsize) |> gpu
README.md:AN = ActNorm(k; logdet=true) |> gpu
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:CUDA = "1, 2, 3, 4, 5"
examples/networks/network_glow_dense.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
examples/networks/network_conditional_glow.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
examples/networks/network_glow.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
examples/jacobian/training_with_jacobian_simple.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
examples/benchmarks/memory_usage_invertiblenetworks.jl:device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
examples/benchmarks/memory_usage_invertiblenetworks.jl:#turn off JULIA cuda optimization to get raw peformance
examples/benchmarks/memory_usage_invertiblenetworks.jl:ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
examples/benchmarks/memory_usage_invertiblenetworks.jl:using CUDA, Printf
examples/benchmarks/memory_usage_invertiblenetworks.jl:export @gpumem
examples/benchmarks/memory_usage_invertiblenetworks.jl:function montitor_gpu_mem(used::Vector{T}, status::Ref) where {T<:Real}
examples/benchmarks/memory_usage_invertiblenetworks.jl:cleanup() = begin GC.gc(true); CUDA.reclaim(); end
examples/benchmarks/memory_usage_invertiblenetworks.jl:macro gpumem(expr)
examples/benchmarks/memory_usage_invertiblenetworks.jl:        Threads.@spawn montitor_gpu_mem(used, monitoring)
examples/benchmarks/memory_usage_invertiblenetworks.jl:        usedmem = @gpumem begin
examples/benchmarks/memory_usage_invertiblenetworks.jl:# control for memory of storing the network parameters on GPU, not relevant to backpropagation
examples/benchmarks/memory_usage_invertiblenetworks.jl:    usedmem = @gpumem begin
examples/benchmarks/memory_usage_normflows.py:#salloc -A rafael -t00:80:00 --gres=gpu:1 --mem-per-cpu=30G srun --pty python 
examples/benchmarks/memory_usage_normflows.py:import nvidia_smi
examples/benchmarks/memory_usage_normflows.py:def _get_gpu_mem(synchronize=True, empty_cache=True):
examples/benchmarks/memory_usage_normflows.py:    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
examples/benchmarks/memory_usage_normflows.py:    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
examples/benchmarks/memory_usage_normflows.py:        mem_all = _get_gpu_mem()
examples/benchmarks/memory_usage_normflows.py:        torch.cuda.synchronize()
examples/benchmarks/memory_usage_normflows.py:    nvidia_smi.nvmlInit()
examples/benchmarks/memory_usage_normflows.py:os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"
examples/benchmarks/memory_usage_normflows.py:    enable_cuda = True
examples/benchmarks/memory_usage_normflows.py:    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
examples/benchmarks/memory_usage_normflows.py:    torch.cuda.synchronize()
examples/benchmarks/memory_usage_normflows.py:    torch.cuda.empty_cache()
examples/applications/conditional_sampling/amortized_glow_mnist_inpainting.jl:function posterior_sampler(G, y, size_x; device=gpu, num_samples=1, batch_size=16)
examples/applications/conditional_sampling/amortized_glow_mnist_inpainting.jl:device = cpu #GPU does not accelerate at this small size. quicker on cpu
examples/applications/non_conditional_sampling/glow_seismic.jl:device = gpu
examples/utils/save_load_network.jl:device = gpu # Will probably by training on GPU, otherwise please change to CPU
src/networks/invertible_network_conditional_hint_multiscale.jl:        XY_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it
src/networks/invertible_network_conditional_glow.jl:        Z_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
src/networks/invertible_network_glow.jl:        Z_dims = fill!(Array{Array}(undef, max(L-1,1)), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
src/layers/invertible_layer_conv1x1.jl:    isa(Mat, CUDA.CuArray) && (Mat2 = CUDA.CuArray(Mat2)) #new Julia 1.10 subarrays require this
src/layers/invertible_layer_conv1x1.jl:        isa(X, CUDA.CuArray) && (Xi = CUDA.CuArray(Xi))
src/layers/invertible_layer_template.jl:# Functor the layer for gpu/cpu offloading
src/utils/dimensionality_operations.jl:########## Haaar wavelet, GPU supported #####################
src/utils/compute_utils.jl:using CUDA
src/utils/compute_utils.jl:cuzeros(::CuArray{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = CUDA.zeros(T, a...)
src/utils/compute_utils.jl:cuones(::CuArray{T, N}, a::Vararg{Int, N2}) where {T, N, N2} = CUDA.ones(T, a...)
src/utils/compute_utils.jl:gemm_outer!(out::CuMatrix{T}, tmp::CuVector{T}, v::CuVector{T}) where T = CUDA.CUBLAS.gemm!('N', 'T', T(1), tmp, v, T(1), out)
src/InvertibleNetworks.jl:import CUDA: CuArray
src/InvertibleNetworks.jl:# gpu

```
