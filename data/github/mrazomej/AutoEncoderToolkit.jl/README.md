# https://github.com/mrazomej/AutoEncoderToolkit.jl

```console
docs/src/rhvae.md:2. Even for `Julia < 1.10`, we could not get `TaylorDiff` to work on `CUDA`
docs/src/rhvae.md:The default both for `cpu` and `gpu` devices is `:finite`.
docs/src/quickstart.md:    To proceed the training on a `CUDA`-compatible device, all we need to do is
docs/src/quickstart.md:    using CUDA
docs/src/quickstart.md:    # Move model to GPU
docs/src/quickstart.md:    vae = vae |> Flux.gpu
docs/src/quickstart.md:    # Move data to GPU
docs/src/quickstart.md:    train_data = train_data |> Flux.gpu
docs/src/quickstart.md:    val_data = val_data |> Flux.gpu
docs/src/hvae.md:2. Even for `Julia < 1.10`, we could not get `TaylorDiff` to work on `CUDA`
docs/src/hvae.md:The default both for `cpu` and `gpu` devices is `:finite`.
docs/src/index.md:## GPU support
docs/src/index.md:`AutoEncoderToolkit.jl` supports GPU training out of the box for `CUDA.jl`-compatible
docs/src/index.md:GPUs. The `CUDA` functionality is provided as an extension. Therefore, to train
docs/src/index.md:a model on the GPU, simply import `CUDA` into the current environment, then move
docs/src/index.md:the model and data to the GPU. The rest of the training pipeline remains the
test/vae.jl:# Check if CUDA is available
test/vae.jl:cuda_functional = haskey(Pkg.project().dependencies, "CUDA")
test/vae.jl:if cuda_functional
test/vae.jl:    println("CUDA available - running tests with CUDA.")
test/vae.jl:    # Import CUDA
test/vae.jl:    using CUDA
test/vae.jl:    if cuda_functional
test/vae.jl:        @testset "GPU | without regularization" begin
test/vae.jl:                # Upload to GPU
test/vae.jl:                vae = Flux.gpu(vae)
test/vae.jl:                        vae, CUDA.cu(data), opt_state; loss_return=true
test/vae.jl:        end # @testset "GPU | without regularization"
test/vae.jl:    end # if cuda_functional
test/hvae.jl:# Check if CUDA is available
test/hvae.jl:cuda_functional = haskey(Pkg.project().dependencies, "CUDA")
test/hvae.jl:    if cuda_functional
test/hvae.jl:        using CUDA
test/hvae.jl:        @testset "GPU | without regularization" begin
test/hvae.jl:                ) |> Flux.gpu
test/hvae.jl:                    hvae, CUDA.cu(data), opt_state; loss_return=true
test/hvae.jl:    end # if cuda_functional
test/mmdvae.jl:# Check if CUDA is available
test/mmdvae.jl:cuda_functional = haskey(Pkg.project().dependencies, "CUDA")
test/mmdvae.jl:if cuda_functional
test/mmdvae.jl:    println("CUDA available - running tests with CUDA.")
test/mmdvae.jl:    # Import CUDA
test/mmdvae.jl:    using CUDA
test/mmdvae.jl:    if cuda_functional
test/mmdvae.jl:        @testset "GPU" begin
test/mmdvae.jl:            x = CUDA.randn(Float32, data_dim, 10)
test/mmdvae.jl:            # Upload to GPU
test/mmdvae.jl:            mmdvae_gpu = Flux.gpu(mmdvae)
test/mmdvae.jl:                result = MMDVAEs.loss(mmdvae_gpu, x)
test/mmdvae.jl:                x_out = CUDA.randn(Float32, data_dim, 10)
test/mmdvae.jl:                result = MMDVAEs.loss(mmdvae_gpu, x, x_out)
test/mmdvae.jl:        end # @testset "GPU"
test/mmdvae.jl:    end # if cuda_functional
test/mmdvae.jl:    if cuda_functional
test/mmdvae.jl:        @testset "GPU" begin
test/mmdvae.jl:            x = CUDA.randn(Float32, data_dim, 10)
test/mmdvae.jl:            # Upload to GPU
test/mmdvae.jl:            mmdvae_gpu = Flux.gpu(mmdvae)
test/mmdvae.jl:                    mmdvae -> MMDVAEs.loss(mmdvae, x), mmdvae_gpu
test/mmdvae.jl:                x_out = CUDA.randn(Float32, data_dim, 10)
test/mmdvae.jl:                    mmdvae -> MMDVAEs.loss(mmdvae, x, x_out), mmdvae_gpu
test/mmdvae.jl:        end # @testset "GPU"
test/mmdvae.jl:    end # if cuda_functional
test/mmdvae.jl:    if cuda_functional
test/mmdvae.jl:        @testset "GPU | without regularization" begin
test/mmdvae.jl:            x = CUDA.randn(Float32, data_dim, 10)
test/mmdvae.jl:            # Upload to GPU
test/mmdvae.jl:            mmdvae_gpu = Flux.gpu(mmdvae)
test/mmdvae.jl:            opt = Flux.Train.setup(Flux.Optimisers.Adam(), mmdvae_gpu)
test/mmdvae.jl:                L = MMDVAEs.train!(mmdvae_gpu, x, opt; loss_return=true)
test/mmdvae.jl:                x_out = CUDA.randn(Float32, data_dim, 10)
test/mmdvae.jl:                L = MMDVAEs.train!(mmdvae_gpu, x, x_out, opt; loss_return=true)
test/mmdvae.jl:        end # @testset "GPU | without regularization"
test/mmdvae.jl:    end # if cuda_functional
test/cuda_ext.jl:# Check if CUDA is available
test/cuda_ext.jl:cuda_functional = haskey(Pkg.project().dependencies, "CUDA")
test/cuda_ext.jl:if cuda_functional
test/cuda_ext.jl:    println("CUDA available - running tests with CUDA.")
test/cuda_ext.jl:    # Import CUDA
test/cuda_ext.jl:    using CUDA
test/cuda_ext.jl:    @testset "AutoEncoderToolkitCUDAExt" begin
test/cuda_ext.jl:                @testset "CUDA.CuVector input" begin
test/cuda_ext.jl:                    diag = CUDA.CuVector{Float32}([1, 2, 3])
test/cuda_ext.jl:                    lower = CUDA.CuVector{Float32}([4, 5, 6])
test/cuda_ext.jl:                    expected = CUDA.CuMatrix{Float32}([1 0 0; 4 2 0; 5 6 3])
test/cuda_ext.jl:                @testset "CUDA.CuMatrix input" begin
test/cuda_ext.jl:                    diag = CUDA.CuMatrix{Float32}([1 4; 2 5; 3 6])
test/cuda_ext.jl:                    lower = CUDA.CuMatrix{Float32}([7 10; 8 11; 9 12])
test/cuda_ext.jl:                    @test result ≈ CUDA.CuArray(expected)
test/cuda_ext.jl:                A = CUDA.randn(Float32, 3, 3, 5)
test/cuda_ext.jl:                @testset "CUDA.CuMatrix input" begin
test/cuda_ext.jl:                    Σ⁻¹ = CUDA.randn(Float32, 3, 3)
test/cuda_ext.jl:                @testset "CUDA.CuArray{3} input" begin
test/cuda_ext.jl:                    Σ⁻¹ = CUDA.randn(Float32, 3, 3, 5)
test/cuda_ext.jl:                @testset "CUDA.CuVector input" begin
test/cuda_ext.jl:                    x = CUDA.randn(Float32, 5)
test/cuda_ext.jl:                @testset "CUDA.CuMatrix input" begin
test/cuda_ext.jl:                    x = CUDA.randn(Float32, 5, 3)
test/cuda_ext.jl:                x = CUDA.randn(Float32, 5)
test/cuda_ext.jl:                μ = CUDA.randn(Float32, 5)
test/cuda_ext.jl:                logσ = CUDA.randn(Float32, 5)
test/cuda_ext.jl:                x = CUDA.randn(Float32, 10, 10)
test/cuda_ext.jl:                y = CUDA.randn(Float32, 10, 20)
test/cuda_ext.jl:                    @test isa(result, CUDA.CuMatrix{Float32})
test/cuda_ext.jl:                    @test isa(result, CUDA.CuMatrix{Float32})
test/cuda_ext.jl:                z = CUDA.randn(Float32, 3, 5)
test/cuda_ext.jl:                centroids_latent = CUDA.randn(Float32, 3, 10)
test/cuda_ext.jl:                M = CUDA.randn(Float32, 3, 3, 10)
test/cuda_ext.jl:                result = AutoEncoderToolkit.RHVAEs._G_inv(CUDA.CuArray, z, centroids_latent, M, T, λ)
test/cuda_ext.jl:end # if cuda_functional
test/rhvae.jl:# Check if CUDA is available
test/rhvae.jl:cuda_functional = haskey(Pkg.project().dependencies, "CUDA")
test/rhvae.jl:if cuda_functional
test/rhvae.jl:    println("CUDA available - running tests with CUDA.")
test/rhvae.jl:    # Import CUDA
test/rhvae.jl:    using CUDA
test/rhvae.jl:    if cuda_functional
test/rhvae.jl:        @testset "GPU | without regularization" begin
test/rhvae.jl:                ) |> Flux.gpu
test/rhvae.jl:                    rhvae, CUDA.cu(data), opt_state; loss_return=true
test/rhvae.jl:    end # if cuda_functional
test/infomaxvae.jl:# Check if CUDA is available
test/infomaxvae.jl:cuda_functional = haskey(Pkg.project().dependencies, "CUDA")
test/infomaxvae.jl:if cuda_functional
test/infomaxvae.jl:    println("CUDA available - running tests with CUDA.")
test/infomaxvae.jl:    # Import CUDA
test/infomaxvae.jl:    using CUDA
test/infomaxvae.jl:    if cuda_functional
test/infomaxvae.jl:        @testset "GPU" begin
test/infomaxvae.jl:            x = CUDA.randn(Float32, data_dim, 10)
test/infomaxvae.jl:            # Upload model to GPU
test/infomaxvae.jl:            infomaxvae_gpu = Flux.gpu(infomaxvae)
test/infomaxvae.jl:                Flux.Optimisers.Adam(), infomaxvae_gpu
test/infomaxvae.jl:                    infomaxvae_gpu, x, opt_infomaxvae; loss_return=true
test/infomaxvae.jl:                x_out = CUDA.randn(Float32, data_dim, 10)
test/infomaxvae.jl:                    infomaxvae_gpu, x, x_out, opt_infomaxvae; loss_return=true
test/infomaxvae.jl:        end # @testset "GPU"
test/infomaxvae.jl:    end # if cuda_functional
test/runtests.jl:# Check if CUDA is available
test/runtests.jl:cuda_functional = haskey(Pkg.project().dependencies, "CUDA")
test/runtests.jl:if cuda_functional
test/runtests.jl:    println("\nCUDA available - running tests with CUDA.\n")
test/runtests.jl:    println("\nCUDA is not available - skipping CUDA tests.\n")
test/runtests.jl:    # Test AutoEncoderToolkitCUDAExt module
test/runtests.jl:    include("cuda_ext.jl")
ext/AutoEncoderToolkitCUDAExt/vae.jl:    µ::CUDA.CuVecOrMat,
ext/AutoEncoderToolkitCUDAExt/vae.jl:    σ::CUDA.CuVecOrMat;
ext/AutoEncoderToolkitCUDAExt/vae.jl:        CUDA.randn(T, size(µ)...)
ext/AutoEncoderToolkitCUDAExt/AutoEncoderToolkitCUDAExt.jl:module AutoEncoderToolkitCUDAExt
ext/AutoEncoderToolkitCUDAExt/AutoEncoderToolkitCUDAExt.jl:# Import CUDA library
ext/AutoEncoderToolkitCUDAExt/AutoEncoderToolkitCUDAExt.jl:using CUDA
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:using CUDA
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:    rrule(::typeof(vec_to_ltri), diag::CUDA.CuVector, lower::CUDA.CuVector)
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:- `diag::CUDA.CuVector`: The diagonal vector.
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:- `lower::CUDA.CuVector`: The lower triangular vector.
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:    ::typeof(vec_to_ltri), diag::CUDA.CuVector, lower::CUDA.CuVector
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Initialize the gradients for 'diag' and 'lower' on the GPU
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        Δdiag = CUDA.zeros(eltype(ΔLtri), n)
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        Δlower = CUDA.zeros(eltype(ΔLtri), length(lower))
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Define the CUDA kernel function for computing the gradients
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Define the size of the blocks and the grid for the CUDA kernel launch
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Launch the CUDA kernel to compute the gradients
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        CUDA.@cuda threads = blocksize blocks = gridsize kernel!(Δdiag, Δlower, ΔLtri, n)
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:    rrule(::typeof(vec_to_ltri), diag::CUDA.CuMatrix, lower::CUDA.CuMatrix)
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:- `diag::CUDA.CuMatrix`: The diagonal matrix.
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:- `lower::CUDA.CuMatrix`: The lower triangular matrix.
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:    ::typeof(vec_to_ltri), diag::CUDA.CuMatrix, lower::CUDA.CuMatrix
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Initialize the gradients for 'diag' and 'lower' on the GPU
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        Δdiag = CUDA.zeros(eltype(ΔLtri), size(diag))
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        Δlower = CUDA.zeros(eltype(ΔLtri), size(lower))
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Define the CUDA kernel function for computing the gradients
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:            k = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Define the size of the blocks and the grid for the CUDA kernel launch
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        # Launch the CUDA kernel to compute the gradients
ext/AutoEncoderToolkitCUDAExt/adjoints.jl:        CUDA.@cuda threads = blocksize blocks = gridsize kernel!(Δdiag, Δlower, ΔLtri, n, cols)
ext/AutoEncoderToolkitCUDAExt/hvae.jl:using CUDA
ext/AutoEncoderToolkitCUDAExt/hvae.jl:    x::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/hvae.jl:    L, ∇L = CUDA.allowscalar() do
ext/AutoEncoderToolkitCUDAExt/hvae.jl:    x_in::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/hvae.jl:    x_out::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/hvae.jl:    L, ∇L = CUDA.allowscalar() do
ext/AutoEncoderToolkitCUDAExt/utils.jl:# Functions that extend the methods in the utils.jl file for GPU arrays
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU implementation of `vec_to_ltri`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:    ::Type{T}, diag::CUDA.CuVector, lower::CUDA.CuVector
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Create a zero matrix of the same type as the diagonal vector on the GPU
ext/AutoEncoderToolkitCUDAExt/utils.jl:    matrix = CUDA.zeros(eltype(diag), n, n)
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Define the CUDA kernel function that will be executed on the GPU
ext/AutoEncoderToolkitCUDAExt/utils.jl:        i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
ext/AutoEncoderToolkitCUDAExt/utils.jl:        j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Define the size of the blocks and the grid for the CUDA kernel launch
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Launch the CUDA kernel with the specified block and grid sizes
ext/AutoEncoderToolkitCUDAExt/utils.jl:    CUDA.@cuda threads = blocksize blocks = gridsize kernel!(
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU implementation of `vec_to_ltri`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:# tensor on the GPU
ext/AutoEncoderToolkitCUDAExt/utils.jl:    ::Type{T}, diag::CUDA.CuMatrix, lower::CUDA.CuMatrix
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Create a 3D tensor of zeros on the GPU with the same type as the diagonal
ext/AutoEncoderToolkitCUDAExt/utils.jl:    tensor = CUDA.zeros(eltype(diag), n, n, cols)
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Define the CUDA kernel function that will be executed on the GPU
ext/AutoEncoderToolkitCUDAExt/utils.jl:        i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
ext/AutoEncoderToolkitCUDAExt/utils.jl:        j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
ext/AutoEncoderToolkitCUDAExt/utils.jl:        k = (CUDA.blockIdx().z - 1) * CUDA.blockDim().z + CUDA.threadIdx().z
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Define the size of the blocks and the grid for the CUDA kernel launch
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Launch the CUDA kernel with the specified block and grid sizes
ext/AutoEncoderToolkitCUDAExt/utils.jl:    CUDA.@cuda threads = blocksize blocks = gridsize kernel!(tensor, diag, lower, n, cols)
ext/AutoEncoderToolkitCUDAExt/utils.jl:    slogdet(A::CUDA.CuArray; check::Bool=false)
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU AbstractArray implementation of `slogdet`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    # Reupload to GPU since the output of each operation is a single scalar
ext/AutoEncoderToolkitCUDAExt/utils.jl:    return logdetA |> Flux.gpu
ext/AutoEncoderToolkitCUDAExt/utils.jl:    _randn_samples(::Type{T}, z::AbstractArray) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:normal distribution. This function is used for GPU arrays.
ext/AutoEncoderToolkitCUDAExt/utils.jl:function utils._randn_samples(::Type{T}, z::CUDA.CuArray) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    return CUDA.randn(eltype(z), size(z))
ext/AutoEncoderToolkitCUDAExt/utils.jl:    sample_MvNormalCanon(Σ⁻¹::CUDA.CuArray{T}) where {T<:Number}
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU AbstractMatrix implementation of `sample_MvNormalCanon`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    r = CUDA.randn(N, size(Σ⁻¹, 1))
ext/AutoEncoderToolkitCUDAExt/utils.jl:    sample_MvNormalCanon(Σ⁻¹::CUDA.CuArray{T,3}) where {T<:Number}
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU AbstractArray implementation of `sample_MvNormalCanon`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:    ::Type{T}, Σ⁻¹::CUDA.CuArray{<:Number,3}
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    Σ = last(CUDA.CUBLAS.matinv_batched(collect(eachslice(Σ⁻¹, dims=3))))
ext/AutoEncoderToolkitCUDAExt/utils.jl:    r = CUDA.randn(N, dim, n_sample)
ext/AutoEncoderToolkitCUDAExt/utils.jl:    unit_vectors(x::CUDA.CuVector)
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU AbstractVector implementation of `unit_vectors`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    return [utils.unit_vector(x, i) for i in 1:length(x)] |> Flux.gpu
ext/AutoEncoderToolkitCUDAExt/utils.jl:    unit_vectors(x::CUDA.CuMatrix)
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU AbstractMatrix implementation of `unit_vectors`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:    ::Type{T}, x::CUDA.CuMatrix
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:    return vectors |> Flux.gpu
ext/AutoEncoderToolkitCUDAExt/utils.jl:        x::CUDA.CuArray;
ext/AutoEncoderToolkitCUDAExt/utils.jl:GPU AbstractVecOrMat implementation of `finite_difference_gradient`.
ext/AutoEncoderToolkitCUDAExt/utils.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/utils.jl:        return CUDA.cu(grad)
ext/AutoEncoderToolkitCUDAExt/utils.jl:        return CUDA.cu(permutedims(reduce(hcat, grad), [2, 1]))
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:GPU-compatible function to compute the Gaussian Kernel between columns of arrays `x` and `y`, defined
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    x::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    y::CUDA.CuArray;
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    xy = CUDA.transpose(x) * y
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    dist = CUDA.transpose(xx) .+ yy .- 2 .* xy
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    x::CUDA.CuArray;
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:        CUDA.randn(eltype(x), size(q_z_x, 1), n_latent_samples)
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    x_in::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    x_out::CUDA.CuArray;
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:        CUDA.randn(eltype(x_in), size(q_z_x, 1), n_latent_samples)
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    x::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    L, ∇L = CUDA.allowscalar() do
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    x_in::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    x_out::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/mmdvae.jl:    L, ∇L = CUDA.allowscalar() do
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:GPU AbstractVector version of the G_inv function.
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:) where {N<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:        CUDA.Diagonal(CUDA.ones(eltype(z), length(z), length(z))) .* λ
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:    # Zygote.dropgrad(CUDA.cu(Matrix(LinearAlgebra.I(length(z)) .* λ)))
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:GPU AbstractMatrix version of the G_inv function.
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:) where {N<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:        CUDA.Diagonal(CUDA.ones(eltype(z), size(z, 1), size(z, 1))) .* λ
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:GPU AbstractMatrix version of the metric_tensor function.
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:) where {T<:CUDA.CuArray}
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:        last(CUDA.CUBLAS.matinv_batched(collect(eachslice(G⁻¹, dims=3))))
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:    x::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:    L, ∇L = CUDA.allowscalar() do
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:    x_in::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:    x_out::CUDA.CuArray,
ext/AutoEncoderToolkitCUDAExt/rhvae.jl:    L, ∇L = CUDA.allowscalar() do
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:AutoEncoderToolkitCUDAExt = "CUDA"
Project.toml:CUDA = "5.3.3"
src/utils.jl:are stored on the CPU or GPU.
src/utils.jl:# GPU Support
src/utils.jl:The function supports both CPU and GPU arrays. For GPU arrays, the data is first
src/utils.jl:and then it is transferred back to the GPU.
src/utils.jl:# GPU Support
src/utils.jl:The function supports both CPU and GPU arrays. 
src/utils.jl:normal distribution. This function is used for non-GPU arrays.
src/utils.jl:# GPU Support
src/utils.jl:The function supports both CPU and GPU arrays.
src/utils.jl:# GPU Support
src/utils.jl:This function supports both CPU and GPU arrays.
src/utils.jl:# GPU Support
src/utils.jl:This function supports both CPU and GPU arrays.
src/utils.jl:# GPU Support
src/decoders.jl:    ] |> Flux.gpu
src/decoders.jl:    ] |> Flux.gpu
src/rhvae.jl:    Flux.gpu(rhvae::RHVAE)
src/rhvae.jl:Move the RHVAE model to the GPU.
src/rhvae.jl:- `rhvae::RHVAE`: The RHVAE model to be moved to the GPU.
src/rhvae.jl:- A new RHVAE model where all the data has been moved to the GPU.
src/rhvae.jl:This function moves all the data of an RHVAE model to the GPU. This includes the
src/rhvae.jl:`metric_chain` for moving to the GPU. Other data fields of the RHVAE model need
src/rhvae.jl:to be manually moved to the GPU.
src/rhvae.jl:function Flux.gpu(rhvae::RHVAE)
src/rhvae.jl:        Flux.gpu(rhvae.vae),
src/rhvae.jl:        Flux.gpu(rhvae.metric_chain),
src/rhvae.jl:        Flux.gpu(rhvae.centroids_data),
src/rhvae.jl:        Flux.gpu(rhvae.centroids_latent),
src/rhvae.jl:        Flux.gpu(rhvae.L),
src/rhvae.jl:        Flux.gpu(rhvae.M),
src/rhvae.jl:# GPU support
src/rhvae.jl:This function supports CPU and GPU arrays.
src/rhvae.jl:# GPU Support
src/rhvae.jl:This function supports CPU and GPU arrays.

```
