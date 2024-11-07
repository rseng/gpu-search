# https://github.com/palumbom/GRASS

```console
test/test_convolve.jl:using CUDA
test/test_convolve.jl:lambdas1, outspec1 = synthesize_spectra(spec1, disk, seed_rng=false, verbose=true, use_gpu=true)
test/runtests.jl:using CUDA
test/runtests.jl:# run the GPU tests if there is a GPU
test/runtests.jl:# if CUDA.functional()
test/runtests.jl:#     include("test_gpu.jl")
scripts/calibrate_iag_blueshifts.jl:using CUDA
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:CUDA = "≥ 5.3.0"
src/convenience.jl:                            use_gpu::Bool=false, precision::DataType=Float64,
src/convenience.jl:    # call appropriate simulation function on cpu or gpu
src/convenience.jl:    if use_gpu
src/convenience.jl:        return synth_gpu(spec, disk, seed_rng, verbose, precision, skip_times, contiguous_only)
src/convenience.jl:function synth_gpu(spec::SpecParams{T}, disk::DiskParams{T}, seed_rng::Bool,
src/convenience.jl:    # make sure there is actually a GPU to use
src/convenience.jl:    @assert CUDA.functional()
src/convenience.jl:       @warn "Single-precision GPU implementation produces large flux and velocity errors!"
src/convenience.jl:    # pre-allocate memory for gpu and pre-compute geometric quantities
src/convenience.jl:    gpu_allocs = GPUAllocs(spec, disk, precision=precision, verbose=verbose)
src/convenience.jl:        tloop_init = zeros(Int, CUDA.length(gpu_allocs.μs))
src/convenience.jl:        keys_cpu = repeat([(:off,:off)], CUDA.length(gpu_allocs.μs))
src/convenience.jl:            μs_cpu = Array(gpu_allocs.μs)
src/convenience.jl:            cbs_cpu = Array(gpu_allocs.z_cbs)
src/convenience.jl:            ax_codes_cpu = convert.(Int64, Array(gpu_allocs.ax_codes))
src/convenience.jl:        tloop_init = gpu_allocs.tloop_init
src/convenience.jl:        soldata = GPUSolarData(soldata_cpu, precision=precision)
src/convenience.jl:        get_keys_and_cbs_gpu!(gpu_allocs, soldata)
src/convenience.jl:            @cusync CUDA.copyto!(gpu_allocs.tloop, tloop_init)
src/convenience.jl:            # generate either seeded rngs, or don't seed them on gpu
src/convenience.jl:                # generate the random numbers on the gpu
src/convenience.jl:                generate_tloop_gpu!(tloop_init, gpu_allocs, soldata)
src/convenience.jl:            # copy the random indices to GPU
src/convenience.jl:            @cusync CUDA.copyto!(gpu_allocs.tloop, tloop_init)
src/convenience.jl:        disk_sim_gpu(spec_temp, disk, soldata, gpu_allocs, flux,
src/structures/GPUSolarData.jl:struct GPUSolarData{T1<:AF}
src/structures/GPUSolarData.jl:function GPUSolarData(soldata::SolarData{T}; precision::DataType=Float64) where T<:AF
src/structures/GPUSolarData.jl:    # copy to GPU and return composite type
src/structures/GPUSolarData.jl:    bis_gpu = CuArray{precision}(bis)
src/structures/GPUSolarData.jl:    int_gpu = CuArray{precision}(int)
src/structures/GPUSolarData.jl:    wid_gpu = CuArray{precision}(wid)
src/structures/GPUSolarData.jl:    dep_contrast_gpu = CuArray{precision}(dep_contrast)
src/structures/GPUSolarData.jl:    cbs_gpu = CuArray{precision}(cbs)
src/structures/GPUSolarData.jl:    len_gpu = CuArray{Int32}(len)
src/structures/GPUSolarData.jl:    disc_ax_gpu = CuArray{Int32}(disc_ax)
src/structures/GPUSolarData.jl:    disc_mu_gpu = CuArray{precision}(disc_mu)
src/structures/GPUSolarData.jl:    return GPUSolarData(bis_gpu, int_gpu, wid_gpu, dep_contrast_gpu,
src/structures/GPUSolarData.jl:                        cbs_gpu, len_gpu, disc_ax_gpu, disc_mu_gpu)
src/structures/GPUAllocs.jl:struct GPUAllocs{T1<:AF}
src/structures/GPUAllocs.jl:function GPUAllocs(spec::SpecParams, disk::DiskParams; precision::DataType=Float64, verbose::Bool=true)
src/structures/GPUAllocs.jl:    # move disk + geometry information to gpu
src/structures/GPUAllocs.jl:        Nθ_gpu = CuArray{Int}(disk.Nθ)
src/structures/GPUAllocs.jl:        R_x_gpu = CuArray{precision}(disk.R_x)
src/structures/GPUAllocs.jl:        O⃗_gpu = CuArray{precision}(disk.O⃗)
src/structures/GPUAllocs.jl:        λs_gpu = CuArray{precision}(spec.lambdas)
src/structures/GPUAllocs.jl:        prof_gpu = CUDA.zeros(precision, Nλ)
src/structures/GPUAllocs.jl:        flux_gpu = CUDA.ones(precision, Nλ, Nt)
src/structures/GPUAllocs.jl:        μs = CUDA.zeros(precision, Nϕ, Nθ_max)
src/structures/GPUAllocs.jl:        wts = CUDA.zeros(precision, Nϕ, Nθ_max)
src/structures/GPUAllocs.jl:        z_rot = CUDA.zeros(precision, Nϕ, Nθ_max)
src/structures/GPUAllocs.jl:        ax_code = CUDA.zeros(Int32, Nϕ, Nθ_max)
src/structures/GPUAllocs.jl:    precompute_quantities_gpu!(disk, μs, wts, z_rot, ax_code)
src/structures/GPUAllocs.jl:    @cusync μs = CUDA.reshape(μs, num_tiles)
src/structures/GPUAllocs.jl:    @cusync wts = CUDA.reshape(wts, num_tiles)
src/structures/GPUAllocs.jl:    @cusync z_rot = CUDA.reshape(z_rot, num_tiles)
src/structures/GPUAllocs.jl:    @cusync ax_code = CUDA.reshape(ax_code, num_tiles)
src/structures/GPUAllocs.jl:    @cusync num_nonzero = CUDA.sum(idx)
src/structures/GPUAllocs.jl:        μs_new = CUDA.zeros(precision, CUDA.sum(idx))
src/structures/GPUAllocs.jl:        wts_new = CUDA.zeros(precision, CUDA.sum(idx))
src/structures/GPUAllocs.jl:        z_rot_new = CUDA.zeros(precision, CUDA.sum(idx))
src/structures/GPUAllocs.jl:        ax_code_new = CUDA.zeros(Int32, CUDA.sum(idx))
src/structures/GPUAllocs.jl:    @cusync @cuda filter_array_gpu!(μs_new, μs, idx, 0)
src/structures/GPUAllocs.jl:    @cusync @cuda filter_array_gpu!(wts_new, wts, idx, 0)
src/structures/GPUAllocs.jl:    @cusync @cuda filter_array_gpu!(z_rot_new, z_rot, idx, 0)
src/structures/GPUAllocs.jl:    @cusync @cuda filter_array_gpu!(ax_code_new, ax_code, idx, 0)
src/structures/GPUAllocs.jl:    @cusync z_cbs = CUDA.zeros(precision, num_nonzero)
src/structures/GPUAllocs.jl:        tloop_gpu = CUDA.zeros(Int32, num_nonzero)
src/structures/GPUAllocs.jl:        tloop_init = CUDA.zeros(Int32, num_nonzero)
src/structures/GPUAllocs.jl:        dat_idx = CUDA.zeros(Int32, num_nonzero)
src/structures/GPUAllocs.jl:        allwavs = CUDA.zeros(precision, num_nonzero, 200)
src/structures/GPUAllocs.jl:        allints = CUDA.zeros(precision, num_nonzero, 200)
src/structures/GPUAllocs.jl:    return GPUAllocs(λs_gpu, prof_gpu, flux_gpu, μs, wts, z_rot, z_cbs, ax_code,
src/structures/GPUAllocs.jl:                     dat_idx, tloop_gpu, tloop_init, allwavs, allints)
src/fig_functions.jl:function spec_loop(spec::SpecParams, disk::DiskParams, Nloop::T; use_gpu::Bool=false) where T<:Integer
src/fig_functions.jl:        lambdas, outspec = synthesize_spectra(spec, disk, seed_rng=false, use_gpu=use_gpu, verbose=false)
src/fig_functions.jl:        # CUDA.memory_status()
src/fig_functions.jl:        # CUDA.reclaim()
src/observing/ObservationPlan.jl:                               use_gpu::Bool=false) where T<:AF
src/observing/ObservationPlan.jl:    wavs, flux = synthesize_spectra(spec, disk, verbose=false, use_gpu=use_gpu,
src/observing/ObservationPlan.jl:    # if use_gpu
src/observing/ObservationPlan.jl:    #     disk_sim_gpu(spec, disk, outspec, skip_times=skip_times)
src/GRASS.jl:using CUDA; CUDA.allowscalar(false)
src/GRASS.jl:include("gpu/gpu_utils.jl")
src/GRASS.jl:# gpu implementation
src/GRASS.jl:include("gpu/gpu_physics.jl")
src/GRASS.jl:include("gpu/gpu_data.jl")
src/GRASS.jl:include("gpu/gpu_precomps.jl")
src/GRASS.jl:include("gpu/gpu_trim.jl")
src/GRASS.jl:include("gpu/gpu_sim.jl")
src/GRASS.jl:include("gpu/gpu_synthesis.jl")
src/GRASS.jl:#         if CUDA.functional()
src/GRASS.jl:#             # TODO figure out what's going wrong with GPU precompilation caching
src/GRASS.jl:#             # lambdas1, outspec = synthesize_spectra(spec, disk, seed_rng=true, verbose=false, use_gpu=true)
src/GRASS.jl:#             lambdas1, outspec = synthesize_spectra(spec, disk, seed_rng=true, verbose=false, use_gpu=false)
src/GRASS.jl:#             lambdas1, outspec = synthesize_spectra(spec, disk, seed_rng=true, verbose=false, use_gpu=false)
src/gpu/gpu_precomps.jl:function precompute_quantities_gpu!(disk::DiskParams{T1}, μs::CuArray{T2,2},
src/gpu/gpu_precomps.jl:    # get precision from GPU allocs
src/gpu/gpu_precomps.jl:    # copy data to GPU
src/gpu/gpu_precomps.jl:    @cusync @captured @cuda threads=threads1 blocks=blocks1 precompute_quantities_gpu!(μs, wts, z_rot, ax_codes, Nϕ,
src/gpu/gpu_precomps.jl:    CUDA.synchronize()
src/gpu/gpu_precomps.jl:function precompute_quantities_gpu!(μs, wts, z_rot, ax_codes, Nϕ, Nθ_max, Nsubgrid,
src/gpu/gpu_precomps.jl:# get indices from GPU blocks + threads
src/gpu/gpu_precomps.jl:        μ_sum = CUDA.zero(CUDA.eltype(μs))
src/gpu/gpu_precomps.jl:        v_sum = CUDA.zero(CUDA.eltype(z_rot))
src/gpu/gpu_precomps.jl:        ld_sum = CUDA.zero(CUDA.eltype(wts))
src/gpu/gpu_precomps.jl:        dA_sum = CUDA.zero(CUDA.eltype(wts))
src/gpu/gpu_precomps.jl:        x_sum = CUDA.zero(CUDA.eltype(μs))
src/gpu/gpu_precomps.jl:        y_sum = CUDA.zero(CUDA.eltype(μs))
src/gpu/gpu_precomps.jl:                x, y, z = sphere_to_cart_gpu(ρs, ϕc, θc)
src/gpu/gpu_precomps.jl:                b = CUDA.zero(CUDA.eltype(μs))
src/gpu/gpu_precomps.jl:                e = CUDA.zero(CUDA.eltype(μs))
src/gpu/gpu_precomps.jl:                def_norm = CUDA.sqrt(d^2.0 + e^2.0 + f^2.0)
src/gpu/gpu_precomps.jl:                rp = 2π * ρs * CUDA.cos(ϕc) / rotation_period_gpu(ϕc, A, B, C)
src/gpu/gpu_precomps.jl:                x, y, z = rotate_vector_gpu(x, y, z, R_x)
src/gpu/gpu_precomps.jl:                μ_sub = calc_mu_gpu(x, y, z, O⃗)
src/gpu/gpu_precomps.jl:                ld = quad_limb_darkening_gpu(μ_sub, u1, u2)
src/gpu/gpu_precomps.jl:                d, e, f = rotate_vector_gpu(d, e, f, R_x)
src/gpu/gpu_precomps.jl:                n1 = CUDA.sqrt(a^2.0 + b^2.0 + c^2.0)
src/gpu/gpu_precomps.jl:                n2 = CUDA.sqrt(d^2.0 + e^2.0 + f^2.0)
src/gpu/gpu_precomps.jl:                dA = calc_dA_gpu(ρs, ϕc, dϕ, dθ)
src/gpu/gpu_precomps.jl:            @inbounds ax_codes[m, n] = find_nearest_ax_gpu(xx, yy)
src/gpu/gpu_precomps.jl:function get_keys_and_cbs_gpu!(gpu_allocs::GPUAllocs{T}, soldata::GPUSolarData{T}) where T<:AF
src/gpu/gpu_precomps.jl:    # parse out gpu allocs
src/gpu/gpu_precomps.jl:    μs = gpu_allocs.μs
src/gpu/gpu_precomps.jl:    z_cbs = gpu_allocs.z_cbs
src/gpu/gpu_precomps.jl:    dat_idx = gpu_allocs.dat_idx
src/gpu/gpu_precomps.jl:    ax_codes = gpu_allocs.ax_codes
src/gpu/gpu_precomps.jl:    @cusync @captured @cuda threads=threads1 blocks=blocks1 get_keys_and_cbs_gpu!(dat_idx, z_cbs, μs, ax_codes,
src/gpu/gpu_precomps.jl:    CUDA.synchronize()
src/gpu/gpu_precomps.jl:function get_keys_and_cbs_gpu!(dat_idx, z_cbs, μs, ax_codes, cbsall, disc_mu, disc_ax)
src/gpu/gpu_precomps.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_precomps.jl:    for i in idx:sdx:CUDA.length(μs)
src/gpu/gpu_precomps.jl:        idx = find_data_index_gpu(μs[i], ax_codes[i], disc_mu, disc_ax)
src/gpu/gpu_utils.jl:# alias the CUDA.@sync macro (stolen from CUDA.jl source)
src/gpu/gpu_utils.jl:        CUDA.synchronize()
src/gpu/gpu_utils.jl:function searchsortednearest_gpu(a,x)
src/gpu/gpu_utils.jl:    idx = CUDA.searchsortedfirst(a,x)
src/gpu/gpu_utils.jl:    if (idx>CUDA.length(a)); return CUDA.length(a); end
src/gpu/gpu_utils.jl:    if (CUDA.abs(a[idx]-x) < CUDA.abs(a[idx-1]-x))
src/gpu/gpu_utils.jl:function filter_array_gpu!(output, input, pred, n)
src/gpu/gpu_utils.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_utils.jl:    for i in idx:sdx:CUDA.length(input)
src/gpu/gpu_utils.jl:        if CUDA.isone(pred[i])
src/gpu/gpu_sim.jl:function disk_sim_gpu(spec::SpecParams{T1}, disk::DiskParams{T1}, soldata::GPUSolarData{T2},
src/gpu/gpu_sim.jl:                      gpu_allocs::GPUAllocs{T2}, flux_cpu::AA{T1,2}; verbose::Bool=false,
src/gpu/gpu_sim.jl:    λs = gpu_allocs.λs
src/gpu/gpu_sim.jl:    prof = gpu_allocs.prof
src/gpu/gpu_sim.jl:    flux = gpu_allocs.flux
src/gpu/gpu_sim.jl:    μs = gpu_allocs.μs
src/gpu/gpu_sim.jl:    wts = gpu_allocs.wts
src/gpu/gpu_sim.jl:    z_rot = gpu_allocs.z_rot
src/gpu/gpu_sim.jl:    z_cbs = gpu_allocs.z_cbs
src/gpu/gpu_sim.jl:    tloop = gpu_allocs.tloop
src/gpu/gpu_sim.jl:    dat_idx = gpu_allocs.dat_idx
src/gpu/gpu_sim.jl:    allwavs = gpu_allocs.allwavs
src/gpu/gpu_sim.jl:    allints = gpu_allocs.allints
src/gpu/gpu_sim.jl:    # alias the input data from GPUSolarData
src/gpu/gpu_sim.jl:    disc_mu_gpu = soldata.mu
src/gpu/gpu_sim.jl:    disc_ax_gpu = soldata.ax
src/gpu/gpu_sim.jl:    lenall_gpu = soldata.len
src/gpu/gpu_sim.jl:    cbsall_gpu = soldata.cbs
src/gpu/gpu_sim.jl:    bisall_gpu = soldata.bis
src/gpu/gpu_sim.jl:    intall_gpu = soldata.int
src/gpu/gpu_sim.jl:    widall_gpu = soldata.wid
src/gpu/gpu_sim.jl:    depcontrast_gpu = soldata.dep_contrast
src/gpu/gpu_sim.jl:    # set number of threads and blocks for len(μ) gpu kernels
src/gpu/gpu_sim.jl:    blocks1 = cld(CUDA.length(μs), prod(threads1))
src/gpu/gpu_sim.jl:    blocks2 = cld(length(lenall_gpu) * maximum(lenall_gpu) * 100, prod(threads2))
src/gpu/gpu_sim.jl:    # set number of threads and blocks for len(μ) * 100 matrix gpu functions
src/gpu/gpu_sim.jl:    blocks3 = cld(CUDA.length(μs) * 100, prod(threads3))
src/gpu/gpu_sim.jl:    # set number of threads and blocks for N*N*Nλ matrix gpu functions
src/gpu/gpu_sim.jl:    blocks4 = cld(CUDA.length(μs) * Nλ, prod(threads4))
src/gpu/gpu_sim.jl:    blocks5 = cld(CUDA.length(prof), prod(threads5))
src/gpu/gpu_sim.jl:        bisall_gpu_loop = CUDA.zeros(T2, CUDA.size(bisall_gpu))
src/gpu/gpu_sim.jl:        intall_gpu_loop = CUDA.zeros(T2, CUDA.size(intall_gpu))
src/gpu/gpu_sim.jl:        widall_gpu_loop = CUDA.zeros(T2, CUDA.size(widall_gpu))
src/gpu/gpu_sim.jl:    @cusync sum_wts = CUDA.sum(wts)
src/gpu/gpu_sim.jl:    @cusync z_cbs_avg = CUDA.sum(z_cbs .* wts) / sum_wts
src/gpu/gpu_sim.jl:            @cusync @captured @cuda threads=threads1 blocks=blocks1 iterate_tloop_gpu!(tloop, dat_idx, lenall_gpu)
src/gpu/gpu_sim.jl:                CUDA.copyto!(bisall_gpu_loop, bisall_gpu)
src/gpu/gpu_sim.jl:                CUDA.copyto!(intall_gpu_loop, intall_gpu)
src/gpu/gpu_sim.jl:                CUDA.copyto!(widall_gpu_loop, widall_gpu)
src/gpu/gpu_sim.jl:            @cusync @cuda threads=threads2 blocks=blocks2 trim_bisector_gpu!(spec.depths[l], spec.variability[l],
src/gpu/gpu_sim.jl:                                                                             depcontrast_gpu, lenall_gpu,
src/gpu/gpu_sim.jl:                                                                             bisall_gpu_loop, intall_gpu_loop,
src/gpu/gpu_sim.jl:                                                                             widall_gpu_loop, bisall_gpu,
src/gpu/gpu_sim.jl:                                                                             intall_gpu, widall_gpu)
src/gpu/gpu_sim.jl:            @cusync @cuda threads=threads3 blocks=blocks3 fill_workspaces!(spec.lines[l], spec.variability[l],
src/gpu/gpu_sim.jl:                                                                           z_rot, z_cbs, lenall_gpu,
src/gpu/gpu_sim.jl:                                                                           bisall_gpu_loop, intall_gpu_loop,
src/gpu/gpu_sim.jl:                                                                           widall_gpu_loop, allwavs, allints)
src/gpu/gpu_sim.jl:            @cusync @cuda threads=threads4 blocks=blocks4 line_profile_gpu!(prof, μs, wts, λs, allwavs, allints)
src/gpu/gpu_sim.jl:            # copy data from GPU to CPU
src/gpu/gpu_sim.jl:            @cusync @cuda threads=threads5 blocks=blocks5 apply_line!(t, prof, flux, sum_wts)
src/gpu/gpu_sim.jl:        @cusync @captured @cuda threads=threads1 blocks=blocks1 iterate_tloop_gpu!(tloop, dat_idx, lenall_gpu)
src/gpu/gpu_sim.jl:    # make sure nothing is still running on GPU
src/gpu/gpu_sim.jl:    # CUDA.synchronize()
src/gpu/gpu_data.jl:function find_nearest_ax_gpu(x, y)
src/gpu/gpu_data.jl:    if (CUDA.iszero(x) & CUDA.iszero(y))
src/gpu/gpu_data.jl:    elseif y >= CUDA.abs(x)
src/gpu/gpu_data.jl:    elseif y <= -CUDA.abs(x)
src/gpu/gpu_data.jl:    elseif x <= -CUDA.abs(y)
src/gpu/gpu_data.jl:    elseif x >= CUDA.abs(y)
src/gpu/gpu_data.jl:function find_data_index_gpu(μ, ax_val, disc_mu, disc_ax)
src/gpu/gpu_data.jl:    mu_ind = searchsortednearest_gpu(disc_mu, μ)
src/gpu/gpu_data.jl:    if mu_ind == CUDA.length(disc_mu)
src/gpu/gpu_data.jl:        return CUDA.length(disc_mu)
src/gpu/gpu_data.jl:function iterate_tloop_gpu!(tloop, dat_idx, lenall)
src/gpu/gpu_data.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_data.jl:    for i in idx:sdx:CUDA.size(dat_idx,1)
src/gpu/gpu_data.jl:        if CUDA.iszero(dat_idx[i])
src/gpu/gpu_data.jl:function check_tloop_gpu!(tloop, dat_idx, lenall)
src/gpu/gpu_data.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_data.jl:    for i in idx:sdx:CUDA.length(dat_idx)
src/gpu/gpu_data.jl:        if CUDA.iszero(dat_idx[i])
src/gpu/gpu_data.jl:function generate_tloop_gpu!(tloop::AA{Int32,1}, gpu_allocs::GPUAllocs{T}, soldata::GPUSolarData{T}) where T<:AF
src/gpu/gpu_data.jl:    dat_idx = gpu_allocs.dat_idx
src/gpu/gpu_data.jl:    blocks1 = cld(CUDA.length(dat_idx), prod(threads1))
src/gpu/gpu_data.jl:    @cusync @captured @cuda threads=threads1 blocks=blocks1 generate_tloop_gpu!(tloop, dat_idx, lenall)
src/gpu/gpu_data.jl:function generate_tloop_gpu!(tloop, dat_idx, lenall)
src/gpu/gpu_data.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_data.jl:    for i in idx:sdx:CUDA.length(dat_idx)
src/gpu/gpu_data.jl:        if CUDA.iszero(dat_idx[i])
src/gpu/gpu_data.jl:        @inbounds tloop[i] = CUDA.floor(Int32, rand() * lenall[idx]) + 1
src/gpu/gpu_physics.jl:function calc_mu_gpu(x, y, z, O⃗)
src/gpu/gpu_physics.jl:    n1 = CUDA.sqrt(O⃗[1]^2.0 + O⃗[2]^2.0 + O⃗[3]^2.0)
src/gpu/gpu_physics.jl:    n2 = CUDA.sqrt(x^2.0 + y^2.0 + z^2.0)
src/gpu/gpu_physics.jl:function sphere_to_cart_gpu(ρ, ϕ, θ)
src/gpu/gpu_physics.jl:    sinϕ = CUDA.sin(ϕ)
src/gpu/gpu_physics.jl:    sinθ = CUDA.sin(θ)
src/gpu/gpu_physics.jl:    cosϕ = CUDA.cos(ϕ)
src/gpu/gpu_physics.jl:    cosθ = CUDA.cos(θ)
src/gpu/gpu_physics.jl:function rotate_vector_gpu(x0, y0, z0, R_x)
src/gpu/gpu_physics.jl:function rotation_period_gpu(ϕ, A, B, C)
src/gpu/gpu_physics.jl:function calc_dA_gpu(ρs, ϕc, dϕ, dθ)
src/gpu/gpu_physics.jl:    return ρs^2.0 * CUDA.sin(π/2.0 - ϕc) * dϕ * dθ
src/gpu/gpu_physics.jl:function quad_limb_darkening_gpu(μ, u1, u2)
src/gpu/gpu_trim.jl:function trim_bisector_gpu!(depth, variability, depcontrast, lenall, bisall_out,
src/gpu/gpu_trim.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_trim.jl:    for i in idx:sdx:CUDA.length(lenall)
src/gpu/gpu_trim.jl:        for j in idy:sdy:CUDA.size(bisall_in, 2)
src/gpu/gpu_trim.jl:            bist_in = CUDA.view(bisall_in, :, j, i)
src/gpu/gpu_trim.jl:            intt_in = CUDA.view(intall_in, :, j, i)
src/gpu/gpu_trim.jl:            bist_out = CUDA.view(bisall_out, :, j, i)
src/gpu/gpu_trim.jl:            intt_out = CUDA.view(intall_out, :, j, i)
src/gpu/gpu_trim.jl:            step = dtrim/(CUDA.length(intt_in) - 1)
src/gpu/gpu_trim.jl:                itp = linear_interp_gpu(intt_in, bist_in)
src/gpu/gpu_trim.jl:                for k in idz:sdz:CUDA.size(bisall_in, 1)
src/gpu/gpu_trim.jl:                    if (1.0 - dtrim) >= CUDA.first(intt_in)
src/gpu/gpu_trim.jl:                for k in idz:sdz:CUDA.size(bisall_in, 1)
src/gpu/gpu_synthesis.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_synthesis.jl:    for i in idx:sdx:CUDA.length(dat_idx)
src/gpu/gpu_synthesis.jl:        if CUDA.iszero(k)
src/gpu/gpu_synthesis.jl:function line_profile_gpu!(prof, μs, wts, λs, allwavs, allints)
src/gpu/gpu_synthesis.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_synthesis.jl:    for i in idx:sdx:CUDA.length(μs)
src/gpu/gpu_synthesis.jl:        allwavs_i = CUDA.view(allwavs, i, :)
src/gpu/gpu_synthesis.jl:        allints_i = CUDA.view(allints, i, :)
src/gpu/gpu_synthesis.jl:        itp = linear_interp_gpu(allwavs_i, allints_i)
src/gpu/gpu_synthesis.jl:        for j in idy:sdy:CUDA.length(λs)
src/gpu/gpu_synthesis.jl:            if ((λs[j] < CUDA.first(allwavs_i)) || (λs[j] > CUDA.last(allwavs_i)))
src/gpu/gpu_synthesis.jl:                @inbounds CUDA.@atomic prof[j] += wts[i]
src/gpu/gpu_synthesis.jl:                @inbounds CUDA.@atomic prof[j] += itp(λs[j]) * wts[i]
src/gpu/gpu_synthesis.jl:    # get indices from GPU blocks + threads
src/gpu/gpu_synthesis.jl:    for i in idx:sdx:CUDA.length(prof)
src/interpolate.jl:function linear_interp_gpu(xs::AA{T,1}, ys::AA{T,1}) where T<:AF
src/interpolate.jl:        if x <= CUDA.first(xs)
src/interpolate.jl:            return CUDA.first(ys)
src/interpolate.jl:        elseif x >= CUDA.last(xs)
src/interpolate.jl:            return CUDA.last(ys)
src/interpolate.jl:            i = CUDA.searchsortedfirst(xs, x) - 1
src/interpolate.jl:            i0 = CUDA.clamp(i, CUDA.firstindex(ys), CUDA.lastindex(ys))
src/interpolate.jl:            i1 = CUDA.clamp(i+1, CUDA.firstindex(ys), CUDA.lastindex(ys))
src/structures.jl:include("structures/GPUAllocs.jl")
src/structures.jl:include("structures/GPUSolarData.jl")

```
