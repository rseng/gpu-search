# https://github.com/FourierFlows/GeophysicalFlows.jl

```console
docs/make.jl:                "GPU" => "gpu.md",
docs/Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
docs/Project.toml:CUDA = "1, 2.4.2, 3.0.0 - 3.6.4, 3.7.1, 4, 5"
docs/src/contributing.md:  with a GPU capability but if that's not a possibility that is available to you then don't 
docs/src/contributing.md:  worry -- simply comment in a PR that you didn't test on GPU.
docs/src/gpu.md:# GPU
docs/src/gpu.md:GPU-functionality is enabled via `FourierFlows.jl`. For more information on how `FourierFlows.jl`
docs/src/gpu.md:handled with GPUs we urge you to the corresponding [`FourierFlows.jl` documentation section ](https://fourierflows.github.io/FourierFlowsDocumentation/stable/gpu/).
docs/src/gpu.md:All `GeophysicalFlows.jl` modules can be run on GPU by providing `GPU()` as the device (`dev`) 
docs/src/gpu.md:julia> GeophysicalFlows.TwoDNavierStokes.Problem(GPU())
docs/src/gpu.md:  ├─────────── grid: grid (on GPU)
docs/src/gpu.md:## Selecting GPU device
docs/src/gpu.md:`FourierFlows.jl` can only utilize a single GPU. If your machine has more than one GPU available, 
docs/src/gpu.md:then using functionality within `CUDA.jl` package enables you can choose the GPU device that 
docs/src/gpu.md:`FourierFlows.jl` should use. The user is referred to the [`CUDA.jl` Documentation](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management); in particular, [`CUDA.devices`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.devices) and [`CUDA.CuDevice`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.CuDevice).
docs/src/index.md:Constantinou et al., (2021). GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs. _Journal of Open Source Software_, **6(60)**, 3053, doi:[10.21105/joss.03053](https://doi.org/10.21105/joss.03053).
docs/src/index.md:  title = {GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs \& GPUs},
test/test_surfaceqg.jl:  Kr = device_array(dev)([CUDA.@allowscalar grid.kr[i] for i=1:grid.nkr, j=1:grid.nl])
test/test_surfaceqg.jl:    CUDA.@allowscalar eta[1, 1] = 0.0
test/test_multilayerqg.jl:  CUDA.@allowscalar U[:, 1] = u1 .+ U1
test/test_multilayerqg.jl:  CUDA.@allowscalar U[:, 2] = u2 .+ U2
test/test_multilayerqg.jl:  CUDA.@allowscalar U[:, 1] = u1 .+ U1
test/test_multilayerqg.jl:  CUDA.@allowscalar U[:, 2] = u2 .+ U2
test/test_multilayerqg.jl:  CUDA.@allowscalar U[:, 3] = u3 .+ U3
test/test_multilayerqg.jl:  CUDA.@allowscalar U[:, 1] = u1 .+ U1
test/test_multilayerqg.jl:  CUDA.@allowscalar U[:, 2] = u2 .+ U2
test/test_multilayerqg.jl:  CUDA.@allowscalar @views @. ψ[:, :, 1] =       cos(2k₀ * x) * cos(2l₀ * y)
test/test_multilayerqg.jl:  CUDA.@allowscalar @views @. ψ[:, :, 2] = 1/2 * cos(2k₀ * x) * cos(2l₀ * y)
test/test_multilayerqg.jl:  return CUDA.@allowscalar isapprox(lateralfluxes[1], 0.00626267, rtol=1e-6) &&
test/test_multilayerqg.jl:         CUDA.@allowscalar isapprox(lateralfluxes[2], 0, atol=1e-12) &&
test/test_multilayerqg.jl:         CUDA.@allowscalar isapprox(verticalfluxes[1], -0.196539, rtol=1e-6)
test/test_multilayerqg.jl:  return CUDA.@allowscalar isapprox(lateralfluxes[1], 0.0313134, atol=1e-7)
test/test_multilayerqg.jl:  CUDA.@allowscalar Ufloats[1] = U1
test/test_multilayerqg.jl:  CUDA.@allowscalar Ufloats[2] = U2
test/test_singlelayerqg.jl:  q_theory = @CUDA.allowscalar @. ampl * cos(kwave * (x - (U[1] + ω / kwave) * clock.t)) * cos(lwave * y)
test/test_singlelayerqg.jl:    CUDA.@allowscalar eta[1, 1] = 0
test/test_singlelayerqg.jl:    CUDA.@allowscalar eta[1, 1] = 0
test/test_twodnavierstokes.jl:  Kr = device_array(dev)([CUDA.@allowscalar grid.kr[i] for i=1:grid.nkr, j=1:grid.nl])
test/test_twodnavierstokes.jl:    CUDA.@allowscalar eta[1, 1] = 0.0
test/test_twodnavierstokes.jl:  Kr = device_array(dev)([CUDA.@allowscalar grid.kr[i] for i=1:grid.nkr, j=1:grid.nl])
test/test_twodnavierstokes.jl:    CUDA.@allowscalar eta[1, 1] = 0.0
test/test_utils.jl:  return CUDA.@allowscalar isapprox(abs.(qhρtest)/abs(qhρtest[1]), (ρtest/ρtest[1]).^(-2), rtol=5e-3)
test/test_barotropicqgql.jl:  Kr = CUDA.@allowscalar device_array(dev)([ grid.kr[i] for i=1:grid.nkr, j=1:grid.nl])
test/test_barotropicqgql.jl:    CUDA.@allowscalar eta[1, 1] = 0
test/runtests.jl:  CUDA,
test/runtests.jl:devices = CUDA.functional() ? (CPU(), GPU()) : (CPU(),)
CITATION.cff:  title: "GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs"
README.md:        <img alt="Buildkite CPU+GPU build status" src="https://img.shields.io/buildkite/4d921fc17b95341ea5477fb62df0e6d9364b61b154e050a123/main?logo=buildkite&label=Buildkite%20CPU%2BGPU">
README.md:For now, GeophysicalFlows.jl is restricted to run on either a single CPU or single GPU. These
README.md:If your machine has more than one GPU available, then functionality within CUDA.jl package 
README.md:enables the user to choose the GPU device that FourierFlows.jl should use. The user is referred
README.md:to the [CUDA.jl Documentation](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management);
README.md:in particular, [`CUDA.devices`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.devices) 
README.md:and [`CUDA.CuDevice`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.CuDevice). 
README.md:The user is also referred to the [GPU section](https://fourierflows.github.io/FourierFlowsDocumentation/stable/gpu/) in the FourierFlows.jl documentation.
README.md:> Constantinou et al., (2021). GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs. _Journal of Open Source Software_, **6(60)**, 3053, doi:[10.21105/joss.03053](https://doi.org/10.21105/joss.03053).
README.md:  title = {{GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs \& GPUs}},
paper/paper.bib:  title={Oceananigans.jl: {Fast} and friendly geophysical fluid dynamics on {GPUs}},
paper/paper.md:title: 'GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs'
paper/paper.md:  - gpu
paper/paper.md:On top of the above-mentioned needs, the recent explosion of machine-learning applications in atmospheric and oceanic sciences advocates for the need that solvers for partial differential equations can be run on GPUs. 
paper/paper.md:`GeophysicalFlows.jl` provides a collection of modules for solving sets of partial differential equations often used as conceptual models. These modules are continuously tested (unit tests and tests for the physics involved) and are well-documented. `GeophysicalFlows.jl` utilizes Julia's functionality and abstraction to enable all modules to run on CPUs or GPUs, and to provide a high level of customizability within modules. The abstractions allow simulations to be tailored for specific research questions, via the choice of parameters, domain properties, and schemes for damping, forcing, time-stepping etc. Simulations can easily be carried out on different computing architectures. Selection of the architecture on which equations are solved is done by providing the argument `CPU()` or `GPU()` during the construction of a particular problem.
paper/paper.md:  is that `GeophysicalFlows.jl` can be run on GPUs or CPUs and leverages a separate package (`FourierFlows.jl`; which is continuously developed) to solve differential equations and compute diagnostics, while `pyqg` can only be run on CPUs and uses a self-contained kernel. 
paper/paper.md:  CPUs (not on GPUs) but can be MPI-parallelized.
paper/paper.md:  approximation. `Oceananigans.jl` also runs on GPUs, and it allows for more variety of boundary
paper/paper.md:  atmosphere. Neither `MAOOAM` nor `qgs` can run on GPUs.
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:CUDA = "1, 2.4.2, 3.0.0 - 3.6.4, 3.7.1, 4, 5"
examples/multilayerqg_2layer.jl:# ## Choosing a device: CPU or GPU
examples/multilayerqg_2layer.jl:dev = CPU()     # Device (CPU/GPU)
examples/multilayerqg_2layer.jl:# `dev = CPU()` and `CuArray` for `dev = GPU()`.
examples/multilayerqg_2layer.jl:# to make sure it is brought back on the CPU when `vars` live on the GPU.
examples/twodnavierstokes_stochasticforcing_budgets.jl:# pkg"add GeophysicalFlows, CUDA, Random, Printf, CairoMakie"
examples/twodnavierstokes_stochasticforcing_budgets.jl:using GeophysicalFlows, CUDA, Random, Printf, CairoMakie
examples/twodnavierstokes_stochasticforcing_budgets.jl:record = CairoMakie.record                # disambiguate between CairoMakie.record and CUDA.record
examples/twodnavierstokes_stochasticforcing_budgets.jl:# ## Choosing a device: CPU or GPU
examples/twodnavierstokes_stochasticforcing_budgets.jl:dev = CPU()     # Device (CPU/GPU)
examples/twodnavierstokes_stochasticforcing_budgets.jl:@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average
examples/twodnavierstokes_stochasticforcing_budgets.jl:if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
examples/twodnavierstokes_stochasticforcing_budgets.jl:# it is brought back on the CPU when the variable lives on the GPU.
examples/singlelayerqg_decaying_barotropic_equivalentbarotropic.jl:# ## Choosing a device: CPU or GPU
examples/singlelayerqg_decaying_barotropic_equivalentbarotropic.jl:dev = CPU()     # Device (CPU/GPU)
examples/singlelayerqg_decaying_barotropic_equivalentbarotropic.jl:# `vars` live on the GPU.
examples/barotropicqgql_betaforced.jl:# pkg"add GeophysicalFlows, CUDA, CairoMakie"
examples/barotropicqgql_betaforced.jl:using GeophysicalFlows, CUDA, Random, Printf, CairoMakie
examples/barotropicqgql_betaforced.jl:record = CairoMakie.record                # disambiguate between CairoMakie.record and CUDA.record
examples/barotropicqgql_betaforced.jl:# ## Choosing a device: CPU or GPU
examples/barotropicqgql_betaforced.jl:dev = CPU()     # Device (CPU/GPU)
examples/barotropicqgql_betaforced.jl:@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average
examples/barotropicqgql_betaforced.jl:if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
examples/barotropicqgql_betaforced.jl:# `vars` live on the GPU.
examples/twodnavierstokes_stochasticforcing.jl:# pkg"add GeophysicalFlows, CUDA, CairoMakie"
examples/twodnavierstokes_stochasticforcing.jl:using GeophysicalFlows, CUDA, Random, Printf, CairoMakie
examples/twodnavierstokes_stochasticforcing.jl:record = CairoMakie.record                # disambiguate between CairoMakie.record and CUDA.record
examples/twodnavierstokes_stochasticforcing.jl:# ## Choosing a device: CPU or GPU
examples/twodnavierstokes_stochasticforcing.jl:dev = CPU()     # Device (CPU/GPU)
examples/twodnavierstokes_stochasticforcing.jl:@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average
examples/twodnavierstokes_stochasticforcing.jl:if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
examples/twodnavierstokes_stochasticforcing.jl:# it is brought back on the CPU when the variable lives on the GPU.
examples/singlelayerqg_betaforced.jl:# pkg"add GeophysicalFlows, CUDA, JLD2, CairoMakie, Statistics"
examples/singlelayerqg_betaforced.jl:using GeophysicalFlows, CUDA, JLD2, CairoMakie, Random, Printf
examples/singlelayerqg_betaforced.jl:record = CairoMakie.record                # disambiguate between CairoMakie.record and CUDA.record
examples/singlelayerqg_betaforced.jl:# ## Choosing a device: CPU or GPU
examples/singlelayerqg_betaforced.jl:dev = CPU()     # Device (CPU/GPU)
examples/singlelayerqg_betaforced.jl:@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average
examples/singlelayerqg_betaforced.jl:if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
examples/singlelayerqg_betaforced.jl:# `vars` live on the GPU.
examples/Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
examples/twodnavierstokes_decaying.jl:# ## Choosing a device: CPU or GPU
examples/twodnavierstokes_decaying.jl:dev = CPU()     # Device (CPU/GPU)
examples/twodnavierstokes_decaying.jl:# the GPU.
examples/singlelayerqg_betadecay.jl:# ## Choosing a device: CPU or GPU
examples/singlelayerqg_betadecay.jl:dev = CPU()     # Device (CPU/GPU)
examples/singlelayerqg_betadecay.jl:# `dev = CPU()` and `CuArray` for `dev = GPU()`.
examples/singlelayerqg_betadecay.jl:# `vars` live on the GPU.
examples/singlelayerqg_decaying_topography.jl:# ## Choosing a device: CPU or GPU
examples/singlelayerqg_decaying_topography.jl:dev = CPU()     # Device (CPU/GPU)
examples/singlelayerqg_decaying_topography.jl:# on the GPU.
examples/singlelayerqg_decaying_topography.jl:# `dev = CPU()` and `CuArray` for `dev = GPU()`.
examples/surfaceqg_decaying.jl:# ## Choosing a device: CPU or GPU
examples/surfaceqg_decaying.jl:dev = CPU()     # Device (CPU/GPU)
examples/surfaceqg_decaying.jl:# plotted with `Array()` to make sure it is brought back on the CPU when `vars` live on the GPU.
src/GeophysicalFlows.jl:problems in periodic domains on CPUs and GPUs. All modules use Fourier-based pseudospectral 
src/GeophysicalFlows.jl:  CUDA,
src/multilayerqg.jl:  CUDA,
src/multilayerqg.jl:- `dev`: (required) `CPU()` (default) or `GPU()`; computer architecture used to time-step `problem`.
src/multilayerqg.jl:  if dev == GPU() && nlayers > 2
src/multilayerqg.jl:    @warn """MultiLayerQG module is not optimized on the GPU yet for configurations with
src/multilayerqg.jl:  Uyy = CUDA.@allowscalar repeat(Uyy, outer=(nx, 1, 1))
src/multilayerqg.jl:    CUDA.@allowscalar @views Qy[:, :, 1] = @. Qy[:, :, 1] - Fp[1] * (U[:, :, 2] - U[:, :, 1])
src/multilayerqg.jl:      CUDA.@allowscalar @views Qy[:, :, j] = @. Qy[:, :, j] - Fp[j] * (U[:, :, j+1] - U[:, :, j]) - Fm[j-1] * (U[:, :, j-1] - U[:, :, j])
src/multilayerqg.jl:    CUDA.@allowscalar @views Qy[:, :, nlayers] = @. Qy[:, :, nlayers] - Fm[nlayers-1] * (U[:, :, nlayers-1] - U[:, :, nlayers])
src/multilayerqg.jl:    CUDA.@allowscalar @views qh[i, j, :] .= params.S[i, j] * ψh[i, j, :]
src/multilayerqg.jl:on the GPU.)
src/multilayerqg.jl:    CUDA.@allowscalar @views ψh[i, j, :] .= params.S⁻¹[i, j] * qh[i, j, :]
src/multilayerqg.jl:on the GPU.)
src/multilayerqg.jl:    k² = CUDA.@allowscalar grid.Krsq[m, n]
src/multilayerqg.jl:    k² = CUDA.@allowscalar grid.Krsq[m, n] == 0 ? 1 : grid.Krsq[m, n]
src/utils.jl:  besseljorder1 = CUDA.@allowscalar A([besselj1(k * r[i, j]) for j=1:grid.ny, i=1:grid.nx])
src/utils.jl:  CUDA.@allowscalar modψ[1, 1] = 0.0
src/twodnavierstokes.jl:  CUDA,
src/twodnavierstokes.jl:  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
src/twodnavierstokes.jl:  CUDA.@allowscalar L[1, 1] = 0
src/twodnavierstokes.jl:  CUDA.@allowscalar prob.sol[1, 1] = 0 # enforce zero domain average
src/twodnavierstokes.jl:  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
src/twodnavierstokes.jl:  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
src/singlelayerqg.jl:  CUDA,
src/singlelayerqg.jl:  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
src/singlelayerqg.jl:  CUDA.@allowscalar L[1, 1] = 0
src/singlelayerqg.jl:  CUDA.@allowscalar L[1, 1] = 0
src/singlelayerqg.jl:  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
src/singlelayerqg.jl:  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
src/singlelayerqg.jl:  CUDA.@allowscalar energy_dragh[1, 1] = 0
src/singlelayerqg.jl:  CUDA.@allowscalar enstrophy_dragh[1, 1] = 0
src/barotropicqgql.jl:  CUDA,
src/barotropicqgql.jl:  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
src/barotropicqgql.jl:  CUDA.@allowscalar L[1, 1] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar @. vars.zetah[1, :] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar @. vars.Zetah[2:end, :] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar @. vars.NZ[2:end, :] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar @. vars.Nz[1, :] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar sol[1, 1] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar @. vars.zetah[1, :] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar @. vars.Zetah[2:end, :] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar vars.zetah[1, 1] = 0.0
src/barotropicqgql.jl:  CUDA.@allowscalar vars.uh[1, 1] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar vars.uh[1, 1] = 0
src/barotropicqgql.jl:  CUDA.@allowscalar vars.uh[1, 1] = 0
src/surfaceqg.jl:    CUDA,
src/surfaceqg.jl:  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
src/surfaceqg.jl:  CUDA.@allowscalar L[1, 1] = 0
src/surfaceqg.jl:  CUDA.@allowscalar prob.sol[1, 1] = 0 # zero domain average
src/surfaceqg.jl:  CUDA.@allowscalar buoyancy_dissipationh[1, 1] = 0

```
