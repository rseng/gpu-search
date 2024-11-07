# https://github.com/AMReX-Astro/MAESTROeX

```console
Util/model_parser/ModelParser.H:/// Define Real vector types for CUDA-compatability. If `AMREX_USE_CUDA`, then
Util/model_parser/ModelParser.H:/// this will be stored in CUDA managed memory.
Util/model_parser/ModelParser.H:#ifdef AMREX_USE_CUDA
Util/model_parser/ModelParser.H:typedef amrex::Gpu::ManagedVector<amrex::Real> RealVector;
Util/model_parser/ModelParser.H:typedef amrex::Gpu::ManagedVector<int> IntVector;
Util/scripts/clang_static_analysis.py:            if 'ignoring #pragma gpu box' not in m.group(1):
CHANGES.md:  * disallow OpenMP + GPUs (#456)
CHANGES.md:  * some GPU modernization (#403, #424)
CHANGES.md:  * Fixed a race condition on GPUs (#402)
CHANGES.md:    is a failure.  Previously there was not an abort on GPUs (#379)
CHANGES.md:  * Bug fix for running code on GPUs
CHANGES.md:  * Remove Make.cuda_rules
CHANGES.md:  * GPU bug fixes
CHANGES.md:  * Fixed non-deterministic issue with GPU runs
CHANGES.md:  * Offloaded more routines to GPU: InitData, MakeThermalCoeffs
CHANGES.md:  * Offloaded more routines to GPU: MakePsi, MakeEtarho, FillUmacGhost, BCFill, Sponge, Dt, BaseState, EOS routines, Average
CHANGES.md:  * Fixed GPU and OMP reduction calls
CHANGES.md:  * Offloaded more routines to GPU: MakeUtrans, MacProj, NodalProj, Makew0, MakeBeta0, AdvectBase, EnforceHSE
CHANGES.md:  * Offloaded more routines to GPU: Fill3dData, MakeEdgeScal, VelPred, PPM
CHANGES.md:  * Various tiling and GPU updates and bugfixes
CHANGES.md:  * Offloading more routines to GPU, including nodal and cell-centered solvers
CHANGES.md:  * Continue to port hydro subroutines to GPU
CHANGES.md:  * GPU-specific changes to Makefile
CHANGES.md:  * Port 3d hydro subroutine to GPU
CHANGES.md:  * Start porting hydro subroutines to GPU (ongoing)
CHANGES.md:  * GPU ports: FirstDtm EstDt
CHANGES.md:  * Various GPU optimizations
CHANGES.md:  * Many thermodynamics and source term subroutines offloaded to GPU:
CHANGES.md:  * Add cuda managed attributes to meth_params.F90 using
CHANGES.md:  * Add RealVector typedef so that Real Vectors can be CUDA managed
CHANGES.md:  * Move put_1d_array_on_sphr routine calls to C++ to avoid calls on the GPU
CHANGES.md:  * If USE_CUDA=TRUE, then set USE_GPU_PRAGMA=TRUE and add CUDA to
CHANGES.md:  * Offloaded plotting routines in make_plot_variables.F90 to GPU
Exec/unit_tests/test_eos/MaestroEvolve.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            ParallelFor(nr, [=] AMREX_GPU_DEVICE(int r) {
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            Gpu::synchronize();
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            ParallelFor(nr, [=] AMREX_GPU_DEVICE(int r) {
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            Gpu::synchronize();
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            ParallelFor(nr, [=] AMREX_GPU_DEVICE(int r) {
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            Gpu::synchronize();
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:        ParallelFor(NumSpec, [=] AMREX_GPU_DEVICE(int comp) {
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:        Gpu::synchronize();
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            ParallelFor(nr, [=] AMREX_GPU_DEVICE(int r) {
Exec/unit_tests/test_basestate/MaestroAdvectBase2.cpp:            Gpu::synchronize();
Exec/unit_tests/test_basestate/MaestroInit.cpp:    for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/unit_tests/test_basestate/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_basestate/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_basestate/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/unit_tests/test_basestate/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_basestate/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_react/MaestroInit.cpp:        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/unit_tests/test_react/MaestroHeating.cpp:        for (MFIter mfi(rho_Hext[lev], TilingIfNotGPU()); mfi.isValid();
Exec/unit_tests/test_react/MaestroHeating.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/unit_tests/test_average/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_average/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_average/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/unit_tests/test_average/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_average/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_projection/MaestroEvolve.cpp:                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
Exec/unit_tests/test_diffusion/MaestroInit.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_diffusion/MaestroInit.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/unit_tests/test_diffusion/MaestroDt.cpp:            for (MFIter mfi(uold[lev], TilingIfNotGPU()); mfi.isValid();
Exec/unit_tests/test_diffusion/MaestroEvolve.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/unit_tests/test_diffusion/MaestroEvolve.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/unit_tests/test_advect/MaestroInit.cpp:    for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/unit_tests/test_advect/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_advect/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/unit_tests/test_advect/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/unit_tests/test_new_basestate/main.cpp:        Gpu::ManagedVector<Real> f_base(nlevs * len * ncomp);
Exec/unit_tests/test_new_basestate/main.cpp:        Gpu::synchronize();
Exec/unit_tests/test_new_basestate/main.cpp:        Gpu::synchronize();
Exec/science/flame/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:            Gpu::synchronize();
Exec/science/flame/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:            Gpu::synchronize();
Exec/science/flame/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                Gpu::synchronize();
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                Gpu::synchronize();
Exec/science/flame/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:        Gpu::synchronize();
Exec/science/flame/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:            Gpu::synchronize();
Exec/science/flame/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroBCFill.cpp:        Gpu::synchronize();
Exec/science/flame/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/flame/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/plane_parallel_star/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/plane_parallel_star/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/wdconvect/inputs_files/inputs_3d_regression.2levels:# GPU parameters 
Exec/science/wdconvect/inputs_files/inputs_3d_regression:# GPU parameters 
Exec/science/wdconvect/inputs_files/inputs_omp_3d_regression:# GPU parameters 
Exec/science/wdconvect/inputs_files/inputs_3d_amr_regression:# GPU parameters 
Exec/science/wdconvect/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/wdconvect/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdconvect/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/wdconvect/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/wdconvect/MaestroInitData.cpp:    GpuArray<Real, 27> alpha;
Exec/science/wdconvect/MaestroInitData.cpp:    GpuArray<Real, 27> beta;
Exec/science/wdconvect/MaestroInitData.cpp:    GpuArray<Real, 27> gamma;
Exec/science/wdconvect/MaestroInitData.cpp:    GpuArray<Real, 27> phix;
Exec/science/wdconvect/MaestroInitData.cpp:    GpuArray<Real, 27> phiy;
Exec/science/wdconvect/MaestroInitData.cpp:    GpuArray<Real, 27> phiz;
Exec/science/wdconvect/MaestroInitData.cpp:    GpuArray<Real, 27> normk;
Exec/science/wdconvect/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/wdconvect/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/wdconvect/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/fully_convective_star/MaestroHeating.cpp:        for (MFIter mfi(scal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/fully_convective_star/MaestroHeating.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/fully_convective_star/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/fully_convective_star/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/fully_convective_star/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/fully_convective_star/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/fully_convective_star/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/fully_convective_star/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/fully_convective_star/MaestroInitData.cpp:            GpuArray<Real, 27> alpha;
Exec/science/fully_convective_star/MaestroInitData.cpp:            GpuArray<Real, 27> beta;
Exec/science/fully_convective_star/MaestroInitData.cpp:            GpuArray<Real, 27> gamma;
Exec/science/fully_convective_star/MaestroInitData.cpp:            GpuArray<Real, 27> phix;
Exec/science/fully_convective_star/MaestroInitData.cpp:            GpuArray<Real, 27> phiy;
Exec/science/fully_convective_star/MaestroInitData.cpp:            GpuArray<Real, 27> phiz;
Exec/science/fully_convective_star/MaestroInitData.cpp:            GpuArray<Real, 27> normk;
Exec/science/fully_convective_star/MaestroInitData.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/toy_convect/MaestroHeating.cpp:        for (MFIter mfi(rho_Hext[lev], TilingIfNotGPU()); mfi.isValid();
Exec/science/toy_convect/MaestroHeating.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/toy_convect/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/toy_convect/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/toy_convect/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/xrb_layered/rms_tool/CMakeLists.txt:   if (AMReX_CUDA)
Exec/science/xrb_layered/rms_tool/CMakeLists.txt:      set_source_files_properties(${_exe}.cpp PROPERTIES LANGUAGE CUDA)
Exec/science/xrb_layered/rms_tool/CMakeLists.txt:if (AMReX_CUDA)
Exec/science/xrb_layered/rms_tool/CMakeLists.txt:   set_source_files_properties(AMReX_PPMUtil.cpp PROPERTIES LANGUAGE CUDA)
Exec/science/xrb_layered/rms_tool/GNUmakefile:USE_CUDA = FALSE
Exec/science/xrb_layered/plotfile_derive/main.cpp:        PhysBCFunct<GpuBndryFuncFab<FabFillNoOp>> physbcf
Exec/science/xrb_layered/plotfile_derive/main.cpp:            (vargeom, bcr, GpuBndryFuncFab<FabFillNoOp>(FabFillNoOp{}));
Exec/science/xrb_layered/plotfile_derive/main.cpp:            PhysBCFunct<GpuBndryFuncFab<FabFillNoOp>> cphysbcf
Exec/science/xrb_layered/plotfile_derive/main.cpp:                (cgeom, bcr, GpuBndryFuncFab<FabFillNoOp>(FabFillNoOp{}));
Exec/science/xrb_layered/plotfile_derive/main.cpp:        for (MFIter mfi(temp_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/xrb_layered/plotfile_derive/main.cpp:            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
Exec/science/xrb_layered/MaestroSponge.cpp:        for (MFIter mfi(sponge[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/xrb_layered/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/xrb_layered/MaestroTagging.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/xrb_layered/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/xrb_layered/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Exec/science/urca/analysis/probin.f90:#ifdef AMREX_USE_CUDA
Exec/science/urca/analysis/probin.f90:#ifdef AMREX_USE_CUDA
Exec/science/urca/analysis/GNUmakefile:# we are not using the CUDA stuff
Exec/science/urca/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/urca/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/urca/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/urca/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/urca/MaestroInitData.cpp:    GpuArray<Real, 27> alpha;
Exec/science/urca/MaestroInitData.cpp:    GpuArray<Real, 27> beta;
Exec/science/urca/MaestroInitData.cpp:    GpuArray<Real, 27> gamma;
Exec/science/urca/MaestroInitData.cpp:    GpuArray<Real, 27> phix;
Exec/science/urca/MaestroInitData.cpp:    GpuArray<Real, 27> phiy;
Exec/science/urca/MaestroInitData.cpp:    GpuArray<Real, 27> phiz;
Exec/science/urca/MaestroInitData.cpp:    GpuArray<Real, 27> normk;
Exec/science/urca/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/urca/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/urca/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/ecsn/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/ecsn/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/ecsn/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/ecsn/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/ecsn/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/ecsn/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/ecsn/MaestroInitData.cpp:        GpuArray<Real, 27> alpha;
Exec/science/ecsn/MaestroInitData.cpp:        GpuArray<Real, 27> beta;
Exec/science/ecsn/MaestroInitData.cpp:        GpuArray<Real, 27> gamma;
Exec/science/ecsn/MaestroInitData.cpp:        GpuArray<Real, 27> phix;
Exec/science/ecsn/MaestroInitData.cpp:        GpuArray<Real, 27> phiy;
Exec/science/ecsn/MaestroInitData.cpp:        GpuArray<Real, 27> phiz;
Exec/science/ecsn/MaestroInitData.cpp:        GpuArray<Real, 27> normk;
Exec/science/ecsn/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/xrb_mixed/MaestroSponge.cpp:        for (MFIter mfi(sponge[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/xrb_mixed/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/xrb_mixed/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
Exec/science/rotating_star/ModelParser.H:/// Define Real vector types for CUDA-compatability. If `AMREX_USE_CUDA`, then
Exec/science/rotating_star/ModelParser.H:/// this will be stored in CUDA managed memory.
Exec/science/rotating_star/ModelParser.H:#ifdef AMREX_USE_CUDA
Exec/science/rotating_star/ModelParser.H:typedef amrex::Gpu::ManagedVector<amrex::Real> RealVector;
Exec/science/rotating_star/ModelParser.H:typedef amrex::Gpu::ManagedVector<int> IntVector;
Exec/science/rotating_star/MaestroHeating.cpp:        for (MFIter mfi(scal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/rotating_star/MaestroHeating.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/rotating_star/MaestroTagging.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/rotating_star/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/rotating_star/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/rotating_star/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/rotating_star/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/rotating_star/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/rotating_star/MaestroInitData.cpp:            GpuArray<Real, 27> alpha;
Exec/science/rotating_star/MaestroInitData.cpp:            GpuArray<Real, 27> beta;
Exec/science/rotating_star/MaestroInitData.cpp:            GpuArray<Real, 27> gamma;
Exec/science/rotating_star/MaestroInitData.cpp:            GpuArray<Real, 27> phix;
Exec/science/rotating_star/MaestroInitData.cpp:            GpuArray<Real, 27> phiy;
Exec/science/rotating_star/MaestroInitData.cpp:            GpuArray<Real, 27> phiz;
Exec/science/rotating_star/MaestroInitData.cpp:            GpuArray<Real, 27> normk;
Exec/science/rotating_star/MaestroInitData.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/code_comp/MaestroMakeGrav.cpp:        ParallelFor(nr_lev, [=] AMREX_GPU_DEVICE(long r) {
Exec/science/code_comp/MaestroMakeGrav.cpp:        Gpu::synchronize();
Exec/science/code_comp/MaestroMakeGrav.cpp:        ParallelFor(nr_lev, [=] AMREX_GPU_DEVICE(long r) {
Exec/science/code_comp/MaestroMakeGrav.cpp:        Gpu::synchronize();
Exec/science/code_comp/MaestroHeating.cpp:        for (MFIter mfi(rho_Hext[lev], TilingIfNotGPU()); mfi.isValid();
Exec/science/code_comp/MaestroHeating.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/code_comp/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/code_comp/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/sub_chandra/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/sub_chandra/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/sub_chandra/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/sub_chandra/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/sub_chandra/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/science/sub_chandra/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/science/sub_chandra/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/science/sub_chandra/MaestroInitData.cpp:            GpuArray<Real, 27> alpha;
Exec/science/sub_chandra/MaestroInitData.cpp:            GpuArray<Real, 27> beta;
Exec/science/sub_chandra/MaestroInitData.cpp:            GpuArray<Real, 27> gamma;
Exec/science/sub_chandra/MaestroInitData.cpp:            GpuArray<Real, 27> phix;
Exec/science/sub_chandra/MaestroInitData.cpp:            GpuArray<Real, 27> phiy;
Exec/science/sub_chandra/MaestroInitData.cpp:            GpuArray<Real, 27> phiz;
Exec/science/sub_chandra/MaestroInitData.cpp:            GpuArray<Real, 27> normk;
Exec/science/sub_chandra/MaestroInitData.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/double_bubble/inputs_2d_regression:# GPU parameters 
Exec/test_problems/double_bubble/MaestroInitData.cpp:// device (if USE_CUDA=TRUE)
Exec/test_problems/double_bubble/MaestroInitData.cpp:AMREX_GPU_DEVICE
Exec/test_problems/double_bubble/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_lo,
Exec/test_problems/double_bubble/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_hi, const Real x,
Exec/test_problems/double_bubble/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/double_bubble/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/double_bubble/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/double_bubble/MaestroInitData.cpp:#ifndef AMREX_USE_GPU
Exec/test_problems/double_bubble/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_lo,
Exec/test_problems/double_bubble/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_hi, const Real x,
Exec/test_problems/double_bubble/MaestroInitData.cpp:#ifndef AMREX_USE_GPU
Exec/test_problems/rt/inputs_2d_regression:# GPU parameters 
Exec/test_problems/rt/MaestroInitData.cpp:// device (if USE_CUDA=TRUE)
Exec/test_problems/rt/MaestroInitData.cpp:AMREX_GPU_DEVICE
Exec/test_problems/rt/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_lo,
Exec/test_problems/rt/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_hi, const Real x,
Exec/test_problems/rt/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/rt/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/rt/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/rt/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_lo,
Exec/test_problems/rt/MaestroInitData.cpp:             const GpuArray<Real, AMREX_SPACEDIM> prob_hi, const Real x,
Exec/test_problems/mach_jet/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroBCFill.cpp:            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/mach_jet/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/mach_jet/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/imposed_external_heating/MaestroMakeS.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/test_problems/imposed_external_heating/MaestroMakeS.cpp:            for (MFIter mfi(delta_gamma1_term[lev], TilingIfNotGPU());
Exec/test_problems/imposed_external_heating/MaestroMakeS.cpp:        for (MFIter mfi(S_cc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/test_problems/imposed_external_heating/MaestroMakeS.cpp:        for (MFIter mfi(correction_cc[lev], TilingIfNotGPU()); mfi.isValid();
Exec/test_problems/imposed_external_heating/MaestroMakeS.cpp:        for (MFIter mfi(S_cc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/test_problems/imposed_external_heating/MaestroHeating.cpp:        for (MFIter mfi(rho_Hext[lev], TilingIfNotGPU()); mfi.isValid();
Exec/test_problems/imposed_external_heating/MaestroHeating.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/test_planar_diag/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/test_planar_diag/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/test_planar_diag/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/incomp_shear_jet/inputs_2d_regression:# GPU parameters 
Exec/test_problems/incomp_shear_jet/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/incomp_shear_jet/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/reacting_bubble/inputs_2d_regression:# GPU parameters 
Exec/test_problems/reacting_bubble/inputs_3d_sdc:# GPU parameters
Exec/test_problems/reacting_bubble/inputs_2d_amr_regression:# GPU parameters 
Exec/test_problems/reacting_bubble/inputs_2d_sdc:# GPU parameters
Exec/test_problems/reacting_bubble/inputs_3d_regression:# GPU parameters 
Exec/test_problems/reacting_bubble/inputs_omp_2d_regression:# GPU parameters 
Exec/test_problems/reacting_bubble/inputs_omp_3d_regression:# GPU parameters 
Exec/test_problems/reacting_bubble/inputs_3d_amr_regression:# GPU parameters 
Exec/test_problems/reacting_bubble/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/reacting_bubble/MaestroTagging.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/reacting_bubble/MaestroTagging.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/reacting_bubble/inputs_3d_gpu_regression:# GPU parameters 
Exec/test_problems/reacting_bubble/inputs_2d_regression.2levels:# GPU parameters 
Exec/test_problems/reacting_bubble/scaling/sc20/run_script4.sh:n_mpi=384 # num nodes * 6 gpu per node
Exec/test_problems/reacting_bubble/scaling/sc20/run_script4.sh:n_gpu=1
Exec/test_problems/reacting_bubble/scaling/sc20/run_script4.sh:MAESTROeX_ex=./Maestro3d.pgi.TPROF.MPI.CUDA.ex
Exec/test_problems/reacting_bubble/scaling/sc20/run_script4.sh:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $MAESTROeX_ex $inputs
Exec/test_problems/reacting_bubble/scaling/sc20/plot.py:    n_gpus_per_node = 6
Exec/test_problems/reacting_bubble/scaling/sc20/plot.py:                n_nodes = max(1, n_ranks / n_gpus_per_node)
Exec/test_problems/reacting_bubble/scaling/sc20/README.md:cuda/10.1.243
Exec/test_problems/reacting_bubble/scaling/sc20/README.md:`cd` up two directories, and `make -j16` will then build the executable `Maestro3d.pgi.TPROF.MPI.CUDA.ex`.
Exec/test_problems/reacting_bubble/scaling/sc20/README.md:Then return to this directory and create symbolic links to `../../Maestro3d.pgi.TPROF.MPI.CUDA.ex`,
Exec/test_problems/reacting_bubble/scaling/sc20/run_script.sh:n_mpi=6 # num nodes * 6 gpu per node
Exec/test_problems/reacting_bubble/scaling/sc20/run_script.sh:n_gpu=1
Exec/test_problems/reacting_bubble/scaling/sc20/run_script.sh:MAESTROeX_ex=./Maestro3d.pgi.TPROF.MPI.CUDA.ex
Exec/test_problems/reacting_bubble/scaling/sc20/run_script.sh:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $MAESTROeX_ex $inputs
Exec/test_problems/reacting_bubble/scaling/sc20/run_script2.sh:n_mpi=48 # num nodes * 6 gpu per node
Exec/test_problems/reacting_bubble/scaling/sc20/run_script2.sh:n_gpu=1
Exec/test_problems/reacting_bubble/scaling/sc20/run_script2.sh:MAESTROeX_ex=./Maestro3d.pgi.TPROF.MPI.CUDA.ex
Exec/test_problems/reacting_bubble/scaling/sc20/run_script2.sh:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $MAESTROeX_ex $inputs
Exec/test_problems/reacting_bubble/scaling/sc20/run_script5.sh:n_mpi=750 # num nodes * 6 gpu per node
Exec/test_problems/reacting_bubble/scaling/sc20/run_script5.sh:n_gpu=1
Exec/test_problems/reacting_bubble/scaling/sc20/run_script5.sh:MAESTROeX_ex=./Maestro3d.pgi.TPROF.MPI.CUDA.ex
Exec/test_problems/reacting_bubble/scaling/sc20/run_script5.sh:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $MAESTROeX_ex $inputs
Exec/test_problems/reacting_bubble/scaling/sc20/run_script3.sh:n_mpi=162 # num nodes * 6 gpu per node
Exec/test_problems/reacting_bubble/scaling/sc20/run_script3.sh:n_gpu=1
Exec/test_problems/reacting_bubble/scaling/sc20/run_script3.sh:MAESTROeX_ex=./Maestro3d.pgi.TPROF.MPI.CUDA.ex
Exec/test_problems/reacting_bubble/scaling/sc20/run_script3.sh:jsrun -n $n_mpi -r $n_rs_per_node -c $n_cores -a 1 -g $n_gpu $MAESTROeX_ex $inputs
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:// device (if USE_CUDA=TRUE)
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:AMREX_GPU_DEVICE
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/reacting_bubble/MaestroInitData.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/test_stability/inputs_2d_C_regression:# GPU parameters 
Exec/test_problems/test_stability/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/test_stability/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/test_convect/inputs_2d_regression:# GPU parameters 
Exec/test_problems/test_convect/MaestroHeating.cpp:        for (MFIter mfi(rho_Hext[lev], TilingIfNotGPU()); mfi.isValid();
Exec/test_problems/test_convect/MaestroHeating.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/test_convect/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Exec/test_problems/test_convect/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/test_problems/test_convect/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Exec/Make.Maestro:ifeq ($(USE_GPU),TRUE)
Exec/Make.Maestro:  # when using GPUs. Throw an error to prevent this case.
Exec/Make.Maestro:    $(error OpenMP is not supported by MAESTROeX when building with GPU support)
Exec/Make.Maestro:ifeq ($(USE_CUDA),TRUE)
Exec/Make.Maestro:ifeq ($(USE_GPU_PRAGMA), TRUE)
README.md:of GPUs.
README.md:  For GPU computing, CUDA 10 or later is required.
paper/paper.md:Our approach to parallelization uses a hybrid MPI/OpenMP approach, with increasing GPU support.
sphinx_docs/source/base_state.rst:   for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
sphinx_docs/source/base_state.rst:       ParallelFor(ncells, [=] AMREX_GPU_DEVICE (int r)
sphinx_docs/source/base_state.rst:   Gpu::synchronize();
sphinx_docs/source/base_state.rst:Note here that we **must** call ``Gpu::synchronize()`` after the GPU kernel. For FABS, the ``MFIter`` loop calls ``Gpu::synchronize()`` at the end to ensure that all GPU streams are complete before moving on. However, as we often iterate over the base state outside of an ``MFIter`` loop, it's necessary that we call it explicitly here.
sphinx_docs/source/base_state.rst:        ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE (int r)
sphinx_docs/source/base_state.rst:        Gpu::synchronize();
sphinx_docs/source/base_state.rst:            ParallelFor(hi-lo+1, [=] AMREX_GPU_DEVICE (int j)
sphinx_docs/source/base_state.rst:            Gpu::synchronize();
sphinx_docs/source/getting_started.rst:For running on GPUs, CUDA 11 or later is required (see :ref:`sec:gpu` for
sphinx_docs/source/index.rst:   gpu
sphinx_docs/source/gpu.rst:.. _sec:gpu:
sphinx_docs/source/gpu.rst:GPU
sphinx_docs/source/gpu.rst:In this chapter, we will present the GPU support in MAESTROeX,
sphinx_docs/source/gpu.rst:to GPU, and some basic profiling and debugging options.
sphinx_docs/source/gpu.rst:Note that currently MAESTROeX only supports NVIDIA GPUs.
sphinx_docs/source/gpu.rst:MAESTROeX has only been tested with NVIDIA/CUDA.  In theory AMD/HIP
sphinx_docs/source/gpu.rst:.. _sec:gpubuild:
sphinx_docs/source/gpu.rst:Building GPU Support
sphinx_docs/source/gpu.rst:To build MAESTROeX with GPU support, add the following argument
sphinx_docs/source/gpu.rst:      USE_CUDA := TRUE
sphinx_docs/source/gpu.rst:not compatible with building with CUDA.
sphinx_docs/source/gpu.rst:``USE_CUDA = TRUE`` and ``USE_OMP = TRUE`` will fail to compile.
sphinx_docs/source/gpu.rst:However, you may use MPI with CUDA for additional parallelization.
sphinx_docs/source/gpu.rst:the CUDA Capability using the ``CUDA_ARCH`` flag. The CUDA Capability will
sphinx_docs/source/gpu.rst:depend on the specific GPU hardware you are running on. On a Linux system, the
sphinx_docs/source/gpu.rst:script found in the CUDA samples directory:
sphinx_docs/source/gpu.rst:``/usr/local/cuda/samples/1_Utilities/deviceQuery`` (its exact location may
sphinx_docs/source/gpu.rst:vary depending on where CUDA is installed on your system). The default value of
sphinx_docs/source/gpu.rst:    CUDA_ARCH := 60
sphinx_docs/source/gpu.rst:.. _sec:gpuporting:
sphinx_docs/source/gpu.rst:.. _sec:gpuprofile:
sphinx_docs/source/gpu.rst:Profiling with GPUs
sphinx_docs/source/gpu.rst:NVIDIA's profiler, ``nvprof``, is recommended when profiling for GPUs.
sphinx_docs/source/gpu.rst:It returns data on how long each kernel launch lasted on the GPU,
sphinx_docs/source/gpu.rst:the number of threads and registers used, the occupancy of the GPU
sphinx_docs/source/gpu.rst:use ``nvprof``, see NVIDIA's User's Guide.
sphinx_docs/source/gpu.rst:around an ``MFIter`` loop that encompasses the entire set of GPU launches
sphinx_docs/source/gpu.rst:        // code that runs on the GPU
sphinx_docs/source/gpu.rst:For now, this is the best way to profile GPU codes using the compiler flag ``TINY_PROFILE = TRUE``.
sphinx_docs/source/gpu.rst:.. _sec:gpudebug:
sphinx_docs/source/gpu.rst:Basic GPU Debugging
sphinx_docs/source/gpu.rst:- Turn off GPU offloading for some part of the code with
sphinx_docs/source/gpu.rst:    Gpu::setLaunchRegion(0);
sphinx_docs/source/gpu.rst:    Gpu::setLaunchRegion(1);
sphinx_docs/source/gpu.rst:  a small problem and examine page faults using NVIDIA's visual profiler, `nvvp`
sphinx_docs/source/gpu.rst:- Run under ``cuda-memcheck``
sphinx_docs/source/gpu.rst:- Run under ``cuda-gdb``
sphinx_docs/source/gpu.rst:- Run with ``CUDA_LAUNCH_BLOCKING=1``.  This means that only one
sphinx_docs/source/makefiles.rst:build process: ``USE_MPI``, ``USE_OMP``, ``USE_CUDA`` for the parallelism,
Source/MaestroRhoHT.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRhoHT.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRhoHT.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:        for (MFIter mfi(state[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRhoHT.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRhoHT.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRhoHT.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:        for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRhoHT.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRhoHT.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:        for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMacProj.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:        for (MFIter mfi(solverrhs[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroMacProj.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:        for (MFIter mfi(rhocc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMacProj.cpp:            ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:            ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMacProj.cpp:            ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:        for (MFIter mfi(utilde_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroVelPred.cpp:            Gpu::synchronize();
Source/MaestroVelPred.cpp:            Gpu::synchronize();
Source/MaestroVelPred.cpp:        for (MFIter mfi(utilde_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroVelPred.cpp:            Gpu::synchronize();
Source/MaestroVelPred.cpp:            Gpu::synchronize();
Source/MaestroVelPred.cpp:            Gpu::synchronize();
Source/MaestroVelPred.cpp:    const amrex::GpuArray<Real, AMREX_SPACEDIM>& dx) const {
Source/MaestroVelPred.cpp:    ParallelFor(mxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(mybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    const Box& domainBox, const amrex::GpuArray<Real, AMREX_SPACEDIM>& dx) const {
Source/MaestroVelPred.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    const Box& domainBox, const amrex::GpuArray<Real, AMREX_SPACEDIM>& dx) const {
Source/MaestroVelPred.cpp:    ParallelFor(mxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(mybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(mzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    const amrex::GpuArray<Real, AMREX_SPACEDIM>& dx) const {
Source/MaestroVelPred.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    const amrex::GpuArray<Real, AMREX_SPACEDIM>& dx) const {
Source/MaestroVelPred.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroVelPred.cpp:#ifndef AMREX_USE_GPU
Source/MaestroMakeS.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeS.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:            for (MFIter mfi(delta_gamma1_term[lev], TilingIfNotGPU());
Source/MaestroMakeS.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:        for (MFIter mfi(S_cc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeS.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:        for (MFIter mfi(correction_cc[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroMakeS.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:        for (MFIter mfi(S_cc[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeS.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:                                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:                        tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeS.cpp:                        tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroAdvection.cpp:        for (MFIter mfi(stateold[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroAdvection.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroAdvection.cpp:                            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroAdvection.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroAdvection.cpp:        for (MFIter mfi(force[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroAdvection.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroAdvection.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:#ifdef AMREX_USE_GPU
Source/MaestroPlot.cpp:        // same type of GPU.
Source/MaestroPlot.cpp:        jobInfoFile << "GPU Information:       "
Source/MaestroPlot.cpp:        jobInfoFile << "GPU model name: " << Gpu::Device::deviceName() << "\n";
Source/MaestroPlot.cpp:        jobInfoFile << "Number of GPUs used: " << Gpu::Device::numDevicesUsed()
Source/MaestroPlot.cpp:            for (MFIter mfi(vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:            for (MFIter mfi(vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:        for (MFIter mfi(vel[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:        for (MFIter mfi(state[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:        for (MFIter mfi(vel_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:        for (MFIter mfi(state[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:        for (MFIter mfi(state[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:            for (MFIter mfi(divw0[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroPlot.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:            for (MFIter mfi(divw0[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroPlot.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:        for (MFIter mfi(pidivu[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPlot.cpp:        for (MFIter mfi(abar[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroPlot.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroGamma.cpp:        for (MFIter mfi(gamma1[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroGamma.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroCelltoEdge.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroCelltoEdge.cpp:            Gpu::synchronize();
Source/MaestroBurner.cpp:        for (MFIter mfi(s_in[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroBurner.cpp:            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) -> ReduceTuple
Source/MaestroBurner.cpp:#ifndef AMREX_USE_GPU
Source/MaestroMakePsi.cpp:    ParallelFor(npts, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakePsi.cpp:    Gpu::synchronize();
Source/MaestroMakePsi.cpp:    ParallelFor(npts, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakePsi.cpp:    Gpu::synchronize();
Source/MaestroDiag.cpp:        for (MFIter mfi(s_in[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroDiag.cpp:            // GPU probably more trouble than it's worth.
Source/MaestroAverage.cpp:            for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroAverage.cpp:#ifdef AMREX_USE_GPU
Source/MaestroAverage.cpp:                // Atomic::Add is non-deterministic on the GPU. If this flag is true,
Source/MaestroAverage.cpp:                    launched = !Gpu::notInLaunchRegion();
Source/MaestroAverage.cpp:                    // turn off GPU
Source/MaestroAverage.cpp:                    if (launched) Gpu::setLaunchRegion(false);
Source/MaestroAverage.cpp:                ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroAverage.cpp:#ifdef AMREX_USE_GPU
Source/MaestroAverage.cpp:                    // turn GPU back on
Source/MaestroAverage.cpp:                    if (launched) Gpu::setLaunchRegion(true);
Source/MaestroAverage.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroAverage.cpp:                Gpu::synchronize();
Source/MaestroAverage.cpp:            for (MFIter mfi(phi[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroAverage.cpp:                ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroAverage.cpp:            ParallelFor(nr_irreg + 1, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroAverage.cpp:            Gpu::synchronize();
Source/MaestroAverage.cpp:            for (MFIter mfi(phi_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroAverage.cpp:#ifdef AMREX_USE_GPU
Source/MaestroAverage.cpp:                // Atomic::Add is non-deterministic on the GPU. If this flag is true,
Source/MaestroAverage.cpp:                    launched = !Gpu::notInLaunchRegion();
Source/MaestroAverage.cpp:                    // turn off GPU
Source/MaestroAverage.cpp:                    if (launched) Gpu::setLaunchRegion(false);
Source/MaestroAverage.cpp:                ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroAverage.cpp:#ifdef AMREX_USE_GPU
Source/MaestroAverage.cpp:                    // turn GPU back on
Source/MaestroAverage.cpp:                    if (launched) Gpu::setLaunchRegion(true);
Source/MaestroAverage.cpp:        ParallelFor(nrf, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroAverage.cpp:        Gpu::synchronize();
Source/MaestroAverage.cpp:        ParallelFor(nrf, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroAverage.cpp:        Gpu::synchronize();
Source/MaestroMakew0.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:            Gpu::synchronize();
Source/MaestroMakew0.cpp:                    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:                    Gpu::synchronize();
Source/MaestroMakew0.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:            Gpu::synchronize();
Source/MaestroMakew0.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:            Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    // need to synchronize gpu values with updated host values
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(nr_finest + 1, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:                ParallelFor(hi - lo, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:            Gpu::synchronize();
Source/MaestroMakew0.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakew0.cpp:            Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    amrex::ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) noexcept {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    // need to synchronize gpu values with updated host values
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    // need to synchronize gpu values with updated host values
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroMakew0.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakew0.cpp:    Gpu::synchronize();
Source/MaestroInit.cpp:        for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroInit.cpp:        for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeEdgeState.cpp:        ParallelFor(nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakeEdgeState.cpp:        Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:        ParallelFor(nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakeEdgeState.cpp:        Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:        ParallelFor(nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakeEdgeState.cpp:        Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:    ParallelFor(nr_fine + 1, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakeEdgeState.cpp:    Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakeEdgeState.cpp:                Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakeEdgeState.cpp:                Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:                ParallelFor(hi - lo + 2, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakeEdgeState.cpp:                Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakeEdgeState.cpp:                Gpu::synchronize();
Source/MaestroMakeEdgeState.cpp:            ParallelFor(hi - lo + 2, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakeEdgeState.cpp:            Gpu::synchronize();
Source/MaestroSlopes.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:            bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:            bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:            bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:            bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:            bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroSlopes.cpp:            bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroMakeEdgeScalars.cpp:        for (MFIter mfi(scal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeEdgeScalars.cpp:                Gpu::synchronize();
Source/MaestroMakeEdgeScalars.cpp:                Gpu::synchronize();
Source/MaestroMakeEdgeScalars.cpp:            for (MFIter mfi(scal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeEdgeScalars.cpp:            for (MFIter mfi(scal_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeEdgeScalars.cpp:    const amrex::GpuArray<Real, AMREX_SPACEDIM> dx, int comp, int bccomp,
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(mxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(mybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    const amrex::GpuArray<Real, AMREX_SPACEDIM> dx, int comp, int bccomp,
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:                       const GpuArray<Real, AMREX_SPACEDIM> dx) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    const Vector<BCRec>& bcs, const amrex::GpuArray<Real, AMREX_SPACEDIM> dx,
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(mxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(mybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(mzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    const amrex::GpuArray<Real, AMREX_SPACEDIM> dx, int comp, int bccomp,
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(imhbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    const Vector<BCRec>& bcs, const amrex::GpuArray<Real, AMREX_SPACEDIM> dx,
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEdgeScalars.cpp:    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T& operator()(
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T& operator()(
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T& operator[](int i) noexcept {
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const T& operator[](
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE T* ptr(int lev, int i = 0,
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE T* dataPtr() noexcept {
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE int nLevels() const noexcept {
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE int length() const noexcept {
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE int nComp() const noexcept {
Source/BaseState.H:AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE BaseStateArray<T> makeBaseStateArray(
Source/BaseState.H:    BaseState(const amrex::Gpu::ManagedVector<T>& src, const int num_levs,
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE int nLevels() const noexcept {
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE int length() const noexcept {
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE int nComp() const noexcept {
Source/BaseState.H:    void copy(const amrex::Gpu::ManagedVector<T>& src);
Source/BaseState.H:    void toVector(amrex::Gpu::ManagedVector<T>& vec) const;
Source/BaseState.H:    AMREX_GPU_HOST_DEVICE [[nodiscard]] AMREX_FORCE_INLINE T* dataPtr() const noexcept {
Source/BaseState.H:    amrex::Gpu::ManagedVector<T> base_data;
Source/BaseState.H:    amrex::Gpu::streamSynchronize();
Source/BaseState.H:BaseState<T>::BaseState(const amrex::Gpu::ManagedVector<T>& src,
Source/BaseState.H:    amrex::Gpu::streamSynchronize();
Source/BaseState.H:    amrex::Gpu::streamSynchronize();
Source/BaseState.H:    amrex::Gpu::streamSynchronize();
Source/BaseState.H:    amrex::Gpu::streamSynchronize();
Source/BaseState.H:        amrex::Gpu::streamSynchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:void BaseState<T>::copy(const amrex::Gpu::ManagedVector<T>& src) {
Source/BaseState.H:void BaseState<T>::toVector(amrex::Gpu::ManagedVector<T>& vec) const {
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/BaseState.H:    amrex::Gpu::synchronize();
Source/MaestroMakeBeta0.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakeBeta0.cpp:                Gpu::synchronize();
Source/MaestroMakeBeta0.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroMakeBeta0.cpp:                Gpu::synchronize();
Source/MaestroMakeUtrans.cpp:        for (MFIter mfi(utilde[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeUtrans.cpp:            ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeUtrans.cpp:        for (MFIter mfi(utilde[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeUtrans.cpp:            ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeUtrans.cpp:        for (MFIter mfi(utilde[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeUtrans.cpp:            Gpu::synchronize();
Source/MaestroMakeUtrans.cpp:            ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeUtrans.cpp:        for (MFIter mfi(utilde[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeUtrans.cpp:            ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeUtrans.cpp:        for (MFIter mfi(utilde[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeUtrans.cpp:            ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroIntra.cpp:        for (MFIter mfi(scal1[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroIntra.cpp:            ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroSponge.cpp:        for (MFIter mfi(sponge[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroSponge.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroSponge.cpp:                            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(
Source/MaestroSponge.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeGrav.cpp:                ParallelFor(nr_lev, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakeGrav.cpp:                Gpu::synchronize();
Source/MaestroMakeGrav.cpp:                ParallelFor(nr_lev, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakeGrav.cpp:                Gpu::synchronize();
Source/param/parse_maestro_params.py:    pf.write("#include <AMReX_Gpu.H>\n")
Source/param/_cpp_parameters:# category: GPU
Source/param/_cpp_parameters:# The nodal solve is non-deterministic on the GPU. Should it instead be run
Source/MaestroMakeEta.cpp:#ifdef AMREX_USE_GPU
Source/MaestroMakeEta.cpp:        launched = !Gpu::notInLaunchRegion();
Source/MaestroMakeEta.cpp:        // turn off GPU
Source/MaestroMakeEta.cpp:        if (launched) Gpu::setLaunchRegion(false);
Source/MaestroMakeEta.cpp:        for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeEta.cpp:            ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEta.cpp:                ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int n) {
Source/MaestroMakeEta.cpp:                Gpu::synchronize();
Source/MaestroMakeEta.cpp:            ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEta.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEta.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakeEta.cpp:            Gpu::synchronize();
Source/MaestroMakeEta.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroMakeEta.cpp:            Gpu::synchronize();
Source/MaestroMakeEta.cpp:#ifdef AMREX_USE_GPU
Source/MaestroMakeEta.cpp:        // turn GPU back on
Source/MaestroMakeEta.cpp:        if (launched) Gpu::setLaunchRegion(true);
Source/MaestroMakeEta.cpp:        for (MFIter mfi(scal_old[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroMakeEta.cpp:            ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeEta.cpp:    ParallelFor(nrf, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroMakeEta.cpp:    Gpu::synchronize();
Source/MaestroMakeEta.cpp:        for (MFIter mfi(scal_old[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroMakeEta.cpp:            ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/BaseStateGeometry.H:              amrex::GpuArray<amrex::Real, 3>& center);
Source/MaestroBaseStateGeometry.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dx_fine,
Source/MaestroBaseStateGeometry.cpp:    const GpuArray<Real, AMREX_SPACEDIM> dx_lev) {
Source/MaestroBaseStateGeometry.cpp:        ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:        for (MFIter mfi(state[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeFlux.cpp:                xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
Source/MaestroMakeFlux.cpp:            ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroMakeFlux.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:        for (MFIter mfi(state[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroMakeFlux.cpp:            ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:            ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroMakeFlux.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroForce.cpp:            ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroForce.cpp:            Gpu::synchronize();
Source/MaestroForce.cpp:        for (MFIter mfi(vel_force_cart[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroForce.cpp:            // offload to GPU
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroForce.cpp:        for (MFIter mfi(scal_force[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroForce.cpp:            // offload to GPU
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroForce.cpp:        for (MFIter mfi(scal_force[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroForce.cpp:        for (MFIter mfi(temp_force[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroForce.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroDt.cpp:#ifdef AMREX_USE_GPU
Source/MaestroDt.cpp:#include <cuda_runtime_api.h>
Source/MaestroDt.cpp:            for (MFIter mfi(uold[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroDt.cpp:                        tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroDt.cpp:                        tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroDt.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroDt.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroDt.cpp:                        tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroDt.cpp:                        tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroDt.cpp:                        tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroDt.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroDt.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroDt.cpp:            for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroDt.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroDt.cpp:                        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroDt.cpp:                        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroDt.cpp:        ParallelFor(base_geom.nr_fine - 2, [=] AMREX_GPU_DEVICE(int i) {
Source/MaestroDt.cpp:        ParallelFor(base_geom.nr_fine - 2, [=] AMREX_GPU_DEVICE(int i) {
Source/MaestroDt.cpp:    Gpu::synchronize();
Source/MaestroHeating.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroAdvectBase.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroAdvectBase.cpp:            amrex::Gpu::synchronize();
Source/MaestroAdvectBase.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroAdvectBase.cpp:            amrex::Gpu::synchronize();
Source/MaestroAdvectBase.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroAdvectBase.cpp:    Gpu::synchronize();
Source/MaestroAdvectBase.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroAdvectBase.cpp:    Gpu::synchronize();
Source/MaestroAdvectBase.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroAdvectBase.cpp:            Gpu::synchronize();
Source/MaestroAdvectBase.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int j) {
Source/MaestroAdvectBase.cpp:            Gpu::synchronize();
Source/MaestroAdvectBase.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroAdvectBase.cpp:    Gpu::synchronize();
Source/MaestroAdvectBase.cpp:    ParallelFor(base_geom.nr_fine, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroAdvectBase.cpp:    Gpu::synchronize();
Source/MaestroFill3dData.cpp:    for (MFIter mfi(s0_cart, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroFill3dData.cpp:            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroFill3dData.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroFill3dData.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroFill3dData.cpp:                    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j,
Source/MaestroFill3dData.cpp:AMREX_GPU_DEVICE
Source/MaestroFill3dData.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:#ifdef AMREX_USE_GPU
Source/MaestroFill3dData.cpp:        // FillBoundary can be non-deterministic on the GPU for non-cell
Source/MaestroFill3dData.cpp:            launched = !Gpu::notInLaunchRegion();
Source/MaestroFill3dData.cpp:            // turn off GPU
Source/MaestroFill3dData.cpp:            if (launched) Gpu::setLaunchRegion(false);
Source/MaestroFill3dData.cpp:#ifdef AMREX_USE_GPU
Source/MaestroFill3dData.cpp:            // turn GPU back on
Source/MaestroFill3dData.cpp:            if (launched) Gpu::setLaunchRegion(true);
Source/MaestroFill3dData.cpp:            for (MFIter mfi(w0cart_mf, TilingIfNotGPU()); mfi.isValid();
Source/MaestroFill3dData.cpp:                ParallelFor(gntbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:        for (MFIter mfi(w0cart_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroFill3dData.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:        for (MFIter mfi(s0_cart[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroFill3dData.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                    ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:            for (MFIter mfi(normal[lev], TilingIfNotGPU()); mfi.isValid();
Source/MaestroFill3dData.cpp:                ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:        for (MFIter mfi(scc_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroFill3dData.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:                ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFill3dData.cpp:        for (MFIter mfi(cc_to_r, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroTagging.cpp:    ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroRegrid.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/MaestroRegrid.cpp:    for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRegrid.cpp:#pragma omp parallel if (Gpu::notInLaunchRegion())
Source/MaestroRegrid.cpp:        for (MFIter mfi(sold[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroRegrid.cpp:                [=] AMREX_GPU_DEVICE(int r) { state_temp(0, r) = base(0, r); });
Source/MaestroRegrid.cpp:    Gpu::synchronize();
Source/MaestroRegrid.cpp:            ParallelFor(nrn, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroRegrid.cpp:            ParallelFor(nrn, [=] AMREX_GPU_DEVICE(int r) {
Source/MaestroRegrid.cpp:        Gpu::synchronize();
Source/MaestroRegrid.cpp:            ParallelFor(hi - lo + 1, [=] AMREX_GPU_DEVICE(int k) {
Source/MaestroRegrid.cpp:            Gpu::synchronize();
Source/MaestroPPM.cpp:                  const amrex::GpuArray<Real, AMREX_SPACEDIM> dx,
Source/MaestroPPM.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPPM.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPPM.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPPM.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPPM.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroPPM.cpp:        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroThermal.cpp:        for (MFIter mfi(scal[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroThermal.cpp:            ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/Maestro.H:/// Define Real vector types for GPU-compatability. If `AMREX_USE_GPU`, then
Source/Maestro.H:/// this will be stored in GPU managed memory.
Source/Maestro.H:#ifdef AMREX_USE_GPU
Source/Maestro.H:using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;
Source/Maestro.H:using IntVector = amrex::Gpu::ManagedVector<int>;
Source/Maestro.H:// function called on GPU only
Source/Maestro.H:AMREX_GPU_DEVICE
Source/Maestro.H:        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_fine,
Source/Maestro.H:        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev);
Source/Maestro.H:                               const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
Source/Maestro.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
Source/Maestro.H:                  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx);
Source/Maestro.H:                               const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
Source/Maestro.H:                                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
Source/Maestro.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
Source/Maestro.H:             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx, const bool is_umac,
Source/Maestro.H:                          const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx) const;
Source/Maestro.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx) const;
Source/Maestro.H:                          const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx) const;
Source/Maestro.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx) const;
Source/Maestro.H:                           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx) const;
Source/Maestro.H:    /// saves on some flops and data movement (GPU)
Source/Maestro.H:    amrex::GpuArray<amrex::Real, 3> center;
Source/MaestroFillData.cpp:            ParallelFor(xbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFillData.cpp:            Gpu::synchronize();
Source/MaestroFillData.cpp:            ParallelFor(ybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroFillData.cpp:            Gpu::synchronize();
Source/MaestroFillData.cpp:            ParallelFor(zbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroBCFill.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroBCFill.cpp:                Gpu::synchronize();
Source/MaestroBCFill.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroBCFill.cpp:                Gpu::synchronize();
Source/MaestroBCFill.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroBCFill.cpp:                Gpu::synchronize();
Source/MaestroBCFill.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroBCFill.cpp:                Gpu::synchronize();
Source/MaestroBCFill.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroBCFill.cpp:                Gpu::synchronize();
Source/MaestroBCFill.cpp:                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroBCFill.cpp:                Gpu::synchronize();
Source/MaestroInitData.cpp:                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroInitData.cpp:    ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroInitData.cpp:                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
Source/MaestroInitData.cpp:    for (MFIter mfi(scal, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroInitData.cpp:        ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroNodalProj.cpp:#ifdef AMREX_USE_GPU
Source/MaestroNodalProj.cpp:    // compRHS is non-deterministic on the GPU. If this flag is true, run
Source/MaestroNodalProj.cpp:        launched = !Gpu::notInLaunchRegion();
Source/MaestroNodalProj.cpp:        // turn off GPU
Source/MaestroNodalProj.cpp:        if (launched) Gpu::setLaunchRegion(false);
Source/MaestroNodalProj.cpp:#ifdef AMREX_USE_GPU
Source/MaestroNodalProj.cpp:        // turn GPU back on
Source/MaestroNodalProj.cpp:        if (launched) Gpu::setLaunchRegion(true);
Source/MaestroNodalProj.cpp:#ifdef AMREX_USE_GPU
Source/MaestroNodalProj.cpp:        launched = !Gpu::notInLaunchRegion();
Source/MaestroNodalProj.cpp:        // turn off GPU
Source/MaestroNodalProj.cpp:        if (launched) Gpu::setLaunchRegion(false);
Source/MaestroNodalProj.cpp:#ifdef AMREX_USE_GPU
Source/MaestroNodalProj.cpp:        // turn GPU back on
Source/MaestroNodalProj.cpp:        if (launched) Gpu::setLaunchRegion(true);
Source/MaestroNodalProj.cpp:        for (MFIter mfi(gphi_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroNodalProj.cpp:            ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroNodalProj.cpp:            ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/MaestroNodalProj.cpp:        for (MFIter mfi(snew_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
Source/MaestroNodalProj.cpp:            ParallelFor(tilebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
Source/BaseStateGeometry.cpp:                             GpuArray<Real, 3>& center) {

```
